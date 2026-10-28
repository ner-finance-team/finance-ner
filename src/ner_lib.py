# -*- coding: utf-8 -*-
"""
NER 실험 공용 코드. 모든 노트북·서버 스크립트가 이 파일을 쓴다(같은 정렬·복원 규칙). 채점은 src/evaluate.py(팀 공용 채점기).

run_experiment(cfg) 가 저장하는 것 (out_dir = reports/results)
  {exp}.json                 설정 + 에폭별 dev 점수(history) + dev/test 점수 + 시간  ← 시각화의 기본 자료
  runs.jsonl                 실행마다 한 줄씩 쌓이는 기록(단계·모델·크기·점수·시간·실행자)
  preds/{exp}_{dev|test}.jsonl   예측 (팀 공용 형식: {"id", "pred": [[s,e,tag],...]})
  scores/{exp}_{dev|test}.json   공용 채점기 결과(태그별·카테고리별·오답 유형 수)
  errors/{exp}_{dev|test}.jsonl  오답 목록(문장마다 경계/유형/오탐/미탐과 글자)
이어서 하기
  - 실험 단위: {exp}.json 이 있으면 건너뜀.
  - 에폭 단위: 매 에폭 체크포인트를 ckpt_dir/{exp} 에 1개만 유지. 끊긴 뒤 같은 설정으로 다시 부르면 마지막 에폭부터 이어서 학습.
    끝나면 체크포인트는 지운다(keep_ckpt=True면 남김).
최종 가중치 (10/23 수정, 끌 수 없음)
  - 학습이 끝나면 예측·채점보다 먼저 최종 모델(가중치+토크나이저)을 model_dir/{exp} 에 저장하고, 다시 읽어 원본과 같은지 확인한다.
  - model_dir 을 안 주면 ckpt_dir 옆의 ner_models 폴더(Colab: 내 드라이브/ner_models, 서버: 프로젝트/ner_models).
  - 확인에 실패하면 체크포인트를 지우지 않고 에러로 멈춘다(가중치가 하나도 안 남는 일이 없게).
  - verify_saved_models() : 이번 세션에 학습·건너뛴 실험의 가중치가 모두 있는지 확인(런타임 반납 전에 부른다).

cfg 키: exp, model, train, dev, test, out_dir, ckpt_dir, head("softmax"|"crf"), loss("ce"|"weighted"), pretok(None|"morph"),
        max_len, lr, epochs, batch, grad_accum, seed, warmup, weight_decay, eval_each_epoch, keep_ckpt, stage, member
"""
import collections, gc, json, math, os, shutil, sys, time, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import TAGS, span_prf, span_prf_by, score as score_split   # 채점 규칙은 evaluate.py 한 곳에서만

LABELS = ["O"] + [f"{p}-{t}" for t in TAGS for p in ("B", "I")]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

DEFAULTS = dict(head="softmax", loss="ce", max_len=512, lr=5e-5, epochs=3, batch=16, grad_accum=1, seed=42,
                warmup=0.1, weight_decay=0.01, eval_each_epoch=True, keep_ckpt=False, stage="", member="", pretok=None,
                save_steps=0,   # save_steps>0: 그 스텝마다 저장(에폭 중간에 끊겨도 이어서). 0이면 에폭마다
                model_dir=None)  # 최종 모델 저장 위치. None이면 ckpt_dir 옆 ner_models (저장 자체는 항상 한다)
SESSION_EXPS = []   # 이번 세션에서 run_experiment 로 다룬 (실험 이름, 가중치 폴더) — verify_saved_models 가 확인

# ───────────────────────── 데이터 ─────────────────────────
def load_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit: break
    return rows

def char_tags(text, spans):
    ct = [None] * len(text)
    for s, e, tag in spans:
        for i in range(s, e):
            ct[i] = (tag, i == s)
    return ct

_KIWI = None
def morph_split(text):
    """형태소 경계에 공백을 끼워 넣는다(글자는 그대로). 반환: (새 문장, 새 문장의 각 글자 → 원문 위치, 끼운 공백은 -1).
    예) '중국인들의' → '중국인 들 의'. 어떤 형태소도 걸쳐 있지 않은 위치에만 끼운다('생긴'=생기+ㄴ 같은 축약은 나누지 않음).
    AI Hub ELECTRA는 사전학습 때 형태소로 나눈 문장을 썼다(학습 코드 processing_mor_corpus.py)."""
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi
        _KIWI = Kiwi()
    toks = _KIWI.tokenize(text)
    inside = set()
    for t in toks:
        inside.update(range(t.start + 1, t.start + t.len))
    cuts = {t.start for t in toks if 0 < t.start < len(text) and t.start not in inside
            and not text[t.start - 1].isspace() and not text[t.start].isspace()}
    out, m = [], []
    for i, ch in enumerate(text):
        if i in cuts:
            out.append(" "); m.append(-1)
        out.append(ch); m.append(i)
    return "".join(out), m

def tokenize_offsets(text, tokenizer, max_len, pretok=None):
    """토큰화 + 원문 기준 글자 위치(offset). pretok='morph'면 형태소로 나눈 문장을 토큰화하고 위치를 원문으로 되돌린다."""
    if pretok != "morph":
        return tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True, return_token_type_ids=False)
    new, m = morph_split(text)
    enc = tokenizer(new, truncation=True, max_length=max_len, return_offsets_mapping=True, return_token_type_ids=False)
    offs = []
    for s, e in enc["offset_mapping"]:
        if s == e:
            offs.append((0, 0)); continue
        idx = [m[i] for i in range(s, e) if m[i] >= 0]
        offs.append((idx[0], idx[-1] + 1) if idx else (0, 0))
    enc["offset_mapping"] = offs
    return enc

def encode(example, tokenizer, max_len, pretok=None):
    """글자 span → 토큰별 BIO 라벨. 특수 토큰은 -100. offset은 항상 원문 기준이라 복원·채점이 같다."""
    enc = tokenize_offsets(example["text"], tokenizer, max_len, pretok)
    ct = char_tags(example["text"], example["spans"])
    labels = []
    for s, e in enc["offset_mapping"]:
        if s == e:
            labels.append(-100); continue
        inside = [ct[i] for i in range(s, e) if ct[i] is not None]
        if not inside:                  labels.append(label2id["O"])
        elif any(b for _, b in inside): labels.append(label2id["B-" + inside[0][0]])
        else:                           labels.append(label2id["I-" + inside[0][0]])
    enc["labels"] = labels
    return enc

def decode_spans(label_ids, offsets):
    """토큰별 BIO → 글자 span. I가 앞 개체와 이어지지 않으면 새 개체로 본다."""
    spans, cur = [], None
    for lid, (s, e) in zip(label_ids, offsets):
        if s == e or lid == -100: continue
        lab = id2label[int(lid)]
        if lab == "O":
            if cur: spans.append(tuple(cur)); cur = None
            continue
        p, tag = lab.split("-", 1)
        if p == "B" or cur is None or cur[2] != tag:
            if cur: spans.append(tuple(cur))
            cur = [s, e, tag]
        else:
            cur[1] = e
    if cur: spans.append(tuple(cur))
    return spans

def upper_bound(rows, tokenizer, max_len, pretok=None):
    """정답을 BIO로 바꿨다 되돌렸을 때의 F1 = 이 토크나이저로 낼 수 있는 최고 점수."""
    golds, preds = [], []
    for r in rows:
        enc = encode(r, tokenizer, max_len, pretok)
        golds.append(r["spans"]); preds.append(decode_spans(enc["labels"], enc["offset_mapping"]))
    return span_prf(golds, preds)["ALL"]

# ───────────────────────── AI Hub 모델 ─────────────────────────
def _dec(n):
    try: return n.encode("cp437").decode("cp949")
    except Exception: return n

def prepare_aihub_model(outer_zip, which="1세부", dest="/content/aihub_models"):
    """'AI모델 및 소스코드.zip' 안의 model/{which}_언어모델.zip 을 풀어 폴더 경로를 돌려준다."""
    target = os.path.join(dest, which)
    if os.path.exists(os.path.join(target, "pytorch_model.bin")):
        return target
    os.makedirs(target, exist_ok=True)
    with zipfile.ZipFile(outer_zip) as oz:
        inner = next(n for n in oz.namelist() if _dec(n).endswith(f"{which}_언어모델.zip"))
        with oz.open(inner) as fh, zipfile.ZipFile(fh) as iz:
            for n in iz.namelist():
                with iz.open(n) as src, open(os.path.join(target, os.path.basename(_dec(n))), "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 24)
    return target

# ───────────────────────── 모델 ─────────────────────────
def build_crf_model(model_name, num_labels):
    """사전학습 본체 + 선형층 + CRF. CRF는 꼬리표끼리의 이동 점수(예: O→I는 낮게)를 배운다."""
    import torch, torch.nn as nn
    from transformers import AutoModel
    from torchcrf import CRF

    class EncoderCRF(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
            self.crf = CRF(num_labels, batch_first=True)

        @staticmethod
        def crf_mask(attention_mask):
            # [CLS](0번)는 잘라내고 마지막 실제 토큰([SEP])도 제외 → 앞쪽 연속 1, 뒤쪽 0
            m = attention_mask[:, 1:].clone().bool()
            lengths = attention_mask.sum(1)
            m[torch.arange(m.size(0)), (lengths - 2).clamp(min=0)] = False
            m[:, 0] = True
            return m

        def emissions(self, input_ids, attention_mask):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return self.classifier(self.dropout(h))[:, 1:].float()

        def forward(self, input_ids, attention_mask, labels=None, **kw):
            em = self.emissions(input_ids, attention_mask)
            loss = None
            if labels is not None:
                tags = labels[:, 1:].clone(); tags[tags < 0] = 0
                loss = -self.crf(em, tags, mask=self.crf_mask(attention_mask), reduction="mean")
            return {"loss": loss, "logits": em}

        def decode(self, input_ids, attention_mask):
            paths = self.crf.decode(self.emissions(input_ids, attention_mask), mask=self.crf_mask(attention_mask))
            return [[-100] + p for p in paths]          # [CLS] 자리 복원

    return EncoderCRF()

def class_weights(train_rows, tokenizer, max_len, pretok=None):
    """꼬리표 가중치 = (전체 토큰/그 꼬리표 토큰)^0.5, 최대 10배, O는 1."""
    import torch
    c = collections.Counter()
    for r in train_rows:
        for l in encode(r, tokenizer, max_len, pretok)["labels"]:
            if l != -100: c[l] += 1
    tot = sum(c.values())
    w = [1.0] + [math.sqrt(tot / max(c[i], 1)) for i in range(1, len(LABELS))]
    m = max(w[1:]); w = [1.0] + [x / m * 10 for x in w[1:]]
    return torch.tensor(w, dtype=torch.float)

# ───────────────────────── 학습·평가 ─────────────────────────
def _to_dataset_stream(rows, tokenizer, max_len, pretok=None):
    """학습용(위치 정보 불필요): 문장마다 작은 정수 배열(int32/int8)로만 보관 → 파이썬 리스트보다 메모리가 훨씬 적다.
    (수십만 문장을 메모리 9GB 서버에서 학습하기 위함. 위치 정보는 학습에 필요 없어 버린다.)"""
    import numpy as np
    from datasets import Dataset
    ids, mask, labs = [], [], []
    for r in rows:
        e = encode(r, tokenizer, max_len, pretok)
        ids.append(np.asarray(e["input_ids"], dtype=np.int32))
        mask.append(np.asarray(e["attention_mask"], dtype=np.int8))
        labs.append(np.asarray(e["labels"], dtype=np.int16))
    return Dataset.from_dict({"input_ids": ids, "attention_mask": mask, "labels": labs})

def _to_dataset(rows, tokenizer, max_len, pretok=None):
    from datasets import Dataset
    encs = [encode(r, tokenizer, max_len, pretok) for r in rows]
    ds = Dataset.from_list([{"input_ids": e["input_ids"], "attention_mask": e["attention_mask"], "labels": e["labels"]} for e in encs])
    return ds, [e["offset_mapping"] for e in encs]

def predict_label_ids(model, ds, collator, batch=64, head="softmax"):
    """예측. 길이순으로 묶어 패딩을 줄이고 원래 순서로 되돌린다."""
    import torch
    was_training = model.training; model.eval(); dev = next(model.parameters()).device
    order = sorted(range(len(ds)), key=lambda i: len(ds[i]["input_ids"]))
    out = [None] * len(ds)
    with torch.no_grad():
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            b = collator([ds[i] for i in idx])
            ii, am = b["input_ids"].to(dev), b["attention_mask"].to(dev)
            if head == "crf":
                preds = model.decode(ii, am)
            else:
                with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    preds = model(input_ids=ii, attention_mask=am).logits.argmax(-1).tolist()
            for j, i in enumerate(idx):
                out[i] = list(preds[j])[:len(ds[i]["input_ids"])]
    if was_training: model.train()
    return out

def save_preds(path, rows, preds):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"id": r["id"], "category": r.get("category"), "pred": [list(x) for x in p]}, ensure_ascii=False) + "\n")

def _last_checkpoint(d):
    if not os.path.isdir(d): return None
    cks = [os.path.join(d, x) for x in os.listdir(d) if x.startswith("checkpoint-")]
    return max(cks, key=lambda p: int(p.rsplit("-", 1)[1])) if cks else None

# ───────────────────────── 최종 가중치 저장·확인 ─────────────────────────
def _ckpt_root(cfg):
    return cfg.get("ckpt_dir") or os.path.join(os.path.dirname(cfg["out_dir"].rstrip("/\\")), "checkpoints")

def model_path(cfg):
    """실험의 최종 가중치 폴더. model_dir 을 안 주면 ckpt_dir 옆 ner_models/{exp}."""
    root = cfg.get("model_dir") or os.path.join(os.path.dirname(os.path.abspath(_ckpt_root(cfg)).rstrip("/\\")), "ner_models")
    return os.path.join(root, cfg["exp"])

def has_weights(mdir):
    return any(os.path.exists(os.path.join(mdir, f)) and os.path.getsize(os.path.join(mdir, f)) > 1_000_000
               for f in ("model.safetensors", "pytorch_model.bin", "crf_model_state.pt", "model_state.pt"))

def _save_and_verify(model, tok, trainer, cfg, mdir, verbose=True):
    """최종 모델을 저장하고 파일을 다시 읽어 메모리의 가중치와 텐서 하나하나가 같은지 확인. 다르면 예외."""
    import torch
    os.makedirs(mdir, exist_ok=True)
    if cfg["head"] == "crf":
        f = os.path.join(mdir, "crf_model_state.pt")
        torch.save(model.state_dict(), f)
        json.dump({"base": cfg["model"], "num_labels": len(LABELS), "labels": LABELS, "max_len": cfg["max_len"], "pretok": cfg["pretok"]},
                  open(os.path.join(mdir, "crf_config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        saved = torch.load(f, map_location="cpu")
    else:
        trainer.save_model(mdir)
        f = os.path.join(mdir, "model.safetensors")
        if os.path.exists(f):
            from safetensors.torch import load_file
            saved = load_file(f)
        else:
            f = os.path.join(mdir, "pytorch_model.bin"); saved = torch.load(f, map_location="cpu")
    tok.save_pretrained(mdir)
    live = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    bad = [k for k, v in saved.items() if k not in live or not torch.equal(v.cpu(), live[k])]
    if not saved or bad or len(saved) < 0.95 * len(live):
        raise RuntimeError(f"가중치 저장 확인 실패({mdir}): 저장 {len(saved)}개 / 모델 {len(live)}개, 다른 텐서 {bad[:3]}. 체크포인트는 지우지 않았다.")
    size = os.path.getsize(f) / 1e6
    if verbose: print(f"최종 가중치 저장·확인 완료: {mdir} ({os.path.basename(f)} {size:.0f}MB, 텐서 {len(saved)}개 일치)")
    return {"path": mdir, "file": os.path.basename(f), "mb": round(size, 1), "tensors": len(saved)}

def verify_saved_models(exps=None, raise_error=True):
    """이번 세션(또는 주어진 실험)의 최종 가중치가 모두 있는지 확인. 런타임 반납·서버 정리 전에 부른다."""
    items = exps if exps is not None else SESSION_EXPS
    missing = [(e, d) for e, d in items if not has_weights(d)]
    for e, d in items: print(("있음  " if (e, d) not in missing else "없음!! ") + e, "→", d)
    if missing and raise_error:
        raise RuntimeError(f"가중치가 없는 실험 {len(missing)}개: {[e for e, _ in missing]} — 런타임을 반납하지 마세요")
    return not missing

def run_experiment(cfg, skip_if_done=True, verbose=True):
    cfg = {**DEFAULTS, **cfg}
    out_dir = cfg["out_dir"]; exp = cfg["exp"]
    path = os.path.join(out_dir, f"{exp}.json")
    mdir = model_path(cfg)
    if (exp, mdir) not in SESSION_EXPS: SESSION_EXPS.append((exp, mdir))
    if skip_if_done and os.path.exists(path):
        if verbose:
            print("이미 있음, 건너뜀:", path)
            if not has_weights(mdir):
                print(f"  ※ 경고: 이 실험의 최종 가중치가 {mdir} 에 없습니다(10/23 이전 코드로 학습). "
                      f"가중치가 필요하면 {path} 를 다른 이름으로 옮긴 뒤 다시 실행하세요.")
        return json.load(open(path, encoding="utf-8"))
    import torch
    from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification,
                              Trainer, TrainerCallback, TrainingArguments, set_seed)
    set_seed(cfg["seed"])
    train_rows, dev_rows, test_rows = load_jsonl(cfg["train"]), load_jsonl(cfg["dev"]), load_jsonl(cfg["test"])
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    t0 = time.time()
    pt = cfg["pretok"]
    train_ds = _to_dataset_stream(train_rows, tok, cfg["max_len"], pt) if len(train_rows) > 100000 \
        else _to_dataset(train_rows, tok, cfg["max_len"], pt)[0]
    dev_ds, dev_offs = _to_dataset(dev_rows, tok, cfg["max_len"], pt)
    test_ds, test_offs = _to_dataset(test_rows, tok, cfg["max_len"], pt)
    encode_sec = round(time.time() - t0, 1)
    collator = DataCollatorForTokenClassification(tok)

    if cfg["head"] == "crf":
        model = build_crf_model(cfg["model"], len(LABELS))
    else:
        model = AutoModelForTokenClassification.from_pretrained(cfg["model"], num_labels=len(LABELS),
                                                                id2label=id2label, label2id=label2id)
    n_params = sum(p.numel() for p in model.parameters())

    ckpt = os.path.join(_ckpt_root(cfg), exp)
    os.makedirs(ckpt, exist_ok=True)
    hist_path = os.path.join(ckpt, "history.json")
    history = json.load(open(hist_path, encoding="utf-8")) if os.path.exists(hist_path) else []

    class DevEval(TrainerCallback):
        """에폭이 끝날 때마다 dev 점수를 재서 기록 → 에폭 수를 고르는 근거(그래프) + 끊겨도 기록 유지."""
        def on_epoch_end(self, args, state, control, model=None, **kw):
            if not cfg["eval_each_epoch"]: return
            ids = predict_label_ids(model, dev_ds, collator, head=cfg["head"])
            m = span_prf([r["spans"] for r in dev_rows], [decode_spans(i, o) for i, o in zip(ids, dev_offs)])
            losses = [h["loss"] for h in state.log_history if "loss" in h]
            history.append({"epoch": round(state.epoch, 2), "step": state.global_step,
                            "train_loss": round(float(losses[-1]), 4) if losses else None,
                            "dev_f1": m["ALL"]["f1"], "dev_macro_f1": m["MACRO_F1"],
                            "dev_precision": m["ALL"]["precision"], "dev_recall": m["ALL"]["recall"]})
            json.dump(history, open(hist_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            if verbose: print(f"  [epoch {state.epoch:.0f}] dev F1 {m['ALL']['f1']} macro {m['MACRO_F1']}")

    cuda = torch.cuda.is_available(); bf16 = cuda and torch.cuda.is_bf16_supported()
    steps = math.ceil(len(train_ds) / (cfg["batch"] * cfg["grad_accum"])) * cfg["epochs"]
    args = TrainingArguments(
        output_dir=ckpt, learning_rate=cfg["lr"], num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch"], gradient_accumulation_steps=cfg["grad_accum"],
        warmup_steps=int(cfg["warmup"] * steps), weight_decay=cfg["weight_decay"],
        bf16=bf16, fp16=cuda and not bf16, eval_strategy="no",
        **({"save_strategy": "steps", "save_steps": cfg["save_steps"], "save_total_limit": 2} if cfg["save_steps"] else
           {"save_strategy": "epoch", "save_total_limit": 1}),
        logging_steps=max(1, steps // 50), report_to="none", seed=cfg["seed"], remove_unused_columns=False)

    trainer_cls = Trainer
    if cfg["loss"] == "weighted":
        w = class_weights(train_rows, tok, cfg["max_len"], pt)
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                labels = inputs.pop("labels"); out = model(**inputs)
                lossf = torch.nn.CrossEntropyLoss(weight=w.to(out.logits.device), ignore_index=-100)
                loss = lossf(out.logits.float().view(-1, len(LABELS)), labels.view(-1))
                return (loss, out) if return_outputs else loss
        trainer_cls = WeightedTrainer
    trainer = trainer_cls(model=model, args=args, train_dataset=train_ds, data_collator=collator, callbacks=[DevEval()])

    resume = _last_checkpoint(ckpt)
    if resume and verbose: print("체크포인트에서 이어서 학습:", resume)
    if resume:   # 이어서 할 때는 체크포인트 뒤에 기록된 에폭을 지운다(같은 에폭이 두 번 기록되지 않게)
        done_epoch = json.load(open(os.path.join(resume, "trainer_state.json"), encoding="utf-8")).get("epoch", 0)
        history[:] = [h for h in history if h["epoch"] <= done_epoch + 1e-6]
    t0 = time.time(); tr = trainer.train(resume_from_checkpoint=resume); train_sec = round(time.time() - t0, 1)
    saved = _save_and_verify(model, tok, trainer, cfg, mdir, verbose)   # 예측·채점보다 먼저: 뒤에서 끊겨도 가중치는 남음

    t0 = time.time()
    dev_pred = [decode_spans(i, o) for i, o in zip(predict_label_ids(model, dev_ds, collator, head=cfg["head"]), dev_offs)]
    test_pred = [decode_spans(i, o) for i, o in zip(predict_label_ids(model, test_ds, collator, head=cfg["head"]), test_offs)]
    infer_sec = round(time.time() - t0, 1)

    meta = {"member": cfg["member"], "stage": cfg["stage"], "model": cfg["model"]}
    for split, rows, pred in (("dev", dev_rows, dev_pred), ("test", test_rows, test_pred)):
        save_preds(os.path.join(out_dir, "preds", f"{exp}_{split}.jsonl"), rows, pred)
        score_split(rows, {r["id"]: set(p) for r, p in zip(rows, pred)}, exp, split, out_dir, meta)

    result = {"exp": exp, "config": {k: v for k, v in cfg.items() if k not in ("out_dir", "ckpt_dir")},
              "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "gpu": torch.cuda.get_device_name(0) if cuda else "cpu", "params": n_params,
              "n_train": len(train_rows), "n_dev": len(dev_rows), "n_test": len(test_rows),
              "encode_sec": encode_sec, "train_sec": train_sec, "infer_sec": infer_sec, "resumed_from": resume,
              "train_samples_per_second": round(tr.metrics.get("train_samples_per_second", 0), 2),
              "train_loss": round(tr.metrics.get("train_loss", 0), 4), "history": history,
              "upper_bound_test": upper_bound(test_rows, tok, cfg["max_len"], pt),
              "dev": span_prf([r["spans"] for r in dev_rows], dev_pred),
              "test": span_prf([r["spans"] for r in test_rows], test_pred),
              "test_by_category": span_prf_by(test_rows, test_pred), "model_path": mdir, "weights": saved}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(result, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"exp": exp, "stage": cfg["stage"], "member": cfg["member"], "model": cfg["model"], "pretok": pt,
                            "head": cfg["head"], "loss": cfg["loss"], "n_train": len(train_rows), "epochs": cfg["epochs"],
                            "lr": cfg["lr"], "seed": cfg["seed"], "dev_f1": result["dev"]["ALL"]["f1"],
                            "test_f1": result["test"]["ALL"]["f1"], "test_macro_f1": result["test"]["MACRO_F1"],
                            "train_sec": train_sec, "gpu": result["gpu"], "finished_at": result["finished_at"]},
                           ensure_ascii=False) + "\n")
    json.dump({k: v for k, v in result.items() if k in ("exp", "config", "dev", "test", "finished_at", "weights")},
              open(os.path.join(mdir, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not has_weights(mdir):   # 한 번 더: 가중치 파일이 없으면 체크포인트도 지우지 않는다
        raise RuntimeError(f"최종 가중치가 없어 체크포인트를 지우지 않습니다: {mdir}")
    if not cfg["keep_ckpt"]:
        shutil.rmtree(ckpt, ignore_errors=True)
    if verbose:
        print(f"[{exp}] dev F1 {result['dev']['ALL']['f1']} | test F1 {result['test']['ALL']['f1']} "
              f"| 학습 {train_sec}s ({result['train_samples_per_second']} 문장/s)")
    del trainer, model; gc.collect()
    if cuda: torch.cuda.empty_cache()
    return result

# ───────────────────────── 결과 모으기 ─────────────────────────
def load_results(out_dir):
    res = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".json"):
            try:
                r = json.load(open(os.path.join(out_dir, fn), encoding="utf-8"))
                if "exp" in r: res.append(r)
            except Exception: pass
    return res

def results_table(results):
    import pandas as pd
    rows = []
    for r in results:
        if "test" not in r: continue
        c = r.get("config", {})
        row = {"exp": r["exp"], "stage": c.get("stage", ""), "member": c.get("member", ""), "pretok": c.get("pretok") or "-",
               "model": c.get("model", r.get("model", "")), "head": c.get("head", "-"), "loss": c.get("loss", "-"),
               "n_train": r.get("n_train"), "epochs": c.get("epochs"), "lr": c.get("lr"), "seed": c.get("seed"),
               "dev_F1": r["dev"]["ALL"]["f1"] if "dev" in r else None, "test_F1": r["test"]["ALL"]["f1"],
               "test_P": r["test"]["ALL"]["precision"], "test_R": r["test"]["ALL"]["recall"],
               "macro_F1": r["test"].get("MACRO_F1"), "train_min": round(r.get("train_sec", 0) / 60, 1)}
        for t in TAGS: row[f"F1_{t}"] = r["test"][t]["f1"]
        rows.append(row)
    return pd.DataFrame(rows)
