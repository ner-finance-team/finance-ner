# -*- coding: utf-8 -*-
"""
[fin_ft_cont.py] fin_ft.py 복사본 + --from_fin(금융 모델 이어 학습) 옵션. 10/27 03 → 5에폭 이어 학습용. 나머지는 fin_ft.py 와 같음.
금융 NER(8태그) 파인튜닝·채점 — 인코더 모델(roberta·mmBERT·1단계 모델) 공용. 팀원 분배 실행용.

  python fin_ft.py --init <모델폴더> --exp <이름> --data <train/dev/test 폴더> --head crf --pretok none --out_root /content/out --mirror <드라이브폴더>

- 데이터: {data}/train.jsonl · dev.jsonl · test.jsonl, 한 줄 = {"text", "spans": [[시작, 끝, 태그], ...]}. 태그는 학습 데이터에서 읽는다.
- 출발 모델(--init): 사전학습(MLM)·1단계 NER 어느 쪽이든 본체(인코더)만 가져오고, 금융 분류 층(+CRF)은 새로 만든다.
- --head crf : 선형층 + CRF(태그 연결 규칙 학습) / softmax : 선형층만.  --pretok morph : 형태소 경계에 공백(1단계 ner_lib.morph_split 과 같은 규칙) 후 토큰화, 위치는 원문으로 되돌림(mmBERT용).
- 라벨: 토큰마다 BIO(개체 첫 토큰 B, 나머지 I). 특수 토큰·글자 없는 토큰은 학습·예측에서 뺀다(CRF는 실제 토큰만 모아서 계산).
- 학습: 에폭마다 dev 채점(span F1: 시작·끝·태그가 정답과 정확히 같아야 맞음)·체크포인트(로컬 + --mirror 드라이브). 끊기면 같은 명령 → 이어서.
- 끝: {out_root}/models/{exp} 저장 → 다시 읽어 텐서 전부 일치 확인 → dev·test 채점(test는 마지막에 한 번) → results/{exp}.json, preds/
- --eval_file 로 추가 채점 파일(예: Fable 라벨 크롤링 350문장)을 줄 수 있다.
"""
import argparse, collections, json, math, os, shutil, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fin_score as S   # 공통 채점(모든 금융 노트북 같은 규칙)

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True); ap.add_argument("--exp", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--head", default="crf", choices=["crf", "softmax"]); ap.add_argument("--pretok", default="none", choices=["none", "morph"])
ap.add_argument("--out_root", required=True); ap.add_argument("--mirror", default="")
ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--crf_lr", type=float, default=1e-3, help="CRF 전이표 학습률(본체보다 크게)")
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--grad_accum", type=int, default=1)
ap.add_argument("--max_len", type=int, default=256); ap.add_argument("--warmup", type=float, default=0.1); ap.add_argument("--weight_decay", type=float, default=0.01)
ap.add_argument("--train_n", type=int, default=0, help="학습 문장 수 제한(0=전부, 학습량 곡선·시험용)")
ap.add_argument("--eval_file", default="", help="추가 채점 파일(jsonl)"); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--workers", type=int, default=2); ap.add_argument("--max_steps", type=int, default=-1, help="시험용")
ap.add_argument("--from_fin", action="store_true", help="--init 이 금융 파인튜닝이 끝난 모델 폴더(fin_ner_config.json 있음)일 때: 분류층·CRF 까지 통째로 읽어 이어 학습")
ap.add_argument("--save_steps", type=int, default=2000, help="이 스텝마다 + 에폭 끝마다 체크포인트(로컬 + 드라이브)")
a = ap.parse_args()
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_pt_utils import LengthGroupedSampler
torch.manual_seed(a.seed)
log = lambda *x: print(time.strftime("%m-%d %H:%M:%S"), *x, flush=True)
O = lambda *p: os.path.join(a.out_root, *p)
ckpt_dir, mdir, res_path = O("checkpoints", a.exp), O("models", a.exp), O("results", f"{a.exp}.json")
for d in (ckpt_dir, os.path.dirname(res_path), O("preds")): os.makedirs(d, exist_ok=True)
if os.path.exists(res_path) and json.load(open(res_path, encoding="utf-8")).get("done"): log("이미 끝난 실험:", res_path); sys.exit(0)
mirror = os.path.join(a.mirror, "checkpoints", a.exp) if a.mirror else ""
def ckpts(d): return sorted([x for x in os.listdir(d) if x.startswith("checkpoint-")], key=lambda x: int(x.split("-")[1])) if d and os.path.isdir(d) else []
if mirror and not ckpts(ckpt_dir) and ckpts(mirror):
    last = ckpts(mirror)[-1]; shutil.copytree(os.path.join(mirror, last), os.path.join(ckpt_dir, last)); log(f"드라이브 체크포인트 {last} 를 가져와 이어서 학습")

# ---------- 데이터 ----------
def load(p, n=0):
    rs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rs[:n] if n else rs
train, dev, test = load(os.path.join(a.data, "train.jsonl"), a.train_n), load(os.path.join(a.data, "dev.jsonl")), load(os.path.join(a.data, "test.jsonl"))
TAGS = S.TAGS   # GUIDE v1.2 고정 8태그
LABELS = ["O"] + [f"{p}-{t}" for t in TAGS for p in ("B", "I")]; L2I = {l: i for i, l in enumerate(LABELS)}
log(f"학습 {len(train):,} · dev {len(dev):,} · test {len(test):,} · 태그 {TAGS} · 머리 {a.head} · 입력 {a.pretok}")

tok = AutoTokenizer.from_pretrained(a.init)
_kiwi = None
def morph_split(text):
    """ner_lib.morph_split 과 같은 규칙: 형태소 경계(어떤 형태소도 걸치지 않은 곳)에 공백. 반환 (새 문장, 새 글자→원문 위치, 공백은 -1)"""
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi; _kiwi = Kiwi()
    toks = _kiwi.tokenize(text); inside = set()
    for t in toks: inside.update(range(t.start + 1, t.start + t.len))
    cuts = {t.start for t in toks if 0 < t.start < len(text) and t.start not in inside and not text[t.start - 1].isspace() and not text[t.start].isspace()}
    out, m = [], []
    for i, ch in enumerate(text):
        if i in cuts: out.append(" "); m.append(-1)
        out.append(ch); m.append(i)
    return "".join(out), m

def encode(r):
    text = r["text"]
    if a.pretok == "morph":
        new, m = morph_split(text)
        enc = tok(new, truncation=True, max_length=a.max_len, return_offsets_mapping=True)
        offs = []
        for s, e in enc["offset_mapping"]:
            idx = [m[i] for i in range(s, e) if m[i] >= 0] if s != e else []
            offs.append((idx[0], idx[-1] + 1) if idx else (0, 0))
    else:
        enc = tok(text, truncation=True, max_length=a.max_len, return_offsets_mapping=True); offs = [tuple(o) for o in enc["offset_mapping"]]
    ct = [None] * len(text)
    for s, e, t in r.get("spans", []):
        for i in range(s, e): ct[i] = (t, i == s)
    labs = []
    for s, e in offs:
        if s == e: labs.append(-100); continue
        inside = [ct[i] for i in range(s, min(e, len(text))) if ct[i] is not None]
        labs.append(L2I["O"] if not inside else L2I[("B-" if any(b for _, b in inside) else "I-") + inside[0][0]] if f"B-{inside[0][0]}" in L2I else L2I["O"])
    # 예측 위치용: 토큰 위치의 앞뒤 공백을 뗀다 (mmBERT 등 SentencePiece 토크나이저는 앞 공백을 토큰 위치에 포함 → 개체 앞에 공백이 붙던 버그, 10/27 수정). 학습 라벨은 위와 같음
    doffs = []
    for s, e in offs:
        while s < e and text[s].isspace(): s += 1
        while e > s and text[e - 1].isspace(): e -= 1
        doffs.append((s, e))
    return {"input_ids": enc["input_ids"], "labels": labs, "offsets": doffs}

t0 = time.time(); E_train = [encode(r) for r in train]; E_dev = [encode(r) for r in dev]; E_test = [encode(r) for r in test]
log(f"토큰화 {time.time()-t0:.0f}초 · 학습 문장당 토큰 {sum(len(e['input_ids']) for e in E_train)/len(E_train):.1f} · 잘린 문장 {sum(len(e['input_ids'])>=a.max_len for e in E_train)}")

# ---------- 모델: 본체 + 선형층 (+CRF) ----------
class FinNER(nn.Module):
    def __init__(self, init, n, head):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(init); self.head = head
        self.dropout = nn.Dropout(0.1); self.classifier = nn.Linear(self.encoder.config.hidden_size, n)
        if head == "crf":
            from torchcrf import CRF; self.crf = CRF(n, batch_first=True)
    def emissions(self, input_ids, attention_mask):
        return self.classifier(self.dropout(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)).float()
    @staticmethod
    def pack(x, valid):
        """실제 글자 토큰(valid)만 앞으로 모은다 → CRF가 특수 토큰 위치와 상관없이 동작"""
        idx = [v.nonzero().squeeze(-1) for v in valid]; T = max(1, max(len(i) for i in idx))
        out = x.new_zeros((x.size(0), T) + x.shape[2:]); m = torch.zeros(x.size(0), T, dtype=torch.bool, device=x.device)
        for b, i in enumerate(idx): out[b, :len(i)] = x[b, i]; m[b, :len(i)] = True
        m[:, 0] = True
        return out, m, idx
    def forward(self, input_ids, attention_mask, labels=None, valid=None, **kw):
        em = self.emissions(input_ids, attention_mask); loss = None
        if labels is not None:
            if self.head == "crf":
                e, m, _ = self.pack(em, valid); y, _, _ = self.pack(labels.clamp(min=0), valid)
                loss = -self.crf(e, y, mask=m, reduction="mean")
            else:
                loss = nn.functional.cross_entropy(em.view(-1, em.size(-1)), labels.view(-1), ignore_index=-100)
        return {"loss": loss, "logits": em}
    @torch.no_grad()
    def predict(self, input_ids, attention_mask, valid):
        em = self.emissions(input_ids, attention_mask); out = []
        if self.head == "crf":
            e, m, idx = self.pack(em, valid); paths = self.crf.decode(e, mask=m)
            for b, (p, i) in enumerate(zip(paths, idx)):
                full = [-100] * input_ids.size(1)
                for k, pos in enumerate(i.tolist()): full[pos] = p[k]
                out.append(full)
        else:
            out = em.argmax(-1).tolist()
        return out

def collate(b):
    T = max(len(x["input_ids"]) for x in b); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([x["input_ids"] + [pad] * (T - len(x["input_ids"])) for x in b])
    att = torch.tensor([[1] * len(x["input_ids"]) + [0] * (T - len(x["input_ids"])) for x in b])
    lab = torch.tensor([x["labels"] + [-100] * (T - len(x["labels"])) for x in b])
    return {"input_ids": ids, "attention_mask": att, "labels": lab, "valid": lab != -100}

def decode(lab_ids, offs):
    spans, cur = [], None
    for lid, (s, e) in zip(lab_ids, offs):
        if s == e or lid < 0: continue
        l = LABELS[lid]
        if l == "O":
            if cur: spans.append(tuple(cur)); cur = None
            continue
        p, t = l.split("-", 1)
        if p == "B" or cur is None or cur[2] != t:
            if cur: spans.append(tuple(cur))
            cur = [s, e, t]
        else: cur[1] = e
    if cur: spans.append(tuple(cur))
    return spans

prf = S.score

def evaluate(model, E, rows, batch=64):
    model.eval(); dev_ = next(model.parameters()).device; order = sorted(range(len(E)), key=lambda i: len(E[i]["input_ids"])); preds = [None] * len(E)
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]; b = collate([E[i] for i in idx]); b = {x: v.to(dev_) for x, v in b.items()}
        with torch.autocast(device_type=dev_.type, dtype=torch.bfloat16, enabled=dev_.type == "cuda"):
            out = model.predict(b["input_ids"], b["attention_mask"], b["valid"])
        for j, i in enumerate(idx): preds[i] = decode(out[j][:len(E[i]["input_ids"])], E[i]["offsets"])
    model.train()
    return prf([r["spans"] for r in rows], preds), preds

model = FinNER(a.init, len(LABELS), a.head)
if a.from_fin:   # 이어 학습: 금융 파인튜닝이 끝난 모델(본체 + 분류층 + CRF)을 통째로 읽는다
    fc = json.load(open(os.path.join(a.init, "fin_ner_config.json"), encoding="utf-8"))
    assert fc["labels"] == LABELS and fc["head"] == a.head and fc["pretok"] == a.pretok, f"이어 학습 설정이 저장된 모델과 다름: {fc}"
    from safetensors.torch import load_file as _lf
    model.load_state_dict(_lf(os.path.join(a.init, "model.safetensors")), strict=True)
    log(f"이어 학습: {a.init} 의 금융 모델 전체(본체·분류층·CRF)를 읽음 (원래 출발 모델 {fc.get('init')})")
for p_ in model.parameters(): p_.data = p_.data.contiguous()   # KoELECTRA 등: 메모리에 흩어진 가중치가 있으면 체크포인트 저장이 거부됨
n_params = sum(p.numel() for p in model.parameters())
class DS(torch.utils.data.Dataset):
    def __init__(s, E): s.E = E
    def __len__(s): return len(s.E)
    def __getitem__(s, i): return {"input_ids": s.E[i]["input_ids"], "labels": s.E[i]["labels"]}
cuda = torch.cuda.is_available(); bf16 = cuda and torch.cuda.is_bf16_supported()
steps_ep = math.ceil(len(E_train) / (a.batch * a.grad_accum)); total = a.max_steps if a.max_steps > 0 else steps_ep * a.epochs
lengths = [len(e["input_ids"]) for e in E_train]

class FTTrainer(Trainer):
    def _get_train_sampler(self, *x, **k):
        g = torch.Generator(); g.manual_seed(a.seed)
        return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps, lengths=lengths, generator=g)
    def create_optimizer(self, *x, **k):
        if self.optimizer is None:
            crf = [p for n, p in self.model.named_parameters() if n.startswith("crf.")]
            nd = [p for n, p in self.model.named_parameters() if not n.startswith("crf.") and (n.endswith("bias") or "norm" in n.lower())]
            de = [p for n, p in self.model.named_parameters() if not n.startswith("crf.") and not (n.endswith("bias") or "norm" in n.lower())]
            groups = [{"params": de, "weight_decay": a.weight_decay}, {"params": nd, "weight_decay": 0.0}]
            if crf: groups.append({"params": crf, "weight_decay": 0.0, "lr": a.crf_lr})
            self.optimizer = torch.optim.AdamW(groups, lr=a.lr)
        return self.optimizer

history = []
_hist_from = None
class EpochEval(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, **kw):
        s, _ = evaluate(model, E_dev, dev); history.append({"epoch": round(state.epoch, 2), "step": state.global_step, "dev_f1": s["ALL"]["f1"], "dev_macro_f1": s["MACRO_F1"]})
        log(f"에폭 {state.epoch:.0f} dev F1 {s['ALL']['f1']} · macro {s['MACRO_F1']}"); control.should_save = True
    def on_save(self, args, state, control, **kw):
        nm = f"checkpoint-{state.global_step}"; src = os.path.join(ckpt_dir, nm)
        if not os.path.isdir(src): return
        json.dump(history, open(os.path.join(src, "epoch_history.json"), "w", encoding="utf-8"), ensure_ascii=False)
        if not mirror: return
        os.makedirs(mirror, exist_ok=True); tmp = os.path.join(mirror, "_tmp_" + nm)
        shutil.rmtree(tmp, ignore_errors=True); shutil.copytree(src, tmp); dst = os.path.join(mirror, nm); shutil.rmtree(dst, ignore_errors=True); os.replace(tmp, dst)
        for old in ckpts(mirror)[:-1]: shutil.rmtree(os.path.join(mirror, old), ignore_errors=True)
        log(f"체크포인트 {nm} → 드라이브 복사")
    def on_log(self, args, state, control, logs=None, **kw):
        if state.global_step and state.global_step % (args.logging_steps * 20) == 0 and hasattr(self, "t"):
            done = state.global_step - self.s0; el = time.time() - self.t
            if done > 0: log(f"스텝 {state.global_step:,}/{state.max_steps:,} · 남은 시간 약 {(state.max_steps-state.global_step)*el/done/3600:.1f}시간")
    def on_train_begin(self, args, state, control, **kw): self.t = time.time(); self.s0 = state.global_step

targs = TrainingArguments(output_dir=ckpt_dir, learning_rate=a.lr, num_train_epochs=a.epochs, max_steps=a.max_steps,
    per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum, warmup_steps=int(a.warmup * total),
    lr_scheduler_type="linear", bf16=bf16, save_strategy="steps", save_steps=min(a.save_steps, steps_ep), save_total_limit=1, eval_strategy="no",
    logging_steps=max(1, min(100, total // 100)), report_to="none", seed=a.seed, remove_unused_columns=False,
    dataloader_num_workers=a.workers if cuda else 0, use_cpu=not cuda)
trainer = FTTrainer(model=model, args=targs, train_dataset=DS(E_train), data_collator=collate, callbacks=[EpochEval()])
resume = os.path.join(ckpt_dir, ckpts(ckpt_dir)[-1]) if ckpts(ckpt_dir) else None
if resume and os.path.exists(os.path.join(resume, "epoch_history.json")): history += json.load(open(os.path.join(resume, "epoch_history.json"), encoding="utf-8"))
log(f"파라미터 {n_params:,} · 에폭당 {steps_ep:,}스텝 · 총 {total:,} · lr {a.lr} · " + (f"이어서 {resume}" if resume else "처음부터"))
T0 = time.time(); trainer.train(resume_from_checkpoint=resume); train_sec = round(time.time() - T0, 1)

# ---------- 저장 → 다시 읽어 확인 ----------
os.makedirs(mdir, exist_ok=True)
from safetensors.torch import save_file, load_file
sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
save_file(sd, os.path.join(mdir, "model.safetensors")); tok.save_pretrained(mdir); model.encoder.config.save_pretrained(mdir)
cfg = {"labels": LABELS, "tags": TAGS, "head": a.head, "pretok": a.pretok, "init": a.init, "max_len": a.max_len, "from_fin": a.from_fin}
json.dump(cfg, open(os.path.join(mdir, "fin_ner_config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
re_sd = load_file(os.path.join(mdir, "model.safetensors"))
diff = [k for k in sd if k not in re_sd or not torch.equal(sd[k].float(), re_sd[k].float())]
if diff: raise SystemExit(f"저장 확인 실패: 다른 텐서 {len(diff)} — 체크포인트는 {ckpt_dir} 에 남아 있음")
log(f"저장 확인: 텐서 {len(sd)}개 일치 → {mdir}")

# ---------- 채점: dev, test(한 번), 추가 파일 ----------
res = {"exp": a.exp, "init": a.init, "head": a.head, "pretok": a.pretok, "tags": TAGS, "n_train": len(train), "params": n_params,
       "config": vars(a), "gpu": torch.cuda.get_device_name(0) if cuda else "cpu", "train_sec": train_sec, "history": history}
sets = [("dev", E_dev, dev), ("test", E_test, test)]
t2k = os.path.join(a.data, "test_2000.jsonl")
if os.path.exists(t2k): r2 = load(t2k); sets.append(("test_2000", [encode(r) for r in r2], r2))   # 모든 모델(gpt-oss 포함) 공통 비교용
if a.eval_file: r3 = load(a.eval_file); sets.append(("extra", [encode(r) for r in r3], r3))
for nm, E, rows in sets:
    s, p = evaluate(model, E, rows); kinds = S.save_preds(O("preds", f"{a.exp}_{nm}.jsonl"), rows, p)
    res[nm] = {**s, "n": len(rows), "error_types": kinds}
    log(f"{nm} {len(rows)}문장 F1 {s['ALL']['f1']} (P {s['ALL']['precision']} R {s['ALL']['recall']}) · macro {s['MACRO_F1']} · 오답 {kinds}")
if a.eval_file: res["extra"]["file"] = a.eval_file
res["log_history"] = trainer.state.log_history   # 학습 손실·학습률·grad norm 전부(그림용)
res.update(model_path=mdir, weights_verified=len(sd), done=True)
json.dump(res, open(res_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log(f"끝 · dev F1 {res['dev']['ALL']['f1']} · test F1 {res['test']['ALL']['f1']} · 학습 {train_sec/60:.1f}분")
print("VERIFY_OK")
