# -*- coding: utf-8 -*-
"""gpt-oss-20b(+경제 사전학습 LoRA) 금융 NER 파인튜닝·채점 — 생성 방식(문장 → "개체=태그 | ..." 로 답하기).

  python ft_gptoss_ner.py --train train.jsonl --dev dev.jsonl --dapt models/dapt_gptoss_econ500k --out_root /content/out --mirror <드라이브폴더> --name ner_v1

데이터 형식(한 줄 = 한 문장): {"text": "...", "spans": [[시작, 끝, "태그"], ...]}  (글자 위치, 끝은 포함 안 함)
순서
 1) 본체(MXFP4 → bf16) 불러오기 → 경제 사전학습 LoRA(--dapt)를 본체에 합침 → 새 LoRA(r=--lora_r)를 붙여 학습
 2) 학습 문장: "### 문장\n{text}\n### 개체\n" 뒤에 정답 "{개체}=TAG | ..." (없으면 "없음"), 손실은 정답 부분에만
    같은 글자가 문장에 여러 번 나오면 두 번째부터 "{개체}@2=TAG" 처럼 몇 번째인지 붙인다(라벨 키트의 @n 규칙)
 3) dev 앞 --eval_n 문장을 답하게 해(그리디) 원문 위치로 되돌리고 span F1(정확히 같은 시작·끝·태그) 채점
    문장에 없는 개체를 쓰면 위치를 못 찾으므로 오탐(fp)으로 센다
 4) {out_root}/models/{name} 에 새 LoRA 저장 → 다시 읽어 텐서 일치 확인 → results/{name}.json, preds/{name}_dev.jsonl
체크포인트는 --save_steps 마다 로컬 + --mirror(드라이브)에 복사, 로컬에 없으면 드라이브에서 가져와 이어서 학습.
"""
import argparse, collections, json, math, os, re, shutil, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fin_score as S   # 공통 채점(모든 금융 노트북 같은 규칙)

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True); ap.add_argument("--dev", required=True)
ap.add_argument("--test2k", default="", help="공통 비교용 test_2000.jsonl (모든 모델이 같은 2,000문장으로 채점)")
ap.add_argument("--eval_file", default="", help="추가 채점 파일(예: Fable 라벨 크롤링 문장)")
ap.add_argument("--dapt", default="", help="경제 사전학습 LoRA 폴더(없으면 원래 gpt-oss)")
ap.add_argument("--model", default="openai/gpt-oss-20b")
ap.add_argument("--out_root", required=True); ap.add_argument("--mirror", default=""); ap.add_argument("--name", required=True)
ap.add_argument("--train_n", type=int, default=0, help="학습 문장 수 제한(0=전부)")
ap.add_argument("--eval_n", type=int, default=1000, help="채점할 dev 문장 수(생성이 느려서 앞 N개)")
ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--micro_batch", type=int, default=8); ap.add_argument("--grad_accum", type=int, default=4)
ap.add_argument("--max_len", type=int, default=384); ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--save_steps", type=int, default=200); ap.add_argument("--gen_batch", type=int, default=16)
ap.add_argument("--max_new", type=int, default=160); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--cpu_test", type=int, default=0)
a = ap.parse_args()

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict

log = lambda *x: print(time.strftime("%m-%d %H:%M:%S"), *x, flush=True)
O = lambda *p: os.path.join(a.out_root, *p)
ckpt_dir, out_dir, res_path = O("ckpt", a.name), O("models", a.name), O("results", f"{a.name}.json")
for d in (ckpt_dir, os.path.dirname(res_path), O("preds")): os.makedirs(d, exist_ok=True)
mirror_ckpt = os.path.join(a.mirror, "ckpt", a.name) if a.mirror else ""
def ckpts(d): return sorted([x for x in os.listdir(d) if x.startswith("checkpoint-")], key=lambda x: int(x.split("-")[1])) if d and os.path.isdir(d) else []
if mirror_ckpt and not ckpts(ckpt_dir) and ckpts(mirror_ckpt):
    last = ckpts(mirror_ckpt)[-1]; shutil.copytree(os.path.join(mirror_ckpt, last), os.path.join(ckpt_dir, last)); log(f"드라이브 체크포인트 {last} 를 가져와 이어서 학습")

# ---------- 데이터 → 질문·정답 글 ----------
def load(p, n=0):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rows[:n] if n else rows
def answer(text, spans):
    parts = []
    for s, e, t in sorted(spans):
        w = text[s:e]; k = [m.start() for m in re.finditer(re.escape(w), text)].index(s) + 1 if w else 1
        parts.append(f"{w}@{k}={t}" if k > 1 else f"{w}={t}")
    return " | ".join(parts) if parts else "없음"
def prompt(text): return f"### 문장\n{text}\n### 개체\n"
def parse(text, out, tags):
    """모델 답 → 원문 span. 위치를 못 찾거나 태그가 목록에 없으면 bad 로 센다."""
    spans, bad = [], 0
    out = out.split("\n")[0].strip()
    if not out or out == "없음": return spans, 0
    for part in out.split(" | "):
        if "=" not in part: bad += 1; continue
        w, t = part.rsplit("=", 1); w, t = w.strip(), t.strip(); k = 1
        m = re.match(r"^(.*)@(\d+)$", w)
        if m: w, k = m.group(1), int(m.group(2))
        pos = [x.start() for x in re.finditer(re.escape(w), text)] if w else []
        if t not in tags or len(pos) < k: bad += 1; continue
        spans.append((pos[k - 1], pos[k - 1] + len(w), t))
    return sorted(set(spans)), bad
train_rows, dev_rows = load(a.train, a.train_n), load(a.dev)
tags = S.TAGS
log(f"학습 {len(train_rows):,}문장 · dev {len(dev_rows):,}문장(채점 앞 {min(a.eval_n, len(dev_rows)):,}) · 태그 {tags}")

# ---------- 모델 ----------
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(a.model)
kw = dict(dtype=torch.float32 if a.cpu_test else torch.bfloat16, attn_implementation="eager")
if getattr(cfg, "quantization_config", None):
    from transformers import Mxfp4Config; kw["quantization_config"] = Mxfp4Config(dequantize=True)
if not a.cpu_test:
    p = torch.cuda.get_device_properties(0); log(f"GPU {p.name} {p.total_memory/2**30:.0f}GB")
    assert p.total_memory > 70 * 2**30, "GPU 메모리 70GB 이상 필요(A100 80GB·H100·G4)"; kw["device_map"] = "cuda"
model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
if a.dapt:
    model = PeftModel.from_pretrained(model, a.dapt).merge_and_unload(); log(f"경제 사전학습 LoRA 합침: {a.dapt}")
L = cfg.num_hidden_layers; layers = sorted({L // 4, L // 2, 3 * L // 4, L - 1})
model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    target_parameters=[f"{i}.mlp.experts.{x}" for i in layers for x in ("gate_up_proj", "down_proj")]))
model.print_trainable_parameters(); model.config.use_cache = False

class SFT(torch.utils.data.Dataset):
    def __init__(s, rows): s.rows = rows
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        r = s.rows[i]; q = tok(prompt(r["text"]), add_special_tokens=False)["input_ids"]
        y = tok(answer(r["text"], r["spans"]) + "\n", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = (q + y)[: a.max_len]; lab = ([-100] * len(q) + y)[: a.max_len]
        return {"input_ids": ids, "labels": lab}
def collate(b):
    L = max(len(x["input_ids"]) for x in b)
    return {"input_ids": torch.tensor([x["input_ids"] + [tok.pad_token_id] * (L - len(x["input_ids"])) for x in b]),
            "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (L - len(x["input_ids"])) for x in b]),
            "labels": torch.tensor([x["labels"] + [-100] * (L - len(x["labels"])) for x in b])}

class Mirror(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        if not mirror_ckpt: return
        nm = f"checkpoint-{state.global_step}"; src = os.path.join(ckpt_dir, nm)
        if not os.path.isdir(src): return
        os.makedirs(mirror_ckpt, exist_ok=True); tmp = os.path.join(mirror_ckpt, "_tmp_" + nm)
        shutil.rmtree(tmp, ignore_errors=True); shutil.copytree(src, tmp); dst = os.path.join(mirror_ckpt, nm); shutil.rmtree(dst, ignore_errors=True); os.replace(tmp, dst)
        for old in ckpts(mirror_ckpt)[:-2]: shutil.rmtree(os.path.join(mirror_ckpt, old), ignore_errors=True)
        log(f"체크포인트 {nm} → 드라이브 복사")
class Speed(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if state.global_step == 3: self.t = time.time()
        if state.global_step in (20, 100) and hasattr(self, "t"):
            sec = (time.time() - self.t) / (state.global_step - 3)
            log(f"[속도] {sec:.2f}초/스텝 · 전체 {state.max_steps:,}스텝 예상 {state.max_steps*sec/3600:.1f}시간")

steps = math.ceil(len(train_rows) / (a.micro_batch * a.grad_accum) * a.epochs)
args = TrainingArguments(output_dir=ckpt_dir, per_device_train_batch_size=a.micro_batch, gradient_accumulation_steps=a.grad_accum,
    learning_rate=a.lr, num_train_epochs=a.epochs, warmup_steps=int(0.03 * steps), lr_scheduler_type="cosine", weight_decay=0.0,
    bf16=not a.cpu_test, use_cpu=bool(a.cpu_test), gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=1 if a.cpu_test else 10, save_strategy="steps", save_steps=a.save_steps, save_total_limit=2,
    report_to="none", seed=a.seed, remove_unused_columns=False)
trainer = Trainer(model=model, args=args, train_dataset=SFT(train_rows), data_collator=collate, callbacks=[Speed(), Mirror()])
t0 = time.time(); trainer.train(resume_from_checkpoint=True if ckpts(ckpt_dir) else None); train_min = (time.time() - t0) / 60

# ---------- 저장 → 다시 읽어 확인 ----------
model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
from safetensors.torch import load_file
mine = {k: v.detach().float().cpu() for k, v in get_peft_model_state_dict(model).items()}
saved = {k: v.float() for k, v in load_file(os.path.join(out_dir, "adapter_model.safetensors")).items()}
diff = [k for k in mine if k not in saved or not torch.equal(mine[k], saved[k])]
verify = not diff and sum(1 for k, v in mine.items() if "lora_B" in k and v.abs().sum() > 0) > 0
log(f"저장 확인: 텐서 {len(mine)}개 · 다른 것 {len(diff)} → {'OK' if verify else '실패'}")

# ---------- dev 채점(생성) ----------
model.eval(); model.config.use_cache = True; tok.padding_side = "left"
def gen_eval(rows, nm):
    preds, outs, bad = [], [], 0
    with torch.no_grad():
        for k in range(0, len(rows), a.gen_batch):
            b = rows[k:k + a.gen_batch]; enc = tok([prompt(r["text"]) for r in b], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            for r, o in zip(b, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                sp, nb = parse(r["text"], o, tags); preds.append(sp); outs.append(o.split("
")[0]); bad += nb
            if k == 0: log(f"[{nm}] 예: {b[0]['text'][:60]} → {outs[0][:120]}")
    sc = S.score([r["spans"] for r in rows], preds)
    kinds = S.save_preds(O("preds", f"{a.name}_{nm}.jsonl"), rows, preds, extra=[{"raw": o} for o in outs])
    log(f"{nm} {len(rows)}문장 F1 {sc['ALL']['f1']} (P {sc['ALL']['precision']} R {sc['ALL']['recall']}) · macro {sc['MACRO_F1']} · 위치 못 찾은 답 {bad}")
    return {"n": len(rows), "score": sc, "error_types": kinds, "unparsed_or_not_found": bad}
t1 = time.time(); evals = {"dev": gen_eval(dev_rows[: a.eval_n], "dev")}
if a.test2k: evals["test_2000"] = gen_eval(load(a.test2k), "test_2000")
if a.eval_file: evals["extra"] = gen_eval(load(a.eval_file), "extra"); evals["extra"]["file"] = a.eval_file
rec = {"name": a.name, "model": a.model, "dapt": a.dapt or None, "train": a.train, "dev": a.dev, "n_train": len(train_rows),
       "tags": tags, "config": vars(a), "train_minutes": round(train_min, 1), "gen_minutes": round((time.time() - t1) / 60, 1),
       "evals": evals, "dev_score": evals["dev"]["score"], "saved_dir": out_dir, "verify_ok": verify, "done": True,
       "log_history": trainer.state.log_history}   # 학습 손실·학습률 전부(그림용)
json.dump(rec, open(res_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("VERIFY_OK" if verify else "VERIFY_FAIL")
