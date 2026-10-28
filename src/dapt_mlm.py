# -*- coding: utf-8 -*-
"""도메인 사전학습(MLM): 경제 문장으로 klue/roberta-base 를 더 학습한다. 라벨(span)은 쓰지 않는다.

  python src/dapt_mlm.py --corpus data/dapt/econ_dapt.txt.gz --name dapt_roberta_econ5m

순서
 1) 문장을 토큰으로 바꿔 이어 붙이고 max_len 블록으로 자른다 → data/dapt/tok_*.u16 (한 번만, 다시 실행하면 재사용)
 2) 블록의 0.5%(최대 5,000개)를 떼어 MLM 검증 손실을 잰다(학습 전 원래 모델 값도 기록)
 3) Trainer로 1에폭 학습. save_steps 마다 체크포인트 저장(최근 2개 유지, 끝나도 지우지 않음).
    끊기면 같은 명령으로 다시 실행 → 마지막 체크포인트부터 이어서 학습
 4) 끝나면 ner_models/{name} 에 모델·토크나이저 저장 → 다시 읽어 텐서 전부 일치하는지 확인(실패하면 에러)
    → 토큰 분류 모델로 본체를 불러올 수 있는지까지 확인 → {name}.json 기록
"""
import argparse, gzip, json, math, os, sys, time
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--model", default="klue/roberta-base")
ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--grad_accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--warmup", type=float, default=0.06)
ap.add_argument("--save_steps", type=int, default=1000)
ap.add_argument("--eval_steps", type=int, default=2000)
ap.add_argument("--head_steps", type=int, default=500,
                help="처음 N스텝은 MLM 출력층만 학습(본체 기울기 0). klue/roberta-base 공개 출력층이 잘 맞지 않아(가린 단어 CE 5.98, klue/bert-base 3.66) 초반 큰 오차가 본체를 흔들지 않게")
ap.add_argument("--max_steps", type=int, default=-1, help="시험용")
ap.add_argument("--limit_lines", type=int, default=0, help="시험용: 앞 N줄만")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

from transformers import (AutoTokenizer, AutoModelForMaskedLM, AutoModelForTokenClassification,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

R = lambda *p: os.path.join(a.root, *p)
ckpt_dir, out_dir = R("checkpoints", a.name), R("ner_models", a.name)
os.makedirs(R("data", "dapt"), exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
log = lambda *x: print(time.strftime("%m-%d %H:%M:%S"), *x, flush=True)
tok = AutoTokenizer.from_pretrained(a.model)
assert tok.vocab_size < 65535

# 1) 토큰화
tag = f"{os.path.basename(a.corpus).split('.')[0]}_{a.model.replace('/', '_')}" + (f"_lim{a.limit_lines}" if a.limit_lines else "")
tok_path, meta_path = R("data", "dapt", f"tok_{tag}.u16"), R("data", "dapt", f"tok_{tag}.json")
if not os.path.exists(meta_path):
    t0 = time.time(); n_lines = n_tok = 0
    with open(tok_path + ".part", "wb") as fo, gzip.open(a.corpus, "rt", encoding="utf-8") as fi:
        buf = []
        def flush():
            global n_tok
            ids = tok(buf, add_special_tokens=True)["input_ids"]
            arr = np.fromiter((i for s in ids for i in s), dtype=np.uint16)
            arr.tofile(fo); n_tok += arr.size
        for line in fi:
            line = line.rstrip("\n")
            if not line: continue
            buf.append(line); n_lines += 1
            if len(buf) == 20000: flush(); buf = []
            if n_lines % 500000 == 0: log(f"토큰화 {n_lines:,}줄")
            if a.limit_lines and n_lines >= a.limit_lines: break
        if buf: flush()
    os.replace(tok_path + ".part", tok_path)
    json.dump({"lines": n_lines, "tokens": n_tok, "sec": round(time.time() - t0)}, open(meta_path, "w"))
meta = json.load(open(meta_path))
ids = np.memmap(tok_path, dtype=np.uint16, mode="r")
n_blocks = len(ids) // a.max_len
blocks = ids[: n_blocks * a.max_len].reshape(n_blocks, a.max_len)
log(f"문장 {meta['lines']:,} · 토큰 {meta['tokens']:,} (문장당 {meta['tokens']/meta['lines']:.1f}) · 블록 {n_blocks:,} × {a.max_len}")

rng = np.random.default_rng(a.seed)
n_eval = min(5000, max(100, n_blocks // 200))
perm = rng.permutation(n_blocks)
eval_idx, train_idx = np.sort(perm[:n_eval]), perm[n_eval:]

class Blocks(torch.utils.data.Dataset):
    def __init__(s, idx): s.idx = idx
    def __len__(s): return len(s.idx)
    def __getitem__(s, i): return {"input_ids": torch.from_numpy(blocks[s.idx[i]].astype(np.int64))}

train_ds, eval_ds = Blocks(train_idx), Blocks(eval_idx)
coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm_probability=0.15)
model = AutoModelForMaskedLM.from_pretrained(a.model)
total_steps = a.max_steps if a.max_steps > 0 else math.ceil(len(train_ds) / (a.batch * a.grad_accum) * a.epochs)
log(f"학습 블록 {len(train_ds):,} · 검증 블록 {n_eval:,} · 총 스텝 {total_steps:,}")

args = TrainingArguments(
    output_dir=ckpt_dir, per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
    gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, num_train_epochs=a.epochs,
    max_steps=a.max_steps, warmup_steps=int(a.warmup * total_steps), weight_decay=0.01, lr_scheduler_type="linear",
    bf16=True, logging_steps=50, eval_strategy="steps", eval_steps=a.eval_steps,
    save_strategy="steps", save_steps=a.save_steps, save_total_limit=2,
    dataloader_num_workers=2, report_to="none", seed=a.seed, remove_unused_columns=False)
from transformers import TrainerCallback
_freeze = {"on": a.head_steps > 0}
for n_, p_ in model.named_parameters():
    if not n_.startswith("lm_head"):   # 단어 임베딩(출력층과 묶임)도 본체 입력이라 예열 동안 고정
        p_.register_hook(lambda g: g * 0 if _freeze["on"] else g)
class HeadWarm(TrainerCallback):
    def on_step_begin(self, args, state, control, **kw):
        on = state.global_step < a.head_steps
        if _freeze["on"] and not on: log(f"스텝 {state.global_step}: 출력층 예열 끝 → 본체도 학습")
        _freeze["on"] = on
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=coll,
                  callbacks=[HeadWarm()])

rec_path = R("reports", "results", f"{a.name}.json")
os.makedirs(os.path.dirname(rec_path), exist_ok=True)
rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
if "eval_loss_before" not in rec:
    torch.manual_seed(a.seed)
    rec["eval_loss_before"] = trainer.evaluate()["eval_loss"]
    log(f"학습 전 MLM 검증 손실 {rec['eval_loss_before']:.4f} (perplexity {math.exp(rec['eval_loss_before']):.2f})")
rec.update(name=a.name, model=a.model, corpus=a.corpus, corpus_meta=meta, max_len=a.max_len, blocks=n_blocks,
           eval_blocks=n_eval, batch=a.batch, grad_accum=a.grad_accum, lr=a.lr, epochs=a.epochs, seed=a.seed)
json.dump(rec, open(rec_path, "w"), ensure_ascii=False, indent=1)

has_ckpt = any(d.startswith("checkpoint-") for d in os.listdir(ckpt_dir))
t0 = time.time()
trainer.train(resume_from_checkpoint=True if has_ckpt else None)
train_min = (time.time() - t0) / 60
torch.manual_seed(a.seed)
rec["eval_loss_after"] = trainer.evaluate()["eval_loss"]
log(f"학습 후 MLM 검증 손실 {rec['eval_loss_after']:.4f} (perplexity {math.exp(rec['eval_loss_after']):.2f})")

# 4) 저장 → 다시 읽어 확인
trainer.save_model(out_dir); tok.save_pretrained(out_dir)
sd = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}
re = AutoModelForMaskedLM.from_pretrained(out_dir).state_dict()
miss = [k for k in sd if k not in re]
diff = [k for k in sd if k in re and not torch.equal(sd[k].float(), re[k].float())]
if miss or diff:
    raise SystemExit(f"저장 확인 실패: 없는 텐서 {len(miss)} · 다른 텐서 {len(diff)} — 체크포인트는 {ckpt_dir} 에 남아 있음")
tc = AutoModelForTokenClassification.from_pretrained(out_dir, num_labels=17)
enc_ok = all(torch.equal(tc.state_dict()[k].float(), re[k].float()) for k in tc.state_dict() if k.startswith("roberta.") and k in re)
log(f"저장 확인: 텐서 {len(sd)}개 일치, 토큰 분류 모델로 본체 불러오기 {'OK' if enc_ok else '불일치'}")
rec.update(saved_dir=out_dir, tensors_checked=len(sd), token_cls_encoder_match=enc_ok, train_minutes=round(train_min, 1),
           steps=trainer.state.global_step, log_history=trainer.state.log_history, done=True)
json.dump(rec, open(rec_path, "w"), ensure_ascii=False, indent=1)
log("끝", out_dir)
