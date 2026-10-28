# -*- coding: utf-8 -*-
"""1단계 NER(7태그) 큰 세트 학습 — 미리 토큰화한 배열(encode_big.py)로 학습. 서버(MIG 10GB·RAM 9GB)·Colab 공용.

  python src/train_big.py --enc data/enc/roberta_5000k --init ner_models/dapt_roberta_econ5m --exp S5_dapt-roberta-base_5000k \
      --lr 5e-5 --epochs 1 --batch 16 --pretok none

채점·라벨 규칙은 1단계와 같다(ner_lib: encode·decode_spans·predict_label_ids, evaluate.span_prf).
- 학습 순서: 비슷한 길이끼리 묶음(LengthGroupedSampler, 50배 묶음 안에서 섞음) → 패딩 낭비를 줄여 빠르게. 배치 16·학습률·예열 10%·감쇠 0.01은 120만과 같음.
- save_steps 마다 체크포인트(최근 2개). 끊기면 같은 명령 → 이어서 학습.
- 끝나면 {model_root}/{exp} 저장 → 다시 읽어 텐서 전부 일치 확인(실패하면 체크포인트를 지우지 않고 에러) → dev·test 예측·채점 → results/{exp}.json.
- 체크포인트는 확인이 끝난 뒤에도 지우지 않는다(--keep_ckpt 1 기본).
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ner_lib
from ner_lib import LABELS, label2id, id2label, load_jsonl, _to_dataset, predict_label_ids, decode_spans, save_preds, upper_bound
from evaluate import span_prf, span_prf_by

ap = argparse.ArgumentParser()
ap.add_argument("--enc", required=True); ap.add_argument("--init", required=True); ap.add_argument("--exp", required=True)
ap.add_argument("--tok", default="", help="토크나이저(없으면 --init)")
ap.add_argument("--pretok", default="none"); ap.add_argument("--dev", default="data/split_v3b/dev.jsonl"); ap.add_argument("--test", default="data/split_v3b/test.jsonl")
ap.add_argument("--out_root", default=".", help="results·ner_models·checkpoints 를 둘 곳")
ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--grad_accum", type=int, default=1)
ap.add_argument("--warmup", type=float, default=0.1); ap.add_argument("--weight_decay", type=float, default=0.01)
ap.add_argument("--save_steps", type=int, default=5000); ap.add_argument("--limit", type=int, default=0, help="시험용: 앞 N문장만")
ap.add_argument("--max_steps", type=int, default=-1); ap.add_argument("--eval_limit", type=int, default=0, help="시험용: dev·test 앞 N문장")
ap.add_argument("--keep_ckpt", type=int, default=1); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()
pretok = None if a.pretok in ("", "none") else a.pretok
from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments, TrainerCallback)
from transformers.trainer_pt_utils import LengthGroupedSampler

log = lambda *x: print(time.strftime("%m-%d %H:%M:%S"), *x, flush=True)
O = lambda *p: os.path.join(a.out_root, *p)
ckpt, mdir, res_path = O("checkpoints", a.exp), O("ner_models", a.exp), O("reports", "results", f"{a.exp}.json")
for d in (ckpt, os.path.dirname(res_path)): os.makedirs(d, exist_ok=True)
if os.path.exists(res_path) and json.load(open(res_path, encoding="utf-8")).get("done"):
    log("이미 끝난 실험:", res_path); sys.exit(0)

meta = json.load(open(os.path.join(a.enc, "meta.json"), encoding="utf-8"))
assert (meta["pretok"] or None) == pretok, f"토큰화 pretok({meta['pretok']})과 --pretok({pretok})이 다름"
ids = np.memmap(os.path.join(a.enc, "ids.bin"), dtype=meta["dtype"], mode="r")
labs = np.memmap(os.path.join(a.enc, "labels.bin"), dtype=np.int8, mode="r")
starts = np.load(os.path.join(a.enc, "starts.npy"))
N = len(starts) - 1 if not a.limit else min(a.limit, len(starts) - 1)
lengths = np.diff(starts[: N + 1]).tolist()
log(f"학습 {N:,}문장 · 토큰 {int(starts[N]):,} (문장당 {starts[N]/N:.1f}) · {a.enc}")

class Enc(torch.utils.data.Dataset):
    def __len__(s): return N
    def __getitem__(s, i):
        b, e = int(starts[i]), int(starts[i + 1])
        x = ids[b:e].astype(np.int64)
        return {"input_ids": x.tolist(), "attention_mask": [1] * (e - b), "labels": labs[b:e].astype(np.int64).tolist()}

tok = AutoTokenizer.from_pretrained(a.tok or a.init)
assert len(tok) == meta["vocab"], f"토크나이저 크기 다름 {len(tok)} vs {meta['vocab']}"
model = AutoModelForTokenClassification.from_pretrained(a.init, num_labels=len(LABELS), id2label=id2label, label2id=label2id)
n_params = sum(p.numel() for p in model.parameters())
collator = DataCollatorForTokenClassification(tok)
cuda = torch.cuda.is_available(); bf16 = cuda and torch.cuda.is_bf16_supported()
steps = a.max_steps if a.max_steps > 0 else math.ceil(N / (a.batch * a.grad_accum) * a.epochs)
args = TrainingArguments(
    output_dir=ckpt, learning_rate=a.lr, num_train_epochs=a.epochs, max_steps=a.max_steps,
    per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
    warmup_steps=int(a.warmup * steps), weight_decay=a.weight_decay, bf16=bf16, fp16=cuda and not bf16,
    eval_strategy="no", save_strategy="steps", save_steps=a.save_steps, save_total_limit=2,
    logging_steps=max(1, min(500, steps // 200)), report_to="none", seed=a.seed, remove_unused_columns=False,
    dataloader_num_workers=a.workers, use_cpu=not cuda)

class LenTrainer(Trainer):
    def _get_train_sampler(self, *args, **kw):
        g = torch.Generator(); g.manual_seed(a.seed)
        return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps, lengths=lengths, generator=g)

class Speed(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if state.global_step and state.global_step % (args.logging_steps * 20) == 0:
            el = time.time() - T0; done = state.global_step - S0
            if done > 0: log(f"스텝 {state.global_step:,}/{state.max_steps:,} · 남은 시간 약 {(state.max_steps-state.global_step)*el/done/3600:.1f}시간")

trainer = LenTrainer(model=model, args=args, train_dataset=Enc(), data_collator=collator, callbacks=[Speed()])
resume = ner_lib._last_checkpoint(ckpt)
S0 = int(resume.rsplit("-", 1)[1]) if resume else 0
log(f"파라미터 {n_params:,} · 총 스텝 {steps:,} · 배치 {a.batch}×{a.grad_accum} · lr {a.lr} · " + (f"이어서 {resume}" if resume else "처음부터"))
T0 = time.time(); tr = trainer.train(resume_from_checkpoint=resume); train_sec = round(time.time() - T0, 1)

# 저장 → 다시 읽어 확인 (예측보다 먼저)
trainer.save_model(mdir); tok.save_pretrained(mdir)
sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
re_sd = AutoModelForTokenClassification.from_pretrained(mdir).state_dict()
diff = [k for k in sd if k not in re_sd or not torch.equal(sd[k], re_sd[k].float())]
if diff: raise SystemExit(f"저장 확인 실패: 다른 텐서 {len(diff)} {diff[:3]} — 체크포인트는 {ckpt} 에 남아 있음")
log(f"저장 확인: 텐서 {len(sd)}개 일치 → {mdir}")

# dev·test 채점 (1단계와 같은 함수)
out = {}
t0 = time.time()
for split, p in (("dev", a.dev), ("test", a.test)):
    rows = load_jsonl(p, a.eval_limit or None)
    ds, offs = _to_dataset(rows, tok, meta["max_len"], pretok)
    pred = [decode_spans(i, o) for i, o in zip(predict_label_ids(model, ds, collator, batch=64), offs)]
    save_preds(O("reports", "results", "preds", f"{a.exp}_{split}.jsonl"), rows, pred)
    out[split] = span_prf([r["spans"] for r in rows], pred)
    if split == "test": out["test_by_category"] = span_prf_by(rows, pred); out["upper_bound_test"] = upper_bound(rows, tok, meta["max_len"], pretok)
    log(f"{split} F1 {out[split]['ALL']['f1']} · macro {out[split]['MACRO_F1']}")
rec = {"exp": a.exp, "init": a.init, "enc": meta, "pretok": pretok, "n_train": N, "params": n_params,
       "config": {k: v for k, v in vars(a).items()}, "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
       "train_sec": train_sec, "resumed_from": resume, "steps": trainer.state.global_step,
       "train_loss": round(tr.metrics.get("train_loss", 0), 4), "infer_sec": round(time.time() - t0, 1),
       "model_path": mdir, "weights_verified": len(sd), "done": True, "log_history": trainer.state.log_history, **out}
json.dump(rec, open(res_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log(f"끝 · dev F1 {out['dev']['ALL']['f1']} · test F1 {out['test']['ALL']['f1']} · 학습 {train_sec/3600:.2f}시간 → {res_path}")
