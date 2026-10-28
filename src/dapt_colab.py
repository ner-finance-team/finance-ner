# -*- coding: utf-8 -*-
"""도메인 사전학습(MLM) — Colab용. 모델 상관없이 쓴다(기본: jhu-clsp/mmBERT-base). 라벨(span)은 쓰지 않는다.

  python dapt_colab.py --corpus /content/drive/MyDrive/ner_dapt/econ_dapt.txt.gz \
                       --out_root /content/drive/MyDrive/ner_dapt --name dapt_mmbert_econ5m

GPU를 최대한 쓰는 설계
 - 문장을 이어 붙여 max_len(기본 512) 블록으로 자른다 → 패딩 0, 모든 토큰이 계산에 쓰임
 - micro_batch=0 이면 GPU에 들어가는 가장 큰 배치를 직접 시험해서 고르고, 남는 만큼 기울기 누적으로 tokens_per_step 을 맞춤
 - bf16 · TF32 · fused AdamW · DataLoader 병렬 · (ModernBERT) 가린 위치만 출력층 계산(sparse_prediction)
가중치를 잃지 않는 설계
 - 체크포인트는 드라이브 {out_root}/ckpt/{name} 에 save_steps 마다(최근 2개, 끝나도 지우지 않음). 끊기면 같은 명령 → 이어서
 - 토큰화 결과도 드라이브에 보관 → 다시 연결해도 토큰화를 건너뜀
 - 끝나면 {out_root}/models/{name} 저장 → 다시 읽어 텐서 전부 일치 확인 → 토큰 분류 모델로 본체 불러오기 확인
   → {out_root}/results/{name}.json 에 verify_ok=true. 하나라도 실패하면 에러로 멈춘다(체크포인트는 그대로)
"""
import argparse, re, gzip, json, math, os, shutil, time
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--out_root", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
ap.add_argument("--work_dir", default="/content/work", help="빠른 로컬 디스크(토큰 캐시 사용)")
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--tokens_per_step", type=int, default=32768, help="옵티마이저 한 스텝에 쓰는 토큰 수(=블록 수×max_len)")
ap.add_argument("--micro_batch", type=int, default=0, help="0이면 GPU에 맞춰 자동")
ap.add_argument("--mlm", type=float, default=0.15)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--warmup", type=float, default=0.06)
ap.add_argument("--head_steps", type=int, default=-1,
                help="처음 N스텝은 출력층만 학습(본체 기울기 0). -1이면 자동: 학습 전 MLM 손실이 3 넘으면 300, 아니면 0 "
                     "(klue/roberta-base 6.60 → 필요, mmBERT-base 1.62 → 불필요)")
ap.add_argument("--save_steps", type=int, default=0, help="0이면 전체의 5%마다")
ap.add_argument("--eval_steps", type=int, default=0, help="0이면 전체의 10%마다")
ap.add_argument("--eval_blocks", type=int, default=2000)
ap.add_argument("--num_workers", type=int, default=4)
ap.add_argument("--attn", default="sdpa")
ap.add_argument("--compile", type=int, default=0)
ap.add_argument("--max_steps", type=int, default=-1, help="시험용")
ap.add_argument("--limit_lines", type=int, default=0, help="시험용")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tok_from", default="", help="토큰 캐시 이름에 쓸 토크나이저 원래 이름(이어 학습할 때 원본 이름을 주면 1에포크 캐시 재사용)")
ap.add_argument("--train_seed", type=int, default=-1, help="학습 순서·가리기 위치 시드. -1이면 --seed. 2번째 에포크는 바꿔야 같은 배치를 반복하지 않음(검증 블록은 --seed 로 고정)")
a = ap.parse_args()

from transformers import (AutoTokenizer, AutoModelForMaskedLM, AutoModelForTokenClassification,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerCallback)

log = lambda *x: print(time.strftime("%m-%d %H:%M:%S"), *x, flush=True)
cuda = torch.cuda.is_available()
bf16 = cuda and torch.cuda.is_bf16_supported()
tf32 = cuda and torch.cuda.get_device_capability(0)[0] >= 8   # A100·L4·H100 (T4는 아님)
if tf32:
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
if cuda:
    p = torch.cuda.get_device_properties(0)
    log(f"GPU {p.name} · 메모리 {p.total_memory/2**30:.1f}GB · bf16 {bf16}")
else:
    log("GPU 없음 — 시험용 CPU 실행")

D = lambda *x: os.path.join(a.out_root, *x)
ckpt_dir, out_dir, cache_dir = D("ckpt", a.name), D("models", a.name), D("cache")
rec_path = D("results", f"{a.name}.json")
for d in (ckpt_dir, cache_dir, os.path.dirname(rec_path), a.work_dir): os.makedirs(d, exist_ok=True)

tok = AutoTokenizer.from_pretrained(a.model)
dt = np.uint16 if len(tok) < 65535 else np.uint32

# 1) 토큰화 (로컬 → 드라이브 캐시)
tag = f"{os.path.basename(a.corpus).split('.')[0]}_{re.sub(r'[^A-Za-z0-9._-]', '_', a.tok_from or a.model)}" + (f"_lim{a.limit_lines}" if a.limit_lines else "")
tok_name, meta_name = f"tok_{tag}.bin", f"tok_{tag}.json"
tok_path, meta_path = os.path.join(a.work_dir, tok_name), os.path.join(a.work_dir, meta_name)
if not os.path.exists(meta_path) and os.path.exists(os.path.join(cache_dir, meta_name)):
    log("드라이브의 토큰 캐시를 로컬로 복사"); shutil.copy(os.path.join(cache_dir, tok_name), tok_path)
    shutil.copy(os.path.join(cache_dir, meta_name), meta_path)
if not os.path.exists(meta_path):
    t0 = time.time(); n_lines = n_tok = 0
    with open(tok_path + ".part", "wb") as fo, gzip.open(a.corpus, "rt", encoding="utf-8") as fi:
        buf = []
        def flush():
            global n_tok
            ids = tok(buf, add_special_tokens=True, return_attention_mask=False)["input_ids"]
            arr = np.fromiter((i for s in ids for i in s), dtype=dt); arr.tofile(fo); n_tok += arr.size
        for line in fi:
            line = line.rstrip("\n")
            if not line: continue
            buf.append(line); n_lines += 1
            if len(buf) == 20000: flush(); buf = []
            if n_lines % 500000 == 0: log(f"토큰화 {n_lines:,}줄 ({time.time()-t0:.0f}s)")
            if a.limit_lines and n_lines >= a.limit_lines: break
        if buf: flush()
    os.replace(tok_path + ".part", tok_path)
    json.dump({"lines": n_lines, "tokens": n_tok, "dtype": np.dtype(dt).name, "sec": round(time.time() - t0)}, open(meta_path, "w"))
    shutil.copy(tok_path, os.path.join(cache_dir, tok_name)); shutil.copy(meta_path, os.path.join(cache_dir, meta_name))
    log("토큰 캐시를 드라이브에 저장")
meta = json.load(open(meta_path))
ids = np.memmap(tok_path, dtype=np.dtype(meta["dtype"]), mode="r")
n_blocks = len(ids) // a.max_len
blocks = ids[: n_blocks * a.max_len].reshape(n_blocks, a.max_len)
log(f"문장 {meta['lines']:,} · 토큰 {meta['tokens']:,} (문장당 {meta['tokens']/meta['lines']:.1f}) · 블록 {n_blocks:,} × {a.max_len}")

rng = np.random.default_rng(a.seed)
perm = rng.permutation(n_blocks)
n_eval = min(a.eval_blocks, max(8, n_blocks // 100))
eval_idx, train_idx = np.sort(perm[:n_eval]), perm[n_eval:]

class Blocks(torch.utils.data.Dataset):
    def __init__(s, idx): s.idx = idx
    def __len__(s): return len(s.idx)
    def __getitem__(s, i): return {"input_ids": torch.from_numpy(blocks[s.idx[i]].astype(np.int64))}
train_ds, eval_ds = Blocks(train_idx), Blocks(eval_idx)
coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm_probability=a.mlm)

kw = dict(attn_implementation=a.attn)
model = AutoModelForMaskedLM.from_pretrained(a.model, **kw)
if hasattr(model.config, "sparse_prediction") or model.config.model_type == "modernbert":
    model.config.sparse_prediction = True
    if hasattr(model, "sparse_prediction"): model.sparse_prediction = True
    log("가린 위치만 출력층 계산(sparse_prediction) 켬")
if cuda: model.cuda()

# 2) GPU에 맞는 micro_batch 자동 선택
per_step_blocks = max(1, a.tokens_per_step // a.max_len)
mb = a.micro_batch
if mb == 0 and cuda:
    model.train(); best, speed = 0, None
    for cand in [c for c in (128, 96, 64, 48, 32, 24, 16, 12, 8, 4, 2, 1) if c <= per_step_blocks]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            b = coll([train_ds[i] for i in range(cand)]); b = {k: v.cuda() for k, v in b.items()}
            for rep in range(2):
                t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16): loss = model(**b).loss
                loss.backward(); torch.cuda.synchronize(); el = time.time() - t0
            model.zero_grad(set_to_none=True)
            used = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            if used < 0.85:   # 옵티마이저 상태(가중치의 2배)와 여유분
                best, speed = cand, cand * a.max_len / el; break
            log(f"배치 {cand}: 메모리 {used:.0%} → 더 작게")
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); log(f"배치 {cand}: 메모리 부족 → 더 작게")
    mb = best or 1
    log(f"micro_batch {mb} 선택 · 측정 처리량 약 {speed:,.0f} 토큰/초" if speed else f"micro_batch {mb}")
elif mb == 0:
    mb = 2; speed = None
else:
    speed = None
grad_accum = max(1, round(per_step_blocks / mb))
total_steps = a.max_steps if a.max_steps > 0 else math.ceil(len(train_ds) / (mb * grad_accum) * a.epochs)
save_steps = a.save_steps or max(50, total_steps // 20)
eval_steps = a.eval_steps or max(50, total_steps // 10)
eta = f" · 예상 학습 시간 약 {len(train_ds)*a.max_len*a.epochs/speed/3600:.1f}시간(측정 처리량 기준, 실제는 조금 더 걸림)" if speed else ""
log(f"학습 블록 {len(train_ds):,} · 검증 블록 {n_eval:,} · 배치 {mb}×누적 {grad_accum} = 스텝당 {mb*grad_accum*a.max_len:,}토큰 · 총 스텝 {total_steps:,} · 저장 {save_steps}스텝마다{eta}")

# 3) 출력층 예열: 처음 head_steps 동안 본체 기울기 0
freeze = {"on": False}   # 학습 전 손실을 잰 뒤 정한다
HEAD = ("lm_head", "head.", "decoder")
for n_, p_ in model.named_parameters():
    if not n_.startswith(HEAD): p_.register_hook(lambda g: g * 0 if freeze["on"] else g)
class HeadWarm(TrainerCallback):
    def on_step_begin(self, args, state, control, **k):
        on = state.global_step < a.head_steps
        if freeze["on"] and not on: log(f"스텝 {state.global_step}: 출력층 예열 끝 → 본체도 학습")
        freeze["on"] = on

TS = a.train_seed if a.train_seed >= 0 else a.seed
args = TrainingArguments(
    output_dir=ckpt_dir, per_device_train_batch_size=mb, per_device_eval_batch_size=mb,
    gradient_accumulation_steps=grad_accum, learning_rate=a.lr, num_train_epochs=a.epochs, max_steps=a.max_steps,
    warmup_steps=int(a.warmup * total_steps), weight_decay=0.01, lr_scheduler_type="linear",
    bf16=bf16, tf32=bool(tf32), optim="adamw_torch_fused" if cuda else "adamw_torch",
    logging_steps=max(1, min(50, total_steps // 100)), eval_strategy="steps", eval_steps=eval_steps,
    save_strategy="steps", save_steps=save_steps, save_total_limit=2,
    dataloader_num_workers=a.num_workers if cuda else 0, dataloader_pin_memory=bool(cuda),
    torch_compile=bool(a.compile), report_to="none", seed=TS, data_seed=TS, remove_unused_columns=False)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=coll, callbacks=[HeadWarm()])

rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
if "eval_loss_before" not in rec:
    torch.manual_seed(a.seed); rec["eval_loss_before"] = trainer.evaluate()["eval_loss"]
    log(f"학습 전 MLM 검증 손실 {rec['eval_loss_before']:.4f}")
if a.head_steps < 0:
    a.head_steps = 300 if rec["eval_loss_before"] > 3.0 else 0
    log(f"출력층 예열 {a.head_steps}스텝 (학습 전 손실 {rec['eval_loss_before']:.2f} 기준 자동)")
freeze["on"] = a.head_steps > 0
rec.update(name=a.name, model=a.model, corpus=a.corpus, corpus_meta=meta, max_len=a.max_len, blocks=n_blocks, eval_blocks=n_eval,
           micro_batch=mb, grad_accum=grad_accum, tokens_per_step=mb * grad_accum * a.max_len, lr=a.lr, mlm=a.mlm,
           epochs=a.epochs, head_steps=a.head_steps, seed=a.seed, train_seed=TS, gpu=torch.cuda.get_device_name(0) if cuda else "cpu",
           verify_ok=False)
json.dump(rec, open(rec_path, "w"), ensure_ascii=False, indent=1)

has_ckpt = any(d.startswith("checkpoint-") for d in os.listdir(ckpt_dir))
if has_ckpt: log("이전 체크포인트에서 이어서 학습")
t0 = time.time()
trainer.train(resume_from_checkpoint=True if has_ckpt else None)
rec["train_minutes_this_run"] = round((time.time() - t0) / 60, 1)
torch.manual_seed(a.seed); rec["eval_loss_after"] = trainer.evaluate()["eval_loss"]
log(f"학습 후 MLM 검증 손실 {rec['eval_loss_after']:.4f}")

# 4) 저장 → 다시 읽어 확인
trainer.save_model(out_dir); tok.save_pretrained(out_dir)
sd = {k: v.detach().float().cpu() for k, v in trainer.model.state_dict().items()}
re_sd = AutoModelForMaskedLM.from_pretrained(out_dir, **kw).state_dict()
miss = [k for k in sd if k not in re_sd]
diff = [k for k in sd if k in re_sd and not torch.equal(sd[k], re_sd[k].float())]
if miss or diff:
    raise SystemExit(f"저장 확인 실패: 없는 텐서 {len(miss)} {miss[:3]} · 다른 텐서 {len(diff)} {diff[:3]} — 체크포인트는 {ckpt_dir} 에 남아 있음")
tc_sd = AutoModelForTokenClassification.from_pretrained(out_dir, num_labels=17, **kw).state_dict()
shared = [k for k in tc_sd if k in re_sd and tc_sd[k].shape == re_sd[k].shape]
enc_ok = len(shared) > 0 and all(torch.equal(tc_sd[k].float(), re_sd[k].float()) for k in shared)
if not enc_ok: raise SystemExit("토큰 분류 모델로 본체를 불러오지 못함")
log(f"저장 확인: 텐서 {len(sd)}개 일치 · 토큰 분류 모델로 본체 {len(shared)}개 텐서 불러오기 OK")
rec.update(saved_dir=out_dir, tensors_checked=len(sd), token_cls_shared_tensors=len(shared), verify_ok=True,
           steps=trainer.state.global_step, log_history=trainer.state.log_history, done=True)
json.dump(rec, open(rec_path, "w"), ensure_ascii=False, indent=1)
log("VERIFY_OK", out_dir)
