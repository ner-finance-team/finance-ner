# -*- coding: utf-8 -*-
"""gpt-oss-20b 도메인 사전학습 (다음 단어 맞히기, LoRA) — Colab A100 80GB / H100 / G4 용.

  python dapt_gptoss_colab.py --corpus .../econ_dapt.txt.gz --out_root .../ner_dapt --name dapt_gptoss_econ500k --lines 500000
  (먼저 --max_steps 50 --probe 1 로 속도만 재고, 같은 명령에서 --probe 를 빼고 본 학습)

- 원문 그대로(형태소 분리 안 함): 생성 방식 NER 로 쓸 모델이라 토큰 경계와 무관(결정기록 10/25).
- 본체(20B)는 고정, LoRA(어텐션 전체 + 일부 층 전문가)만 학습·저장 → 드라이브에는 덧붙임 가중치만 올라간다.
- 문장 끝에 EOS 를 붙여 이어 붙이고 max_len 블록으로 자름. 블록 0.5%(최대 400)는 검증용.
- 체크포인트는 {out_root}/ckpt/{name} (최근 2개). 끊기면 같은 명령 → 이어서 학습.
- 끝나면 {out_root}/models/{name} 에 LoRA 저장 → 파일을 다시 읽어 학습된 텐서와 전부 일치하는지 확인 → results/{name}.json, "VERIFY_OK" 출력.
"""
import argparse, gzip, json, math, os, shutil, time
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--out_root", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--model", default="openai/gpt-oss-20b")
ap.add_argument("--work_dir", default="/content/work")
ap.add_argument("--lines", type=int, default=500_000, help="말뭉치 앞 N줄 (이미 무작위 순서)")
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--micro_batch", type=int, default=4)
ap.add_argument("--grad_accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=float, default=0.03)
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--expert_layers", default="auto", help="LoRA 를 붙일 전문가 층 번호(쉼표). auto = 층 수의 1/4·2/4·3/4·마지막")
ap.add_argument("--save_steps", type=int, default=100)
ap.add_argument("--eval_blocks", type=int, default=400)
ap.add_argument("--max_steps", type=int, default=-1)
ap.add_argument("--probe", type=int, default=0, help="1이면 속도만 재고 저장 없이 끝냄")
ap.add_argument("--cpu_test", type=int, default=0, help="로컬 시험용(작은 모델, CPU)")
ap.add_argument("--mirror", default="", help="체크포인트를 저장할 때마다 이 폴더(드라이브)에도 복사. 로컬에 체크포인트가 없으면 여기서 가져와 이어서 학습")
a = ap.parse_args()

from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, TrainerCallback)
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict

log = lambda *x: print(time.strftime("%m-%d %H:%M:%S"), *x, flush=True)
dev = "cpu" if a.cpu_test else "cuda"
if not a.cpu_test:
    p = torch.cuda.get_device_properties(0)
    log(f"GPU {p.name} {p.total_memory/2**30:.0f}GB")
    assert p.total_memory > 70 * 2**30, "GPU 메모리 70GB 이상 필요(A100 80GB·H100·G4). A100 40GB·L4·T4 불가"
    torch.backends.cuda.matmul.allow_tf32 = True
ckpt_dir = os.path.join(a.out_root, "ckpt", a.name)
out_dir = os.path.join(a.out_root, "models", a.name)
res_path = os.path.join(a.out_root, "results", f"{a.name}.json")
for d in (ckpt_dir, os.path.dirname(res_path), a.work_dir): os.makedirs(d, exist_ok=True)
mirror_ckpt = os.path.join(a.mirror, "ckpt", a.name) if a.mirror else ""
def _ckpts(d): return sorted([x for x in os.listdir(d) if x.startswith("checkpoint-")], key=lambda x: int(x.split("-")[1])) if d and os.path.isdir(d) else []
if mirror_ckpt and not _ckpts(ckpt_dir) and _ckpts(mirror_ckpt):
    last = _ckpts(mirror_ckpt)[-1]
    shutil.copytree(os.path.join(mirror_ckpt, last), os.path.join(ckpt_dir, last))
    print(time.strftime("%m-%d %H:%M:%S"), f"드라이브 체크포인트 {last} 를 로컬로 가져옴 → 이어서 학습", flush=True)

tok = AutoTokenizer.from_pretrained(a.model)
eos = tok.eos_token_id

# 1) 토큰화 → uint32 파일 (드라이브 cache 에도 보관해 재접속 때 재사용)
tag = f"{os.path.basename(a.corpus).split('.')[0]}_{a.model.replace('/', '_')}_{a.lines}"
cache = os.path.join(a.out_root, "cache"); os.makedirs(cache, exist_ok=True)
tok_path, meta_path = os.path.join(a.work_dir, f"tok_{tag}.u32"), os.path.join(a.work_dir, f"tok_{tag}.json")
if not os.path.exists(meta_path) and os.path.exists(os.path.join(cache, f"tok_{tag}.json")):
    for f in (tok_path, meta_path): shutil.copy(os.path.join(cache, os.path.basename(f)), f)
    log("토큰 캐시를 드라이브에서 가져옴")
if not os.path.exists(meta_path):
    t0, n_lines, n_tok, buf = time.time(), 0, 0, []
    with open(tok_path + ".part", "wb") as fo, gzip.open(a.corpus, "rt", encoding="utf-8") as fi:
        def flush():
            global n_tok
            ids = tok(buf, add_special_tokens=False)["input_ids"]
            arr = np.fromiter((i for s in ids for i in s + [eos]), dtype=np.uint32); arr.tofile(fo); n_tok += arr.size
        for line in fi:
            line = line.rstrip("\n")
            if not line: continue
            buf.append(line); n_lines += 1
            if len(buf) == 20000: flush(); buf = []
            if n_lines >= a.lines: break
        if buf: flush()
    os.replace(tok_path + ".part", tok_path)
    json.dump({"lines": n_lines, "tokens": n_tok, "sec": round(time.time() - t0)}, open(meta_path, "w"))
    for f in (tok_path, meta_path): shutil.copy(f, os.path.join(cache, os.path.basename(f)))
meta = json.load(open(meta_path))
ids = np.memmap(tok_path, dtype=np.uint32, mode="r")
n_blocks = len(ids) // a.max_len
blocks = ids[: n_blocks * a.max_len].reshape(n_blocks, a.max_len)
rng = np.random.default_rng(42)
perm = rng.permutation(n_blocks)
n_eval = min(a.eval_blocks, max(2, n_blocks // 200))
eval_idx, train_idx = np.sort(perm[:n_eval]), perm[n_eval:]
log(f"문장 {meta['lines']:,} · 토큰 {meta['tokens']:,} (문장당 {meta['tokens']/meta['lines']:.1f}) · 블록 {n_blocks:,}×{a.max_len} · 검증 {n_eval}")

class Blocks(torch.utils.data.Dataset):
    def __init__(s, idx): s.idx = idx
    def __len__(s): return len(s.idx)
    def __getitem__(s, i): return {"input_ids": torch.from_numpy(blocks[s.idx[i]].astype(np.int64))}

# 2) 모델 (MXFP4 → bf16 로 풀어서 GPU 에 올림) + LoRA
cfg = AutoConfig.from_pretrained(a.model)
kw = dict(dtype=torch.bfloat16 if not a.cpu_test else torch.float32, attn_implementation="eager", use_cache=False)
if getattr(cfg, "quantization_config", None):
    from transformers import Mxfp4Config
    kw["quantization_config"] = Mxfp4Config(dequantize=True)
if not a.cpu_test: kw["device_map"] = "cuda"
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
log(f"모델 불러옴 {time.time()-t0:.0f}초")
L = cfg.num_hidden_layers
layers = sorted({L // 4, L // 2, 3 * L // 4, L - 1}) if a.expert_layers == "auto" else [int(x) for x in a.expert_layers.split(",")]
lcfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                  target_parameters=[f"{i}.mlp.experts.{p}" for i in layers for p in ("gate_up_proj", "down_proj")])
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
log(f"전문가 LoRA 층 {layers}")

per_step = a.micro_batch * a.grad_accum
total_steps = a.max_steps if a.max_steps > 0 else math.ceil(len(train_idx) / per_step)
args = TrainingArguments(
    output_dir=ckpt_dir, per_device_train_batch_size=a.micro_batch, per_device_eval_batch_size=a.micro_batch,
    gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, num_train_epochs=1, max_steps=a.max_steps,
    warmup_steps=int(a.warmup * total_steps), lr_scheduler_type="cosine", weight_decay=0.0,
    bf16=not a.cpu_test, use_cpu=bool(a.cpu_test), gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=10 if not a.cpu_test else 1,
    eval_strategy="no", save_strategy="no" if a.probe else "steps", save_steps=a.save_steps, save_total_limit=2,
    dataloader_num_workers=2 if not a.cpu_test else 0, report_to="none", seed=42, remove_unused_columns=False)

class Mirror(TrainerCallback):
    """체크포인트를 저장할 때마다 드라이브에도 복사(임시 이름으로 복사 후 바꿔 끊겨도 반쪽 파일이 남지 않게), 드라이브에는 최근 2개만"""
    def on_save(self, args, state, control, **kw):
        if not mirror_ckpt or a.probe: return
        name = f"checkpoint-{state.global_step}"; src = os.path.join(ckpt_dir, name)
        if not os.path.isdir(src): return
        os.makedirs(mirror_ckpt, exist_ok=True); tmp = os.path.join(mirror_ckpt, "_tmp_" + name)
        shutil.rmtree(tmp, ignore_errors=True); shutil.copytree(src, tmp); dst = os.path.join(mirror_ckpt, name); shutil.rmtree(dst, ignore_errors=True); os.replace(tmp, dst)
        for old in _ckpts(mirror_ckpt)[:-2]: shutil.rmtree(os.path.join(mirror_ckpt, old), ignore_errors=True)
        log(f"체크포인트 {name} → 드라이브 복사")

class Speed(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        s = state.global_step
        if s == 5: self.t5 = time.time()
        if s in (25, 50, 100) or (s == state.max_steps and hasattr(self, "t5")):
            if not hasattr(self, "t5") or s <= 5: return
            sec = (time.time() - self.t5) / (s - 5)
            full = math.ceil(len(train_idx) / per_step)
            log(f"[속도] 스텝 {s}: {sec:.2f}초/스텝 · {per_step*a.max_len/sec:,.0f}토큰/초 · "
                f"{meta['lines']:,}문장 전체 {full:,}스텝 예상 {full*sec/3600:.1f}시간 · GPU 최고 메모리 "
                f"{(torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0):.1f}GB")

coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
trainer = Trainer(model=model, args=args, train_dataset=Blocks(train_idx), eval_dataset=Blocks(eval_idx),
                  data_collator=coll, callbacks=[Speed(), Mirror()])

rec = json.load(open(res_path)) if os.path.exists(res_path) else {}
if "eval_loss_before" not in rec and not a.probe:
    rec["eval_loss_before"] = trainer.evaluate()["eval_loss"]   # LoRA B=0 → 원래 모델과 같은 값
    log(f"학습 전 검증 손실 {rec['eval_loss_before']:.4f}")
rec.update(name=a.name, model=a.model, corpus=a.corpus, corpus_meta=meta, max_len=a.max_len, blocks=n_blocks,
           micro_batch=a.micro_batch, grad_accum=a.grad_accum, lr=a.lr, lora_r=a.lora_r, expert_layers=layers)
if not a.probe: json.dump(rec, open(res_path, "w"), ensure_ascii=False, indent=1)

has_ckpt = (not a.probe) and any(d.startswith("checkpoint-") for d in os.listdir(ckpt_dir))
t0 = time.time()
trainer.train(resume_from_checkpoint=True if has_ckpt else None)
if a.probe:
    log("속도 측정 끝 (저장 안 함)"); raise SystemExit(0)
rec["eval_loss_after"] = trainer.evaluate()["eval_loss"]
log(f"학습 후 검증 손실 {rec['eval_loss_after']:.4f} (학습 전 {rec['eval_loss_before']:.4f})")

# 3) 저장 → 파일을 다시 읽어 확인
model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
mine = {k: v.detach().float().cpu() for k, v in get_peft_model_state_dict(model).items()}
from safetensors.torch import load_file
saved = {k: v.float() for k, v in load_file(os.path.join(out_dir, "adapter_model.safetensors")).items()}
miss = [k for k in mine if k not in saved]; diff = [k for k in mine if k in saved and not torch.equal(mine[k], saved[k])]
nonzero_B = sum(1 for k, v in mine.items() if "lora_B" in k and v.abs().sum() > 0)
ok = not miss and not diff and len(mine) > 0 and nonzero_B > 0
rec.update(saved_dir=out_dir, adapter_tensors=len(mine), lora_B_trained=nonzero_B, verify_ok=ok,
           train_minutes=round((time.time() - t0) / 60, 1), steps=trainer.state.global_step,
           log_history=trainer.state.log_history, done=ok)
json.dump(rec, open(res_path, "w"), ensure_ascii=False, indent=1)
if not ok:
    raise SystemExit(f"저장 확인 실패: 없는 텐서 {len(miss)} · 다른 텐서 {len(diff)} · 학습된 B {nonzero_B} — 체크포인트는 {ckpt_dir}")
log(f"저장 확인: LoRA 텐서 {len(mine)}개 일치 (학습된 B {nonzero_B}개) → {out_dir}")
print("VERIFY_OK")
