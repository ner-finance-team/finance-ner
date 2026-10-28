# -*- coding: utf-8 -*-
"""
GPU 서버(A100 등)에서 노트북 없이 실험을 돌리는 실행 스크립트. 노트북 M1·M2·D1·D2와 같은 설정·같은 결과 이름을 쓴다.
결과는 reports/results/*.json 에 저장되고, 이미 있는 실험은 건너뛴다(끊겨도 다시 실행하면 이어서 진행).

준비 (서버에서 1회)
  pip install -U torch transformers datasets accelerate pytorch-crf pandas kiwipiepy
  프로젝트 폴더에 필요한 것: src/, data/split_v1/, data/exp_sets/(D1·D2), "대규모 웹데이터 기반 한국어 말뭉치 데이터/AI모델 및 소스코드.zip"(AI Hub 모델)

사용 예
  python src/run_experiments.py --group M1 --keys klue_roberta            # 기준선 1회 (시간 재기)
  python src/run_experiments.py --group M1                                # 후보 모델 전부
  python src/run_experiments.py --group D1                                # 학습량 곡선
  python src/run_experiments.py --group D2
  python src/run_experiments.py --group M2 --base klue/roberta-base
  python src/run_experiments.py --list                                    # 실험 목록만 보기
  공통 옵션: --member 이름 (기록용), --ckpt-dir 경로 (기본 checkpoints/, 끊기면 같은 명령을 다시 실행하면 이어서 학습)
여러 GPU면 CUDA_VISIBLE_DEVICES=0 ... / CUDA_VISIBLE_DEVICES=1 ... 로 그룹을 나눠 동시에 돌린다.
"""
import argparse, json, os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

def build(group, base, aihub_dir):
    SPLIT = os.path.join(PROJECT, "data", "split_v1"); EXP = os.path.join(PROJECT, "data", "exp_sets")
    RES = os.path.join(PROJECT, "reports", "results")
    common = dict(dev=f"{SPLIT}/dev.jsonl", test=f"{SPLIT}/test.jsonl", out_dir=RES, max_len=512, lr=5e-5, epochs=3, batch=16, seed=42)
    tr = f"{SPLIT}/train.jsonl"
    if group == "M1":
        models = {"klue_roberta": dict(model="klue/roberta-base"), "klue_bert": dict(model="klue/bert-base"),
                  "koelectra": dict(model="monologg/koelectra-base-v3-discriminator"),
                  "aihub_electra_1": dict(model=("AIHUB", "1세부")), "aihub_electra_2": dict(model=("AIHUB", "2세부")),
                  "klue_roberta_large": dict(model="klue/roberta-large", lr=2e-5, batch=8, grad_accum=2),
                  "klue_roberta_morph": dict(model="klue/roberta-base", pretok="morph"),          # 형태소 입력
                  "aihub_electra_1_morph": dict(model=("AIHUB", "1세부"), pretok="morph")}
        return {k: {**common, "train": tr, **v, "exp": f"M1_{k}"} for k, v in models.items()}
    if group == "M2":
        tag = base.split("/")[-1]
        return {"crf": {**common, "train": tr, "model": base, "head": "crf", "exp": f"M2_{tag}_crf"},
                "weighted": {**common, "train": tr, "model": base, "loss": "weighted", "exp": f"M2_{tag}_weighted"}}
    if group == "D1":
        return {str(n): {**common, "train": f"{EXP}/lc_train_{n}.jsonl", "model": base, "exp": f"D1_lc_{n}"}
                for n in (1500, 6000, 20000, 60000, 200000, 300000, 600000, 1000000)}
    if group == "D2":
        return {"A_seed42": {**common, "train": tr, "model": base, "exp": "D2_set_A_seed42"},
                "B_seed42": {**common, "train": f"{EXP}/var_train_B.jsonl", "model": base, "exp": "D2_set_B_seed42"},
                "C_seed42": {**common, "train": f"{EXP}/var_train_C.jsonl", "model": base, "exp": "D2_set_C_seed42"},
                "A_seed1": {**common, "train": tr, "model": base, "seed": 1, "exp": "D2_set_A_seed1"},
                "A_seed2": {**common, "train": tr, "model": base, "seed": 2, "exp": "D2_set_A_seed2"}}
    if group == "D3":   # 데이터 개선: 모델 고정, 학습 라벨만 바꾼 30만 (src/make_d3_sets.py). 비교 기준은 D1_lc_300000
        D3 = os.path.join(PROJECT, "data", "d3_sets")
        return {k: {**common, "train": f"{D3}/{k}_train.jsonl", "model": base, "exp": f"D3_{k}_300k"}
                for k in ("D3a", "D3b", "D3c", "D3ab")}
    if group == "D5":   # 단어 중간 출현을 뺀 고친 정답(split_v3b)으로 학습·평가. 비교: D1_lc_300000을 v3b로 다시 채점한 값
        V = os.path.join(PROJECT, "data", "split_v3b")
        c5 = {**common, "dev": f"{V}/dev.jsonl", "test": f"{V}/test.jsonl"}
        return {"v3b_300k": {**c5, "train": f"{V}/train.jsonl", "model": base, "exp": "D5_v3b_300k"},
                "v3b_600k": {**c5, "train": f"{V}/train_600k.jsonl", "model": base, "exp": "D5_v3b_600k"},
                "v3b_crf_300k": {**c5, "train": f"{V}/train.jsonl", "model": base, "head": "crf", "exp": "D5_v3b_crf_300k"}}
    raise SystemExit(f"모르는 그룹: {group}")

# 기준선과 완전히 같은 실험(같은 데이터·순서·시드)은 결과를 복사해 재사용
SAME_AS_BASELINE = {"D1_lc_60000", "D2_set_A_seed42"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["M1", "M2", "D1", "D2", "D3", "D5"]); ap.add_argument("--keys", nargs="*")
    ap.add_argument("--base", default="klue/roberta-base"); ap.add_argument("--aihub-dir", default=os.path.join(PROJECT, "aihub_models"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--member", default="", help="실행한 사람 이름 (runs.jsonl·점수 파일에 기록)")
    ap.add_argument("--ckpt-dir", default=os.path.join(PROJECT, "checkpoints"), help="에폭 체크포인트 위치(끊기면 여기서 이어서)")
    ap.add_argument("--aihub-zip", default=os.path.join(PROJECT, "대규모 웹데이터 기반 한국어 말뭉치 데이터", "AI모델 및 소스코드.zip"),
                    help="AI Hub 'AI모델 및 소스코드.zip' 위치")
    ap.add_argument("--train", default="", help="학습 파일 바꾸기 (예: data/exp_sets/lc_train_200000.jsonl)")
    ap.add_argument("--suffix", default="", help="실험 이름 꼬리표 (예: _200k → M1_koelectra_200k)")
    ap.add_argument("--epochs", type=int, default=0, help="에폭 바꾸기 (0이면 기본 3)")
    ap.add_argument("--micro-batch", type=int, default=0,
                    help="GPU 메모리가 작을 때: 한 번에 올리는 배치. 유효 배치(batch×grad_accum)는 그대로 유지하도록 누적 횟수를 자동 조정")
    a = ap.parse_args()
    groups = [a.group] if a.group else ["M1", "M2", "D1", "D2"]
    if a.list:
        for g in groups:
            for k, c in build(g, a.base, a.aihub_dir).items(): print(g, k, c["exp"], c["model"], os.path.basename(c["train"]))
        return
    import ner_lib
    ner_lib.DEFAULTS.update(member=a.member, ckpt_dir=a.ckpt_dir)
    STAGE = {"M1": "M1_모델비교", "M2": "M2_헤드손실", "D1": "D1_학습량곡선", "D2": "D2_표본흔들림", "D3": "D3_데이터개선", "D5": "D5_정답수정v3b"}
    for g in groups:
        ner_lib.DEFAULTS["stage"] = STAGE[g]
        cfgs = build(g, a.base, a.aihub_dir)
        for k in (a.keys or list(cfgs)):
            c = dict(cfgs[k])
            if a.train: c["train"] = a.train if os.path.isabs(a.train) else os.path.join(PROJECT, a.train)
            if a.epochs: c["epochs"] = a.epochs
            if a.suffix: c["exp"] = c["exp"] + a.suffix
            if a.micro_batch and a.micro_batch < c.get("batch", 16):
                eff = c.get("batch", 16) * c.get("grad_accum", 1)
                c["batch"], c["grad_accum"] = a.micro_batch, max(1, eff // a.micro_batch)
            if isinstance(c["model"], tuple):
                c["model"] = ner_lib.prepare_aihub_model(a.aihub_zip, c["model"][1], a.aihub_dir)
            base_json = os.path.join(c["out_dir"], "M1_klue_roberta.json")
            dst = os.path.join(c["out_dir"], f"{c['exp']}.json")
            if c["exp"] in SAME_AS_BASELINE and a.base == "klue/roberta-base" and os.path.exists(base_json) and not os.path.exists(dst):
                r = json.load(open(base_json, encoding="utf-8")); r["exp"] = c["exp"]
                json.dump(r, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print("기준선 결과 재사용:", c["exp"]); continue
            ner_lib.run_experiment(c)

if __name__ == "__main__":
    main()
