# -*- coding: utf-8 -*-
"""
데이터 실험용 학습 세트 (dev·test는 split_v1 그대로 고정).

1) 학습량 곡선: 작은 세트가 큰 세트에 포함되도록(nested) 만든다.
   lc_train_1500 ⊂ lc_train_6000 ⊂ lc_train_20000 ⊂ lc_train_60000(= split_v1 train) ⊂ lc_train_200000
2) 표본 흔들림: split_v1 train(=세트 A)과 서로 겹치지 않는 6만 세트 B, C.

추가 문장은 data/spans_full(학습용 TL1)에서 split_v1과 같은 규칙(카테고리 층화, 상용구·5글자 미만·중복 제외)으로 뽑고,
split_v1의 train/dev/test 문장과 글자가 같은 문장은 모두 뺀다.
사용: python src/make_experiment_sets.py
"""
import argparse, glob, json, os, sys, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_splits import collect, pick, rkey, describe

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def save(rows, p):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/spans_full"); ap.add_argument("--base", default="data/split_v1")
    ap.add_argument("--out", default="data/exp_sets"); ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--lc-sizes", default="1500,6000,20000,60000,200000")
    ap.add_argument("--var-size", type=int, default=60000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    sizes = sorted(int(x) for x in a.lc_sizes.split(","))
    A = load(os.path.join(a.base, "train.jsonl"))
    dev, test = load(os.path.join(a.base, "dev.jsonl")), load(os.path.join(a.base, "test.jsonl"))
    banned = {r["text"] for r in A + dev + test}

    big = max(sizes); n_extra = max(0, big - len(A)) + 2 * a.var_size
    files = sorted(glob.glob(os.path.join(a.src, "train", "*.jsonl.gz")))
    merged, cnt = collect(files, n_extra // 17 + 1, a.seed, a.workers)
    per = (int(n_extra * 1.02) + len(merged) - 1) // len(merged)     # 전역 중복 제거로 빠질 몫 2% 여유
    extra, st = pick(merged, per, frozenset(banned))
    seen, uniq = set(banned), []
    for r in sorted(extra, key=lambda r: rkey(a.seed + 1, r["id"])):   # 카테고리 섞기 + 전역 중복 제거
        if r["text"] in seen: continue
        seen.add(r["text"]); uniq.append(r)
    if len(uniq) < n_extra:
        raise SystemExit(f"추가 문장이 부족합니다: {len(uniq)} < {n_extra}")

    stats = {"args": vars(a), "rules": ["dev/test는 split_v1 고정", "lc 세트는 포함 관계(앞에서부터 자름)",
             "var B·C는 A·서로·lc_200000과 겹치지 않음", "추가 문장은 split_v1 train/dev/test와 같은 문장 제외"],
             "scan": dict(cnt), "pick": {k: v for k, v in st.items() if not k.startswith("cat:")}}
    n_lc_extra = max(0, big - len(A))
    lc_pool = A + uniq[:n_lc_extra]                       # A(6만) 다음에 추가분을 이어 붙임 → 앞에서 자르면 포함 관계
    for n in sizes:
        rows = lc_pool[:n]
        save(rows, os.path.join(a.out, f"lc_train_{n}.jsonl")); stats[f"lc_train_{n}"] = describe(rows)
    B = uniq[n_lc_extra:n_lc_extra + a.var_size]; C = uniq[n_lc_extra + a.var_size:n_lc_extra + 2 * a.var_size]
    for name, rows in (("var_train_B", B), ("var_train_C", C)):
        save(rows, os.path.join(a.out, f"{name}.jsonl")); stats[name] = describe(rows)
    # 겹침 검사
    tA, tB, tC, tL = ({r["text"] for r in x} for x in (A, B, C, lc_pool))
    stats["overlap_check"] = {"A∩B": len(tA & tB), "A∩C": len(tA & tC), "B∩C": len(tB & tC),
                              "lc200k∩B": len(tL & tB), "lc200k∩C": len(tL & tC),
                              "anyTrain∩test": len((tL | tB | tC) & {r['text'] for r in test})}
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: (v if k in ("overlap_check", "elapsed_sec") else {kk: vv for kk, vv in v.items() if kk in ("sentences", "entities")})
                      for k, v in stats.items() if k not in ("args", "rules", "scan", "pick")}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
