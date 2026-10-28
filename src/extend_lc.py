# -*- coding: utf-8 -*-
"""
학습량 곡선 세트를 더 크게 늘린다: lc_train_{N} = lc_train_200000 + 새 문장 (포함 관계 유지).
새 문장은 spans_full(학습용 TL1)에서 같은 규칙(카테고리 층화, 상용구·5글자 미만·중복 제외)으로 뽑고,
기존 lc_train_200000, var_train_B·C, split_v1 dev·test 와 글자가 같은 문장은 모두 뺀다.
사용: python src/extend_lc.py --to 300000
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_splits import collect, pick, rkey, describe

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", type=int, default=300000); ap.add_argument("--base", default="data/exp_sets/lc_train_200000.jsonl")
    ap.add_argument("--seed", type=int, default=300); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(); t0 = time.time()
    base = load(a.base)
    others = load("data/exp_sets/var_train_B.jsonl") + load("data/exp_sets/var_train_C.jsonl") + \
             load("data/split_v1/dev.jsonl") + load("data/split_v1/test.jsonl")
    banned = {r["text"] for r in base + others}
    need = a.to - len(base)
    files = sorted(glob.glob("data/spans_full/train/*.jsonl.gz"))
    merged, _ = collect(files, need // 17 + 1, a.seed, a.workers)
    per = (int(need * 1.05) + len(merged) - 1) // len(merged)
    extra, _ = pick(merged, per, frozenset(banned))
    seen, uniq = set(banned), []
    for r in sorted(extra, key=lambda r: rkey(a.seed + 1, r["id"])):
        if r["text"] in seen: continue
        seen.add(r["text"]); uniq.append(r)
    if len(uniq) < need: raise SystemExit(f"부족: {len(uniq)} < {need}")
    rows = base + uniq[:need]
    out = f"data/exp_sets/lc_train_{a.to}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    test_texts = {r["text"] for r in load("data/split_v1/test.jsonl")}
    st = {"file": out, "base": a.base, "added": need, **describe(rows),
          "overlap_check": {"new∩test": len({r["text"] for r in uniq[:need]} & test_texts),
                            "new∩base": len({r["text"] for r in uniq[:need]} & {r["text"] for r in base}),
                            "prefix_is_base": all(x["id"] == y["id"] for x, y in zip(rows, base))},
          "elapsed_sec": round(time.time() - t0, 1)}
    st.pop("categories", None)
    json.dump(st, open(f"data/exp_sets/stats_lc_{a.to}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(st, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
