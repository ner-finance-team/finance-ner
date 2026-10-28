# -*- coding: utf-8 -*-
"""
학습량 곡선 세트를 크게 늘린다(메모리 절약판): train_{N}k = 기존 세트(앞부분 그대로) + 새 문장.
기존 extend_lc.py 와 같은 규칙:
  - spans_full/train(TL1)에서 카테고리마다 같은 수를 무작위로(시드 해시 rkey) 뽑는다.
  - 상용구·5글자 미만·같은 문장 중복 제외.
  - 기존 세트, 표본 B·C, dev·test(v1=v3b)와 글자가 같은 문장 제외.
추가 규칙(10/22):
  - 2단계 골드셋 510문장(data/goldset/goldset_master.jsonl)과 같은 문장도 제외(2단계 평가 오염 방지).
  - 정답 span은 고친 규칙(split_v3b: 단어 중간 출현 제외)으로 다시 만든다.
메모리: 1차로 카테고리별 후보 수만 세고, 2차에서 rkey가 문턱보다 작은 문장만 모은다(전체를 메모리에 올리지 않음).
사용: python src/extend_lc2.py --to 1200000 --base data/split_v3b/train_600k.jsonl --out data/split_v3b/train_1200k.jsonl
"""
import argparse, glob, gzip, json, os, sys, time, collections
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_splits import rkey
from make_split_v3b import spans_v3b

SEED = 1200   # main()에서 --seed(기본: 목표 크기/1000)로 바꾼다. 같은 시드를 다시 쓰면 이미 뽑힌 문장과 겹친다(10/23 실패 원인)
def ok(r): return (not r["boiler"]) and len(r["text"]) >= 5

def count(path):
    c = collections.Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if ok(r): c[r["category"]] += 1
    return c

def gather(job):
    path, thr, seed = job          # 시드는 작업마다 직접 넘긴다(Windows 다중 프로세스는 전역 변수를 새로 읽음)
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not ok(r): continue
            k = rkey(seed, r["id"])
            if k < thr[r["category"]]:
                out.append((k, {x: r[x] for x in ("id", "category", "site", "text", "labels", "unmatched")}))
    return out

def load_texts(p):
    with open(p, encoding="utf-8") as f: return {json.loads(l)["text"] for l in f}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--to", type=int, required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(); t0 = time.time()
    global SEED
    SEED = a.seed or a.to // 1000          # 120만 → 1200(기존과 같음), 200만 → 2000
    print(f"시드 {SEED}", flush=True)
    base = [json.loads(l) for l in open(a.base, encoding="utf-8")]
    EXCL = ("data/exp_sets/var_train_B.jsonl", "data/exp_sets/var_train_C.jsonl", "data/split_v1/dev.jsonl",
            "data/split_v1/test.jsonl", "data/goldset/goldset_master.jsonl")
    excl = set().union(*(load_texts(p) for p in EXCL))
    # 기존 부분 정리(10/22 검증에서 발견): 평가·골드셋과 같은 문장, 카테고리가 달라 남아 있던 중복 문장(앞의 것만 남김)
    n0 = len(base); seen0, clean = set(), []
    for r in base:
        if r["text"] in excl or r["text"] in seen0: continue
        seen0.add(r["text"]); clean.append(r)
    base = clean
    print(f"기존 {n0:,}문장 중 평가·골드셋과 같거나 중복인 문장 {n0-len(base)}개 제외", flush=True)
    banned = {r["text"] for r in base} | excl
    need = a.to - len(base)
    files = sorted(glob.glob("data/spans_full/train/*.jsonl.gz"))
    with Pool(a.workers) as P: counts = sum(P.map(count, files), collections.Counter())
    cats = sorted(counts); per = need // len(cats) + 1
    # 문턱: 카테고리 c에서 per×1.25개 정도가 걸리도록 (중복·제외 몫 여유)
    thr = {c: int(min(1.0, per * 1.25 / counts[c]) * (2 ** 64 - 1)) for c in cats}
    print(f"카테고리 {len(cats)}개, 카테고리당 {per:,}개 목표, 1차 {time.time()-t0:.0f}초", flush=True)
    with Pool(a.workers) as P: got = [x for part in P.imap_unordered(gather, [(f, thr, SEED) for f in files]) for x in part]
    by = collections.defaultdict(list)
    for k, r in got: by[r["category"]].append((k, r))
    new, seen, short = [], set(banned), {}
    for c in cats:
        n = 0
        for k, r in sorted(by[c], key=lambda x: x[0]):
            if r["text"] in seen: continue
            seen.add(r["text"]); n += 1
            r["spans"] = spans_v3b(r["text"], [tuple(x) for x in r["labels"]])
            r.pop("unmatched", None); new.append(r)
            if n >= per: break
        if n < per: short[c] = n
    if short: print("부족한 카테고리:", short)
    new = sorted(new, key=lambda r: rkey(SEED + 1, r["id"]))[:need]
    if len(new) < need: raise SystemExit(f"부족: {len(new)} < {need}")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in base + new: f.write(json.dumps({k: r[k] for k in ("id", "category", "site", "text", "labels", "spans")}, ensure_ascii=False) + "\n")
    st = {"out": a.out, "base": a.base, "base_n_before_gold_removal": n0, "base_n": len(base), "added": len(new), "total": len(base) + len(new), "seed": SEED,
          "candidates_by_category": dict(counts), "elapsed_sec": round(time.time() - t0, 1)}
    json.dump(st, open(a.out.replace(".jsonl", "_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in st.items() if k != "candidates_by_category"}, ensure_ascii=False))

if __name__ == "__main__":
    main()
