# -*- coding: utf-8 -*-
"""1단계 학습 세트를 크게(500만) 만든다 — 메모리 절약판(이 PC 여유 메모리 3GB 기준).

  python src/make_train_big.py --to 5000000 --base data/split_v3b/train_1200k.jsonl --out data/split_v3b/train_5000k.jsonl

규칙 (extend_lc2.py 와 같음 + 추가 1개)
  - spans_full/train(TL1)에서 카테고리마다 같은 수를 무작위(rkey 시드)로 뽑는다. 상용구·5글자 미만·같은 문장 제외.
  - 기존 세트(앞부분 그대로), 표본 B·C, 1단계 dev·test(v1·v3b), 2단계 골드셋과 글자가 같은 문장 제외.
  - [추가 10/25] Claude 금융 표본(pool 12,000 + 시범 500 + 맞추기 10)과 같은 문장도 제외(2단계 dev로 쓸 문장).
    기존 120만 부분에 있으면 그 문장도 빼고 개수를 기록한다.
  - 정답 span 은 고친 규칙(spans_v3b: 단어 중간 출현 제외)으로 원본 라벨에서 다시 만든다.
메모리: 문장 대신 8바이트 지문(blake2b)만 집합에 둔다. 후보는 (키, 카테고리, 지문, 파일, 줄) 숫자 배열로만 모은다.
  뽑은 문장은 무작위 순서 번호로 64개 임시 묶음에 나눠 쓴 뒤 묶음마다 정렬해 이어 붙인다(전체를 메모리에 올리지 않음).
"""
import argparse, glob, gzip, hashlib, json, os, sys, time, collections, shutil
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_splits import rkey
from make_split_v3b import spans_v3b

def H(t): return int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "big")
def ok(r): return (not r["boiler"]) and len(r["text"]) >= 5
EXCL = ["data/exp_sets/var_train_B.jsonl", "data/exp_sets/var_train_C.jsonl", "data/split_v1/dev.jsonl", "data/split_v1/test.jsonl",
        "data/split_v3b/dev.jsonl", "data/split_v3b/test.jsonl", "data/goldset/goldset_master.jsonl"]
CLAUDE = ["data/fin_silver/pool.jsonl", "data/fin_silver/claude_silver_500.jsonl", "data/fin_silver/claude_calib10_v1.jsonl"]
CDT = np.dtype([("k", "u8"), ("cat", "u1"), ("h", "u8"), ("f", "u2"), ("ln", "u4")])

def count(path):
    c = collections.Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if ok(r): c[r["category"]] += 1
    return c

def gather(job):
    fi, path, thr, cats, seed = job
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for ln, line in enumerate(f):
            r = json.loads(line)
            if not ok(r): continue
            k = rkey(seed, r["id"])
            if k < thr[r["category"]]:
                out.append((k, cats[r["category"]], H(r["text"]), fi, ln))
    return np.array(out, dtype=CDT)

def extract(job):
    path, want = job          # want: {줄 번호: 순서 번호}
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for ln, line in enumerate(f):
            if ln in want:
                r = json.loads(line)
                r["spans"] = [list(x) for x in spans_v3b(r["text"], [tuple(x) for x in r["labels"]])]
                rows.append((want[ln], json.dumps({k: r[k] for k in ("id", "category", "site", "text", "labels", "spans")}, ensure_ascii=False)))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", type=int, required=True); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(); t0 = time.time()
    seed = a.seed or a.to // 1000
    log = lambda *x: print(f"[{time.time()-t0:6.0f}s]", *x, flush=True)
    log(f"시드 {seed}")
    def hashes(paths): return {H(json.loads(l)["text"]) for p in paths for l in open(p, encoding="utf-8")}
    excl, claude = hashes(EXCL), hashes(CLAUDE)
    log(f"제외 지문: 평가·골드셋 {len(excl):,} · Claude 표본 {len(claude):,}")

    # 기존 부분: 그대로 앞에 둔다(평가·골드셋·Claude 표본·중복은 뺌)
    tmp = a.out + ".parts"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    seen, n0, drop = set(), 0, collections.Counter()
    with open(os.path.join(tmp, "base.jsonl"), "w", encoding="utf-8") as fo:
        for l in open(a.base, encoding="utf-8"):
            n0 += 1; r = json.loads(l); h = H(r["text"])
            if h in excl: drop["평가·골드셋"] += 1; continue
            if h in claude: drop["Claude 표본"] += 1; continue
            if h in seen: drop["중복"] += 1; continue
            seen.add(h); fo.write(l if l.endswith("\n") else l + "\n")
    nb = len(seen); need = a.to - nb
    log(f"기존 {n0:,} → {nb:,} (뺀 것 {dict(drop)}) · 새로 {need:,}")
    banned = seen | excl | claude

    files = sorted(glob.glob("data/spans_full/train/*.jsonl.gz"))
    with Pool(a.workers) as P: counts = sum(P.map(count, files), collections.Counter())
    cats = sorted(counts); cid = {c: i for i, c in enumerate(cats)}
    per = need // len(cats) + 1
    thr = {c: int(min(1.0, per * 1.25 / counts[c]) * (2 ** 64 - 1)) for c in cats}
    log(f"후보 세기 끝: 카테고리 {len(cats)}개, 카테고리당 {per:,}개 목표, 최소 후보 {min(counts.values()):,}")
    with Pool(a.workers) as P:
        cand = np.concatenate(P.map(gather, [(i, f, thr, cid, seed) for i, f in enumerate(files)]))
    log(f"1차 후보 {len(cand):,}개 ({cand.nbytes/2**20:.0f}MB)")

    pick, short = [], {}
    for c in cats:
        sub = cand[cand["cat"] == cid[c]]; sub = sub[np.argsort(sub["k"], kind="stable")]
        n = 0
        for row in sub:
            h = int(row["h"])
            if h in banned: continue
            banned.add(h); pick.append(row); n += 1
            if n >= per: break
        if n < per: short[c] = n
    if short: log("부족한 카테고리:", short)
    pick = np.array(pick, dtype=CDT)
    order = np.array([rkey(seed + 1, f"{int(r['f'])}:{int(r['ln'])}") for r in pick], dtype="u8")
    pick = pick[np.argsort(order, kind="stable")][:need]
    if len(pick) < need: raise SystemExit(f"부족: {len(pick):,} < {need:,}")
    log(f"선택 {len(pick):,}개")

    # 무작위 순서 번호(pos)로 64개 묶음에 나눠 쓰기 → 묶음마다 정렬
    NB = 64; bucket_size = (need + NB - 1) // NB
    jobs = collections.defaultdict(dict)
    for pos, r in enumerate(pick): jobs[int(r["f"])][int(r["ln"])] = pos
    outs = [open(os.path.join(tmp, f"b{i:02d}.tsv"), "w", encoding="utf-8") for i in range(NB)]
    done = 0
    with Pool(a.workers) as P:
        for rows in P.imap_unordered(extract, [(files[f], w) for f, w in jobs.items()]):
            for pos, js in rows: outs[pos // bucket_size].write(f"{pos}\t{js}\n")
            done += len(rows)
    for f in outs: f.close()
    if done != need: raise SystemExit(f"추출 수 불일치 {done} != {need}")
    log("추출 끝 → 합치기")
    with open(a.out + ".part", "w", encoding="utf-8") as fo:
        shutil.copyfileobj(open(os.path.join(tmp, "base.jsonl"), encoding="utf-8"), fo)
        for i in range(NB):
            rows = [l.rstrip("\n").split("\t", 1) for l in open(os.path.join(tmp, f"b{i:02d}.tsv"), encoding="utf-8")]
            rows.sort(key=lambda x: int(x[0]))
            for _, js in rows: fo.write(js + "\n")
    os.replace(a.out + ".part", a.out); shutil.rmtree(tmp, ignore_errors=True)
    st = {"out": a.out, "base": a.base, "base_n_before": n0, "base_dropped": dict(drop), "base_n": nb, "added": need, "total": nb + need,
          "seed": seed, "excluded_claude_pool": len(claude), "candidates_by_category": dict(counts), "elapsed_sec": round(time.time() - t0, 1)}
    json.dump(st, open(a.out.replace(".jsonl", "_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(json.dumps({k: v for k, v in st.items() if k != "candidates_by_category"}, ensure_ascii=False))

if __name__ == "__main__":
    main()
