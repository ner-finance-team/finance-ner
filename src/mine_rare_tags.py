# -*- coding: utf-8 -*-
"""
드문 태그(NW·TR·FD) 문장을 전체 학습용 span 파일(spans_full/train, TL1 5,856만 문장)에서 모두 뽑는다.
목적: 드문 태그 보강(실제 문장 추가)과 태그 전문가 모델 학습 재료. 증강(인공 문장)보다 먼저 실제 문장을 쓴다.
규칙: 표본과 같게 상용구·5글자 미만 제외, 위치를 못 찾은 라벨이 있는 문장 제외,
      팀 기준 학습(split_v3 30만)·dev·test·60만 세트와 글자가 같은 문장 제외, 풀 안에서도 중복 제거.
사용: python src/mine_rare_tags.py → data/rare_pool/{NW,TR,FD}.jsonl, data/rare_pool/stats.json
"""
import glob, gzip, json, os, time, collections
from multiprocessing import Pool

TAGS = ("NW", "TR", "FD")
KEYS = tuple(f'"{t}"]' for t in TAGS)

def scan(path):
    rows, n = [], collections.Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            n["lines"] += 1
            if not any(k in line for k in KEYS): continue
            r = json.loads(line)
            if r["boiler"] or len(r["text"]) < 5 or r.get("unmatched"): continue
            tags = {t for _, _, t in r["spans"]}
            if tags & set(TAGS):
                rows.append({k: r[k] for k in ("id", "category", "site", "text", "labels", "spans")})
    return rows, n

def load_texts(p):
    with open(p, encoding="utf-8") as f: return {json.loads(l)["text"] for l in f}

def main():
    t0 = time.time()
    banned = set()
    for p in ("data/split_v3/train.jsonl", "data/split_v1/dev.jsonl", "data/split_v1/test.jsonl", "data/exp_sets/lc_train_600000.jsonl"):
        banned |= load_texts(p)
    files = sorted(glob.glob("data/spans_full/train/*.jsonl.gz"))
    pool, seen, tot = {t: [] for t in TAGS}, set(), collections.Counter()
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as P:
        for i, (rows, n) in enumerate(P.imap_unordered(scan, files)):
            tot.update(n)
            for r in rows:
                if r["text"] in banned or r["text"] in seen: continue
                seen.add(r["text"])
                for t in {t for _, _, t in r["spans"]} & set(TAGS): pool[t].append(r)
            if (i + 1) % 20 == 0: print(f"  {i+1}/{len(files)} 파일, {time.time()-t0:.0f}초", flush=True)
    os.makedirs("data/rare_pool", exist_ok=True)
    st = {"scanned_lines": tot["lines"], "elapsed_sec": round(time.time() - t0, 1), "tags": {}}
    for t in TAGS:
        with open(f"data/rare_pool/{t}.jsonl", "w", encoding="utf-8") as f:
            for r in pool[t]: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ents = [r["text"][s:e] for r in pool[t] for s, e, tt in r["spans"] if tt == t]
        c = collections.Counter(ents)
        st["tags"][t] = {"sentences": len(pool[t]), "entities": len(ents), "distinct_strings": len(c),
                         "top": c.most_common(20)}
    json.dump(st, open("data/rare_pool/stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({t: {k: v for k, v in s.items() if k != "top"} for t, s in st["tags"].items()}, ensure_ascii=False),
          st["scanned_lines"], st["elapsed_sec"])

if __name__ == "__main__":
    main()
