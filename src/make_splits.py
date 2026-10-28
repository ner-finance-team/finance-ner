# -*- coding: utf-8 -*-
"""
data/spans_full (전체 span) → 팀 공용 train / dev / test 표본.

규칙
- train·dev: 학습용 TL1에서, test: 검증용 VL1에서 뽑는다(AI Hub의 Validation을 우리는 test로 쓴다).
- 카테고리 층화: 17개 카테고리에서 같은 수씩.
- 제외: 상용구(boiler) 문장, 5글자 미만 문장, 같은 문장(text) 중복.
- 누수 방지: test 문장 중 train·dev와 글자가 똑같은 문장은 뺀다.
- 개체 없는 문장은 자연 비율 그대로 둔다(실제 입력도 그렇기 때문). 희소 태그 보강은 별도 실험(E4)으로 한다.
- 무작위 선택은 (seed, id) 해시로 정해 누가 다시 돌려도 같은 결과가 나온다.

사용: python src/make_splits.py --out data/split_v1
"""
import argparse, glob, gzip, hashlib, heapq, json, os, time, collections
from multiprocessing import Pool

KEEP = ("id", "category", "site", "text", "labels", "spans")

def rkey(seed, _id):
    return int.from_bytes(hashlib.blake2b(f"{seed}:{_id}".encode(), digest_size=8).digest(), "big")

def scan(job):
    """shard 하나에서 카테고리별로 무작위 키가 가장 작은 k개를 남긴다(병합하면 전체에서 무작위 표본과 같다)."""
    path, k, seed = job
    heaps = collections.defaultdict(list)       # 최대 힙(음수 키)으로 작은 키 k개 유지
    n = collections.Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n["seen"] += 1
            if r["boiler"]: n["drop_boiler"] += 1; continue
            if len(r["text"]) < 5: n["drop_short"] += 1; continue
            key = rkey(seed, r["id"])
            h = heaps[r["category"]]
            item = (-key, {x: r[x] for x in KEEP})
            if len(h) < k: heapq.heappush(h, item)
            elif -key > h[0][0]: heapq.heapreplace(h, item)
    return {c: [(-nk, rec) for nk, rec in h] for c, h in heaps.items()}, dict(n)

def collect(files, per_cat, seed, workers):
    buf = int(per_cat * 1.3) + 50                # 중복 제거로 빠질 몫까지 넉넉히
    merged, cnt = collections.defaultdict(list), collections.Counter()
    with Pool(workers) as pool:
        for got, n in pool.imap_unordered(scan, [(f, buf, seed) for f in files]):
            cnt.update(n)
            for c, items in got.items():
                merged[c].extend(items)
    return merged, cnt

def pick(merged, per_cat, banned=frozenset()):
    out, stat = [], collections.Counter()
    for c in sorted(merged):
        seen = set(); got = 0
        for _, rec in sorted(merged[c], key=lambda x: x[0]):
            if rec["text"] in seen: stat["drop_dup"] += 1; continue
            if rec["text"] in banned: stat["drop_overlap_with_train"] += 1; continue
            seen.add(rec["text"]); out.append(rec); got += 1
            if got >= per_cat: break
        stat[f"cat:{c}"] = got
    return out, stat

def describe(rows):
    tags = collections.Counter(t for r in rows for _, _, t in r["spans"])
    return {"sentences": len(rows), "entities": sum(tags.values()), "tags": dict(tags),
            "sent_no_entity": sum(1 for r in rows if not r["spans"]),
            "avg_len": round(sum(len(r["text"]) for r in rows) / max(len(rows), 1), 1),
            "categories": dict(collections.Counter(r["category"] for r in rows))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/spans_full"); ap.add_argument("--out", default="data/split_v1")
    ap.add_argument("--n-train", type=int, default=60000); ap.add_argument("--n-dev", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=10000); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    tr_files = sorted(glob.glob(os.path.join(a.src, "train", "*.jsonl.gz")))
    te_files = sorted(glob.glob(os.path.join(a.src, "valid", "*.jsonl.gz")))

    merged, c1 = collect(tr_files, (a.n_train + a.n_dev) // 17 + 1, a.seed, a.workers)
    ncat = len(merged); per = (a.n_train + a.n_dev + ncat - 1) // ncat
    trdev, s1 = pick(merged, per)
    rnd = sorted(trdev, key=lambda r: rkey(a.seed + 1, r["id"]))      # 카테고리 섞기
    dev, train = rnd[:a.n_dev], rnd[a.n_dev:a.n_dev + a.n_train]

    merged_t, c2 = collect(te_files, a.n_test // 17 + 1, a.seed + 2, a.workers)
    per_t = (a.n_test + len(merged_t) - 1) // len(merged_t)
    banned = frozenset(r["text"] for r in train + dev)
    test, s2 = pick(merged_t, per_t, banned)
    test = sorted(test, key=lambda r: rkey(a.seed + 3, r["id"]))[:a.n_test]

    stats = {"args": vars(a), "source": "data/spans_full (src/build_all_spans.py)",
             "rules": ["train/dev ← TL1, test ← VL1", "카테고리 층화", "boiler 제외", "5글자 미만 제외",
                       "같은 문장 중복 제외", "test에서 train/dev와 같은 문장 제외", "개체 없는 문장 자연 비율"],
             "scan_train": dict(c1), "pick_train": dict(s1), "scan_test": dict(c2), "pick_test": dict(s2)}
    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        with open(os.path.join(a.out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[name] = describe(rows)
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("categories",)} if isinstance(v, dict) and k in ("train", "dev", "test") else v
                      for k, v in stats.items() if k not in ("args", "rules")}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
