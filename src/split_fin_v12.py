# -*- coding: utf-8 -*-
"""금융 라벨 30만(v1.2)을 학습 28만 / 검증 1만 / 테스트 1만으로 나눈다 — 원본 파일 단위(같은 출처 문장이 갈라지지 않게).

  python src/split_fin_v12.py --zip "G:/내 드라이브/NER_라벨_통합_v1.2/labeled_all_v1.2.zip" --out data/fin_v12

- 문장 id "TL1_{원본파일}_{t|c}_{문서}_{문장}" 의 원본파일(5,070개)을 묶음 단위로 섞어(시드 고정) dev·test 가 1만 문장을 넘을 때까지 채운다.
- 검사: 세 세트 사이 id·문장·원본파일 겹침 0, 크롤링 테스트(fin_test_crawl)와 문장 겹침, 태그 비율·문장당 개체 수 비교.
출력: train.jsonl · dev.jsonl · test.jsonl (id·text·spans·member·site) + split_stats.json
"""
import argparse, collections, json, os, random, zipfile

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True); ap.add_argument("--out", default="data/fin_v12")
ap.add_argument("--n_dev", type=int, default=10000); ap.add_argument("--n_test", type=int, default=10000); ap.add_argument("--seed", type=int, default=20261027)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

z = zipfile.ZipFile(a.zip); name = [i.filename for i in z.infolist() if i.filename.endswith(".jsonl")][0]
rows = [json.loads(l) for l in z.open(name)]
grp = collections.defaultdict(list)
for i, r in enumerate(rows): grp[r["id"].split("_")[1]].append(i)
files = sorted(grp); random.Random(a.seed).shuffle(files)
part, need, cur = {}, [("test", a.n_test), ("dev", a.n_dev)], 0
fi = 0
for nm, n in need:
    got = 0
    while got < n:
        f = files[fi]; fi += 1; part[f] = nm; got += len(grp[f])
for f in files[fi:]: part[f] = "train"
sets = {"train": [], "dev": [], "test": []}
for f, nm in part.items(): sets[nm] += [rows[i] for i in grp[f]]
for nm in sets: sets[nm].sort(key=lambda r: r["id"])

def stat(rs):
    c = collections.Counter(t for r in rs for _, _, t in r["spans"]); n = sum(c.values())
    return {"sentences": len(rs), "entities": n, "per_sentence": round(n / len(rs), 3), "no_entity": round(sum(not r["spans"] for r in rs) / len(rs), 3),
            "files": len({r["id"].split("_")[1] for r in rs}), "tag_pct": {t: round(100 * c[t] / n, 2) for t in sorted(c)},
            "members": dict(collections.Counter(r["member"] for r in rs))}
st = {nm: stat(rs) for nm, rs in sets.items()}
ids = {nm: {r["id"] for r in rs} for nm, rs in sets.items()}; tx = {nm: {r["text"] for r in rs} for nm, rs in sets.items()}
fs = {nm: {r["id"].split("_")[1] for r in rs} for nm, rs in sets.items()}
pairs = [("train", "dev"), ("train", "test"), ("dev", "test")]
st["overlap"] = {f"{x}-{y}": {"id": len(ids[x] & ids[y]), "text": len(tx[x] & tx[y]), "file": len(fs[x] & fs[y])} for x, y in pairs}
crawl = "data/fin_test_crawl/test_crawl.jsonl"
if os.path.exists(crawl):
    ct = {json.loads(l)["text"] for l in open(crawl, encoding="utf-8")}
    st["overlap_with_crawl_test"] = {nm: len(tx[nm] & ct) for nm in sets}
st["total"] = sum(len(v) for v in sets.values()); st["seed"] = a.seed; st["source"] = a.zip
assert st["total"] == len(rows) == len({r["id"] for r in rows})
assert all(v["id"] == 0 and v["file"] == 0 for v in st["overlap"].values())
for nm, rs in sets.items():
    with open(os.path.join(a.out, f"{nm}.jsonl"), "w", encoding="utf-8") as f:
        for r in rs: f.write(json.dumps({k: r[k] for k in ("id", "text", "spans", "member", "site")}, ensure_ascii=False) + "\n")
json.dump(st, open(os.path.join(a.out, "split_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(st, ensure_ascii=False, indent=1))
