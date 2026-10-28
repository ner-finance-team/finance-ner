# -*- coding: utf-8 -*-
"""경제 문장 540만에서 금융 라벨링용 100만 문장을 뽑아 5명에게 20만씩 나눈다.
제외: 2단계 골드셋(평가), 1단계 dev·test(split_v1=v3b), Claude 직접 라벨 표본 pool(12,000), 20~200자 밖, (이름) 2개 이상, 같은 문장 중복.
출력: fin_label_kit/data/shard_{이름}.jsonl  (k=분담 안 번호, id, site, text) + split_stats.json
"""
import json, random, hashlib, collections
MEMBERS = ["sejin_choi", "kim_byeongseok", "kim_hyeonho", "jin_taehyeon", "park_gyutae"]
N, SEED = 1_000_000, 20261023
def texts(p): return {json.loads(l)["text"] for l in open(p, encoding="utf-8")}
excl = texts("data/goldset/goldset_master.jsonl") | texts("data/split_v1/dev.jsonl") | texts("data/split_v1/test.jsonl") | texts("data/fin_silver/pool.jsonl")
rng = random.Random(SEED); keep, seen, n_ok, drop = [], set(), 0, collections.Counter()
for line in open("data/econ_pool/econ_all.jsonl", encoding="utf-8"):
    r = json.loads(line); t = r["text"]
    if t in excl: drop["평가·골드셋·Claude표본"] += 1; continue
    if not (20 <= len(t) <= 200): drop["길이"] += 1; continue
    if t.count("(이름)") > 1: drop["(이름)2개+"] += 1; continue
    h = hashlib.md5(t.encode()).digest()
    if h in seen: drop["중복"] += 1; continue
    seen.add(h); n_ok += 1
    item = (r["id"], r.get("site"), t)
    if len(keep) < N: keep.append(item)
    else:
        j = rng.randrange(n_ok)
        if j < N: keep[j] = item
rng.shuffle(keep)
st = {"candidates": n_ok, "dropped": dict(drop), "selected": len(keep), "seed": SEED, "per_member": {}}
for i, m in enumerate(MEMBERS):
    part = keep[i * 200000:(i + 1) * 200000]
    with open(f"fin_label_kit/data/shard_{m}.jsonl", "w", encoding="utf-8") as f:
        for k, (id_, site, t) in enumerate(part, 1): f.write(json.dumps({"k": k, "id": id_, "site": site, "text": t}, ensure_ascii=False) + "\n")
    st["per_member"][m] = {"n": len(part), "sites": dict(collections.Counter(s for _, s, _ in part))}
# 겹침 재검사(독립)
allt = [t for _, _, t in keep]
st["check"] = {"unique_texts": len(set(allt)), "overlap_with_excluded": len(set(allt) & excl)}
json.dump(st, open("fin_label_kit/data/split_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(st, ensure_ascii=False, indent=1))
