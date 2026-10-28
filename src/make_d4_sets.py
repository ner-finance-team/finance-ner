# -*- coding: utf-8 -*-
"""
D4 드문 태그 보강 실험 세트 (모델 고정 klue/roberta-base, 3에폭. 비교 기준 D1_lc_300000)
  D4_rare : 팀 기준 30만 + 드문 태그(NW·TR·FD) 실제 문장 7만   ← data/rare_pool
  D4_rand : 팀 기준 30만 + 무작위 실제 문장 7만               ← lc_train_600000 의 뒤 30만(같은 표본 규칙)에서 앞 7만
            → 두 세트의 문장 수가 같으므로, 차이는 '어떤 문장을 더했나'에서만 생긴다.
  X_TR    : TR 전문가용. TR 라벨만 남긴 학습 세트(다른 태그 라벨은 지움).
            TR 문장(풀 5만 + 30만 안의 TR 문장) + TR 없는 문장(30만에서 같은 수) → TR만 배우게 한다.
드문 태그 문장 거르기: 한 글자 FD('시' 등), TR '이름'((이름) 비식별 표시), FD 'lg화학' 같은 확인된 오라벨이 든 문장은 뺀다.
같은 개체 문자열이 너무 많이 반복되지 않게 문자열당 문장 수에 상한(CAP)을 둔다(예: 인플루언서 7,589문장 → 상한까지만).
사용: python src/make_d4_sets.py → data/d4_sets/*.jsonl, stats_d4.json
"""
import collections, json, os, random

OUT = "data/d4_sets"
ADD = {"NW": 15000, "TR": 30000, "FD": 25000}          # 합 7만
CAP = {"NW": 20, "TR": 10, "FD": 20}                   # 개체 문자열당 최대 문장 수
BAD = {("TR", "이름"), ("FD", "lg화학"), ("FD", "LG화학")}
X_TR_POOL = 50000

def load(p):
    with open(p, encoding="utf-8") as f: return [json.loads(l) for l in f]

def dump(rows, p):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    open(p + ".done", "w").close()

def clean(r):
    for s, e, t in r["spans"]:
        if t in ("FD", "TR", "NW") and e - s == 1: return False
        if (t, r["text"][s:e]) in BAD: return False
    return True

def tag_count(rows):
    c = collections.Counter(t for r in rows for _, _, t in r["spans"]); return {t: c[t] for t in ("CV", "OG", "TM", "AF", "FD", "TR", "NW")}

def pick_capped(pool, tag, n, cap, used, rng):
    rng.shuffle(pool)
    per, out = collections.Counter(), []
    for r in pool:
        if len(out) >= n: break
        if r["text"] in used or not clean(r): continue
        keys = {r["text"][s:e] for s, e, t in r["spans"] if t == tag}
        if any(per[k] >= cap for k in keys): continue
        for k in keys: per[k] += 1
        used.add(r["text"]); out.append(r)
    return out, len(per)

def main():
    os.makedirs(OUT, exist_ok=True); rng = random.Random(7)
    base = load("data/split_v3/train.jsonl")
    used = {r["text"] for r in base}
    st = {"base": len(base), "base_tags": tag_count(base), "rare": {}}
    rare = []
    for tag in ("NW", "TR", "FD"):          # 드문 것부터 채움
        pool = load(f"data/rare_pool/{tag}.jsonl")
        got, distinct = pick_capped(pool, tag, ADD[tag], CAP[tag], used, rng)
        rare += got
        st["rare"][tag] = {"added_sentences": len(got), "distinct_strings": distinct, "target": ADD[tag]}
        print(f"  {tag}: {len(got):,}문장 (서로 다른 {tag} 문자열 {distinct:,}개)")
    n_add = len(rare)
    extra = load("data/exp_sets/lc_train_600000.jsonl")[300000:]
    assert not ({r["text"] for r in extra[:1000]} & {r["text"] for r in base}), "60만 뒤쪽이 30만과 겹침"
    rand = [r for r in extra if r["text"] not in used][:n_add]
    rare_set, rand_set = base + rare, base + rand
    dump(rare_set, f"{OUT}/D4_rare_train.jsonl"); dump(rand_set, f"{OUT}/D4_rand_train.jsonl")
    st["D4_rare"] = {"sentences": len(rare_set), "tags": tag_count(rare_set)}
    st["D4_rand"] = {"sentences": len(rand_set), "tags": tag_count(rand_set)}

    # TR 전문가: TR 라벨만 남김
    tr_pool = load("data/rare_pool/TR.jsonl")
    used_x = set(used)
    tr_extra, _ = pick_capped(tr_pool, "TR", X_TR_POOL, CAP["TR"], used_x, random.Random(11))
    tr_base = [r for r in base if any(t == "TR" for _, _, t in r["spans"]) and clean(r)]
    pos = tr_extra + tr_base
    neg_all = [r for r in base if not any(t == "TR" for _, _, t in r["spans"])]
    neg = random.Random(12).sample(neg_all, len(pos))
    x = [{**r, "spans": [sp for sp in r["spans"] if sp[2] == "TR"]} for r in pos + neg]
    random.Random(13).shuffle(x)
    dump(x, f"{OUT}/X_TR_train.jsonl")
    st["X_TR"] = {"sentences": len(x), "tr_sentences": len(pos), "neg_sentences": len(neg), "tr_entities": sum(len(r["spans"]) for r in x)}
    json.dump(st, open(f"{OUT}/stats_d4.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in st.items() if k != "rare"}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
