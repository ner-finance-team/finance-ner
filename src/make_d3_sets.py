# -*- coding: utf-8 -*-
"""
D3 데이터 개선 실험용 학습 세트 만들기. 모델은 고정(klue/roberta-base), 학습 라벨만 바꾼다.
규칙의 통계(문자열별 태그 분포)는 학습 데이터(split_v3 train 30만)에서만 낸다. dev·test는 건드리지 않는다.

  D3a 다수결 정제 : 같은 문자열이 MIN_A번 이상 라벨되고 한 태그가 SHARE 이상이면, 나머지 태그를 그 태그로 바꾼다.
  D3b 누락 보완   : 2글자 이상, MIN_B번 이상 라벨, 한 태그가 SHARE 이상, 그리고 단어 시작 위치에 나왔을 때
                    라벨된 비율이 LAB_RATE 이상인 문자열(일반 명사일 가능성이 낮은 것)이
                    단어 시작 위치에 라벨 없이 나오면 그 태그로 라벨을 추가(기존 개체와 겹치면 건너뜀).
  D3c 한 글자 제거 : 한 글자 개체 라벨을 지운다(O로).
  D3ab            : D3a 다음 D3b.

같은 규칙을 dev·test에 "적용했다면" 몇 개가 바뀌는지도 센다(평가 데이터에도 같은 노이즈가 있는지 보는 용도, 파일은 만들지 않음).

사용: python src/make_d3_sets.py            → data/d3_sets/{D3a,D3b,D3c,D3ab}_train.jsonl, stats_d3.json, 예시 파일
"""
import json, os, random, collections

TRAIN = "data/split_v3/train.jsonl"
DEV, TEST = "data/split_v1/dev.jsonl", "data/split_v1/test.jsonl"
OUT = "data/d3_sets"
SHARE, MIN_A, MIN_B, LAB_RATE = 0.95, 10, 20, 0.5
WORD_START_PREV = set(" \t\"'(“‘[<«·/")

def load(p):
    with open(p, encoding="utf-8") as f: return [json.loads(l) for l in f]

def dump(rows, p):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def is_word_start(text, i):
    return i == 0 or text[i - 1] in WORD_START_PREV

def tag_stats(rows):
    """문자열 → Counter(태그)"""
    c = collections.defaultdict(collections.Counter)
    for r in rows:
        for s, e, t in r["spans"]: c[r["text"][s:e]][t] += 1
    return c

def majority_map(stats, min_n):
    m = {}
    for w, cnt in stats.items():
        n = sum(cnt.values())
        if n < min_n: continue
        t, k = cnt.most_common(1)[0]
        if k / n >= SHARE: m[w] = t
    return m

def apply_a(rows, amap):
    out, changed, ex = [], collections.Counter(), []
    for r in rows:
        sp = []
        for s, e, t in r["spans"]:
            w = r["text"][s:e]; nt = amap.get(w, t)
            if nt != t:
                changed[(w, t, nt)] += 1
                if len(ex) < 300: ex.append({"id": r["id"], "text": r["text"], "span": [s, e], "str": w, "from": t, "to": nt})
            sp.append([s, e, nt])
        out.append({**r, "spans": sp})
    return out, changed, ex

def build_b_candidates(rows, stats):
    """단어 시작 위치 출현 중 라벨 비율 계산 → 누락 보완 후보."""
    base = {w: t for w, t in majority_map(stats, MIN_B).items() if len(w) >= 2}
    by2 = collections.defaultdict(list)
    for w in base: by2[w[:2]].append(w)
    for k in by2: by2[k].sort(key=len, reverse=True)
    occ, lab = collections.Counter(), collections.Counter()
    for r in rows:
        text = r["text"]; starts = {(s, e) for s, e, _ in r["spans"]}
        for i in range(len(text) - 1):
            if not is_word_start(text, i): continue
            for w in by2.get(text[i:i + 2], ()):
                if text.startswith(w, i):
                    occ[w] += 1
                    if (i, i + len(w)) in starts: lab[w] += 1
                    break
    cand = {w: base[w] for w in base if occ[w] and lab[w] / occ[w] >= LAB_RATE}
    rate = {w: (lab[w], occ[w]) for w in base}
    return cand, rate

def apply_b(rows, cand):
    by2 = collections.defaultdict(list)
    for w in cand: by2[w[:2]].append(w)
    for k in by2: by2[k].sort(key=len, reverse=True)
    out, added, ex = [], collections.Counter(), []
    for r in rows:
        text = r["text"]; sp = [list(x) for x in r["spans"]]
        taken = [False] * (len(text) + 1)
        for s, e, _ in sp:
            for j in range(s, e): taken[j] = True
        i = 0
        while i < len(text) - 1:
            hit = None
            if is_word_start(text, i) and not taken[i]:
                for w in by2.get(text[i:i + 2], ()):
                    if text.startswith(w, i) and not any(taken[i:i + len(w)]): hit = w; break
            if hit:
                sp.append([i, i + len(hit), cand[hit]]); added[(hit, cand[hit])] += 1
                for j in range(i, i + len(hit)): taken[j] = True
                if len(ex) < 300: ex.append({"id": r["id"], "text": text, "span": [i, i + len(hit)], "str": hit, "tag": cand[hit]})
                i += len(hit)
            else: i += 1
        sp.sort(key=lambda x: (x[0], x[1]))
        out.append({**r, "spans": sp})
    return out, added, ex

def apply_c(rows):
    out, removed = [], collections.Counter()
    for r in rows:
        sp = []
        for s, e, t in r["spans"]:
            if e - s == 1: removed[(r["text"][s:e], t)] += 1
            else: sp.append([s, e, t])
        out.append({**r, "spans": sp})
    return out, removed

def n_ent(rows): return sum(len(r["spans"]) for r in rows)

def top(counter, k=25): return [[list(key) if isinstance(key, tuple) else key, v] for key, v in counter.most_common(k)]

def main():
    os.makedirs(OUT, exist_ok=True)
    train = load(TRAIN); dev = load(DEV); test = load(TEST)
    stats = tag_stats(train)
    amap = majority_map(stats, MIN_A)
    cand, rate = build_b_candidates(train, stats)
    print(f"학습 {len(train):,}문장, 개체 {n_ent(train):,}개 | D3a 규칙 문자열 {len(amap):,}개 | D3b 후보 문자열 {len(cand):,}개")

    S = {"params": dict(SHARE=SHARE, MIN_A=MIN_A, MIN_B=MIN_B, LAB_RATE=LAB_RATE), "train_entities": n_ent(train),
         "rule_strings": {"D3a": len(amap), "D3b": len(cand)}, "sets": {}, "if_applied_to_eval": {}}
    a, ch_a, ex_a = apply_a(train, amap)
    b, add_b, ex_b = apply_b(train, cand)
    c, rm_c = apply_c(train)
    ab, add_ab, _ = apply_b(a, cand)
    for name, rows, n_changed, cnt in [("D3a", a, sum(ch_a.values()), ch_a), ("D3b", b, sum(add_b.values()), add_b),
                                       ("D3c", c, sum(rm_c.values()), rm_c), ("D3ab", ab, sum(ch_a.values()) + sum(add_ab.values()), None)]:
        dump(rows, f"{OUT}/{name}_train.jsonl")
        S["sets"][name] = {"entities": n_ent(rows), "changed": n_changed, "top_changes": top(cnt) if cnt else None}
        print(f"  {name}: 개체 {n_ent(train):,} → {n_ent(rows):,}, 바뀐 라벨 {n_changed:,}개")
    dump(ex_a, f"{OUT}/examples_D3a.jsonl"); dump(ex_b, f"{OUT}/examples_D3b.jsonl")
    for split, rows in (("dev", dev), ("test", test)):
        _, ca, _ = apply_a(rows, amap); _, cb, _ = apply_b(rows, cand); _, cc = apply_c(rows)
        S["if_applied_to_eval"][split] = {"entities": n_ent(rows), "D3a_changed": sum(ca.values()),
                                          "D3b_added": sum(cb.values()), "D3c_removed": sum(cc.values())}
        print(f"  (같은 규칙을 {split}에 적용했다면) 개체 {n_ent(rows):,} 중 D3a {sum(ca.values()):,} / D3b +{sum(cb.values()):,} / D3c -{sum(cc.values()):,}")
    with open(f"{OUT}/stats_d3.json", "w", encoding="utf-8") as f: json.dump(S, f, ensure_ascii=False, indent=1)
    with open(f"{OUT}/D3b_candidates.json", "w", encoding="utf-8") as f:
        json.dump({w: {"tag": t, "labeled": rate[w][0], "word_start_occ": rate[w][1]} for w, t in sorted(cand.items(), key=lambda x: -rate[x[0]][1])},
                  f, ensure_ascii=False, indent=0)
    for name in ("D3a", "D3b", "D3c", "D3ab"):
        open(f"{OUT}/{name}_train.jsonl.done", "w").close()

if __name__ == "__main__":
    main()
