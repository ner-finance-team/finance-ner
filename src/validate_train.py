# -*- coding: utf-8 -*-
"""
학습 데이터 파일 독립 검증. 만든 코드와 상관없이 결과 파일만 보고 규칙이 지켜졌는지 확인한다. 하나라도 FAIL이면 실험하지 않는다.
검사
  1 줄 수 = 기대값            2 id·문장 중복 없음
  3 앞부분 = 기존 세트(골드셋 문장 뺀 것)와 같은 순서·같은 id
  4 dev·test·표본 B·C·2단계 골드셋과 같은 문장 없음
  5 span 형식: 0 ≤ 시작 < 끝 ≤ 문장 길이, 태그 7종, 정렬, 서로 안 겹침
  6 span 글자가 그 문장의 원본 라벨(글자·태그)에 있음
  7 span = 고친 규칙(spans_v3b)으로 원본 라벨에서 다시 만든 것과 같음
  8 라벨 없는 문장에 span 없음, 5글자 미만 문장 없음, 상용구 패턴 문장 없음
  9 분포: 추가된 부분의 카테고리별 문장 수, 태그 비율이 기존 부분과 비슷한지(태그 비율 차이 2%p 이내)
사용: python src/validate_train.py --file data/split_v3b/train_1200k.jsonl --base data/split_v3b/train_600k.jsonl --n 1200000
"""
import argparse, collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_split_v3b import spans_v3b
TAGS = {"CV", "OG", "TM", "AF", "FD", "TR", "NW"}
BOILER = re.compile(r"(저작권자|무단\s*전재|재배포\s*금지|ⓒ|©)")

def texts(p): return {json.loads(l)["text"] for l in open(p, encoding="utf-8")}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--file", required=True); ap.add_argument("--base"); ap.add_argument("--n", type=int)
    a = ap.parse_args()
    R = [json.loads(l) for l in open(a.file, encoding="utf-8")]
    res = []
    def check(name, ok, detail=""): res.append((ok, name, detail)); print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)
    check("1 줄 수", a.n is None or len(R) == a.n, f"{len(R):,}")
    ids = collections.Counter(r["id"] for r in R); tx = collections.Counter(r["text"] for r in R)
    check("2 id 중복 없음", max(ids.values()) == 1, f"중복 {sum(v-1 for v in ids.values() if v>1)}")
    check("2 문장 중복 없음", max(tx.values()) == 1, f"중복 {sum(v-1 for v in tx.values() if v>1)}")
    gold = texts("data/goldset/goldset_master.jsonl")
    nb = 0
    if a.base:
        excl = set().union(*(texts(p) for p in ("data/split_v1/dev.jsonl", "data/split_v1/test.jsonl", "data/exp_sets/var_train_B.jsonl",
                                                  "data/exp_sets/var_train_C.jsonl", "data/goldset/goldset_master.jsonl")))
        B, s0 = [], set()
        for l in open(a.base, encoding="utf-8"):
            r = json.loads(l)
            if r["text"] in excl or r["text"] in s0: continue
            s0.add(r["text"]); B.append(r["id"])
        nb = len(B)
        check("3 앞부분 = 기존 세트(평가·골드셋·중복 제외)", [r["id"] for r in R[:nb]] == B, f"기존 {nb:,}")
    bad = {}
    for name, p in [("dev", "data/split_v1/dev.jsonl"), ("test", "data/split_v1/test.jsonl"), ("표본B", "data/exp_sets/var_train_B.jsonl"),
                    ("표본C", "data/exp_sets/var_train_C.jsonl"), ("골드셋", "data/goldset/goldset_master.jsonl")]:
        bad[name] = len(texts(p) & set(tx))
    check("4 평가·골드셋과 겹침 없음", all(v == 0 for v in bad.values()), str(bad))
    e5 = e6 = e7 = e8 = 0; ex = []
    for r in R:
        t = r["text"]; sp = r["spans"]; labs = {(x[0], x[1]) for x in r["labels"]}
        if any(not (0 <= s < e <= len(t)) or tg not in TAGS for s, e, tg in sp) or sp != sorted(sp) or \
           any(sp[i][1] > sp[i + 1][0] for i in range(len(sp) - 1)): e5 += 1
        if any((t[s:e], tg) not in labs for s, e, tg in sp): e6 += 1; ex.append(("6", r["id"]))
        if sp != spans_v3b(t, [tuple(x) for x in r["labels"]]): e7 += 1; ex.append(("7", r["id"]))
        if (not r["labels"] and sp) or len(t) < 5 or BOILER.search(t): e8 += 1
    check("5 span 형식", e5 == 0, f"문제 문장 {e5}")
    check("6 span 글자가 원본 라벨에 있음", e6 == 0, f"문제 문장 {e6}")
    check("7 고친 규칙(v3b)과 일치", e7 == 0, f"문제 문장 {e7}")
    check("8 라벨 없음·짧은 문장·상용구 없음", e8 == 0, f"문제 문장 {e8}")
    if a.base and nb < len(R):
        old, new = R[:nb], R[nb:]
        cat = collections.Counter(r["category"] for r in new)
        spread = (max(cat.values()) - min(cat.values())) / max(1, max(cat.values()))
        check("9 추가분 카테고리 균등(최대·최소 차 5% 이내)", spread <= 0.05, f"{len(cat)}개, 최소 {min(cat.values()):,} 최대 {max(cat.values()):,}")
        def ratio(rows):
            c = collections.Counter(tg for r in rows for _, _, tg in r["spans"]); n = sum(c.values()); return {k: c[k] / n for k in TAGS}, n / len(rows)
        ro, eo = ratio(old); rn, en = ratio(new)
        diff = max(abs(ro[k] - rn[k]) for k in TAGS)
        check("9 태그 비율 비슷(차 2%p 이내)", diff <= 0.02, f"최대 차 {diff*100:.2f}%p, 문장당 개체 기존 {eo:.3f} / 추가 {en:.3f}")
        lo = sum(len(r['text']) for r in old) / len(old); ln = sum(len(r['text']) for r in new) / len(new)
        check("9 평균 문장 길이 비슷(차 5% 이내)", abs(lo - ln) / lo <= 0.05, f"기존 {lo:.1f} / 추가 {ln:.1f}")
    n_fail = sum(1 for ok, _, _ in res if not ok)
    print(f"\n결과: {len(res)-n_fail}/{len(res)} 통과" + ("" if not n_fail else f" — FAIL {n_fail}개, 실험하지 말 것. 예: {ex[:5]}"))
    sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
    main()
