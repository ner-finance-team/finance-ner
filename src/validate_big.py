# -*- coding: utf-8 -*-
"""큰 학습 세트 + dev·test 독립 검증(한 줄씩 읽음 — 메모리 절약). 만든 코드(make_train_big.py)를 쓰지 않고 결과 파일만 본다.
하나라도 FAIL 이면 학습하지 않는다.

학습 세트 (--file)
  1 줄 수 = --n                          2 id 중복 없음 · 문장 중복 없음
  3 앞부분 = 기존 세트(--base)에서 평가·골드셋·Claude 표본·중복을 뺀 것과 같은 순서·같은 id
  4 1단계 dev·test(v1·v3b)·표본 B·C·2단계 골드셋·Claude 금융 표본과 같은 문장 없음
  5 span 형식: 0 ≤ 시작 < 끝 ≤ 문장 길이, 태그 7종, 정렬, 서로 안 겹침
  6 span 글자·태그가 그 문장의 원본 라벨에 있음
  7 span = 고친 규칙(spans_v3b)으로 원본 라벨에서 다시 만든 것과 같음
  8 라벨 없는 문장에 span 없음 · 5글자 미만 없음 · 상용구 없음
  9 추가분: 카테고리 균등(차 5% 이내) · 태그 비율(차 2%p 이내) · 평균 길이(차 5% 이내)
 10 추가분 문장이 원본(spans_full/train)에 같은 id·같은 글자·같은 라벨로 있음(뽑을 때 줄이 어긋나지 않았는지)
dev·test (--dev --test)
 11 각각 5·6·7·8 규칙 · id·문장 중복 없음 · dev와 test 사이 겹침 없음 · 골드셋·Claude 표본과 겹침 없음
"""
import argparse, collections, glob, gzip, hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_split_v3b import spans_v3b
TAGS = {"CV", "OG", "TM", "AF", "FD", "TR", "NW"}
BOILER = re.compile(r"(저작권자|무단\s*전재|재배포\s*금지|ⓒ|©)")
def H(t): return int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "big")
def rows(p):
    with open(p, encoding="utf-8") as f:
        for l in f: yield json.loads(l)
def hashes(paths): return {H(r["text"]) for p in paths for r in rows(p)}
EVAL = ["data/split_v1/dev.jsonl", "data/split_v1/test.jsonl", "data/split_v3b/dev.jsonl", "data/split_v3b/test.jsonl",
        "data/exp_sets/var_train_B.jsonl", "data/exp_sets/var_train_C.jsonl", "data/goldset/goldset_master.jsonl"]
CLAUDE = ["data/fin_silver/pool.jsonl", "data/fin_silver/claude_silver_500.jsonl", "data/fin_silver/claude_calib10_v1.jsonl"]

res = []
def check(name, ok, detail=""):
    res.append(ok); print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)

def row_errors(r):
    """5·6·7·8 규칙 위반 번호 집합"""
    t, sp = r["text"], [tuple(x) for x in r["spans"]]; labs = {(x[0], x[1]) for x in r["labels"]}; e = set()
    if any(not (0 <= s < en <= len(t)) or tg not in TAGS for s, en, tg in sp) or sp != sorted(sp) or \
       any(sp[i][1] > sp[i + 1][0] for i in range(len(sp) - 1)): e.add(5)
    if any((t[s:en], tg) not in labs for s, en, tg in sp): e.add(6)
    if sp != [tuple(x) for x in spans_v3b(t, [tuple(x) for x in r["labels"]])]: e.add(7)
    if (not r["labels"] and sp) or len(t) < 5 or BOILER.search(t): e.add(8)
    return e

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--file"); ap.add_argument("--base"); ap.add_argument("--n", type=int)
    ap.add_argument("--dev"); ap.add_argument("--test")
    a = ap.parse_args()
    ev, cl = hashes(EVAL), hashes(CLAUDE)
    if a.file:
        base_ids = []
        if a.base:
            seen = set()
            for r in rows(a.base):
                h = H(r["text"])
                if h in ev or h in cl or h in seen: continue
                seen.add(h); base_ids.append(r["id"])
        nb = len(base_ids)
        n = 0; ids, tx = set(), set(); dup_id = dup_tx = ov_ev = ov_cl = prefix_bad = 0
        err = collections.Counter(); ex = []
        cat = collections.Counter(); tag = [collections.Counter(), collections.Counter()]; ln = [0, 0]; cnt = [0, 0]
        new_ids = {}
        for i, r in enumerate(rows(a.file)):
            n += 1; h = H(r["text"]); hi = H(r["id"])
            if hi in ids: dup_id += 1
            if h in tx: dup_tx += 1
            ids.add(hi); tx.add(h)
            if h in ev: ov_ev += 1
            if h in cl: ov_cl += 1
            if i < nb and r["id"] != base_ids[i]: prefix_bad += 1
            for e in row_errors(r):
                err[e] += 1
                if len(ex) < 5: ex.append((e, r["id"]))
            part = 0 if i < nb else 1
            if part: cat[r["category"]] += 1; new_ids[hi] = (h, H(json.dumps(r["labels"], ensure_ascii=False)))
            tag[part].update(tg for _, _, tg in r["spans"]); ln[part] += len(r["text"]); cnt[part] += 1
            if n % 1_000_000 == 0: print(f"  {n:,}줄 검사", flush=True)
        check("1 줄 수", a.n is None or n == a.n, f"{n:,}")
        check("2 id 중복 없음", dup_id == 0, f"중복 {dup_id}"); check("2 문장 중복 없음", dup_tx == 0, f"중복 {dup_tx}")
        if a.base: check("3 앞부분 = 기존 세트(평가·골드셋·Claude·중복 제외)", prefix_bad == 0 and n >= nb, f"기존 {nb:,}, 어긋남 {prefix_bad}")
        check("4 평가·골드셋과 겹침 없음", ov_ev == 0, f"{ov_ev}"); check("4 Claude 금융 표본과 겹침 없음", ov_cl == 0, f"{ov_cl}")
        for k, name in ((5, "span 형식"), (6, "span 글자가 원본 라벨에 있음"), (7, "고친 규칙(v3b)과 일치"), (8, "라벨 없음·짧은 문장·상용구 없음")):
            check(f"{k} {name}", err[k] == 0, f"문제 문장 {err[k]}" + (f" 예 {ex[:3]}" if err[k] else ""))
        if cnt[1]:
            spread = (max(cat.values()) - min(cat.values())) / max(cat.values())
            check("9 추가분 카테고리 균등(차 5% 이내)", spread <= 0.05, f"{len(cat)}개, 최소 {min(cat.values()):,} 최대 {max(cat.values()):,}")
            if cnt[0]:
                r0 = {k: tag[0][k] / sum(tag[0].values()) for k in TAGS}; r1 = {k: tag[1][k] / sum(tag[1].values()) for k in TAGS}
                d = max(abs(r0[k] - r1[k]) for k in TAGS)
                check("9 태그 비율 비슷(차 2%p 이내)", d <= 0.02, f"최대 차 {d*100:.2f}%p · 문장당 개체 {sum(tag[0].values())/cnt[0]:.3f} / {sum(tag[1].values())/cnt[1]:.3f}")
                l0, l1 = ln[0] / cnt[0], ln[1] / cnt[1]
                check("9 평균 길이 비슷(차 5% 이내)", abs(l0 - l1) / l0 <= 0.05, f"{l0:.1f} / {l1:.1f}")
            found = bad = 0
            for p in sorted(glob.glob("data/spans_full/train/*.jsonl.gz")):
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    for l in f:
                        r = json.loads(l); hi = H(r["id"])
                        if hi in new_ids:
                            found += 1
                            if new_ids[hi] != (H(r["text"]), H(json.dumps(r["labels"], ensure_ascii=False))): bad += 1
            check("10 추가분이 원본과 같은 id·글자·라벨", found == len(new_ids) and bad == 0, f"찾음 {found:,}/{len(new_ids):,}, 다름 {bad}")
    if a.dev and a.test:
        sets = {}
        for name, p in (("dev", a.dev), ("test", a.test)):
            err = collections.Counter(); ex = []; ids, tx = set(), set(); dup = 0
            for r in rows(p):
                h = H(r["text"])
                if h in tx or r["id"] in ids: dup += 1
                tx.add(h); ids.add(r["id"])
                for e in row_errors(r):
                    err[e] += 1
                    if len(ex) < 3: ex.append((e, r["id"]))
            sets[name] = tx
            check(f"11 {name} 규칙 5·6·7·8", sum(err.values()) == 0, f"{len(tx):,}문장, 위반 {dict(err)} {ex}")
            check(f"11 {name} 중복 없음", dup == 0, f"{dup}")
            gold = hashes(["data/goldset/goldset_master.jsonl"])
            check(f"11 {name} 골드셋·Claude 표본과 겹침 없음", not (tx & gold) and not (tx & cl), f"골드셋 {len(tx & gold)} · Claude {len(tx & cl)}")
        check("11 dev·test 서로 겹침 없음", not (sets["dev"] & sets["test"]), f"{len(sets['dev'] & sets['test'])}")
    nf = res.count(False)
    print(f"\n결과: {len(res)-nf}/{len(res)} 통과" + (" — FAIL 있음, 학습하지 말 것" if nf else ""))
    sys.exit(1 if nf else 0)

if __name__ == "__main__":
    main()
