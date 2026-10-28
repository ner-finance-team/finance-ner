# -*- coding: utf-8 -*-
"""
split_v3b: dev·test 정답 span을 고친 비교용 판 (팀 기준 split_v3는 그대로 둔다).
문제: 원본 라벨은 (글자, 태그)만 있어 문장 안 모든 출현을 정답으로 잡았는데, 단어 경계 검사가 없어
      '대전시'·'제시' 안의 '시'(FD)처럼 단어 중간에서 시작하는 출현까지 정답이 됐다.
고친 규칙: 같은 라벨 글자가 여러 번 나오면, 단어 중간(앞 글자가 한글·영문·숫자)에서 시작하는 출현은 뺀다.
          모든 출현이 단어 중간이면(원래 라벨이 그 자리를 가리킨 것) 첫 출현을 남긴다. 겹치면 긴 span 우선(기존과 같음).
사용: python src/make_split_v3b.py → data/split_v3b/{dev,test}.jsonl, stats.json
"""
import json, os, re, collections
WORD = re.compile(r"[가-힣a-zA-Z0-9]")
TAGS = {"CV", "OG", "TM", "AF", "FD", "TR", "NW"}

def spans_v3b(text, labels):
    cand = []
    for tx, tg in labels:
        if not tx or tg not in TAGS: continue
        occ, p = [], text.find(tx)
        while p >= 0: occ.append(p); p = text.find(tx, p + 1)
        if not occ: continue
        good = [p for p in occ if p == 0 or not WORD.match(text[p - 1])]
        for p in (good or occ[:1]): cand.append((p, p + len(tx), tg))
    cand.sort(key=lambda x: (-(x[1] - x[0]), x[0])); keep = []
    for s, e, t in cand:
        if all(e <= a or b <= s for a, b, _ in keep): keep.append((s, e, t))
    return [list(x) for x in sorted(keep)]

def main():
    os.makedirs("data/split_v3b", exist_ok=True); st = {}
    for sp in ("dev", "test"):
        rows = [json.loads(l) for l in open(f"data/split_v1/{sp}.jsonl", encoding="utf-8")]
        removed = collections.Counter(); n_old = n_new = changed = 0
        with open(f"data/split_v3b/{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                new = spans_v3b(r["text"], [tuple(x) for x in r["labels"]])
                old = {tuple(x) for x in r["spans"]}; nw = {tuple(x) for x in new}
                for s, e, t in old - nw: removed[(r["text"][s:e], t)] += 1
                n_old += len(old); n_new += len(nw); changed += old != nw
                f.write(json.dumps({**r, "spans": new}, ensure_ascii=False) + "\n")
        st[sp] = {"entities_v3": n_old, "entities_v3b": n_new, "removed": n_old - n_new, "sentences_changed": changed,
                  "top_removed": [[list(k), v] for k, v in removed.most_common(20)]}
        print(sp, {k: v for k, v in st[sp].items() if k != "top_removed"}, removed.most_common(8))
    json.dump(st, open("data/split_v3b/stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
