# -*- coding: utf-8 -*-
"""
학습 데이터에도 split_v3b 규칙(단어 중간 출현 제외)을 적용한다. make_split_v3b.spans_v3b 를 그대로 쓴다.
  data/split_v3/train.jsonl          → data/split_v3b/train.jsonl (30만, 팀 기준 학습)
  data/exp_sets/lc_train_600000.jsonl → data/split_v3b/train_600k.jsonl (60만, 최종 후보용)
사용: python src/fix_train_v3b.py
"""
import json, collections, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_split_v3b import spans_v3b

def fix(src, dst):
    removed = collections.Counter(); n_old = n_new = changed = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            new = spans_v3b(r["text"], [tuple(x) for x in r["labels"]])
            old = {tuple(x) for x in r["spans"]}; nw = {tuple(x) for x in new}
            for s, e, t in old - nw: removed[(r["text"][s:e], t)] += 1
            n_old += len(old); n_new += len(nw); changed += old != nw
            out.write(json.dumps({**r, "spans": new}, ensure_ascii=False) + "\n")
    open(dst + ".done", "w").close()
    return {"entities_before": n_old, "entities_after": n_new, "removed": n_old - n_new, "sentences_changed": changed,
            "top_removed": [[list(k), v] for k, v in removed.most_common(20)]}

if __name__ == "__main__":
    st = json.load(open("data/split_v3b/stats.json", encoding="utf-8"))
    st["train_300k"] = fix("data/split_v3/train.jsonl", "data/split_v3b/train.jsonl")
    st["train_600k"] = fix("data/exp_sets/lc_train_600000.jsonl", "data/split_v3b/train_600k.jsonl")
    json.dump(st, open("data/split_v3b/stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k in ("train_300k", "train_600k"): print(k, {a: b for a, b in st[k].items() if a != "top_removed"}, st[k]["top_removed"][:8])
