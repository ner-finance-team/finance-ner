# 확정 후처리 P1(조사 자르기)을 모든 RUN·세트에 적용 → 원래/후처리 점수, 후처리 예측 파일 저장, 앙상블(03+12+03s)도
import json, os, sys, glob, collections
from transformers import AutoTokenizer
from kiwipiepy import Kiwi
FT = "G:/내 드라이브/NER_라벨_통합_v1.2/finetune"; sys.path.insert(0, FT); import fin_score as S
RES = os.path.join(FT, "results"); X = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
kiwi = Kiwi(); KC = {}
def kt(t):
    if t not in KC: KC[t] = kiwi.tokenize(t)
    return KC[t]
def p1(t, sp, tk):
    offs = [o for o in tk(t, return_offsets_mapping=True)["offset_mapping"] if o[0] != o[1]]; out = []
    for a, b, g in sp:
        tok = next(((s, e) for s, e in offs if s < b <= e), None)
        if tok and tok[1] - tok[0] > 1:
            j = [k for k in kt(t) if k.tag.startswith("J") and tok[0] < k.start < b and k.start > a and k.start + k.len <= b]
            if j: b = min(k.start for k in j)
        out.append([a, b, g])
    return out
def trim(t, sp):
    o = []
    for a, b, g in sp:
        while a < b and t[a].isspace(): a += 1
        while b > a and t[b-1].isspace(): b -= 1
        if a < b: o.append([a, b, g])
    return o
table = {}; post_rows = {}
for d in sorted(glob.glob(os.path.join(RES, "F*"))):
    E = os.path.basename(d); k = E.split("_")[0]
    if k == "F09": continue   # 생성형: 토큰 조각 문제 없음(조사 붙은 예측 0건)
    tk = AutoTokenizer.from_pretrained(os.path.join(d, "model")); table[k] = {}; post_rows[k] = {}
    for s in ("dev", "test", "test_2000", "extra"):
        p = os.path.join(d, f"{E}_{s}.jsonl")
        if not os.path.exists(p): p = os.path.join(X, f"{E}_{s}.jsonl")
        if not os.path.exists(p): continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        if k == "F04":
            for r in rows: r["pred"] = trim(r["text"], r["pred"])
        g = [r["gold"] for r in rows]; s0 = S.score(g, [r["pred"] for r in rows])
        pp = [p1(r["text"], r["pred"], tk) for r in rows]; s1 = S.score(g, pp)
        ch = sum(x != y for a, b in zip(rows, pp) for x, y in zip(a["pred"], b))
        table[k][s] = {"원래": s0["ALL"]["f1"], "후처리": s1["ALL"]["f1"], "원래_macro": s0["MACRO_F1"], "후처리_macro": s1["MACRO_F1"], "바뀐개체": ch,
                       "후처리_tags": {t: s1[t]["f1"] for t in S.TAGS}}
        post_rows[k][s] = pp
        S.save_preds(os.path.join(OUT, f"{E}_{s}_후처리.jsonl"), [{"id": r["id"], "text": r["text"], "spans": r["gold"]} for r in rows], pp)
    print(k, {s: (v["원래"], v["후처리"], v["바뀐개체"]) for s, v in table[k].items()}, flush=True)
# 앙상블 03 + 12 + 03s (각자 후처리 후 과반)
def load_gold(k, s):
    E = [os.path.basename(x) for x in glob.glob(os.path.join(RES, k + "_*"))][0]
    p = os.path.join(RES, E, f"{E}_{s}.jsonl")
    if not os.path.exists(p): p = os.path.join(X, f"{E}_{s}.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")]
ens = {}
for s in ("dev", "test", "test_2000", "extra"):
    keys = ["F03", "F12", "F03s"]
    if not all(s in post_rows[k] for k in keys): continue
    base = load_gold("F03", s); out_raw, out_pp = [], []
    for i, r in enumerate(base):
        for src, out in (("raw", out_raw), ("pp", out_pp)):
            c = collections.Counter()
            for k in keys:
                pred = load_gold(k, s)[i]["pred"] if src == "raw" else post_rows[k][s][i]
                c.update(tuple(a) for a in pred)
            out.append([list(e) for e, v in c.items() if v >= 2])
    g = [r["gold"] for r in base]
    a, b = S.score(g, out_raw), S.score(g, out_pp)
    ens[s] = {"원래": a["ALL"]["f1"], "후처리": b["ALL"]["f1"], "원래_macro": a["MACRO_F1"], "후처리_macro": b["MACRO_F1"]}
    print("앙상블", s, ens[s], flush=True)
json.dump({"runs": table, "ensemble_03_12_03s": ens, "rule": "P1 조사 자르기: 예측 개체의 마지막 토큰 안에서 Kiwi 조사(J*)가 시작하면 그 앞에서 자름(F04는 앞뒤 공백 보정 후)"},
          open(os.path.join(OUT, "후처리점수_종합.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
