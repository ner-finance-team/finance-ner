import json, os, glob, collections, sys
FT = "G:/내 드라이브/NER_라벨_통합_v1.2/finetune"; sys.path.insert(0, FT); import fin_score as S
SP = sys.argv[1]; PP = os.path.join(SP, "postproc"); X = os.path.join(SP, "extra_scored"); RES = os.path.join(FT, "results")
def trim(t, sp):
    o = []
    for a, b, g in sp:
        while a < b and t[a].isspace(): a += 1
        while b > a and t[b-1].isspace(): b -= 1
        if a < b: o.append([a, b, g])
    return o
def orig(E, s):
    p = os.path.join(RES, E, f"{E}_{s}.jsonl")
    if not os.path.exists(p): p = os.path.join(X, f"{E}_{s}.jsonl")
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    if E.startswith("F04"):
        for r in rows: r["pred"] = trim(r["text"], r["pred"])
    return rows
table = {}; P = {}
for f in sorted(glob.glob(os.path.join(PP, "*_후처리.jsonl"))):
    b = os.path.basename(f)[:-len("_후처리.jsonl")]
    for s in ("test_2000", "dev", "test", "extra"):
        if b.endswith("_" + s): E = b[:-len(s) - 1]; break
    k = E.split("_")[0]; pp = [json.loads(l) for l in open(f, encoding="utf-8")]; o = orig(E, s)
    g = [r["gold"] for r in o]; s0 = S.score(g, [r["pred"] for r in o]); s1 = S.score(g, [r["pred"] for r in pp])
    table.setdefault(k, {})[s] = {"원래": s0["ALL"]["f1"], "후처리": s1["ALL"]["f1"], "원래_macro": s0["MACRO_F1"], "후처리_macro": s1["MACRO_F1"], "후처리_tags": {t: s1[t]["f1"] for t in S.TAGS}}
    P.setdefault(k, {})[s] = (o, pp)
ens = {}
for s in ("dev", "test", "test_2000", "extra"):
    ks = ["F03", "F12", "F03s"]
    if not all(s in P.get(k, {}) for k in ks): continue
    base = P["F03"][s][0]; g = [r["gold"] for r in base]
    for mode, idx in (("원래", 0), ("후처리", 1)):
        out = []
        for i in range(len(base)):
            c = collections.Counter(tuple(a) for k in ks for a in P[k][s][idx][i]["pred"]); out.append([list(e) for e, v in c.items() if v >= 2])
        sc = S.score(g, out); ens.setdefault(s, {})[mode] = sc["ALL"]["f1"]; ens[s][mode + "_macro"] = sc["MACRO_F1"]
json.dump({"runs": table, "ensemble_03_12_03s": ens}, open(os.path.join(PP, "후처리점수_종합.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{'RUN':5s} " + "  ".join(f"{s:>17s}" for s in ("dev", "test", "test_2000", "extra")))
for k in sorted(table, key=lambda k: -table[k].get("test", {}).get("후처리", 0)):
    print(f"{k:5s} " + "  ".join((f"{table[k][s]['원래']:.4f}→{table[k][s]['후처리']:.4f}" if s in table[k] else " " * 13).rjust(17) for s in ("dev", "test", "test_2000", "extra")))
print("앙상블", " ".join(f"{s}:{v['원래']:.4f}→{v['후처리']:.4f}" for s, v in ens.items()))
