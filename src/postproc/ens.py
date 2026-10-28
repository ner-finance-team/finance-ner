import json, os, itertools, collections, sys
sys.path.insert(0, r"G:\내 드라이브\NER_라벨_통합_v1.2\finetune"); import fin_score as S
R = r"G:\내 드라이브\NER_라벨_통합_v1.2\finetune\results"; X = sys.argv[1]
names = {"F03":"F03_dapt_roberta-large_crf","F07":"F07_dapt-stage1-5000k_roberta-large_crf","F11":"F11_klue-roberta-base_crf","F10":"F10_klue-bert-base_crf",
         "F02":"F02_dapt_roberta-base_crf","F03c":"F03c_dapt_roberta-large_crf_ep5","F12":"F12_monologg-koelectra-base-v3-discriminator_crf","F03s":"F03s_dapt_roberta-large_softmax"}
def load(k, s):
    p = os.path.join(R, names[k], f"{names[k]}_{s}.jsonl")
    if not os.path.exists(p): p = os.path.join(X, f"{names[k]}_{s}.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")]
D = {s: {k: load(k, s) for k in names} for s in ("dev", "test", "extra")}
def run(keys, s, mode):
    base = D[s][keys[0]]; out = []
    for i, r in enumerate(base):
        sets = [{tuple(a) for a in D[s][k][i]["pred"]} for k in keys]
        if mode == "vote":   # 과반
            c = collections.Counter(e for st in sets for e in st); out.append([list(e) for e, v in c.items() if v * 2 > len(keys)])
        elif mode == "inter": out.append([list(e) for e in set.intersection(*sets)])
        elif mode == "union":  # 겹치면 첫 모델 우선
            acc = list(sets[0])
            for st in sets[1:]:
                for e in st:
                    if not any(a[0] < e[1] and e[0] < a[1] for a in acc): acc.append(e)
            out.append([list(e) for e in acc])
    sc = S.score([r["gold"] for r in base], out); return sc["ALL"]["f1"], sc["MACRO_F1"]
single = {k: S.score([r["gold"] for r in D["dev"][k]], [r["pred"] for r in D["dev"][k]])["ALL"]["f1"] for k in names}
res = []
for n in (2, 3):
    for keys in itertools.combinations(sorted(names, key=lambda k: -single[k]), n):
        modes = ["inter", "union"] if n == 2 else ["vote"]
        for m in modes: res.append((run(list(keys), "dev", m)[0], keys, m))
res.sort(reverse=True)
print("dev 단일:", {k: round(v, 4) for k, v in sorted(single.items(), key=lambda x: -x[1])})
print("dev 상위 조합 8개:")
for f, k, m in res[:8]: print(f"  {f:.4f} {'+'.join(k)} {m}")
best = res[0]; keys, m = list(best[1]), best[2]
print("dev로 고른 조합:", "+".join(keys), m)
for s in ("dev", "test", "extra"): print(f"  {s}: 앙상블 {run(keys, s, m)} vs 03 단일 {run(['F03'], s, 'vote')}")
for keys, m in ((["F03","F12","F10"], "vote"), (["F03","F12"], "union"), (["F03","F07","F11"], "vote")):
    print("참고", "+".join(keys), m, {s: run(keys, s, m)[0] for s in ("dev", "test", "extra")})
