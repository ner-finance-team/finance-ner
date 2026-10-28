# -*- coding: utf-8 -*-
"""
태그 전문가 모델 결합: 기본 모델 예측에서 그 태그만 전문가 예측으로 바꾼다. 방법은 dev로 고르고 test는 확인만.
  replace_main_wins   : 기본 모델의 그 태그 예측을 지우고 전문가 예측을 넣되, 다른 태그와 겹치면 기본 모델을 남김
  replace_expert_wins : 같은데, 겹치면 전문가 예측을 남김(겹친 기본 모델 예측은 지움)
  union               : 기본 모델의 그 태그 예측은 두고, 전문가 예측 중 겹치지 않는 것만 더함
사용: python src/combine_expert.py --main M2_roberta-base_crf_200k --expert X_TR_expert --tag TR
결과: reports/results/combine_{tag}.json, 고른 방법의 예측·점수는 reports/results 에 C_{main}+{expert} 이름으로 저장
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_pred, load_jsonl, span_prf, score

def ov(a, b): return a[0] < b[1] and b[0] < a[1]

def combine(main, exp, tag, how):
    other = [x for x in main if x[2] != tag]; mine = [x for x in main if x[2] == tag]; ex = [x for x in exp if x[2] == tag]
    if how == "union":
        return mine + other + [x for x in ex if not any(ov(x, y) for y in mine + other)]
    if how == "replace_main_wins":
        return other + [x for x in ex if not any(ov(x, y) for y in other)]
    if how == "replace_expert_wins":
        return ex + [y for y in other if not any(ov(x, y) for x in ex)]
    raise ValueError(how)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--main", required=True); ap.add_argument("--expert", required=True)
    ap.add_argument("--tag", default="TR"); ap.add_argument("--results", default="reports/results"); ap.add_argument("--split-dir", default="data/split_v1")
    a = ap.parse_args(); R = a.results
    G = {sp: load_jsonl(f"{a.split_dir}/{sp}.jsonl") for sp in ("dev", "test")}
    M = {sp: load_pred(f"{R}/preds/{a.main}_{sp}.jsonl") for sp in G}
    E = {sp: load_pred(f"{R}/preds/{a.expert}_{sp}.jsonl") for sp in G}
    def run(sp, how):
        return [combine(list(M[sp].get(r["id"], ())), list(E[sp].get(r["id"], ())), a.tag, how) for r in G[sp]]
    out = {"main": a.main, "expert": a.expert, "tag": a.tag}
    base = {sp: span_prf([r["spans"] for r in G[sp]], [list(M[sp].get(r["id"], ())) for r in G[sp]]) for sp in G}
    out["main_only"] = {sp: {"f1": base[sp]["ALL"]["f1"], a.tag: base[sp][a.tag]["f1"]} for sp in G}
    res = {}
    for how in ("replace_main_wins", "replace_expert_wins", "union"):
        res[how] = {sp: span_prf([r["spans"] for r in G[sp]], run(sp, how)) for sp in G}
        out[how] = {sp: {"f1": res[how][sp]["ALL"]["f1"], a.tag: res[how][sp][a.tag]["f1"]} for sp in G}
    best = max(res, key=lambda h: res[h]["dev"]["ALL"]["f1"])
    out["chosen_on_dev"] = best
    name = f"C_{a.main}+{a.expert}_{best}"
    for sp in G:
        pm = {r["id"]: set(map(tuple, p)) for r, p in zip(G[sp], run(sp, best))}
        with open(f"{R}/preds/{name}_{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in G[sp]: f.write(json.dumps({"id": r["id"], "category": r.get("category"), "pred": sorted(map(list, pm[r["id"]]))}, ensure_ascii=False) + "\n")
        score(G[sp], pm, name, sp, R, {"stage": "X_전문가결합", "main": a.main, "expert": a.expert, "how": best})
    json.dump(out, open(f"{R}/combine_{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
