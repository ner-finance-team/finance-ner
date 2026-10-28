# -*- coding: utf-8 -*-
"""
'태그마다 따로 정하는' 조합이 얼마나 도움이 되는지, 이미 저장된 예측만으로 미리 재 본다(학습 없음, GPU 불필요).
  A) 태그별 최고 모델: 태그마다 dev F1이 가장 높은 모델의 그 태그 예측만 모아 합친다.
  B) 태그별 투표 기준: 태그마다 '몇 개 모델 이상이 같은 span을 냈으면 채택'할지 dev에서 고른다.
  비교: 단일 최고 모델, 전체 다수결(태그 구분 없이 같은 기준).
조합 방법은 dev로만 정하고, test는 정한 방법을 한 번 적용해 확인만 한다.
겹치는 예측은 (A: 그 태그 dev F1, B: 표 수)가 높은 쪽을 남긴다.
사용: python src/per_tag_combo.py  → reports/per_tag_combo.json
"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import span_prf, load_pred, load_jsonl, TAGS

MODELS = ["M1_klue_roberta", "M1_klue_bert", "M1_koelectra", "M1_aihub_electra_1", "M1_aihub_electra_2",
          "M1_klue_roberta_morph", "M1_aihub_electra_1_morph", "M2_roberta-base_crf", "M2_roberta-base_weighted",
          "D1_lc_200000", "D1_lc_300000", "M2_roberta-base_crf_200k"]

def resolve(cands):
    """cands: [(priority, (s,e,t))] → 겹치지 않게 우선순위 높은 것부터."""
    keep = []
    for _, (s, e, t) in sorted(cands, key=lambda x: -x[0]):
        if all(e <= a or b <= s for a, b, _ in keep): keep.append((s, e, t))
    return keep

def main():
    G = {sp: load_jsonl(f"data/split_v1/{sp}.jsonl") for sp in ("dev", "test")}
    P = {sp: {m: load_pred(f"reports/results/preds/{m}_{sp}.jsonl") for m in MODELS} for sp in G}
    gold = {sp: [r["spans"] for r in G[sp]] for sp in G}
    ids = {sp: [r["id"] for r in G[sp]] for sp in G}
    def sc(sp, preds): return span_prf(gold[sp], preds)
    out = {"models": MODELS}

    single = {m: {sp: sc(sp, [P[sp][m].get(i, set()) for i in ids[sp]]) for sp in G} for m in MODELS}
    best = max(MODELS, key=lambda m: single[m]["dev"]["ALL"]["f1"])
    out["single_best"] = {"model": best, **{sp: {"f1": single[best][sp]["ALL"]["f1"], "macro": single[best][sp]["MACRO_F1"]} for sp in G}}

    # A) 태그별 최고 모델 (dev로 고름)
    pick = {t: max(MODELS, key=lambda m: single[m]["dev"][t]["f1"]) for t in TAGS}
    pri = {t: single[pick[t]]["dev"][t]["f1"] for t in TAGS}
    def combo_a(sp):
        return [resolve([(pri[t], x) for t in TAGS for x in P[sp][pick[t]].get(i, ()) if x[2] == t]) for i in ids[sp]]
    out["A_per_tag_best"] = {"pick": pick, **{sp: (lambda r: {"f1": r["ALL"]["f1"], "macro": r["MACRO_F1"],
                                    "by_tag": {t: r[t]["f1"] for t in TAGS}})(sc(sp, combo_a(sp))) for sp in G}}

    # B) 투표: 전체 공통 기준 k vs 태그별 기준 k_t (dev로 고름)
    votes = {sp: [collections.Counter(x for m in MODELS for x in P[sp][m].get(i, ())) for i in ids[sp]] for sp in G}
    def combo_vote(sp, kmap):
        return [resolve([(v, x) for x, v in vc.items() if v >= kmap[x[2]]]) for vc in votes[sp]]
    n = len(MODELS)
    glob = {k: sc("dev", combo_vote("dev", {t: k for t in TAGS}))["ALL"]["f1"] for k in range(1, n + 1)}
    kg = max(glob, key=glob.get)
    out["B_global_vote"] = {"k": kg, **{sp: (lambda r: {"f1": r["ALL"]["f1"], "macro": r["MACRO_F1"]})(sc(sp, combo_vote(sp, {t: kg for t in TAGS}))) for sp in G}}
    # 태그별 k: 한 태그씩 바꿔 가며 dev 전체 F1이 가장 높은 값(좌표 상승 2바퀴)
    kt = {t: kg for t in TAGS}
    for _ in range(2):
        for t in TAGS:
            kt[t] = max(range(1, n + 1), key=lambda k: sc("dev", combo_vote("dev", {**kt, t: k}))["ALL"]["f1"])
    out["B_per_tag_vote"] = {"k": kt, **{sp: (lambda r: {"f1": r["ALL"]["f1"], "macro": r["MACRO_F1"],
                                   "by_tag": {t: r[t]["f1"] for t in TAGS}})(sc(sp, combo_vote(sp, kt))) for sp in G}}
    out["single_by_tag_dev"] = {t: {"best_model": pick[t], "f1": pri[t], "best_single_overall": single[best]["dev"][t]["f1"]} for t in TAGS}
    json.dump(out, open("reports/per_tag_combo.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("single_best", "A_per_tag_best", "B_global_vote", "B_per_tag_vote")}, ensure_ascii=False, indent=1))
    print(json.dumps(out["single_by_tag_dev"], ensure_ascii=False))

if __name__ == "__main__":
    main()
