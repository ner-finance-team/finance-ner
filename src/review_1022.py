# -*- coding: utf-8 -*-
"""
10/22 1단계 결과 종합: 모든 실험(최세진 + 팀원)을 같은 채점기 점수로 한 표에 모으고,
30만 기준선(D1_lc_300000) 대비 차이, 태그별 점수, 오답 유형, 모델 조합(앙상블)을 낸다.
조합은 dev로 정하고 test는 확인만. 결과: reports/review_1022.json
"""
import collections, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_pred, load_jsonl, span_prf, TAGS

R = "reports/results"
BASE = "D1_lc_300000"
NOTE = {  # 사람이 보는 설명 (실행자 / 무엇을 바꿨나)
    "D1_lc_300000": ("sejin_choi", "klue/roberta-base, 30만 (기준선)"),
    "D1_lc_600000": ("sejin_choi", "klue/roberta-base, 60만"),
    "M2_roberta-base_crf_300k": ("sejin_choi", "roberta-base + CRF"),
    "M2_roberta-base_weighted_300k": ("sejin_choi", "roberta-base + 가중 손실"),
    "M1_koelectra_300k": ("sejin_choi", "KoELECTRA"),
    "D3_D3b_300k": ("sejin_choi", "데이터 정제: 누락 보완(+23,502 라벨)"),
    "D4_rare_363k": ("sejin_choi", "30만 + 드문 태그 문장 63,199"),
    "D4_rand_363k": ("sejin_choi", "30만 + 무작위 문장 63,199 (비교군)"),
    "M1_hong_klue-roberta-large_lr2e-5_300k": ("kim_byeongseok", "klue/roberta-large, lr 2e-5"),
    "T_jake_roberta_lr2e5_300k": ("park_gyutae", "roberta-base, lr 2e-5"),
    "T_jake_roberta_lr3e5_300k": ("park_gyutae", "roberta-base, lr 3e-5"),
    "T_jin_roberta_weighted_300k": ("jin_taehyeon", "roberta-base + 가중 손실"),
    "T_jin_roberta_ce_lossacc_300k": ("jin_taehyeon", "roberta-base, loss·acc 기록"),
    "T_jin_roberta_weighted_lossacc_300k": ("jin_taehyeon", "roberta-base + 가중 손실, loss·acc 기록"),
    "T_jin_roberta_ep5_300k": ("jin_taehyeon", "roberta-base, 5에폭"),
}

def sc(name, split):
    p = f"{R}/scores/{name}_{split}.json"
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

def main():
    rows = []
    for name, (who, what) in NOTE.items():
        d, t = sc(name, "dev"), sc(name, "test")
        if not d or not t: print("없음:", name); continue
        rows.append({"exp": name, "who": who, "what": what,
                     "dev_f1": d["metrics"]["ALL"]["f1"], "test_f1": t["metrics"]["ALL"]["f1"],
                     "test_p": t["metrics"]["ALL"]["precision"], "test_r": t["metrics"]["ALL"]["recall"],
                     "test_macro": t["metrics"]["MACRO_F1"], "test_tag": {k: t["metrics"][k]["f1"] for k in TAGS},
                     "dev_errors": d.get("error_counts")})
    base = next(r for r in rows if r["exp"] == BASE)
    for r in rows:
        r["d_dev"] = round((r["dev_f1"] - base["dev_f1"]) * 100, 2); r["d_test"] = round((r["test_f1"] - base["test_f1"]) * 100, 2)
        r["verdict"] = "기준선보다 높음" if r["d_dev"] >= 0.7 else ("기준선보다 낮음" if r["d_dev"] <= -0.7 else "차이 없음")
    rows.sort(key=lambda r: -r["dev_f1"])
    out = {"base": BASE, "rule": "dev F1 차이 0.7점 이상이면 의미 있는 차이(표본 흔들림 std 0.37점의 2배)", "table": rows}

    # 조합(앙상블): 구조가 다른 상위 모델들
    G = {sp: load_jsonl(f"data/split_v1/{sp}.jsonl") for sp in ("dev", "test")}
    CAND = ["M2_roberta-base_crf_300k", "M1_hong_klue-roberta-large_lr2e-5_300k", "D1_lc_600000", "T_jin_roberta_ep5_300k",
            "M1_koelectra_300k", "D4_rare_363k", "T_jake_roberta_lr3e5_300k"]
    CAND = [m for m in CAND if os.path.exists(f"{R}/preds/{m}_dev.jsonl")]
    P = {sp: {m: load_pred(f"{R}/preds/{m}_{sp}.jsonl") for m in CAND} for sp in G}
    def score(sp, preds): return span_prf([r["spans"] for r in G[sp]], preds)["ALL"]["f1"]
    # 두 모델씩 오라클 재현율(둘 중 하나라도 맞힘)과 오답 겹침률
    pair = {}
    for a, b in itertools.combinations(CAND, 2):
        hit_a = hit_b = both_miss = any_miss = n = 0
        for r in G["dev"]:
            g = {tuple(x) for x in r["spans"]}; pa = P["dev"][a].get(r["id"], set()); pb = P["dev"][b].get(r["id"], set())
            for x in g:
                n += 1; ia, ib = x in pa, x in pb
                hit_a += ia; hit_b += ib
                if not ia or not ib: any_miss += 1
                if not ia and not ib: both_miss += 1
        pair[f"{a} + {b}"] = {"oracle_recall": round(1 - both_miss / n, 4), "error_overlap": round(both_miss / max(any_miss, 1), 4)}
    out["pairs_dev"] = dict(sorted(pair.items(), key=lambda x: -x[1]["oracle_recall"]))
    # 다수결: 상위 k개 모델로 v표 이상 (dev로 k·v 선택)
    best = None
    for k in range(3, len(CAND) + 1):
        ms = CAND[:k]
        for v in range(2, k + 1):
            f = score("dev", [ {x for x, c in collections.Counter(s for m in ms for s in P["dev"][m].get(r["id"], ())).items() if c >= v} for r in G["dev"]])
            if not best or f > best[0]: best = (f, k, v)
    f, k, v = best; ms = CAND[:k]
    test_f = score("test", [{x for x, c in collections.Counter(s for m in ms for s in P["test"][m].get(r["id"], ())).items() if c >= v} for r in G["test"]])
    out["vote_ensemble"] = {"models": ms, "min_votes": v, "dev_f1": round(f, 4), "test_f1": round(test_f, 4),
                            "best_single": {"exp": rows[0]["exp"], "dev_f1": rows[0]["dev_f1"], "test_f1": rows[0]["test_f1"]}}
    json.dump(out, open("reports/review_1022.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{'실험':42s} {'실행자':15s} dev    test   Δdev   Δtest  macro  판정")
    for r in rows:
        print(f"{r['exp'][:42]:42s} {r['who']:15s} {r['dev_f1']:.4f} {r['test_f1']:.4f} {r['d_dev']:+5.2f} {r['d_test']:+5.2f} {r['test_macro']:.4f} {r['verdict']}")
    print("\n오라클 재현율 상위 5쌍:"); [print(" ", k, v) for k, v in list(out["pairs_dev"].items())[:5]]
    print("오답 겹침률 낮은 5쌍:"); [print(" ", k, v) for k, v in sorted(pair.items(), key=lambda x: x[1]["error_overlap"])[:5]]
    print("\n다수결 조합:", out["vote_ensemble"])

if __name__ == "__main__":
    main()
