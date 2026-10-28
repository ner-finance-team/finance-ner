# -*- coding: utf-8 -*-
"""
여러 모델이 한목소리로(AGREE 비율 이상) 정답 라벨과 다르게 답한 곳 = 라벨 오류 후보. 골드 평가셋 검수 때 먼저 볼 목록.
저장된 예측 파일(reports/results/preds)만 쓰므로 GPU가 필요 없다.

후보 종류
  누락 : 정답에 없는 개체를 모델들이 같은 위치·같은 태그로 찾음 (정답 개체와 겹치지 않음) → 라벨 빠뜨림 의심
  유형 : 정답과 같은 범위인데 모델들이 다른 같은 태그로 답함 → 태그 잘못 의심
  경계 : 정답과 겹치고 태그도 같은데 모델들이 다른 같은 범위로 답함 → 범위 잘못 의심
  과잉 : 정답 개체를 거의 모든 모델이 아무것도 겹치게 예측하지 않음 → 개체가 아닌데 라벨 의심

사용: python src/find_label_errors.py --split dev   (test도 같게)
출력: reports/label_error_candidates/{split}.jsonl, {split}_summary.json
"""
import argparse, collections, json, os

MODELS = ["M1_klue_roberta", "M1_klue_bert", "M1_koelectra", "M1_aihub_electra_1", "M1_aihub_electra_2",
          "M1_klue_roberta_morph", "M1_aihub_electra_1_morph", "M2_roberta-base_crf", "M2_roberta-base_weighted",
          "D1_lc_200000", "D1_lc_300000", "M2_roberta-base_crf_200k"]

def load(p):
    with open(p, encoding="utf-8") as f: return [json.loads(l) for l in f]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="dev"); ap.add_argument("--agree", type=float, default=0.8)
    a = ap.parse_args()
    gold = {r["id"]: r for r in load(f"data/split_v1/{a.split}.jsonl")}
    models = [m for m in MODELS if os.path.exists(f"reports/results/preds/{m}_{a.split}.jsonl")]
    P = {m: {r["id"]: {tuple(x) for x in r["pred"]} for r in load(f"reports/results/preds/{m}_{a.split}.jsonl")} for m in models}
    need = max(2, round(a.agree * len(models)))
    out, cnt, by_tag = [], collections.Counter(), collections.Counter()
    for sid, g in gold.items():
        text = g["text"]; G = {tuple(x) for x in g["spans"]}
        votes = collections.Counter(s for m in models for s in P[m].get(sid, ()))
        def overl(s, e, S): return [x for x in S if x[0] < e and s < x[1]]
        for (s, e, t), v in votes.items():
            if v < need or (s, e, t) in G: continue
            ov = overl(s, e, G)
            if not ov: kind, gtag = "누락", None
            elif any(x[0] == s and x[1] == e for x in ov): kind, gtag = "유형", [x[2] for x in ov if x[0] == s and x[1] == e][0]
            elif any(x[2] == t for x in ov): kind, gtag = "경계", t
            else: continue
            gold_str = [text[x[0]:x[1]] + "/" + x[2] for x in ov]
            out.append({"id": sid, "kind": kind, "votes": f"{v}/{len(models)}", "model": [s, e, t], "model_str": text[s:e],
                        "gold": gold_str, "text": text, "category": g.get("category")})
            cnt[kind] += 1; by_tag[(kind, t)] += 1
        for (s, e, t) in G:
            hit = sum(1 for m in models if overl(s, e, P[m].get(sid, ())))
            if hit <= len(models) - need:
                out.append({"id": sid, "kind": "과잉", "votes": f"{len(models) - hit}/{len(models)} 모델이 예측 안 함", "model": None,
                            "model_str": None, "gold": [text[s:e] + "/" + t], "text": text, "category": g.get("category")})
                cnt["과잉"] += 1; by_tag[("과잉", t)] += 1
    os.makedirs("reports/label_error_candidates", exist_ok=True)
    with open(f"reports/label_error_candidates/{a.split}.jsonl", "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_gold = sum(len(r["spans"]) for r in gold.values())
    sents = len({r["id"] for r in out})
    summ = {"split": a.split, "models": models, "need_votes": need, "gold_entities": n_gold, "sentences": len(gold),
            "candidates": dict(cnt), "candidate_sentences": sents,
            "by_kind_tag": {f"{k}_{t}": v for (k, t), v in sorted(by_tag.items(), key=lambda x: -x[1])}}
    with open(f"reports/label_error_candidates/{a.split}_summary.json", "w", encoding="utf-8") as f: json.dump(summ, f, ensure_ascii=False, indent=1)
    print(f"[{a.split}] 모델 {len(models)}개, {need}개 이상 동의 | 정답 개체 {n_gold:,} | 후보 {dict(cnt)} | 후보가 있는 문장 {sents:,}/{len(gold):,}")

if __name__ == "__main__":
    main()
