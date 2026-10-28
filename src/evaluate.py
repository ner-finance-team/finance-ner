# -*- coding: utf-8 -*-
"""
NER 팀 공용 채점기 (파이썬 표준 라이브러리만 사용 — 어떤 모델·코드로 만든 예측이든 같은 기준으로 채점)

[예측 파일 형식] JSON Lines, 한 줄에 한 문장
  {"id": "VL1_...", "pred": [[시작, 끝, "태그"], ...]}
  - id는 정답 파일(split_v1의 dev.jsonl / test.jsonl)의 id와 같아야 한다.
  - 위치는 원문 글자 위치, 끝은 포함하지 않는다(text[시작:끝]). 태그는 CV OG TM AF FD TR NW.

[채점 기준]
  - 엄격 일치: (시작, 끝, 태그) 셋이 모두 같아야 정답.
  - micro F1(전체 개체 단위, 주 지표), 태그별 P/R/F1, macro F1(7개 태그 F1 평균), 카테고리별 F1.
  - 오답 유형: 경계(태그 같음·범위 다름) / 유형(범위 같음·태그 다름) / 경계+유형 / 오탐(정답과 안 겹침) / 미탐(예측과 안 겹침)

[사용]
  채점:  python src/evaluate.py score --gold data/split_v1/test.jsonl --pred my_test.jsonl --name M1_hong_kobert --split test                 --member hong --meta model=monologg/kobert lr=5e-5 epochs=3 seed=42 stage=M1_모델비교
         → reports/results/scores/{name}_{split}.json  (점수)
           reports/results/errors/{name}_{split}.jsonl (오답 목록, 문장마다 한 줄)
  비교:  python src/evaluate.py compare --gold data/split_v1/dev.jsonl --pred a_dev.jsonl b_dev.jsonl c_dev.jsonl
         → 모델끼리 오답이 얼마나 겹치는지, 합쳤을 때(합집합·교집합·다수결) 점수
"""
import argparse, collections, json, os, sys, time

TAGS = ["CV", "OG", "TM", "AF", "FD", "TR", "NW"]
ERROR_TYPES = ["경계", "유형", "경계+유형", "오탐", "미탐"]

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_pred(path):
    """예측 파일 → {id: set((s,e,tag))}. 'pred' 대신 'spans'·'entities' 키도 받아 준다."""
    out = {}
    for r in load_jsonl(path):
        sp = r.get("pred", r.get("spans", r.get("entities", [])))
        out[r["id"]] = {(int(s), int(e), str(t)) for s, e, t in sp}
    return out

def _prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(P, 4), "recall": round(R, 4), "f1": round(2 * P * R / (P + R), 4) if P + R else 0.0,
            "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}

def span_prf(gold_list, pred_list):
    tp, fp, fn = collections.Counter(), collections.Counter(), collections.Counter()
    for gold, pred in zip(gold_list, pred_list):
        g, p = set(map(tuple, gold)), set(map(tuple, pred))
        for x in p & g: tp[x[2]] += 1
        for x in p - g: fp[x[2]] += 1
        for x in g - p: fn[x[2]] += 1
    out = {"ALL": _prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))}
    for t in TAGS: out[t] = _prf(tp[t], fp[t], fn[t])
    out["MACRO_F1"] = round(sum(out[t]["f1"] for t in TAGS) / len(TAGS), 4)
    return out

def span_prf_by(rows, pred_list, key="category"):
    groups = collections.defaultdict(lambda: ([], []))
    for r, p in zip(rows, pred_list):
        groups[r.get(key, "-")][0].append(r["spans"]); groups[r.get(key, "-")][1].append(p)
    return {k: span_prf(g, p)["ALL"] for k, (g, p) in sorted(groups.items())}

def error_records(text, gold, pred):
    """한 문장의 오답 목록. 각 항목에 유형과 글자를 함께 적어 사람이 바로 읽을 수 있게 한다."""
    g, p = set(map(tuple, gold)), set(map(tuple, pred))
    errs = []
    ov = lambda a, b: a[0] < b[1] and b[0] < a[1]
    for x in sorted(p - g):
        same_range = [y for y in g if y[0] == x[0] and y[1] == x[1]]
        overl = [y for y in g if ov(x, y)]
        if same_range:                                   typ, ref = "유형", same_range[0]
        elif any(y[2] == x[2] for y in overl):           typ, ref = "경계", next(y for y in overl if y[2] == x[2])
        elif overl:                                      typ, ref = "경계+유형", overl[0]
        else:                                            typ, ref = "오탐", None
        errs.append({"type": typ, "pred": list(x), "pred_text": text[x[0]:x[1]],
                     "gold": list(ref) if ref else None, "gold_text": text[ref[0]:ref[1]] if ref else None})
    for y in sorted(g - p):
        if not any(ov(x, y) for x in p):
            errs.append({"type": "미탐", "pred": None, "pred_text": None, "gold": list(y), "gold_text": text[y[0]:y[1]]})
    return errs

def score(gold_rows, pred_map, name, split, out_root, extra=None):
    """채점 + 점수 json + 오답 jsonl 저장. 반환값은 점수 dict."""
    missing = [r["id"] for r in gold_rows if r["id"] not in pred_map]
    preds = [sorted(pred_map.get(r["id"], set())) for r in gold_rows]
    res = {"name": name, "split": split, "n_sentences": len(gold_rows), "n_missing_pred": len(missing),
           "scored_at": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {}),
           "metrics": span_prf([r["spans"] for r in gold_rows], preds),
           "by_category": span_prf_by(gold_rows, preds)}
    etc = collections.Counter(); etag = collections.Counter()
    os.makedirs(os.path.join(out_root, "errors"), exist_ok=True); os.makedirs(os.path.join(out_root, "scores"), exist_ok=True)
    with open(os.path.join(out_root, "errors", f"{name}_{split}.jsonl"), "w", encoding="utf-8") as f:
        for r, p in zip(gold_rows, preds):
            errs = error_records(r["text"], r["spans"], p)
            for e in errs:
                etc[e["type"]] += 1
                etag[(e["type"], (e["gold"] or e["pred"])[2])] += 1
            if errs:
                f.write(json.dumps({"id": r["id"], "category": r.get("category"), "text": r["text"], "errors": errs},
                                   ensure_ascii=False) + "\n")
    res["error_counts"] = {t: etc[t] for t in ERROR_TYPES}
    res["error_counts_by_tag"] = {t: {k: etag[(k, t)] for k in ERROR_TYPES} for t in TAGS}
    json.dump(res, open(os.path.join(out_root, "scores", f"{name}_{split}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if missing: print(f"[경고] 예측이 없는 문장 {len(missing)}개 (빈 예측으로 채점). 예: {missing[:3]}", file=sys.stderr)
    return res

def compare(gold_rows, pred_maps, names):
    """모델끼리 오답이 겹치는 정도와 단순 앙상블 점수. 어떤 모델을 합칠지는 dev로 정한다(test로 정하면 점수가 부풀려진다)."""
    gold = [set(map(tuple, r["spans"])) for r in gold_rows]
    ids = [r["id"] for r in gold_rows]
    P = [[pm.get(i, set()) for i in ids] for pm in pred_maps]
    total = sum(len(g) for g in gold)
    out = {"n_gold_entities": total, "single": {}, "pairs": {}, "ensembles": {}}
    for n, pl in zip(names, P):
        out["single"][n] = span_prf(gold, pl)["ALL"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_only = b_only = both = neither = 0
            for g, pa, pb in zip(gold, P[i], P[j]):
                for x in g:
                    ia, ib = x in pa, x in pb
                    both += ia and ib; a_only += ia and not ib; b_only += ib and not ia; neither += not ia and not ib
            out["pairs"][f"{names[i]} | {names[j]}"] = {
                "둘다_맞춤": both, f"{names[i]}만_맞춤": a_only, f"{names[j]}만_맞춤": b_only, "둘다_틀림": neither,
                "오라클_재현율": round((total - neither) / total, 4) if total else 0.0,
                "오답_겹침률": round(neither / max(1, neither + a_only + b_only), 4)}
    k = len(P)
    if k >= 2:
        inter = [set.intersection(*[P[m][s] for m in range(k)]) for s in range(len(ids))]
        union = [set.union(*[P[m][s] for m in range(k)]) for s in range(len(ids))]
        out["ensembles"]["교집합(모두 동의)"] = span_prf(gold, inter)["ALL"]
        out["ensembles"]["합집합(하나라도)"] = span_prf(gold, union)["ALL"]
        if k >= 3:
            need = k // 2 + 1; maj = []
            for s in range(len(ids)):
                c = collections.Counter(x for m in range(k) for x in P[m][s]); maj.append({x for x, v in c.items() if v >= need})
            out["ensembles"][f"다수결({need}/{k})"] = span_prf(gold, maj)["ALL"]
    return out

def main():
    ap = argparse.ArgumentParser(description="NER 팀 공용 채점기")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--gold", required=True); s.add_argument("--pred", required=True)
    s.add_argument("--name", required=True); s.add_argument("--split", required=True, choices=["dev", "test"])
    s.add_argument("--out", default="reports/results"); s.add_argument("--member", default="")
    s.add_argument("--meta", nargs="*", default=[], help="설정 기록 key=value (예: model=klue/bert-base lr=3e-5 epochs=3 seed=42 stage=M1_모델비교)")
    c = sub.add_parser("compare"); c.add_argument("--gold", required=True); c.add_argument("--pred", nargs="+", required=True)
    c.add_argument("--names", nargs="*"); c.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "score":
        meta = dict(kv.split("=", 1) for kv in a.meta)
        r = score(load_jsonl(a.gold), load_pred(a.pred), a.name, a.split, a.out,
                  {"member": a.member, "pred_file": os.path.basename(a.pred), **meta})
        m = r["metrics"]
        print(f"{a.name} [{a.split}] F1 {m['ALL']['f1']} P {m['ALL']['precision']} R {m['ALL']['recall']} macro {m['MACRO_F1']}")
        print("오답 유형:", r["error_counts"])
    else:
        names = a.names or [os.path.splitext(os.path.basename(p))[0] for p in a.pred]
        r = compare(load_jsonl(a.gold), [load_pred(p) for p in a.pred], names)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        if a.out: json.dump(r, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
