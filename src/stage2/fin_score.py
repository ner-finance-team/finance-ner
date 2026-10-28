# -*- coding: utf-8 -*-
"""금융 NER 공통 채점 — 모든 금융 노트북(인코더 fin_ft.py · gpt-oss ft_gptoss_ner.py)이 이 파일 하나로 채점한다.

규칙
- 개체 = (시작, 끝, 태그) 글자 위치. 정답과 세 값이 **정확히 같아야** 맞음(tp). 예측에만 있으면 fp, 정답에만 있으면 fn.
- ALL = 8태그 전체를 합친 micro P·R·F1. MACRO_F1 = 고정 8태그 F1의 단순 평균(그 세트에 안 나온 태그는 F1 0으로 셈).
- 태그 목록은 GUIDE v1.2 의 8개로 고정(모델·데이터에 따라 바뀌지 않게).
오답 기록: errors(gold, pred) → 문장마다 fp(틀리게 찾은 것)·fn(놓친 것)과 유형(경계 오류·태그 오류·없는 개체·놓침).
"""
import collections, json, os

TAGS = ["DATE", "FIN_ORG", "FIN_PRODUCT", "INDICATOR", "MONEY", "ORG", "POLICY", "RATE"]
EVAL_SETS = {"test_2000": "test_2000.jsonl"}   # 모든 모델 공통 비교용 고정 부분집합(테스트 1만에서 시드 고정 2,000문장)

def _f(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(P, 4), "recall": round(R, 4), "f1": round(2 * P * R / (P + R) if P + R else 0.0, 4), "tp": tp, "fp": fp, "fn": fn}

def score(golds, preds):
    c = collections.defaultdict(lambda: [0, 0, 0])
    for g, p in zip(golds, preds):
        g = {tuple(x) for x in g}; p = {tuple(x) for x in p}
        for x in p: c[x[2]][0 if x in g else 1] += 1
        for x in g - p: c[x[2]][2] += 1
    out = {t: _f(*c[t]) for t in TAGS}
    out["ALL"] = _f(*[sum(c[t][i] for t in c) for i in range(3)])
    out["MACRO_F1"] = round(sum(out[t]["f1"] for t in TAGS) / len(TAGS), 4)
    return out

def errors(text, gold, pred):
    """한 문장의 오답 목록. 유형: boundary(겹치지만 위치 다름, 태그 같음) · tag(위치 같고 태그 다름) · boundary+tag · spurious(정답과 안 겹침) · missed(놓침)"""
    g = {tuple(x) for x in gold}; p = {tuple(x) for x in pred}; out = []
    for s, e, t in sorted(p - g):
        ov = [x for x in g if x[0] < e and s < x[1]]
        kind = "spurious" if not ov else "tag" if any(x[0] == s and x[1] == e for x in ov) else "boundary" if any(x[2] == t for x in ov) else "boundary+tag"
        out.append({"type": kind, "pred": [s, e, t], "pred_text": text[s:e], "gold": [list(x) for x in ov]})
    for s, e, t in sorted(g - p):
        if not any(x[0] < e and s < x[1] for x in p): out.append({"type": "missed", "gold": [s, e, t], "gold_text": text[s:e]})
    return out

def save_preds(path, rows, preds, extra=None):
    """문장마다 정답·예측·오답(errors)을 저장. extra: 문장별 추가 정보 목록(예: 생성 모델의 원래 답)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kinds = collections.Counter()
    with open(path, "w", encoding="utf-8") as f:
        for i, (r, p) in enumerate(zip(rows, preds)):
            er = errors(r["text"], r["spans"], p); kinds.update(e["type"] for e in er)
            d = {"id": r.get("id"), "text": r["text"], "gold": [list(x) for x in r["spans"]], "pred": [list(x) for x in p], "errors": er}
            if extra: d.update(extra[i])
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return dict(kinds)
