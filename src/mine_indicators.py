# -*- coding: utf-8 -*-
"""
경제 문장에서 경제지표(INDICATOR) 후보 표현을 모은다 → 사람이 검토해 사전으로 쓰고, 그 사전으로 선라벨한다.
방법: (1) 지표 이름으로 확실한 씨앗 목록, (2) '…지수·…금리·…률·…수지·…이익·…물가'로 끝나는 표현을 자동 수집(조사 제거).
사용: python src/mine_indicators.py [입력 jsonl] → data/fin_labels/indicator_candidates.json
"""
import json, re, sys, collections
SRC = sys.argv[1] if len(sys.argv) > 1 else "data/econ_sample_100k.jsonl"
SEED = ["기준금리", "코스피", "코스닥", "나스닥", "다우지수", "s&p500", "환율", "원·달러 환율", "원/달러 환율", "원달러 환율",
        "소비자물가지수", "소비자물가", "생산자물가", "물가상승률", "경제성장률", "성장률", "gdp", "국내총생산", "실업률", "고용률",
        "경상수지", "무역수지", "국고채 금리", "가계부채", "가계대출", "주택담보대출 금리", "대출금리", "예금금리", "시가총액",
        "매출액", "매출", "영업이익", "순이익", "당기순이익", "영업이익률", "per", "pbr", "roe", "eps", "주가", "주가지수",
        "아파트값", "집값", "전셋값", "매매가격지수", "전세가율", "청약경쟁률", "연체율", "부채비율", "자기자본비율", "bis비율"]
SUFFIX = r"(?:지수|금리|물가|상승률|증가율|감소율|성장률|률|율|수지|이익|이익률|매출액|가격|값|비율|지표|잔액)"
JOSA = r"(?:은|는|이|가|을|를|의|에|에서|으로|로|와|과|도|만|까지|부터|보다|이며|이다|였다|이었다)?"
pat = re.compile(rf"((?:[가-힣a-z0-9&·/]+\s)?[가-힣a-z0-9&·/]*{SUFFIX}){JOSA}(?=[\s,\.\)\]'\"’”]|$)")
NOT = {"비율", "가격", "값", "이익", "지표", "금리", "물가", "지수", "률", "율", "잔액", "수지", "확률", "효율", "비율이", "법률", "자율", "규율", "비율로", "선율", "능률", "환율"}
rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
seed_c = collections.Counter(); auto_c = collections.Counter(); ex = {}
for r in rows:
    t = r["text"]
    for s in SEED:
        if s in t: seed_c[s] += 1; ex.setdefault(s, t)
    for m in pat.finditer(t):
        w = m.group(1).strip()
        last = w.split()[-1]
        if last in NOT and " " not in w: continue
        if len(w) < 3: continue
        auto_c[w] += 1; ex.setdefault(w, t)
auto = [(w, c) for w, c in auto_c.most_common() if c >= 5 and w not in seed_c]
out = {"source": SRC, "sentences": len(rows),
       "seed": [{"term": w, "count": c, "example": ex[w][:150]} for w, c in seed_c.most_common()],
       "auto_candidates_need_review": [{"term": w, "count": c, "example": ex[w][:150]} for w, c in auto[:400]]}
json.dump(out, open("data/fin_labels/indicator_candidates.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
n_any = sum(1 for r in rows if any(s in r["text"] for s in seed_c))
print(f"문장 {len(rows):,} | 씨앗 지표가 하나라도 나온 문장 {n_any:,} ({n_any/len(rows)*100:.1f}%)")
print("씨앗 상위:", seed_c.most_common(30))
print("자동 후보 상위:", auto[:80])
