# -*- coding: utf-8 -*-
"""
경제지표(INDICATOR) 사전 v0. mine_indicators.py 결과를 보고 사람이(여기서는 Claude가 초안) 고른 핵심 지표 이름을 무리별로 정리.
경계 규칙: 지표 이름만(앞의 시점·지역·형용사 '지난해·서울·높은·평균'은 빼고, 시점은 DATE로 따로). 값(3.5%, 1조원)은 RATE·MONEY로 따로.
사용: python src/build_indicator_dict.py [입력 jsonl] → data/fin_labels/indicator_dict_v0.json (무리별 지표 + 입력 문장에서의 등장 수)
"""
import json, sys, collections, re
SRC = sys.argv[1] if len(sys.argv) > 1 else "data/econ_sample_100k.jsonl"
DICT = {
 "기업 실적·재무": ["매출액", "매출", "영업이익률", "영업이익", "당기순이익", "순이익", "영업손실", "순손실", "부채비율", "자기자본비율", "BIS비율", "bis비율",
                 "PER", "per", "PBR", "pbr", "ROE", "roe", "EPS", "eps", "시장 점유율", "점유율", "배당수익률", "지분율"],
 "주식·시장": ["코스피", "코스닥", "나스닥", "다우지수", "S&P500", "s&p500", "주가지수", "주가", "시가총액", "거래대금", "수익률"],
 "금리": ["기준금리", "법정 최고금리", "최고금리", "대출금리", "예금금리", "주택담보대출 금리", "보금자리론 금리", "정기예금 금리", "국고채 금리", "가산금리", "만기이자율", "금리"],
 "환율·거시": ["원·달러 환율", "원/달러 환율", "원달러 환율", "환율", "소비자물가지수", "소비자물가", "생산자물가", "물가상승률", "경제성장률", "성장률", "국내총생산", "GDP", "gdp",
            "실업률", "고용률", "경상수지", "무역수지", "가계부채", "가계대출", "연체율"],
 "부동산": ["매매가격지수", "아파트값", "아파트 매매가격", "매매가격", "전셋값", "전세가격", "전세가율", "집값", "주택가격", "분양가격", "분양가", "공시가격", "공시지가",
          "청약경쟁률", "청약 경쟁률", "낙찰가율", "용적률", "건폐율"],
 "보험": ["손해율", "자동차보험 손해율", "실손보험 손해율"],
}
ALL = sorted({w for v in DICT.values() for w in v}, key=len, reverse=True)
group = {w: g for g, v in DICT.items() for w in v}
rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
cnt = collections.Counter(); sent_hit = 0
for r in rows:
    t = r["text"]; taken = [False] * len(t); hit = False
    for w in ALL:   # 긴 것부터, 겹치면 긴 쪽만
        for m in re.finditer(re.escape(w), t):
            if any(taken[m.start():m.end()]): continue
            # 단어 안쪽 일부('매출'이 '매출채권' 앞부분)는 제외: 뒤 글자가 한글이면 조사만 허용
            nxt = t[m.end():m.end() + 1]
            if nxt and re.match(r"[가-힣]", nxt) and not re.match(r"[은는이가을를의에과와도만로]", nxt): continue
            for i in range(m.start(), m.end()): taken[i] = True
            cnt[w] += 1; hit = True
    sent_hit += hit
by_group = {g: sorted(((w, cnt[w]) for w in v), key=lambda x: -x[1]) for g, v in DICT.items()}
out = {"version": "v0 (Claude 초안, 팀 검토 전)", "source": SRC, "sentences": len(rows), "sentences_with_indicator": sent_hit,
       "groups": {g: [{"term": w, "count": c} for w, c in lst] for g, lst in by_group.items()},
       "group_totals": {g: sum(c for _, c in lst) for g, lst in by_group.items()}}
json.dump(out, open("data/fin_labels/indicator_dict_v0.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"문장 {len(rows):,} 중 지표가 있는 문장 {sent_hit:,} ({sent_hit/len(rows)*100:.1f}%), 지표 출현 {sum(cnt.values()):,}회")
for g, lst in by_group.items(): print(f"  {g}: {out['group_totals'][g]:,}회 | " + ", ".join(f"{w} {c}" for w, c in lst[:8]))
