---
name: fin-ner-label
description: 2팀 금융 NER 실버셋 라벨링. 배정된 경제 뉴스 문장(data/shard_<이름>.jsonl)을 50문장씩 읽고 GUIDE.md 기준으로 FIN_ORG·ORG·FIN_PRODUCT·MONEY·RATE·DATE·INDICATOR·POLICY 라벨을 직접 달아 label_tool.py로 저장·검사한다. 사용자가 "라벨링 시작", "이어서 라벨", "다음 묶음", "/fin-ner-label" 이라고 하면 쓴다.
---

# 금융 NER 라벨링 스킬

## 0. 준비 (세션마다 처음 한 번)
1. 사용자 영어 이름(`--me`)을 확인한다: sejin_choi, kim_byeongseok, kim_hyeonho, jin_taehyeon, park_gyutae 중 하나. 모르면 묻는다. `data/shard_<이름>.jsonl` 이 있는지 확인한다.
2. `GUIDE.md` 를 **처음부터 끝까지** 읽는다. `examples.json` 의 `examples`(정답 예시 31개)를 읽는다. `calibration` 의 정답은 맞추기 전에는 보지 않는다.
3. 맞추기: `python label_tool.py calib --me <이름>` → 10문장 라벨을 `labels/calib.txt` 에 쓰고 `python label_tool.py calib-score --me <이름> --file labels/calib.txt`. F1 0.90 미만이면 틀린 항목의 규칙을 GUIDE.md에서 찾아 읽고 다시 맞춘다. 결과를 `notes/calib_log.md` 에 한 줄 적는다(날짜, F1).
4. `python label_tool.py status --me <이름>` 으로 이어서 할 묶음을 확인한다.

## 1. 한 묶음 처리 (반복)
1. `python label_tool.py next --me <이름>` 를 실행해 50문장을 받는다.
2. 각 문장을 읽고, 아래 순서로 판단한다.
   - 숫자 먼저: MONEY(단위 있는 금액) · RATE(% / %p / bp) · DATE(날짜·기간·상대시점)
   - 기관: 금융이면 FO, 아니면 O (GUIDE 2장 FO 목록·4장 결정표)
   - 지표 이름 I, 금융상품 P, 제도·법·세금 PO
   - 경계: 조사 빼기, 수식어 빼기, 괄호 약어 따로, 겹치면 긴 쪽, 애매하면 달지 않음
3. `labels/b<번호5자리>.txt` 에 `번호|글자=코드|...` 형식으로 50줄을 쓴다. 개체 없는 문장은 `번호|`. 글자는 원문 그대로, 나오는 순서대로.
4. `python label_tool.py add --me <이름> --batch <번호> --file labels/b<번호5자리>.txt`
5. 에러가 나면 메시지대로 고치고 다시 add. **확인:** 줄은 하나씩 판단해 틀린 것만 고쳐 다시 add.
6. 사용자에게는 묶음마다 한 줄만 보고한다: `batch N 완료 — 개체 X개 (확인 Y건 처리)`.

## 2. 20묶음마다
- `status` 로 태그 비율을 기준과 비교한다(CLAUDE.md 품질 점검).
- GUIDE.md 2·3·4장을 다시 읽는다.
- 기준이 애매했던 문장이 있었으면 `notes/기준_제안.md` 에 "문장 번호 / 표현 / 고민 / 내가 택한 것" 형식으로 적는다.

## 3. 하지 말 것
- 규칙 코드·정규식으로 라벨을 대량 생성하지 않는다. 다른 모델 API를 부르지 않는다.
- GUIDE.md, examples.json, data/ 를 고치지 않는다.
- 묶음을 건너뛰거나 일부 문장만 저장하지 않는다(도구가 막는다).
- 속도를 위해 판단을 대충 하지 않는다. 한 묶음은 정확하게, 다음 묶음은 그다음.

## 4. 끝낼 때
- `status` 결과(완료 묶음 수, 다음 묶음 번호)를 사용자에게 알려 준다.
- 사용자가 원하면 `export` 로 `labeled_<이름>.jsonl` 을 만들고 zip으로 묶는 방법을 안내한다.
