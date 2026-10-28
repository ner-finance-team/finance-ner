# 금융 NER 라벨링 키트 (팀원용) — 2026-10-23

경제 뉴스 100만 문장을 5명이 20만 문장씩 나눠, **각자 자기 Claude Code 세션에서 Claude가 직접 라벨을 달게** 하는 키트입니다.

## 1. 받을 것 (노션 "금융 라벨링 100만 — 분담과 키트" 페이지)
- `fin_label_kit.zip` — 이 폴더 전체(CLAUDE.md, 스킬, GUIDE.md, examples.json, label_tool.py)
- `shard_<내이름>.zip` — 내 분담 20만 문장

| 이름 | 분담 파일 | 문장 수 |
|---|---|---|
| sejin_choi (최세진) | shard_sejin_choi.jsonl | 200,000 |
| kim_byeongseok (김병석) | shard_kim_byeongseok.jsonl | 200,000 |
| kim_hyeonho (김현호) | shard_kim_hyeonho.jsonl | 200,000 |
| jin_taehyeon (진태현) | shard_jin_taehyeon.jsonl | 200,000 |
| park_gyutae (박규태) | shard_park_gyutae.jsonl | 200,000 |

100만 문장은 AI Hub 말뭉치 경제 카테고리 540만 문장에서 무작위로 뽑았습니다. **2단계 골드셋(평가용) 510문장, 1단계 dev·test, Claude가 따로 라벨 중인 12,000문장과 겹치는 문장은 0개**이고, 중복 문장도 없습니다(`data/split_stats.json`).

## 2. 설치 (5분)
1. `fin_label_kit.zip` 을 원하는 곳에 푼다. 예: `C:\fin_label_kit\`
2. `shard_<내이름>.zip` 을 풀어 나온 jsonl을 `fin_label_kit\data\` 에 넣는다.
3. Python 3.8 이상이면 된다(추가 설치 없음).
4. VS Code 또는 터미널에서 **`fin_label_kit` 폴더를 열고** Claude Code를 시작한다. 그래야 CLAUDE.md와 스킬이 자동으로 적용된다.

## 3. 시작
Claude에게 이렇게 말한다:
```
/fin-ner-label 내 이름은 kim_byeongseok 이야. 라벨링 시작해 줘.
```
Claude가 기준서 읽기 → 맞추기 10문장 점검(F1 0.90 이상이면 통과) → 50문장씩 라벨 → 저장·검사를 반복합니다. 중간에 끊겨도 `/fin-ner-label 이어서` 라고 하면 다음 묶음부터 이어갑니다(완료한 묶음은 `out/<이름>/` 에 저장).

## 4. 사람이 할 일
- 가끔 `python label_tool.py status --me <이름>` 결과를 봅니다(진행률, 태그 비율).
- `labels/` 의 txt 몇 개를 열어 이상한 라벨이 없는지 눈으로 봅니다.
- 기준이 애매한 사례는 Claude가 `notes/기준_제안.md` 에 적어 둡니다. 회의 때 가져옵니다.
- **기준서(GUIDE.md)는 각자 고치지 않습니다.** 바뀌면 노션에 새 판이 올라옵니다.

## 5. 결과 보내기
```
python label_tool.py export --me <이름>
```
`labeled_<이름>.jsonl` 을 zip으로 묶어 노션 "결과 올리기"에 올립니다. 중간 결과(예: 1만 문장마다)를 올려도 됩니다.

## 6. 현실적인 속도 (참고)
- Claude가 직접 다는 방식은 50문장 한 묶음에 보통 1~2분 걸립니다(최세진 세션 측정: 2,850문장).
- 20만 문장은 4,000묶음이라 **한 사람이 한 번에 끝낼 수 있는 양이 아닙니다.** Claude 요금제의 사용량 한도에 따라 하루에 처리할 수 있는 양이 정해집니다. 하루 처리량을 status로 재서 팀에 공유하고, 목표량은 그 숫자를 보고 다시 정합니다.
- 맞추기 F1과 태그 비율을 지키는 것이 양보다 중요합니다. 5명 라벨이 서로 다르면 합쳐서 학습할 때 오히려 점수가 떨어집니다.

## 7. 파일 설명
| 파일 | 내용 |
|---|---|
| CLAUDE.md | Claude가 이 폴더에서 지킬 규칙(자동 적용) |
| .claude/skills/fin-ner-label/SKILL.md | 라벨링 절차 스킬 (/fin-ner-label) |
| GUIDE.md | 라벨 기준서 v1 (태그 정의, 경계 규칙, 결정표, 입력 형식) |
| examples.json | 정답 예시 31문장 + 맞추기 10문장(정답 포함 — 맞추기 전에 보지 않기) |
| label_tool.py | next / show / add / status / calib / calib-score / export |
| data/shard_<이름>.jsonl | 내 분담 문장 (k=번호, id, site, text) |
