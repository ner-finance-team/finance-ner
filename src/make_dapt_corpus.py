# -*- coding: utf-8 -*-
"""도메인 사전학습(MLM)용 경제 문장 만들기 — 라벨 데이터·평가 데이터와 겹치지 않게.

입력: data/econ_pool/econ_all.jsonl (경제 5,398,845문장. 상용구·15자 미만·중복·1단계 dev/test 이미 제외)
뺄 것 (평가에 쓸 문장 — 사전학습에서 미리 읽히면 점수가 부풀 수 있음):
  - 2단계 금융 골드셋 510문장 (최종 평가용)
  - Claude 직접 라벨 표본 pool 12,000 + 시범 500 + 맞추기 10 (dev 검증용으로 쓸 예정)
넣는 것: 실버 100만(5명 분담 shard)은 학습용이라 글자만 읽혀도 문제없어 포함한다(2026-10-25 결정).
목표 500만. 남는 문장이 그보다 적으면 남은 것 전부를 쓴다.
출력: data/dapt/econ_dapt.txt.gz (한 줄 = 한 문장, 무작위 순서), data/dapt/stats.json
"""
import gzip, json, os, random, time, glob

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *p: os.path.join(HERE, *p)
TARGET, SEED = 5_000_000, 20261025
norm = lambda t: t.strip().replace("\n", " ")

def texts(path):
    return {norm(json.loads(l)["text"]) for l in open(path, encoding="utf-8")}

t0 = time.time()
excl, parts = set(), {}
srcs = {"goldset": [P("data", "goldset", "goldset_master.jsonl")],
        "claude_pool": [P("data", "fin_silver", "pool.jsonl"), P("data", "fin_silver", "claude_silver_500.jsonl"),
                        P("data", "fin_silver", "claude_calib10_v1.jsonl")]}
for name, files in srcs.items():
    s = set()
    for f in files: s |= texts(f)
    parts[name] = len(s); excl |= s
print("뺄 문장", parts, "합", len(excl))

keep, dropped = [], 0
for l in open(P("data", "econ_pool", "econ_all.jsonl"), encoding="utf-8"):
    t = norm(json.loads(l)["text"])
    if t in excl: dropped += 1; continue
    keep.append(t)
rng = random.Random(SEED)
pick = rng.sample(keep, TARGET) if len(keep) > TARGET else (rng.shuffle(keep) or keep)
assert not (set(pick) & excl)
os.makedirs(P("data", "dapt"), exist_ok=True)
with gzip.open(P("data", "dapt", "econ_dapt.txt.gz"), "wt", encoding="utf-8", compresslevel=6) as fo:
    for t in pick: fo.write(t + "\n")
st = {"source": "data/econ_pool/econ_all.jsonl", "excluded_sets": parts, "excluded_total": len(excl),
      "econ_dropped": dropped, "available": len(keep), "target": TARGET, "picked": len(pick), "seed": SEED,
      "chars_mean": round(sum(map(len, pick)) / len(pick), 1), "elapsed_sec": round(time.time() - t0)}
json.dump(st, open(P("data", "dapt", "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(st)
