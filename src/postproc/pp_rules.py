# 후보 후처리 규칙을 03 dev 예측에 하나씩/함께 적용해 효과 측정 (test 안 봄)
import json, sys, re, collections
from transformers import AutoTokenizer
from kiwipiepy import Kiwi
sys.path.insert(0, r"G:\내 드라이브\NER_라벨_통합_v1.2\finetune"); import fin_score as S
import os
R = "G:/내 드라이브/NER_라벨_통합_v1.2/finetune"
E = "F03_dapt_roberta-large_crf"
rows = [json.loads(l) for l in open(os.path.join(R, "results", E, E + "_dev.jsonl"), encoding="utf-8")]
tk = AutoTokenizer.from_pretrained(os.path.join(R, "results", E, "model")); kiwi = Kiwi()
# 학습 데이터의 관례 확인: RATE/MONEY 정답이 +/- 부호를 포함하나
tr = [json.loads(l) for l in open(os.path.join(R, "data", "train.jsonl"), encoding="utf-8")]
sign_in = sign_out = 0; close_in = close_out = 0; dae_in = 0
for r in tr:
    t = r["text"]
    for a, b, g in r["spans"]:
        if g in ("RATE", "MONEY"):
            if t[a] in "+-−▲▼△▽": sign_in += 1
            elif a > 0 and t[a-1] in "+-−▲▼△▽": sign_out += 1
            if t[a:b].endswith("대"): dae_in += 1
        s_ = t[a:b]
        if s_.count("(") > s_.count(")") and b < len(t) and t[b] == ")": close_out += 1
        if "(" in s_ and s_.endswith(")"): close_in += 1
print(f"[학습 데이터 관례] RATE·MONEY 부호 포함 {sign_in} vs 부호 바로 앞(제외) {sign_out} · 괄호 닫힘 포함 {close_in} vs 미포함(바로 뒤 ')') {close_out} · '대'로 끝남 {dae_in}")
def r_josa(t, sp, cache):
    offs = [o for o in tk(t, return_offsets_mapping=True)["offset_mapping"] if o[0] != o[1]]; out = []
    for a, b, g in sp:
        tok = next(((s, e) for s, e in offs if s < b <= e), None)
        if tok and tok[1] - tok[0] > 1:
            if "k" not in cache: cache["k"] = kiwi.tokenize(t)
            j = [k for k in cache["k"] if k.tag.startswith("J") and tok[0] < k.start < b and k.start > a and k.start + k.len <= b]
            if j: b = min(k.start for k in j)
        out.append([a, b, g])
    return out
def r_paren(t, sp, c):
    return [[a, b + 1, g] if t[a:b].count("(") > t[a:b].count(")") and b < len(t) and t[b] == ")" else [a, b, g] for a, b, g in sp]
def r_sign(t, sp, c):  # 학습 관례가 '제외'면 앞 부호 제거
    return [[a + 1, b, g] if g in ("RATE", "MONEY") and t[a] in "+-−▲▼△▽" and b - a > 1 else [a, b, g] for a, b, g in sp]
def r_dae(t, sp, c):
    return [[a, b - 1, g] if g in ("RATE", "MONEY") and t[a:b].endswith("대") and b - a > 1 else [a, b, g] for a, b, g in sp]
rules = {"P1 조사 자르기": r_josa, "P2 닫는 괄호 포함": r_paren, "P3 RATE·MONEY 앞 부호 제거": r_sign, "P4 RATE·MONEY 끝 '대' 제거": r_dae}
gold = [r["gold"] for r in rows]; base = [r["pred"] for r in rows]
b0 = S.score(gold, base)["ALL"]["f1"]; print(f"03 dev 원래 {b0}")
def apply(fs):
    out = []
    for r, p in zip(rows, base):
        c = {}; q = [list(x) for x in p]
        for f in fs: q = f(r["text"], q, c)
        out.append(q)
    return out
for n, f in rules.items():
    p = apply([f]); ch = sum(x != y for a, b in zip(base, p) for x, y in zip(a, b)); s = S.score(gold, p)
    print(f"  {n}: dev {s['ALL']['f1']} ({s['ALL']['f1'] - b0:+.4f}) · 바뀐 개체 {ch}")
p = apply(list(rules.values())); print(f"  모두 적용: dev {S.score(gold, p)['ALL']['f1']}")
