# -*- coding: utf-8 -*-
"""사전학습 말뭉치를 형태소 단위로 띄운 판으로 바꾼다 (mmBERT용).

  python src/make_dapt_morph.py            # data/dapt/econ_dapt.txt.gz → data/dapt/econ_dapt_morph.txt.gz

규칙은 ner_lib.morph_split 과 똑같다(형태소 경계에 공백만 끼우고 글자는 그대로, 축약 형태소 안은 안 나눔).
파인튜닝·추론도 같은 함수(pretok="morph")를 써야 사전학습 때 본 문장 모양과 맞는다.
"""
import gzip, json, os, sys, time
from kiwipiepy import Kiwi

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
SRC = os.path.join(HERE, "data", "dapt", "econ_dapt.txt.gz")
DST = os.path.join(HERE, "data", "dapt", "econ_dapt_morph.txt.gz")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 시험용: 앞 N줄만

def split_with(text, toks):   # ner_lib.morph_split 의 자르기 부분과 동일
    inside = set()
    for t in toks:
        inside.update(range(t.start + 1, t.start + t.len))
    cuts = {t.start for t in toks if 0 < t.start < len(text) and t.start not in inside
            and not text[t.start - 1].isspace() and not text[t.start].isspace()}
    return "".join((" " + ch) if i in cuts else ch for i, ch in enumerate(text))

def lines():
    with gzip.open(SRC, "rt", encoding="utf-8") as f:
        for n, l in enumerate(f):
            if LIMIT and n >= LIMIT: break
            yield l.rstrip("\n")

kiwi = Kiwi(num_workers=os.cpu_count())
t0, n, added = time.time(), 0, 0
out = DST if not LIMIT else DST.replace(".txt.gz", f"_lim{LIMIT}.txt.gz")
with gzip.open(out + ".part", "wt", encoding="utf-8", compresslevel=6) as fo:
    src = list(lines()) if LIMIT else None
    it = src if LIMIT else lines()
    buf = []
    def flush(buf):
        global n, added
        for text, toks in zip(buf, kiwi.tokenize(buf)):
            new = split_with(text, toks)
            assert new.replace(" ", "") == text.replace(" ", "")
            added += len(new) - len(text); n += 1
            fo.write(new + "\n")
    for t in it:
        buf.append(t)
        if len(buf) == 50000:
            flush(buf); buf = []
            if n % 500000 == 0:
                print(f"{n:,}줄 · {n/(time.time()-t0):,.0f}줄/초", flush=True)
    if buf: flush(buf)
os.replace(out + ".part", out)

# ner_lib.morph_split 과 결과가 같은지 앞 2,000줄로 확인
from ner_lib import morph_split
with gzip.open(out, "rt", encoding="utf-8") as f:
    got = [next(f).rstrip("\n") for _ in range(min(2000, n))]
ref = [morph_split(t)[0] for t in list(lines())[:len(got)]] if LIMIT else None
if ref is None:
    with gzip.open(SRC, "rt", encoding="utf-8") as f:
        ref = [morph_split(next(f).rstrip("\n"))[0] for _ in range(len(got))]
same = sum(a == b for a, b in zip(got, ref))
st = {"source": SRC, "out": out, "lines": n, "spaces_added": added, "spaces_per_line": round(added / max(n, 1), 2),
      "check_vs_ner_lib": f"{same}/{len(got)}", "workers": os.cpu_count(), "sec": round(time.time() - t0)}
print(st)
json.dump(st, open(out.replace(".txt.gz", "_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
if same != len(got): raise SystemExit("ner_lib.morph_split 과 결과가 다름")
