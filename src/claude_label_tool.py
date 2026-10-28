# -*- coding: utf-8 -*-
"""Claude 직접 라벨 도구.
  sample:  python src/claude_label_tool.py sample --n 12000      → data/fin_silver/pool.jsonl (골드셋 제외, 무작위)
  show:    python src/claude_label_tool.py show --batch 1 --size 50   → 번호와 문장 출력
  add:     python src/claude_label_tool.py add --batch 1 --file 라벨.txt  → 위치 계산·검사 후 batches/b0001.jsonl
라벨 형식(한 줄에 한 문장): 번호|글자=코드|글자=코드 ...  (코드: FO O P M R D I PO, 문장에 나오는 순서대로)
"""
import argparse, json, os, random, re, sys
CODE = {"FO": "FIN_ORG", "O": "ORG", "P": "FIN_PRODUCT", "M": "MONEY", "R": "RATE", "D": "DATE", "I": "INDICATOR", "PO": "POLICY"}
D = "data/fin_silver"; WORD = re.compile(r"[가-힣a-zA-Z0-9]")

def sample(n):
    gold = {json.loads(l)["text"] for l in open("data/goldset/goldset_master.jsonl", encoding="utf-8")}
    rng = random.Random(20261023); keep = []; seen = 0
    with open("data/econ_pool/econ_all.jsonl", encoding="utf-8") as f:   # 저수지 표집(전체를 메모리에 안 올림)
        for line in f:
            r = json.loads(line); t = r["text"]
            if t in gold or not (20 <= len(t) <= 200) or t.count("(이름)") > 1: continue
            seen += 1
            if len(keep) < n: keep.append(r)
            else:
                j = rng.randrange(seen)
                if j < n: keep[j] = r
    rng.shuffle(keep)
    with open(f"{D}/pool.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(keep, 1): f.write(json.dumps({"k": i, "id": r["id"], "site": r.get("site"), "text": r["text"]}, ensure_ascii=False) + "\n")
    print("후보", seen, "→", len(keep))

def pool(): return [json.loads(l) for l in open(f"{D}/pool.jsonl", encoding="utf-8")]

def show(b, size):
    P = pool()[(b - 1) * size: b * size]
    for r in P: print(f"{r['k']}|{r['text']}")

def add(b, size, path):
    P = {r["k"]: r for r in pool()[(b - 1) * size: b * size]}
    lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    got, warn, out = set(), [], []
    for l in lines:
        parts = l.split("|"); k = int(parts[0]); r = P[k]; t = r["text"]; got.add(k)
        sp, cur = [], 0
        for p in parts[1:]:
            if not p: continue
            tx, code = p.rsplit("=", 1); code = code.strip(); nth = 0
            if "@" in code: code, nth = code.split("@"); nth = int(nth)   # 글자=코드@n : 문장에서 n번째 출현(1부터)
            tg = CODE[code]
            if nth:
                occ = [i for i in range(len(t)) if t.startswith(tx, i)]; s = occ[nth - 1] if len(occ) >= nth else -1
            else:
                s = t.find(tx, cur)
                if s < 0: s = t.find(tx)
            if s < 0: raise SystemExit(f"[{k}] 문장에 없음: {tx!r}")
            sp.append([s, s + len(tx), tg]); cur = s + len(tx)
            if s > 0 and WORD.match(t[s - 1]): warn.append(f"[{k}] 단어 중간에서 시작: {t[max(0,s-3):s+len(tx)]!r} → {tx}")
        sp.sort()
        for a, c in zip(sp, sp[1:]):
            if a[1] > c[0]: raise SystemExit(f"[{k}] 겹침: {t[a[0]:a[1]]} / {t[c[0]:c[1]]}")
        out.append({"id": r["id"], "k": k, "site": r["site"], "text": t, "spans": sp, "labeler": "claude_opus_5_5", "guide": "GUIDE_claude.md v1"})
    miss = sorted(set(P) - got)
    if miss: raise SystemExit(f"빠진 문장 번호: {miss}")
    os.makedirs(f"{D}/batches", exist_ok=True)
    with open(f"{D}/batches/b{b:04d}.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(out, key=lambda x: x["k"]): f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = sum(len(r["spans"]) for r in out)
    print(f"batch {b}: {len(out)}문장, 개체 {n}, 개체 없는 문장 {sum(not r['spans'] for r in out)}")
    for w in warn: print("  확인:", w)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=1); ap.add_argument("--size", type=int, default=50); ap.add_argument("--file")
    a = ap.parse_args()
    {"sample": lambda: sample(a.n), "show": lambda: show(a.batch, a.size), "add": lambda: add(a.batch, a.size, a.file)}[a.cmd]()
