# -*- coding: utf-8 -*-
"""금융 NER 라벨링 도구 (팀원 공용). 표준 라이브러리만 쓴다.

  python label_tool.py next   --me 이름            다음 할 묶음(50문장) 번호와 문장 출력
  python label_tool.py show   --me 이름 --batch N  N번 묶음 문장 출력
  python label_tool.py add    --me 이름 --batch N --file labels/b00001.txt   위치 계산·검사 후 저장
  python label_tool.py status --me 이름            진행률, 태그 분포, 최근 묶음 검사 요약
  python label_tool.py calib  --me 이름            맞추기 10문장 출력(정답 없이)
  python label_tool.py calib-score --me 이름 --file labels/calib.txt   기준 정답과 일치율(F1)
  python label_tool.py export --me 이름            labeled_이름.jsonl 하나로 합치기(보낼 파일)

라벨 입력 형식(한 줄에 한 문장): 번호|글자=코드|글자=코드 ...   코드: FO O P M R D I PO, 글자=코드@n 은 n번째 출현
"""
import argparse, collections, glob, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = {"FO": "FIN_ORG", "O": "ORG", "P": "FIN_PRODUCT", "M": "MONEY", "R": "RATE", "D": "DATE", "I": "INDICATOR", "PO": "POLICY"}
SIZE = 50
WORD = re.compile(r"[가-힣a-zA-Z0-9]")
FIN_HINT = re.compile(r"(은행|증권|證|생명|화재|손보|캐피탈|카드|저축은행|자산운용|운용|금융|투자증권|보험)$")
MONEY_OK = re.compile(r"\d.*(원|달러|엔|유로|위안|센트|弗|파운드|조|억|만)$")
RATE_OK = re.compile(r"\d\s*(%|%p|%포인트|% 포인트|포인트|bp|%p\))$")

def shard(me): return os.path.join(HERE, "data", f"shard_{me}.jsonl")
def outdir(me): d = os.path.join(HERE, "out", me); os.makedirs(d, exist_ok=True); return d
def load_shard(me):
    p = shard(me)
    if not os.path.exists(p): sys.exit(f"분담 파일이 없습니다: {p}  (노션에서 shard_{me}.zip 을 받아 data/ 에 풀었는지 확인)")
    return [json.loads(l) for l in open(p, encoding="utf-8")]
def done_batches(me): return sorted(int(os.path.basename(f)[1:6]) for f in glob.glob(os.path.join(outdir(me), "b*.jsonl")))

def show(me, b):
    rows = load_shard(me)[(b - 1) * SIZE: b * SIZE]
    if not rows: sys.exit("모든 묶음을 끝냈습니다.")
    print(f"### batch {b}  (문장 {rows[0]['k']}~{rows[-1]['k']})  — 라벨은 labels/b{b:05d}.txt 에 저장 후 add")
    for r in rows: print(f"{r['k']}|{r['text']}")

def parse(lines, P):
    got, warn, out = set(), [], []
    for l in lines:
        parts = l.split("|"); k = int(parts[0])
        if k not in P: raise SystemExit(f"[{k}] 이 묶음의 문장 번호가 아닙니다")
        if k in got: raise SystemExit(f"[{k}] 같은 번호가 두 번 있습니다")
        r = P[k]; t = r["text"]; got.add(k); sp, cur = [], 0
        for p in parts[1:]:
            p = p.strip()
            if not p: continue
            if "=" not in p: raise SystemExit(f"[{k}] '글자=코드' 형식이 아닙니다: {p!r}")
            tx, code = p.rsplit("=", 1); code = code.strip(); nth = 0
            if "@" in code: code, nth = code.split("@"); nth = int(nth)
            if code not in CODE: raise SystemExit(f"[{k}] 모르는 코드 {code!r} (FO O P M R D I PO 중 하나)")
            if tx != tx.strip() or not tx: raise SystemExit(f"[{k}] 글자 앞뒤에 공백이 있습니다: {tx!r}")
            if nth:
                occ = [i for i in range(len(t)) if t.startswith(tx, i)]; s = occ[nth - 1] if len(occ) >= nth else -1
            else:
                s = t.find(tx, cur)
                if s < 0: s = t.find(tx)
            if s < 0: raise SystemExit(f"[{k}] 문장에 없는 글자: {tx!r}")
            tg = CODE[code]; sp.append([s, s + len(tx), tg]); cur = s + len(tx)
            if s > 0 and WORD.match(t[s - 1]) and tg not in ("MONEY", "RATE"): warn.append(f"[{k}] 단어 중간에서 시작: …{t[max(0, s - 4):s + len(tx)]} → {tx}={code}")
            if tg == "MONEY" and not MONEY_OK.search(tx): warn.append(f"[{k}] 금액 형식 확인: {tx}")
            if tg == "RATE" and not RATE_OK.search(tx): warn.append(f"[{k}] 비율 형식 확인: {tx}")
            if tg == "ORG" and FIN_HINT.search(tx): warn.append(f"[{k}] 금융회사 이름 같은데 O: {tx} (FO 아닌지 확인)")
            if tx == "(이름)" or "(이름)" in tx: raise SystemExit(f"[{k}] (이름) 은 라벨하지 않습니다")
        sp.sort()
        for a, c in zip(sp, sp[1:]):
            if a[1] > c[0]: raise SystemExit(f"[{k}] 겹침: {t[a[0]:a[1]]} / {t[c[0]:c[1]]} — 긴 쪽 하나만 남기세요")
        out.append({"id": r["id"], "k": k, "site": r.get("site"), "text": t, "spans": sp})
    miss = sorted(set(P) - got)
    if miss: raise SystemExit(f"빠진 문장 번호: {miss}  (개체가 없으면 '번호|' 만 적기)")
    return out, warn

def add(me, b, path, labeler):
    P = {r["k"]: r for r in load_shard(me)[(b - 1) * SIZE: b * SIZE]}
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8") if l.strip()]
    out, warn = parse(lines, P)
    with open(os.path.join(outdir(me), f"b{b:05d}.jsonl"), "w", encoding="utf-8") as f:
        for r in sorted(out, key=lambda x: x["k"]):
            f.write(json.dumps({**r, "labeler": labeler, "member": me, "guide": "GUIDE.md v1"}, ensure_ascii=False) + "\n")
    c = collections.Counter(t for r in out for *_, t in r["spans"])
    print(f"저장: batch {b} — {len(out)}문장, 개체 {sum(c.values())}, 개체 없는 문장 {sum(not r['spans'] for r in out)} | " + " ".join(f"{k}:{v}" for k, v in c.most_common()))
    for w in warn: print("  확인:", w)
    if warn: print("  → 확인 항목이 맞으면 그대로 두고, 틀렸으면 txt를 고쳐 add 를 다시 실행(덮어씀).")

def status(me):
    rows = load_shard(me); d = done_batches(me); total = (len(rows) + SIZE - 1) // SIZE
    c = collections.Counter(); n = ents = empty = 0
    for f in glob.glob(os.path.join(outdir(me), "b*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); n += 1; ents += len(r["spans"]); empty += not r["spans"]; c.update(t for *_, t in r["spans"])
    print(f"{me}: 묶음 {len(d)}/{total} 완료, 문장 {n:,}/{len(rows):,} ({n / len(rows) * 100:.2f}%)")
    if n:
        print(f"  문장당 개체 {ents / n:.2f}, 개체 없는 문장 {empty / n * 100:.1f}%  (Claude 기준 2,850문장: 문장당 약 2.3개, 개체 없음 약 22%)")
        print("  태그 비율:", " ".join(f"{k} {v / ents * 100:.1f}%" for k, v in c.most_common()))
        print("  (Claude 기준: DATE 23 / ORG 21 / INDICATOR 14 / MONEY 11 / RATE 11 / FIN_ORG 10 / FIN_PRODUCT 7 / POLICY 3 %)")
    nxt = next((b for b in range(1, total + 1) if b not in set(d)), None)
    print("  다음 묶음:", nxt)
    return nxt

def calib():
    ex = json.load(open(os.path.join(HERE, "examples.json"), encoding="utf-8"))["calibration"]
    print("### 맞추기 10문장 — 기준서만 보고 라벨을 달아 labels/calib.txt 에 저장 후 calib-score (examples.json 의 정답은 보지 않는다)")
    for i, r in enumerate(ex, 1): print(f"{i}|{r['text']}")

def calib_score(path):
    ex = json.load(open(os.path.join(HERE, "examples.json"), encoding="utf-8"))["calibration"]
    P = {i: {"id": r["id"], "text": r["text"]} for i, r in enumerate(ex, 1)}
    out, _ = parse([l.rstrip("\n") for l in open(path, encoding="utf-8") if l.strip()], P)
    tp = fp = fn = 0
    for r in out:
        G = {tuple(x) for x in ex[r["k"] - 1]["spans"]}; Pp = {tuple(x) for x in r["spans"]}
        tp += len(G & Pp); fp += len(Pp - G); fn += len(G - Pp)
        for s, e, t in sorted(Pp - G): print(f"  [{r['k']}] 기준에 없음: {r['text'][s:e]}={t}")
        for s, e, t in sorted(G - Pp): print(f"  [{r['k']}] 빠뜨림:     {r['text'][s:e]}={t}")
    p = tp / max(1, tp + fp); rc = tp / max(1, tp + fn); f1 = 2 * p * rc / max(1e-9, p + rc)
    print(f"맞추기 일치율: 정밀도 {p:.3f} 재현율 {rc:.3f} F1 {f1:.3f}  (0.90 미만이면 GUIDE.md를 다시 읽고 다른 점을 확인)")

def export(me):
    files = sorted(glob.glob(os.path.join(outdir(me), "b*.jsonl"))); n = 0
    p = os.path.join(HERE, f"labeled_{me}.jsonl")
    with open(p, "w", encoding="utf-8") as fo:
        for f in files:
            for l in open(f, encoding="utf-8"): fo.write(l); n += 1
    print(f"{p}: {n:,}문장 ({len(files)}묶음). 이 파일을 zip으로 묶어 노션 '결과 올리기'에 올린다.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--me"); ap.add_argument("--batch", type=int)
    ap.add_argument("--file"); ap.add_argument("--labeler", default="claude")
    a = ap.parse_args()
    if a.cmd in ("next", "show", "add", "status", "export") and not a.me: sys.exit("--me 내이름(영어) 를 주세요")
    if a.cmd == "next":
        b = status(a.me)
        if b: show(a.me, b)
    elif a.cmd == "show": show(a.me, a.batch)
    elif a.cmd == "add": add(a.me, a.batch, a.file, a.labeler)
    elif a.cmd == "status": status(a.me)
    elif a.cmd == "calib": calib()
    elif a.cmd == "calib-score": calib_score(a.file)
    elif a.cmd == "export": export(a.me)
    else: sys.exit(__doc__)
