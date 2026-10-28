# -*- coding: utf-8 -*-
"""큰 학습 세트를 미리 토큰·라벨 배열로 바꿔 둔다(학습 서버 RAM 9GB·CPU 2.3코어라 학습 때 토큰화하지 않게).

  python src/encode_big.py --file data/split_v3b/train_5000k.jsonl --tok klue/roberta-base --out data/enc/roberta_5000k
  python src/encode_big.py --file ... --tok jhu-clsp/mmBERT-base --pretok morph --out data/enc/mmbert_morph_5000k

라벨 붙이는 방식은 1단계와 똑같다: ner_lib.encode(문장, 토크나이저, max_len, pretok) 를 그대로 쓴다.
출력: ids.bin(토큰, uint16/uint32) · labels.bin(int8, -100 포함) · starts.npy(문장 시작 위치, int64, 길이 N+1) · meta.json
검사: 처음 2,000문장은 저장된 배열을 다시 읽어 ner_lib.encode 결과와 한 개도 다르지 않은지 확인.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

G = {}
def init(tok_name, max_len, pretok):
    from transformers import AutoTokenizer
    import ner_lib
    G.update(tok=AutoTokenizer.from_pretrained(tok_name), max_len=max_len, pretok=pretok or None, ner_lib=ner_lib)

def work(lines):
    ids, labs, lens = [], [], []
    for l in lines:
        e = G["ner_lib"].encode(json.loads(l), G["tok"], G["max_len"], G["pretok"])
        ids.extend(e["input_ids"]); labs.extend(e["labels"]); lens.append(len(e["input_ids"]))
    return ids, labs, lens

def chunks(path, n):
    buf = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            buf.append(l)
            if len(buf) == n: yield buf; buf = []
    if buf: yield buf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True); ap.add_argument("--tok", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--pretok", default=""); ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer
    V = len(AutoTokenizer.from_pretrained(a.tok)); dt = np.uint16 if V < 65535 else np.uint32
    n = tok = trunc = 0; starts = [0]
    with open(os.path.join(a.out, "ids.bin"), "wb") as fi, open(os.path.join(a.out, "labels.bin"), "wb") as fl, \
         Pool(a.workers, initializer=init, initargs=(a.tok, a.max_len, a.pretok)) as P:
        for ids, labs, lens in P.imap(work, chunks(a.file, 5000)):
            np.asarray(ids, dtype=dt).tofile(fi); np.asarray(labs, dtype=np.int8).tofile(fl)
            for L in lens:
                starts.append(starts[-1] + L); trunc += L >= a.max_len
            n += len(lens); tok += len(ids)
            if n % 500_000 < 5000: print(f"[{time.time()-t0:5.0f}s] {n:,}문장", flush=True)
    np.save(os.path.join(a.out, "starts.npy"), np.asarray(starts, dtype=np.int64))
    meta = {"file": a.file, "tok": a.tok, "pretok": a.pretok or None, "max_len": a.max_len, "vocab": V, "dtype": np.dtype(dt).name,
            "sentences": n, "tokens": tok, "tokens_per_sentence": round(tok / n, 2), "truncated": int(trunc), "sec": round(time.time() - t0)}
    # 검사: 앞 2,000문장 다시 계산해 비교
    init(a.tok, a.max_len, a.pretok)
    ids = np.memmap(os.path.join(a.out, "ids.bin"), dtype=dt, mode="r"); labs = np.memmap(os.path.join(a.out, "labels.bin"), dtype=np.int8, mode="r")
    bad = 0
    for i, l in enumerate(next(chunks(a.file, 2000))):
        e = G["ner_lib"].encode(json.loads(l), G["tok"], a.max_len, a.pretok or None)
        s, t = starts[i], starts[i + 1]
        if list(ids[s:t]) != e["input_ids"] or list(labs[s:t]) != e["labels"]: bad += 1
    assert len(starts) - 1 == n and starts[-1] == tok == ids.size == labs.size
    meta["check_first_2000"] = f"{2000-bad}/2000"
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(meta)
    if bad: raise SystemExit(f"다시 계산과 다름: {bad}문장")

if __name__ == "__main__":
    main()
