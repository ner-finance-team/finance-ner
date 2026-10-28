# -*- coding: utf-8 -*-
"""
말뭉치 TL1 전체(spans_full/train)에서 '경제' 카테고리 문장을 모두 뽑는다. 2단계 금융 라벨링의 재료(골드셋·학습용 문장을 여기서 고른다).
제외: 상용구 문장, 15글자 미만, 같은 문장 중복. 1단계의 dev·test와 같은 문장도 제외(평가 오염 방지).
기존 7종 라벨(spans)은 참고용으로 같이 저장한다(2단계 정답으로 그대로 쓰지 않음).
사용: python src/extract_econ.py → data/econ_pool/econ_all.jsonl, stats.json
"""
import glob, gzip, json, os, time, collections
from multiprocessing import Pool

KEY = '"category": "경제"'

def scan(path):
    rows, n = [], collections.Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if KEY not in line: continue
            r = json.loads(line)
            if r["category"] != "경제": continue
            n["econ"] += 1
            if r["boiler"]: n["drop_boiler"] += 1; continue
            if len(r["text"]) < 15: n["drop_short"] += 1; continue
            rows.append({"id": r["id"], "site": r["site"], "file": r["file"], "text": r["text"], "spans_7tag": r["spans"]})
    return rows, n

def main():
    t0 = time.time()
    banned = set()
    for p in ("data/split_v1/dev.jsonl", "data/split_v1/test.jsonl"):
        with open(p, encoding="utf-8") as f: banned |= {json.loads(l)["text"] for l in f}
    files = sorted(glob.glob("data/spans_full/train/*.jsonl.gz"))
    os.makedirs("data/econ_pool", exist_ok=True)
    seen, tot, kept = set(), collections.Counter(), 0
    sites = collections.Counter()
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as P, open("data/econ_pool/econ_all.jsonl", "w", encoding="utf-8") as out:
        for i, (rows, n) in enumerate(P.imap_unordered(scan, files)):
            tot.update(n)
            for r in rows:
                if r["text"] in seen or r["text"] in banned: tot["drop_dup_or_eval"] += 1; continue
                seen.add(r["text"]); kept += 1; sites[r["site"]] += 1
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (i + 1) % 40 == 0: print(f"  {i+1}/{len(files)} 파일, {kept:,}문장, {time.time()-t0:.0f}초", flush=True)
    st = {"kept": kept, **dict(tot), "sites": dict(sites.most_common()), "elapsed_sec": round(time.time() - t0, 1)}
    json.dump(st, open("data/econ_pool/stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open("data/econ_pool/econ_all.jsonl.done", "w").close()
    print(json.dumps(st, ensure_ascii=False))

if __name__ == "__main__":
    main()
