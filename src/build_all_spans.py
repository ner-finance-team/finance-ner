# -*- coding: utf-8 -*-
"""
원본 라벨 zip 전체(TL1·VL1)의 모든 문장을 글자 span으로 바꿔 저장한다. 필터링은 하지 않고 표시만 남긴다.

출력: data/spans_full/{train,valid}/part-XXXXX.jsonl.gz  (zip 안 json 파일 묶음 단위 shard)
      data/spans_full/stats.json
한 줄 = {
  "id":       "TL1_000123_c_0004_0002"  (zip_파일번호_title/content_문서번호_문장번호)
  "category", "site", "file", "part": "title"|"content",
  "text":     문장,
  "labels":   원본 라벨 [[text, tag], ...]  (다른 규칙으로 다시 변환할 때 사용)
  "spans":    [[start, end, tag], ...]  모든 출현 태깅, 겹치면 긴 span 우선 (prepare_data.make_spans 와 같은 규칙)
  "unmatched": 문장에서 못 찾은 라벨 수,
  "boiler":   상용구(기사 꼬리말) 여부 (prepare_data.BOILER 정규식)
}
중복 문장 제거는 표본을 뽑을 때 한다(전체를 한 번에 메모리에 올리지 않기 위해).

사용:
  python src/build_all_spans.py                 # 전체
  python src/build_all_spans.py --limit-files 20 --out data/spans_test   # 시험
"""
import argparse, gzip, json, os, sys, time, zipfile, collections
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import DEF_BASE, TRAIN_ZIP, VAL_ZIP, TAGS, BOILER, make_spans, dec  # 규칙을 한 곳에서 관리

def work(job):
    zpath, prefix, names, out_path = job
    z = zipfile.ZipFile(zpath)
    st = collections.Counter()
    tagc = collections.Counter()
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=5) as fo:
        for fname in names:
            info = z.getinfo(fname)
            disp = dec(fname)
            cat = disp.split("/")[0]
            try:
                obj = json.loads(z.read(info).decode("utf-8-sig"))
            except Exception:
                st["bad_file"] += 1; continue
            st["files"] += 1
            fidx = os.path.splitext(os.path.basename(disp))[0]
            for di, doc in enumerate(obj.get("named_entity", [])):
                site = doc.get("source_site", "")
                for part in ("title", "content"):
                    for si, s in enumerate(doc.get(part, []) or []):
                        text = s.get("sentence", "") or ""
                        labels = [[l.get("text", ""), l.get("tag")] for l in (s.get("labels") or [])]
                        spans = make_spans(text, [{"text": a, "tag": b} for a, b in labels])
                        unmatched = sum(1 for a, b in labels if b in TAGS and a and a not in text)
                        boiler = bool(BOILER.search(text))
                        rec = {"id": f"{prefix}_{fidx}_{part[0]}_{di:04d}_{si:04d}", "category": cat, "site": site,
                               "file": disp, "part": part, "text": text, "labels": labels, "spans": spans,
                               "unmatched": unmatched, "boiler": boiler}
                        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        st["sentences"] += 1; st["labels"] += len(labels); st["spans"] += len(spans)
                        st["unmatched"] += unmatched; st["boiler"] += boiler
                        if not spans: st["no_entity"] += 1
                        for _, _, t in spans: tagc[t] += 1
    return dict(st), dict(tagc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEF_BASE)
    ap.add_argument("--out", default="data/spans_full")
    ap.add_argument("--files-per-shard", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit-files", type=int, default=0, help="시험용: zip마다 앞의 N개 파일만")
    a = ap.parse_args()
    t0 = time.time(); stats = {"args": vars(a)}
    for split, rel, prefix in (("valid", VAL_ZIP, "VL1"), ("train", TRAIN_ZIP, "TL1")):
        zpath = os.path.join(a.base, rel)
        names = sorted(i.filename for i in zipfile.ZipFile(zpath).infolist() if i.filename.lower().endswith(".json"))
        if a.limit_files: names = names[:a.limit_files]
        od = os.path.join(a.out, split); os.makedirs(od, exist_ok=True)
        jobs = [(zpath, prefix, names[i:i + a.files_per_shard], os.path.join(od, f"part-{i // a.files_per_shard:05d}.jsonl.gz"))
                for i in range(0, len(names), a.files_per_shard)]
        tot, tags = collections.Counter(), collections.Counter()
        t1 = time.time()
        with Pool(a.workers) as pool:
            for k, (st, tg) in enumerate(pool.imap_unordered(work, jobs), 1):
                tot.update(st); tags.update(tg)
                print(f"[{split}] shard {k}/{len(jobs)} | 문장 {tot['sentences']:,} | {time.time() - t1:.0f}s", flush=True)
        stats[split] = {"zip": rel, "json_files": len(names), "shards": len(jobs), **dict(tot), "tags": dict(tags),
                        "elapsed_sec": round(time.time() - t1, 1)}
        json.dump(stats, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
