# -*- coding: utf-8 -*-
"""
AI Hub 웹데이터 기반 한국어 말뭉치(NE 라벨) → 층화 표본 + 상용구 필터 + 문자 span + JSONL
- zip을 풀지 않고 파일 단위로 스트리밍한다.
- 출력: data/processed/{train,dev,test}.jsonl  각 줄 = {"id","text","spans":[[s,e,tag],...],"category","site","split_src"}
         data/processed/stats.json (표본 통계)
사용:
  python src/prepare_data.py --n-train 60000 --n-dev 10000 --n-test 10000 --out data/processed
  python src/prepare_data.py --n-train 2000 --n-dev 300 --n-test 300 --out data/processed_small   # 빠른 점검
옵션:
  --no-filter        상용구·중복 필터 끄기 (baseline 조건)
  --first-only       모든 출현 대신 첫 출현만 태깅
  --min-rare 2000    NW/TR/FD 포함 문장 최소 보장 수 (학습셋)
"""
import argparse, json, os, random, re, zipfile, collections, time

DEF_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "030.웹데이터 기반 한국어 말뭉치 데이터", "01.데이터")
TRAIN_ZIP = os.path.join("1.Training", "라벨링데이터", "TL1.zip")
VAL_ZIP = os.path.join("2.Validation", "라벨링데이터", "VL1.zip")
TAGS = ["CV", "OG", "TM", "AF", "FD", "TR", "NW"]
RARE = {"NW", "TR", "FD"}

# ── 상용구(기사 꼬리말) 필터: 데이터 리포트 4-1 근거 ──
BOILER = re.compile(r"(저작권자|무단\s*전재|무단전재|재배포\s*금지|\(이메일\)|copyright|ⓒ|©|^=\s*\.$|^\s*[\w.\-]+@\s*$)", re.I)
EMAIL_ID = re.compile(r"^[\w.\-]+@$")          # 'yns4656@' 류 라벨 제거

def dec(name):
    try: return name.encode("cp437").decode("cp949")
    except Exception: return name

def iter_sentences(zpath, files):
    """zip 안의 json 파일 목록을 읽어 (category, site, sentence, labels) 생성"""
    z = zipfile.ZipFile(zpath)
    for info in files:
        cat = dec(info.filename).split("/")[0]
        try:
            obj = json.loads(z.read(info).decode("utf-8-sig"))
        except Exception:
            continue
        for doc in obj.get("named_entity", []):
            site = doc.get("source_site", "")
            for part in ("title", "content"):
                for s in doc.get(part, []) or []:
                    yield cat, site, s.get("sentence", ""), s.get("labels") or []

def make_spans(sent, labels, first_only=False):
    """text+tag → 문자 오프셋 span. 모든 출현(기본) 또는 첫 출현. 겹침은 긴 span 우선."""
    cand = []
    for l in labels:
        tx, tg = l.get("text", ""), l.get("tag")
        if not tx or tg not in TAGS or EMAIL_ID.match(tx):
            continue
        start = 0
        while True:
            p = sent.find(tx, start)
            if p < 0: break
            cand.append((p, p + len(tx), tg))
            if first_only: break
            start = p + 1
    # 겹침 해소: 길이 긴 것 우선, 그다음 앞선 것
    cand.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    chosen = []
    for s, e, tg in cand:
        if all(e <= cs or s >= ce for cs, ce, _ in chosen):
            chosen.append((s, e, tg))
    chosen.sort()
    return [list(x) for x in chosen]

def sample_split(zpath, n_target, seed, use_filter, first_only, min_rare, want_rare_boost):
    """카테고리 층화: 카테고리별 목표 n_target/17, 소수 태그 문장 최소 보장"""
    rnd = random.Random(seed)
    z = zipfile.ZipFile(zpath)
    files = [i for i in z.infolist() if i.filename.lower().endswith(".json")]
    by_cat = collections.defaultdict(list)
    for i in files: by_cat[dec(i.filename).split("/")[0]].append(i)
    cats = sorted(by_cat)
    per_cat = max(1, n_target // len(cats))
    out, seen, stat = [], set(), collections.Counter()
    rare_count = collections.Counter()
    for cat in cats:
        fl = by_cat[cat][:]; rnd.shuffle(fl)
        got = 0
        for cat2, site, sent, labels in iter_sentences(zpath, fl):
            stat["seen"] += 1
            if use_filter:
                if BOILER.search(sent): stat["drop_boiler"] += 1; continue
                if sent in seen: stat["drop_dup"] += 1; continue
            seen.add(sent)
            if len(sent) < 5: stat["drop_short"] += 1; continue
            spans = make_spans(sent, labels, first_only)
            tags_here = {t for _, _, t in spans}
            is_rare = bool(tags_here & RARE)
            # 카테고리 할당량 채웠으면 소수 태그 문장만 추가 수집
            if got >= per_cat and not (want_rare_boost and is_rare and any(rare_count[t] < min_rare for t in tags_here & RARE)):
                if got >= per_cat and sum(1 for t in RARE if rare_count[t] >= min_rare) == len(RARE): break
                if got >= per_cat: continue
            out.append({"id": f"{os.path.basename(zpath)[:3]}_{len(out):07d}", "text": sent, "spans": spans,
                        "category": cat, "site": site, "split_src": os.path.basename(zpath)})
            got += 1
            for t in tags_here & RARE: rare_count[t] += 1
            for t in tags_here: stat[f"sent_with_{t}"] += 1
            if not spans: stat["sent_no_entity"] += 1
    rnd.shuffle(out)
    return out, stat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEF_BASE)
    ap.add_argument("--n-train", type=int, default=60000); ap.add_argument("--n-dev", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=10000); ap.add_argument("--out", default="data/processed")
    ap.add_argument("--no-filter", action="store_true"); ap.add_argument("--first-only", action="store_true")
    ap.add_argument("--min-rare", type=int, default=2000); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    use_filter = not a.no_filter
    # train+dev: Training zip에서 한 번에 뽑고 나눔 / test: Validation zip
    trdev, st1 = sample_split(os.path.join(a.base, TRAIN_ZIP), a.n_train + a.n_dev, a.seed, use_filter, a.first_only,
                              a.min_rare, True)
    test, st2 = sample_split(os.path.join(a.base, VAL_ZIP), a.n_test, a.seed + 1, use_filter, a.first_only, 0, False)
    train, dev = trdev[:a.n_train], trdev[a.n_train:a.n_train + a.n_dev]
    stats = {"args": vars(a), "elapsed_sec": round(time.time() - t0, 1)}
    for name, rows, st in (("train", train, st1), ("dev", dev, None), ("test", test, st2)):
        with open(os.path.join(a.out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tagc = collections.Counter(t for r in rows for _, _, t in r["spans"])
        cats = collections.Counter(r["category"] for r in rows)
        stats[name] = {"sentences": len(rows), "entities": sum(tagc.values()), "tags": dict(tagc),
                       "sent_no_entity": sum(1 for r in rows if not r["spans"]), "categories": dict(cats),
                       "avg_len": round(sum(len(r["text"]) for r in rows) / max(len(rows), 1), 1)}
        if st is not None: stats[name]["sampling"] = dict(st)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: (v if k in ("args", "elapsed_sec") else {kk: vv for kk, vv in v.items() if kk != "categories"})
                      for k, v in stats.items()}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
