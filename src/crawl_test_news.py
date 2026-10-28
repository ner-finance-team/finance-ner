# -*- coding: utf-8 -*-
"""최종 테스트용 네이버 경제 기사 문장 수집 (약 2,000문장, 최근 6년).

  python src/crawl_test_news.py --target 2000 --out data/fin_test_crawl

- 경제 하위 분야 8개(금융·증권·부동산·산업/재계·중기/벤처·글로벌 경제·생활경제·경제 일반)에 똑같이 나눈다(분야당 target/8).
- 2020-11-01 ~ 오늘 사이에서 날짜를 무작위로 골라(시드 고정) 그날 그 분야 속보 목록에서 기사 1개를 무작위로 고른다.
- 기사 단위: 제목 1문장 + 본문 문장(Kiwi로 나눔) 앞에서부터 최대 --per_article 개. 사진 설명·기자 이메일·저작권 문구·15자 미만 제외.
- 같은 문장 중복 제외. 사전학습 말뭉치(econ_all)·Claude 표본·골드셋과 글자가 같은 문장 제외.
출력: test_crawl.jsonl (한 줄 = 한 문장: id·text·kind(title/body)·section·date·press·url·article_title), articles.jsonl, stats.json
"""
import argparse, datetime, gzip, hashlib, json, os, random, re, time, collections
import requests
from bs4 import BeautifulSoup
from kiwipiepy import Kiwi

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
SECTIONS = {"금융": 259, "증권": 258, "부동산": 260, "산업/재계": 261, "중기/벤처": 771, "글로벌 경제": 262, "생활경제": 310, "경제 일반": 263}
DROP = re.compile(r"(저작권|무단\s*전재|재배포|ⓒ|©|@|https?://|▶|☞|\[사진|사진=|그래픽=|자료=|제공=|기자\s*=|특파원\s*=|구독|좋아요)")
H = lambda t: hashlib.blake2b(t.encode(), digest_size=8).digest()

def get(url):
    for k in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=15); r.raise_for_status(); time.sleep(0.6); return r.text
        except Exception:
            time.sleep(2 + 2 * k)
    return None

def article(url):
    h = get(url)
    if not h: return None
    s = BeautifulSoup(h, "html.parser")
    t, b = s.select_one("#title_area"), s.select_one("#dic_area")
    if not t or not b: return None
    for x in b.select("span.end_photo_org, em.img_desc, figure, table, script, style, .vod_player_wrap, .ab_photo, strong"): x.decompose()
    for br in b.find_all("br"): br.replace_with("\n")
    d = s.select_one("span._ARTICLE_DATE_TIME"); p = s.select_one(".media_end_head_top_logo img")
    return {"url": url, "title": t.get_text(" ", strip=True), "lines": [re.sub(r"\s+", " ", x).strip() for x in b.get_text().split("\n")],
            "date": d.get("data-date-time") if d else None, "press": p.get("alt") if p else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000); ap.add_argument("--per_article", type=int, default=20)
    ap.add_argument("--start", default="2020-11-01"); ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default="data/fin_test_crawl")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = random.Random(a.seed); t0 = time.time()
    # 제외할 문장 지문: 사전학습 말뭉치 원천(econ_all)·Claude 표본·골드셋
    excl = set()
    for p in ["data/econ_pool/econ_all.jsonl", "data/fin_silver/pool.jsonl", "data/fin_silver/claude_silver_500.jsonl",
              "data/fin_silver/claude_calib10_v1.jsonl", "data/goldset/goldset_master.jsonl"]:
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"): excl.add(H(json.loads(l)["text"].strip()))
    print(f"제외 지문 {len(excl):,}개 ({time.time()-t0:.0f}초)", flush=True)
    kiwi = Kiwi(); start = datetime.date.fromisoformat(a.start); days = (datetime.date.today() - start).days
    per_sec = a.target // len(SECTIONS); seen, rows, arts, drop = set(), [], [], collections.Counter()
    for sec, sid in SECTIONS.items():
        n_sec, tried = 0, 0
        while n_sec < per_sec and tried < 200:
            tried += 1
            day = start + datetime.timedelta(days=rng.randrange(days))
            h = get(f"https://news.naver.com/breakingnews/section/101/{sid}?date={day:%Y%m%d}")
            links = [x["href"].split("?")[0] for x in BeautifulSoup(h or "", "html.parser").select("li.sa_item a.sa_text_title")]
            if not links: drop["빈 목록"] += 1; continue
            art = article(rng.choice(links))
            if not art: drop["기사 못 읽음"] += 1; continue
            cand = [("title", art["title"])]
            for line in art["lines"]:
                if line: cand += [("body", x.text.strip()) for x in kiwi.split_into_sents(line)]
            got = []
            for kind, text in cand:
                if kind == "body" and len([g for g in got if g[0] == "body"]) >= a.per_article: break
                if len(text) < (10 if kind == "title" else 15): drop["짧음"] += 1; continue
                if DROP.search(text): drop["상용구·사진·기자"] += 1; continue
                h8 = H(text)
                if h8 in excl: drop["사전학습·표본·골드셋과 같음"] += 1; continue
                if h8 in seen: drop["중복"] += 1; continue
                seen.add(h8); got.append((kind, text))
            if len(got) < 4: drop["문장 4개 미만 기사"] += 1; continue
            oid, aid = art["url"].rstrip("/").split("/")[-2:]
            for k, (kind, text) in enumerate(got):
                rows.append({"id": f"T{oid}_{aid}_{k:02d}", "text": text, "kind": kind, "section": sec, "date": art["date"],
                             "press": art["press"], "url": art["url"], "article_title": art["title"]})
            arts.append({"section": sec, "date": art["date"], "press": art["press"], "url": art["url"], "title": art["title"], "sentences": len(got)})
            n_sec += len(got)
        print(f"{sec}: 기사 {sum(1 for x in arts if x['section']==sec)}개 · 문장 {n_sec} ({time.time()-t0:.0f}초)", flush=True)
    with open(os.path.join(a.out, "test_crawl.jsonl"), "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "articles.jsonl"), "w", encoding="utf-8") as f:
        for r in arts: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    years = collections.Counter((r["date"] or "????")[:4] for r in arts)
    st = {"sentences": len(rows), "articles": len(arts), "titles": sum(r["kind"] == "title" for r in rows),
          "by_section": dict(collections.Counter(r["section"] for r in rows)), "articles_by_year": dict(sorted(years.items())),
          "press": len({r["press"] for r in arts}), "dropped": dict(drop), "period": f"{a.start} ~ {datetime.date.today()}",
          "seed": a.seed, "sec": round(time.time() - t0)}
    json.dump(st, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(st, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
