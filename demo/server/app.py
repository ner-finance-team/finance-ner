# -*- coding: utf-8 -*-
"""금융 NER 데모 API (FastAPI). 내 PC에서만 돈다. 웹 페이지와 크롬 확장이 같이 쓴다.

실행:  demo/run_server.bat  또는  cd demo/server && uvicorn app:app --port 8000
모델 폴더 = config.json 과 가중치(model.safetensors 또는 pytorch_model.bin)가 든 폴더.
  찾는 곳: NER_MODELS_DIR(;로 여러 개) → demo/models/ 바로 아래.
  시작 모델: NER_MODEL > 이름이 F03_ 로 시작하는 폴더(1위 모델) > 이름순 첫 번째. 실행 중에는 /api/model 로 바꾼다.
  형태소 입력 모델: 모델 폴더에 demo.json {"pretok": "morph"} 를 두거나 폴더 이름에 mmbert 가 들어가면 morph.
  Claude 비교(선택): 환경 변수 ANTHROPIC_API_KEY 가 있을 때만 켜진다. 키는 서버에만 있고 확장으로 가지 않는다.

GET  /api/health                  지금 모델 이름·태그 목록
GET  /api/models                  고를 수 있는 모델 목록 (부를 때마다 폴더를 다시 훑음)
POST /api/model    {"name":"..."}                                  → 모델 교체 (웹 페이지·확장 모두 적용)
GET  /api/labels                  태그별 표시 이름·색 (labels.json)
GET  /api/latest?section=101      네이버 뉴스 섹션 최신 기사 목록(101=경제)
POST /api/ner      {"texts":[{"id":"title","text":"..."}, ...]}   → 개체 + 리포트
POST /api/article  {"url":"https://n.news.naver.com/..."}        → 기사 가져와 제목·문단 분석 → 개체 + 리포트
POST /api/compare  {"texts":[...], "ours":[...]}                   → 같은 글을 Claude(기준서 v1.2)가 라벨링한 결과와 비교
"""
import json, os, re, time, collections, gc, threading
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ner_engine import NerEngine, EnsembleEngine
from claude_engine import ClaudeEngine, read_key, MODEL as CLAUDE_MODEL

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = [p for p in os.environ.get("NER_MODELS_DIR", "").split(";") if p] + [os.path.normpath(os.path.join(HERE, "..", "models"))]
API_NAME = "claude-api"                     # 상용 API 항목(폴더가 아님)
ENS_NAME = "ensemble-03-12-03s"             # 앙상블 항목: 폴더 이름이 F03_ · F12_ · F03s_ 로 시작하는 모델 3개의 다수결
ENS_PREFIX = ("F03_", "F12_", "F03s_")
CLAUDE_PRICE = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}   # $/100만 토큰(입력·출력)

def find_models():
    """모델 이름(폴더 이름) → 경로. NER_MODEL 을 먼저, 그다음 MODEL_DIRS 바로 아래. 이름이 겹치면 앞쪽이 이긴다."""
    found, env = {}, os.environ.get("NER_MODEL")
    if env and os.path.isfile(os.path.join(env, "config.json")): found[os.path.basename(os.path.normpath(env))] = os.path.abspath(env)
    for d in MODEL_DIRS:
        if not os.path.isdir(d): continue
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            has_w = any(os.path.isfile(os.path.join(p, w)) for w in ("model.safetensors", "pytorch_model.bin"))   # 가중치 없는 폴더는 목록에서 뺌
            if os.path.isfile(os.path.join(p, "config.json")) and has_w: found.setdefault(n, os.path.abspath(p))
    return found

def pretok_of(path):
    """형태소 입력 여부: demo.json → (NER_MODEL 이면) NER_PRETOK → 폴더 이름에 mmbert"""
    f = os.path.join(path, "demo.json")
    if os.path.isfile(f): return json.load(open(f, encoding="utf-8")).get("pretok") or None
    env = os.environ.get("NER_MODEL")
    if env and os.path.abspath(env) == path and os.environ.get("NER_PRETOK"): return os.environ["NER_PRETOK"]
    return "morph" if "mmbert" in os.path.basename(path).lower() else None

MODELS = find_models()
if not MODELS: raise SystemExit(f"모델 폴더가 없습니다. 다음 중 한 곳 아래에 모델 폴더(config.json + 가중치)를 두세요: {MODEL_DIRS}")
def ens_members():
    """앙상블에 쓸 모델 3개(F03_·F12_·F03s_ 로 시작하는 폴더). 하나라도 없으면 None"""
    pick = [next((n for n in MODELS if n.startswith(p)), None) for p in ENS_PREFIX]
    return pick if all(pick) else None
env = os.environ.get("NER_MODEL")
_first = next((n for n in MODELS if n.startswith("F03_")), None)   # 금융 1위 모델이 있으면 그걸로 시작
CURRENT = os.path.basename(os.path.normpath(env)) if env and os.path.basename(os.path.normpath(env)) in MODELS else _first or next(iter(MODELS))
engine = NerEngine(MODELS[CURRENT], pretok=pretok_of(MODELS[CURRENT]))
LOCK = threading.Lock()                    # 모델을 바꾸는 동안 다른 교체 요청은 기다림
LABELS = {k: v for k, v in json.load(open(os.path.join(HERE, "labels.json"), encoding="utf-8")).items() if not k.startswith("_")}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}

app = FastAPI(title="금융 NER 데모")
# 크롬 확장(네이버 기사 페이지에서 실행)과 로컬 웹 페이지가 부를 수 있게
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Text(BaseModel):
    id: str
    text: str
class NerReq(BaseModel):
    texts: list[Text]
class ArticleReq(BaseModel):
    url: str
class ModelReq(BaseModel):
    name: str
class CompareReq(BaseModel):
    texts: list[Text]
    ours: list[dict]          # 지금 화면에 칠한 우리 모델 결과: [{id, entities:[{start,end,tag,text,score}]}]
    force: bool = False       # True 면 저장된 Claude 결과를 쓰지 않고 새로 호출

def report(result):
    """태그별로 같은 개체를 모아 횟수·첫 위치·평균 확신도를 정리"""
    by = collections.defaultdict(dict)
    for t in result["texts"]:
        for e in t["entities"]:
            d = by[e["tag"]].setdefault(e["text"], {"text": e["text"], "count": 0, "first": {"id": t["id"], "start": e["start"]}, "_s": []})
            d["count"] += 1; d["_s"].append(e["score"])
    out = []
    for tag in sorted(by, key=lambda k: -sum(x["count"] for x in by[k].values())):
        items = sorted(by[tag].values(), key=lambda x: (-x["count"], x["first"]["id"] != "title", x["first"]["start"]))
        for x in items: s = x.pop("_s"); x["score"] = round(sum(s) / len(s), 4)
        out.append({"tag": tag, "total": sum(x["count"] for x in items), "unique": len(items), "items": items})
    return {"entities": sum(len(t["entities"]) for t in result["texts"]), "by_tag": out}

def drop_overlaps(ents):
    """겹치는 개체는 긴 쪽만 남긴다(형태소 입력 모델이 문장 첫 글자에 1글자 개체를 겹쳐 내는 경우 등)"""
    keep = []
    for e in sorted(ents, key=lambda e: (e["start"] - e["end"], e["start"])):
        if all(e["end"] <= k["start"] or k["end"] <= e["start"] for k in keep): keep.append(e)
    return sorted(keep, key=lambda e: e["start"])

def run(texts):
    try: r = engine.analyze(texts)
    except RuntimeError as e: raise HTTPException(502, str(e))   # 상용 API 오류(키·속도 제한·연결)를 화면에 그대로
    for t in r["texts"]: t["entities"] = drop_overlaps(t["entities"])
    r["report"] = report(r)
    return r

def fetch_article(url):
    """네이버 기사에서 제목·문단만 꺼낸다(사진·사진 설명·표·스크립트 제외, <br> 로 문단 나눔)"""
    if not re.match(r"https?://(n\.)?news\.naver\.com/", url) and "naver.com" not in url:
        raise HTTPException(400, "네이버 뉴스 기사 주소만 받습니다")
    h = requests.get(url, headers=UA, timeout=10); h.raise_for_status()
    s = BeautifulSoup(h.text, "html.parser")
    title, body = s.select_one("#title_area"), s.select_one("#dic_area")
    if not title or not body: raise HTTPException(422, "제목·본문을 찾지 못함(기사 페이지가 아니거나 구조가 바뀜)")
    for x in body.select("span.end_photo_org, em.img_desc, figure, table, script, style, .vod_player_wrap, .ab_photo"): x.decompose()
    for br in body.find_all("br"): br.replace_with("\n")
    paras = [re.sub(r"\s+", " ", p).strip() for p in body.get_text().split("\n")]
    paras = [p for p in paras if p]
    def attr(sel, a):
        x = s.select_one(sel); return x.get(a) if x else None
    meta = {"url": url,
            "press": attr(".media_end_head_top_logo img", "alt") or attr('meta[name="twitter:creator"]', "content"),
            "date": attr("span._ARTICLE_DATE_TIME", "data-date-time"),            # 입력 시각 (한국 시간)
            "modified": attr("span._ARTICLE_MODIFY_DATE_TIME", "data-modify-date-time"),
            "fetched": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}  # 우리가 가져온 시각
    return title.get_text(strip=True), paras, meta

def load_engine(name):
    if name == API_NAME: return ClaudeEngine(read_key())
    if name == ENS_NAME:
        ens = ens_members()
        if not ens: raise RuntimeError("앙상블 모델 3개(F03_ · F12_ · F03s_ 폴더)가 다 있어야 합니다")
        return EnsembleEngine([NerEngine(MODELS[n], pretok=pretok_of(MODELS[n])) for n in ens], ENS_NAME)
    return NerEngine(MODELS[name], pretok=pretok_of(MODELS[name]))

@app.get("/api/health")
def health():
    return {"ok": True, "model": CURRENT, "pretok": engine.pretok, "tags": engine.tags}

@app.get("/api/models")
def models():
    global MODELS
    MODELS = find_models()
    ens = ens_members()
    return {"current": CURRENT, "models": [{"name": n, "kind": "local", "path": p, "pretok": pretok_of(p)} for n, p in MODELS.items()]
            + ([{"name": ENS_NAME, "kind": "ensemble", "members": ens}] if ens else [])
            + [{"name": API_NAME, "kind": "api", "model": CLAUDE_MODEL, "ready": bool(read_key())}]}

@app.post("/api/model")
def set_model(req: ModelReq):
    global engine, CURRENT
    if req.name == API_NAME:
        if not read_key(): raise HTTPException(400, "API 키가 없습니다. 환경 변수 ANTHROPIC_API_KEY 를 넣고 서버를 다시 켜세요")
    elif req.name != ENS_NAME:
        if req.name not in MODELS: MODELS.update(find_models())
        if req.name not in MODELS: raise HTTPException(404, f"모델 폴더를 찾지 못함: {req.name}")
    with LOCK:                                # 실패하면 지금 모델을 그대로 쓴다
        if getattr(engine, "name", None) != req.name:
            try: new = load_engine(req.name)
            except Exception as e: raise HTTPException(500, f"모델을 읽지 못함({req.name}): {e}")
            engine = new; gc.collect()        # 노트북 메모리를 아끼려고 한 번에 한 모델만 둔다
        CURRENT = req.name
    return health()

_claude = None
def diff_entities(ours, theirs):
    """우리 모델(ours) vs Claude(theirs) 개체 비교 → 같음 / 태그만 다름 / 범위 다름 / 우리만 / Claude만"""
    O = {(e["start"], e["end"]): e for e in ours}; C = {(e["start"], e["end"]): e for e in theirs}
    same, tagd, bound, only_o, only_c, used_c = [], [], [], [], [], set()
    for k, e in O.items():
        if k in C:
            used_c.add(k)
            (same if C[k]["tag"] == e["tag"] else tagd).append({"text": e["text"], "ours": e["tag"], "claude": C[k]["tag"], "start": e["start"]})
            continue
        ov = [c for kc, c in C.items() if kc[0] < k[1] and k[0] < kc[1] and kc not in O]
        if ov:
            for c in ov: used_c.add((c["start"], c["end"]))
            bound.append({"ours_text": e["text"], "ours": e["tag"], "claude_text": " / ".join(c["text"] for c in ov),
                          "claude": " / ".join(c["tag"] for c in ov), "start": e["start"]})
        else: only_o.append({"text": e["text"], "ours": e["tag"], "start": e["start"]})
    for kc, c in C.items():
        if kc not in used_c and not any(kc[0] < k[1] and k[0] < kc[1] for k in O): only_c.append({"text": c["text"], "claude": c["tag"], "start": c["start"]})
    return same, tagd, bound, only_o, only_c

@app.post("/api/compare")
def compare(req: CompareReq):
    """지금 화면의 우리 모델 결과를 Claude(기준서 v1.2) 라벨과 비교 → 일치 점수 + 다르게 단 개체 목록"""
    global _claude
    if not read_key(): raise HTTPException(400, "Claude 비교에는 API 키가 필요합니다. 환경 변수 ANTHROPIC_API_KEY 를 넣고 서버를 다시 켜세요")
    texts = [t.model_dump() for t in req.texts]
    if _claude is None:
        try: _claude = ClaudeEngine(read_key())
        except Exception as e: raise HTTPException(500, f"Claude 준비 실패: {e}")
    try: c = _claude.analyze(texts, force=req.force)       # 같은 기사면 저장된 결과(재호출·요금 없음)
    except RuntimeError as e: raise HTTPException(502, str(e))
    c = json.loads(json.dumps(c))                           # 저장본을 건드리지 않게 복사
    for t in c["texts"]: t["entities"] = drop_overlaps(t["entities"])
    ours = {t["id"]: t["entities"] for t in req.ours}
    out = {"same": [], "tag": [], "boundary": [], "only_ours": [], "only_claude": []}; n_o = n_c = 0
    for t in c["texts"]:
        o = ours.get(t["id"], []); n_o += len(o); n_c += len(t["entities"])
        for key, items in zip(out, diff_entities(o, t["entities"])):
            out[key] += [dict(i, id=t["id"]) for i in items]
    m = len(out["same"]); p = m / n_o if n_o else 0; r = m / n_c if n_c else 0
    u = c.get("usage", {}); pin, pout = CLAUDE_PRICE.get(CLAUDE_MODEL, (0, 0))
    cost = ((u.get("input_tokens") or 0) * pin + (u.get("output_tokens") or 0) * pout + (u.get("cache_read_input_tokens") or 0) * pin * 0.1) / 1e6
    return {"model": CURRENT, "claude": c["model"], "n_ours": n_o, "n_claude": n_c, "match": m,
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "precision": round(p, 4), "recall": round(r, 4),
            "seconds": c["seconds"], "usage": u, "cost_usd_est": round(cost, 4), "cached": bool(c.get("cached")), "saved_at": c.get("saved_at"), **out}

@app.get("/api/labels")
def labels():
    return {t: LABELS.get(t, {"name": t, "bg": "#e5e7eb", "fg": "#374151"}) for t in engine.tags}

@app.get("/api/latest")
def latest(section: int = 101, n: int = 20):
    h = requests.get(f"https://news.naver.com/section/{section}", headers=UA, timeout=10); h.raise_for_status()
    # 섹션 위쪽 헤드라인(시간 없음)은 빼고, 아래 최신순 목록(언론사·"n분전" 있음)만 쓴다
    s = BeautifulSoup(h.text, "html.parser"); seen, out = set(), []
    for li in s.select("li.sa_item"):
        if "_SECTION_HEADLINE" in (li.get("class") or []): continue
        a, ago = li.select_one("a.sa_text_title"), li.select_one(".sa_text_datetime")
        if not a: continue
        u = a["href"].split("?")[0]
        if u in seen: continue
        seen.add(u)
        press = li.select_one(".sa_text_press")
        out.append({"url": u, "title": a.get_text(" ", strip=True), "press": press.get_text(strip=True) if press else None,
                    "ago": ago.get_text(strip=True) if ago else None})
        if len(out) >= n: break
    return {"fetched": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "items": out}

@app.post("/api/ner")
def ner(req: NerReq):
    return run([t.model_dump() for t in req.texts])

@app.post("/api/article")
def article(req: ArticleReq):
    title, paras, meta = fetch_article(req.url)
    r = run([{"id": "title", "text": title}] + [{"id": f"p{i}", "text": p} for i, p in enumerate(paras)])
    r["article"] = meta
    return r

web = os.path.join(HERE, "..", "web")
if os.path.isdir(web): app.mount("/", StaticFiles(directory=web, html=True), name="web")
