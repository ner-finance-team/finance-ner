# -*- coding: utf-8 -*-
"""상용 API(Anthropic Claude)로 금융 8태그 개체 찾기. NerEngine.analyze() 와 같은 형식으로 돌려주므로 화면은 그대로 쓴다.

기준: server/fin_guide/GUIDE.md (금융 NER 라벨 기준서, 1~4장) + examples.json (정답 예시). 금융 학습 데이터도 Claude가 이 기준서로 라벨했다.
      기준서가 바뀌면 이 두 파일만 바꿔 끼운다(코드 수정 불필요, 서버 재시작 후 반영).
키: 환경 변수 ANTHROPIC_API_KEY 에서만 읽는다(파일로 저장하지 않음). 키는 서버에만 있고 크롬 확장으로 가지 않는다.
모델: CLAUDE_MODEL (기본 claude-opus-5-5) · 생각 깊이: CLAUDE_EFFORT (기본 low = 빠르게, medium/high 는 느리지만 꼼꼼)
방식: 제목·본문을 한 번에 보내 (글 id, 개체 글자, 태그, 확신도) 목록을 JSON으로 받고, 원문에서 글자를 찾아 위치를 붙인다.
      기준서·예시는 매번 같으므로 프롬프트 캐시를 건다(반복 호출 요금 절약).
확신도 = 모델이 스스로 적은 값(0~1). 학습 모델의 확률과는 성격이 다르다.
거절(refusal) 대비로 서버 측 대체 모델(fallbacks="default")을 켜 둔다.
"""
import json, os, re, time
import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))
GUIDE_FILE = os.path.join(HERE, "fin_guide", "GUIDE.md")
EXAMPLES_FILE = os.path.join(HERE, "fin_guide", "examples.json")
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("CLAUDE_EFFORT", "low")
TAGS = ["DATE", "FIN_ORG", "FIN_PRODUCT", "INDICATOR", "MONEY", "ORG", "POLICY", "RATE"]   # 기준서 1장의 태그 8개

def build_system():
    """기준서 1~4장(5장 '출력 형식'부터는 도구용이라 뺌) + 정답 예시 → 지시문"""
    guide = re.split(r"\n## 5\.", open(GUIDE_FILE, encoding="utf-8").read())[0]
    ex = json.load(open(EXAMPLES_FILE, encoding="utf-8"))["examples"]
    shots = "\n".join(json.dumps({"text": e["text"], "entities": [{"text": e["text"][s:t], "tag": g} for s, t, g in e["spans"]]}, ensure_ascii=False) for e in ex)
    head = ("너는 한국어 뉴스 기사에서 금융 개체명을 찾는 도구다. 아래 기준서를 그대로 따른다.\n"
            "입력은 [{id, text}] JSON 이다. 찾은 개체마다 id, text, tag, confidence 를 적는다.\n"
            f"- tag 는 코드(FO, O …)가 아니라 태그 이름({', '.join(TAGS)})으로 적는다.\n"
            "- text 는 원문 글자 그대로, 원문에 나오는 순서대로 적는다. 같은 글자가 여러 번 나오면 나올 때마다 적는다.\n"
            "- confidence 는 그 태그가 맞을 가능성(0~1).\n\n")
    return [{"type": "text", "text": head + guide},
            {"type": "text", "text": "## 정답 예시 (한 줄에 한 문장)\n" + shots, "cache_control": {"type": "ephemeral"}}]

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["entities"], "properties": {"entities": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "text", "tag", "confidence"], "properties": {
        "id": {"type": "string"}, "text": {"type": "string"}, "tag": {"type": "string", "enum": TAGS}, "confidence": {"type": "number"}}}}}}

def read_key():
    return os.environ.get("ANTHROPIC_API_KEY", "").strip() or None

def locate(texts, items):
    """모델이 적은 개체 글자를 원문에서 찾아 위치를 붙인다. 원문에 없는 글자는 버리고, 겹치면 앞쪽·긴 것을 남긴다."""
    res = [{"id": t["id"], "text": t["text"], "entities": []} for t in texts]
    idx, cur = {t["id"]: i for i, t in enumerate(texts)}, {}   # cur: (글 번호, 개체 글자) → 다음에 찾기 시작할 위치
    for it in items:
        i, s = idx.get(it["id"]), it["text"].strip()
        if i is None or not s: continue
        src, got = texts[i]["text"], res[i]["entities"]
        p = src.find(s, cur.get((i, s), 0))
        while p >= 0 and any(p < e["end"] and e["start"] < p + len(s) for e in got):   # 이미 찾은 개체 자리면 다음 자리
            p = src.find(s, p + 1)
        if p < 0: continue
        cur[(i, s)] = p + len(s)
        got.append({"start": p, "end": p + len(s), "tag": it["tag"], "text": s, "score": round(min(max(float(it["confidence"]), 0.0), 1.0), 4)})
    for t in res:
        keep, end = [], -1
        for e in sorted(t["entities"], key=lambda e: (e["start"], e["start"] - e["end"])):
            if e["start"] >= end: keep.append(e); end = e["end"]
        t["entities"] = keep
    return res

# ---------- Claude 결과 저장(캐시): 같은 기사는 다시 호출하지 않는다 ----------
import hashlib
CACHE_DIR = os.path.join(HERE, "claude_cache")          # 서버를 다시 켜도 남는다. 지워도 됨(다음에 새로 호출)
_MEM = {}
def _guide_sig():
    h = hashlib.sha1()
    for f in (GUIDE_FILE, EXAMPLES_FILE):
        if os.path.isfile(f): h.update(open(f, "rb").read())
    return h.hexdigest()[:12]
def cache_key(texts):
    raw = json.dumps({"m": MODEL, "e": EFFORT, "g": _guide_sig(), "t": [[t["id"], t["text"]] for t in texts]}, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()
def cache_get(key):
    if key in _MEM: return _MEM[key]
    p = os.path.join(CACHE_DIR, key + ".json")
    if os.path.isfile(p):
        try: _MEM[key] = json.load(open(p, encoding="utf-8")); return _MEM[key]
        except Exception: return None
    return None
def cache_put(key, res):
    _MEM[key] = res; os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(dict(res, saved_at=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(CACHE_DIR, key + ".json"), "w", encoding="utf-8"), ensure_ascii=False)

class ClaudeEngine:
    def __init__(self, key):
        self.client = anthropic.Anthropic(api_key=key)
        self.system = build_system()            # 기준서·예시 파일이 없으면 여기서 에러(모델 교체 실패로 보임)
        self.tags, self.pretok, self.name = TAGS, None, f"Claude API ({MODEL}, 금융 8태그)"

    def analyze(self, texts, force=False):
        """같은 글(제목·본문)·같은 Claude 모델·생각 깊이·기준서면 저장된 결과를 쓴다(API 재호출·요금 없음). force=True 면 새로 호출."""
        key = cache_key(texts)
        if not force:
            hit = cache_get(key)
            if hit: return dict(hit, cached=True)
        res = self._call(texts)
        cache_put(key, res)
        return dict(res, cached=False)

    def _call(self, texts):
        t0 = time.time()
        try:
            r = self.client.beta.messages.create(
                model=MODEL, max_tokens=16000, system=self.system,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": json.dumps([{"id": t["id"], "text": t["text"]} for t in texts], ensure_ascii=False)}])
        except anthropic.AuthenticationError: raise RuntimeError("API 키가 올바르지 않습니다")
        except anthropic.PermissionDeniedError: raise RuntimeError("이 API 키로는 이 모델을 쓸 수 없습니다")
        except anthropic.RateLimitError: raise RuntimeError("요청이 너무 많습니다(속도 제한). 잠시 뒤 다시 하세요")
        except anthropic.APIStatusError as e: raise RuntimeError(f"API 오류 {e.status_code}: {e.message}")
        except anthropic.APIConnectionError: raise RuntimeError("Anthropic 서버에 연결하지 못했습니다(인터넷 확인)")
        if r.stop_reason == "refusal": raise RuntimeError("모델이 이 글의 처리를 거절했습니다")
        if r.stop_reason == "max_tokens": raise RuntimeError("응답이 너무 길어 잘렸습니다")
        items = json.loads(next(b.text for b in r.content if b.type == "text"))["entities"]
        return {"model": self.name, "tags": self.tags, "sentences": sum(1 for t in texts for l in t["text"].split("\n") if l.strip()),
                "seconds": round(time.time() - t0, 3), "texts": locate(texts, items),
                "usage": {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens,
                          "cache_read_input_tokens": r.usage.cache_read_input_tokens}}
