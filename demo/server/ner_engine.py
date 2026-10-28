# -*- coding: utf-8 -*-
"""데모용 NER 엔진: 글 → 문장 나누기(Kiwi) → 모델 → 개체 위치(원문 기준)·태그·확신도.

모델은 환경 변수 NER_MODEL(폴더), 형태소 입력 여부는 NER_PRETOK("morph" 또는 빈 값)로 고른다.
태그 목록은 모델 설정(id2label)에서 읽으므로, 금융 모델로 바꿔도 이 파일은 그대로 쓴다.
확신도 = 개체를 이루는 토큰들이 그 라벨을 받을 확률(softmax)의 평균.
"""
import os, sys, time
import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, AutoModel

class FinNER(torch.nn.Module):
    """금융 파인튜닝 모델(src/fin_ft.py 와 같은 구조): 본체 + 선형층 (+CRF). 폴더에 fin_ner_config.json 이 있으면 이걸로 읽는다."""
    def __init__(self, model_dir, n, head):
        super().__init__()
        self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_dir)); self.head = head
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, n)
        if head == "crf":
            from torchcrf import CRF; self.crf = CRF(n, batch_first=True)
        from safetensors.torch import load_file
        self.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    def forward(self, input_ids, attention_mask):
        return self.classifier(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state).float()
    def crf_decode(self, em, valid):
        """실제 글자 토큰만 모아 CRF 로 경로를 고르고 원래 위치로 되돌림(학습 코드 pack 과 같음). 나머지 위치는 0(O)"""
        out = torch.zeros(em.shape[:2], dtype=torch.long)
        for b in range(em.size(0)):
            i = valid[b].nonzero().squeeze(-1)
            if len(i) == 0: continue
            path = self.crf.decode(em[b:b + 1, i])[0]
            out[b, i] = torch.tensor(path)
        return out

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))       # ner_lib.morph_split 재사용(학습 때와 같은 형태소 분리)

class NerEngine:
    def __init__(self, model_dir, pretok=None, max_len=512, device=None):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        fc = os.path.join(model_dir, "fin_ner_config.json")
        self.fin = json.load(open(fc, encoding="utf-8")) if os.path.exists(fc) else None
        if self.fin:                                                   # 금융 파인튜닝 모델(CRF/softmax)
            self.model = FinNER(model_dir, len(self.fin["labels"]), self.fin["head"]).eval()
            self.id2label = dict(enumerate(self.fin["labels"]))
            pretok = pretok or (self.fin.get("pretok") if self.fin.get("pretok") != "none" else None)
            max_len = min(max_len, self.fin.get("max_len", max_len))
        else:                                                          # 1단계 모델(일반 토큰 분류)
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tags = sorted({l.split("-", 1)[1] for l in self.id2label.values() if l != "O"})
        self.pretok, self.max_len, self.model_dir = pretok or None, max_len, model_dir
        self.postproc = os.environ.get("NER_POSTPROC", "1") != "0" and not self.pretok   # 조사 자르기(형태소 입력 모델은 해당 없음)
        self.name = os.path.basename(os.path.normpath(model_dir))
        from kiwipiepy import Kiwi
        self.kiwi = Kiwi()

    def _encode(self, text):
        """토큰화 + 원문 기준 위치. 형태소 입력이면 ner_lib 와 같은 방식으로 나눈 뒤 위치를 원문으로 되돌림."""
        if not self.pretok:
            enc = self.tok(text, truncation=True, max_length=self.max_len, return_offsets_mapping=True)
            return enc["input_ids"], self._trim(text, [tuple(o) for o in enc["offset_mapping"]])
        from ner_lib import morph_split
        new, m = morph_split(text)
        enc = self.tok(new, truncation=True, max_length=self.max_len, return_offsets_mapping=True)
        offs = []
        for s, e in enc["offset_mapping"]:
            idx = [m[i] for i in range(s, e) if m[i] >= 0] if s != e else []
            offs.append((idx[0], idx[-1] + 1) if idx else (0, 0))
        return enc["input_ids"], self._trim(text, offs)

    @staticmethod
    def _trim(text, offs):
        """토큰 위치의 앞뒤 공백 제거 (mmBERT 등은 앞 공백을 토큰 위치에 포함 → 개체 앞에 공백이 붙는 문제 방지)"""
        out = []
        for s, e in offs:
            while s < e and text[s].isspace(): s += 1
            while e > s and text[e - 1].isspace(): e -= 1
            out.append((s, e))
        return out

    @torch.no_grad()
    def _predict_sents(self, sents, batch=32):
        out = []
        for k in range(0, len(sents), batch):
            encs = [self._encode(s) for s in sents[k:k + batch]]
            L = max(len(i) for i, _ in encs)
            pad = self.tok.pad_token_id
            ids = torch.tensor([i + [pad] * (L - len(i)) for i, _ in encs], device=self.device)
            att = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i, _ in encs], device=self.device)
            if self.fin:
                em = self.model(ids, att); prob = em.softmax(-1)
                if self.fin["head"] == "crf":
                    valid = torch.tensor([[o[0] != o[1] for o in offs] + [False] * (L - len(offs)) for _, offs in encs], device=self.device)
                    lab = self.model.crf_decode(em, valid).to(self.device)
                else:
                    lab = prob.argmax(-1)
                pmax = prob.gather(-1, lab.unsqueeze(-1)).squeeze(-1)   # 고른 라벨의 확률 = 확신도
            else:
                prob = self.model(input_ids=ids, attention_mask=att).logits.float().softmax(-1)
                pmax, lab = prob.max(-1)
            for j, (i, offs) in enumerate(encs):
                sp = self._decode(lab[j, :len(i)].tolist(), pmax[j, :len(i)].tolist(), offs)
                out.append(self._josa_cut(sents[k + j], sp, offs) if self.postproc else sp)
        return out

    def _josa_cut(self, sent, spans, offs):
        """확정 후처리(조사 자르기): 개체의 마지막 토큰이 조사까지 한 조각으로 묶였으면(예 '##리에') 조사 앞에서 자른다.
        평가와 같은 규칙: 마지막 토큰 안에서 Kiwi 조사(J*)가 시작하면 그 위치까지만. 03 test 0.9258 → 0.9289."""
        toks = [(s, e) for s, e in offs if s != e]; ks = None
        for sp in spans:
            a, b = sp["start"], sp["end"]
            tok = next(((s, e) for s, e in toks if s < b <= e), None)
            if not tok or tok[1] - tok[0] < 2: continue
            if ks is None: ks = self.kiwi.tokenize(sent)
            j = [k.start for k in ks if k.tag.startswith("J") and tok[0] < k.start < b and k.start > a and k.start + k.len <= b]
            if j: sp["end"] = min(j)
        return spans

    def _decode(self, labs, probs, offs):
        """토큰별 BIO → 글자 span (ner_lib.decode_spans 와 같은 규칙: I가 앞 개체와 안 이어지면 새 개체)"""
        spans, cur = [], None
        for lid, p, (s, e) in zip(labs, probs, offs):
            if s == e: continue
            lab = self.id2label[lid]
            if lab == "O":
                if cur: spans.append(cur); cur = None
                continue
            pre, tag = lab.split("-", 1)
            if pre == "B" or cur is None or cur["tag"] != tag:
                if cur: spans.append(cur)
                cur = {"start": s, "end": e, "tag": tag, "_p": [p]}
            else:
                cur["end"] = e; cur["_p"].append(p)
        if cur: spans.append(cur)
        for sp in spans:
            ps = sp.pop("_p"); sp["score"] = round(sum(ps) / len(ps), 4)
        return spans

    def analyze(self, texts):
        """texts: [{"id": "title", "text": "..."}, ...] → 글마다 개체 목록(원문 위치) + 처리 시간"""
        t0 = time.time()
        jobs = []                                  # (글 번호, 문장 시작 위치, 문장)
        for ti, t in enumerate(texts):
            pos = 0
            for line in t["text"].split("\n"):          # 줄이 바뀌면 다른 문장(소제목 줄끼리 개체가 이어지지 않게)
                if line.strip():
                    for s in self.kiwi.split_into_sents(line):
                        if s.text.strip(): jobs.append((ti, pos + s.start, s.text))
                pos += len(line) + 1
        preds = self._predict_sents([j[2] for j in jobs])
        res = [{"id": t["id"], "text": t["text"], "entities": []} for t in texts]
        for (ti, base, sent), spans in zip(jobs, preds):
            for sp in spans:
                s, e = base + sp["start"], base + sp["end"]
                res[ti]["entities"].append({"start": s, "end": e, "tag": sp["tag"], "text": texts[ti]["text"][s:e], "score": sp["score"]})
        return {"model": self.name, "tags": self.tags, "sentences": len(jobs),
                "seconds": round(time.time() - t0, 3), "texts": res}


class EnsembleEngine:
    """여러 모델의 다수결: 같은 (시작, 끝, 태그)를 과반이 낸 개체만 남긴다. 확신도 = 낸 모델들의 확신도 평균 × 표 비율.
    평가에서 03 + 12 + 03s(dev로 고른 조합): test 0.9303 · 외부 350 0.9343 (후처리 포함)."""
    def __init__(self, engines, name):
        self.engines, self.name, self.pretok = engines, name, None
        self.tags = sorted({t for e in engines for t in e.tags})

    def analyze(self, texts):
        t0 = time.time(); outs = [e.analyze(texts) for e in self.engines]; need = len(self.engines) // 2 + 1
        res = []
        for i, t in enumerate(texts):
            votes = {}
            for o in outs:
                for e in o["texts"][i]["entities"]:
                    votes.setdefault((e["start"], e["end"], e["tag"]), []).append(e["score"])
            ents = [{"start": s, "end": e, "tag": g, "text": t["text"][s:e], "score": round(sum(v) / len(v) * len(v) / len(outs), 4)}
                    for (s, e, g), v in votes.items() if len(v) >= need]
            res.append({"id": t["id"], "text": t["text"], "entities": sorted(ents, key=lambda x: x["start"])})
        return {"model": self.name, "tags": self.tags, "sentences": outs[0]["sentences"], "seconds": round(time.time() - t0, 3), "texts": res,
                "members": [o["model"] for o in outs]}
