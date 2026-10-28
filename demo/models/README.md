# 모델 폴더

학습된 모델 가중치는 용량(모델당 약 0.4~1.3GB) 때문에 저장소에 넣지 않았습니다. 팀 공유 드라이브에서 받아 이 폴더 아래에 **폴더째** 두세요.

```
demo/models/
├─ F03_dapt_roberta-large_crf/                       ← 1위 모델 (서버가 이걸로 시작)
│   ├─ config.json
│   ├─ model.safetensors                              ← 가중치 (없으면 목록에 안 보임)
│   ├─ fin_ner_config.json                            ← 태그·헤드(CRF/softmax) 정보
│   └─ tokenizer.json, tokenizer_config.json …
├─ F12_monologg-koelectra-base-v3-discriminator_crf/  ← 앙상블 구성원
└─ F03s_dapt_roberta-large_softmax/                   ← 앙상블 구성원
```

- 폴더 이름 앞부분이 `F03_`, `F12_`, `F03s_` 인 세 모델이 모두 있으면 목록에 **앙상블(ensemble-03-12-03s)** 이 생깁니다.
- 다른 곳에 둔 모델을 쓰려면: `set NER_MODELS_DIR=D:\models;E:\more` (`;`로 여러 곳).
- 형태소 입력 모델(mmBERT 계열)은 폴더 이름에 `mmbert` 가 있거나 폴더 안에 `demo.json` = `{"pretok": "morph"}` 가 있으면 형태소로 나눠 넣습니다. 서브워드 입력 mmBERT(RUN 13)는 `{"pretok": ""}` 로 끕니다.
