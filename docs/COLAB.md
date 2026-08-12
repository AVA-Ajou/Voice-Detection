# Colab에서 학습 돌리기 — 처음 쓰는 사람용

Colab은 구글이 빌려주는 원격 컴퓨터다. 브라우저에서 파이썬을 실행하는데, **GPU가 달려 있다.**
맥에는 없는 물건이라 여기서 학습을 돌린다. 무료 등급은 T4(16GB)를 준다.

무료 등급의 제약을 먼저 알아둘 것.

```
  90분 아무것도 안 하면      → 세션 끊김
  최대 12시간               → 강제 종료
  세션이 끊기면             → 파일이 전부 사라짐   ← 이게 제일 중요
```

그래서 **결과물을 구글 드라이브에 저장**하도록 아래 순서를 짰다. 끊겨도 어댑터는 남는다.

---

## 0. 준비 — 한 번만 하면 되는 것

### Gemma 사용 승인 — 이제 필요 없다

예전 베이스(`google/gemma-3-4b-it`)는 승인이 필요한 저장소여서 허깅페이스 가입·라이선스 동의·
토큰 발급을 먼저 해야 했다. 지금 쓰는 **`google/gemma-4-E2B-it`은 게이트되지 않는다** —
토큰 없이 그냥 받아진다. 이 절은 통째로 건너뛰어도 된다.

> 승인이 필요한 모델로 바꾸게 되면 https://huggingface.co/settings/tokens 에서 `Read` 토큰을
> 만들어 Colab 비밀 변수(`HF_TOKEN`)에 넣는 절차가 다시 필요해진다.

### 파일을 구글 드라이브에 올리기

Colab은 내 맥의 파일을 못 본다. 드라이브를 거쳐야 한다.

내 드라이브에 `Voice-Detection` 폴더를 만들고 아래를 올린다.

```
Voice-Detection/
├── common.py
├── train_lora.py
├── evaluate.py
├── infer.py
└── finetune/
    ├── binary_train.jsonl
    └── binary_val.jsonl
```

> `binary_*.jsonl` 이 없으면 맥에서 먼저 `python3 build_binary_set.py` 를 돌린다.
> 전사본 원본(202MB)은 올릴 필요 없다. 학습셋만 있으면 된다.

---

## 1. 노트북 만들고 GPU 켜기

1. https://colab.research.google.com → **새 노트북**
2. 메뉴 **런타임 → 런타임 유형 변경**
3. 하드웨어 가속기를 **T4 GPU** 로 선택 → 저장

> ⚠️ 이걸 안 하면 CPU로 돌아간다. 학습이 며칠 걸린다. 반드시 확인할 것.

아래 셀들을 하나씩 붙여넣고 `Shift+Enter` 로 실행한다.

---

## 2. GPU가 잡혔는지 확인

```python
!nvidia-smi
```

`Tesla T4` 와 `15360MiB` 비슷한 게 보이면 성공. 아무것도 안 나오면 1번을 다시 한다.

---

## 3. 라이브러리 설치

```python
!pip install -q -U "transformers>=4.50" peft bitsandbytes accelerate
```

2~3분 걸린다. 빨간 경고가 몇 줄 떠도 무시해도 된다.

---

## 4. 드라이브 연결

```python
from google.colab import drive
drive.mount('/content/drive')
```

실행하면 팝업이 뜬다. 구글 계정 선택 → 권한 허용. `Mounted at /content/drive` 가 나오면 됐다.

---

## 5. 허깅페이스 로그인 — 건너뛴다

지금 베이스(`google/gemma-4-E2B-it`)는 게이트되지 않아 로그인 없이 받아진다.
승인이 필요한 모델로 바꿨을 때만 이걸 먼저 실행한다.

```python
from huggingface_hub import login
login()   # 입력창에 hf_xxxxx 토큰. 화면에 안 보이는 게 정상이다
```

---

## 6. 작업 폴더로 이동

```python
%cd /content/drive/MyDrive/Voice-Detection
!ls
```

올려둔 파일들이 보이면 준비 끝이다.

---

## 7. 학습

```python
!python train_lora.py --out /content/drive/MyDrive/Voice-Detection/adapter-gemma4
```

`--out` 을 드라이브 경로로 주는 게 핵심이다. **세션이 끊겨도 어댑터가 살아남는다.**

처음엔 모델을 받느라 10분쯤 조용하다(9.6GB). 그 다음 이런 게 흐른다.

```
정답 토큰  '예' → 238643   '아니오' → 237534
trainable params: 24,158,208 || all params: 5,1... || trainable%: 0.47
{'loss': 0.5073, 'grad_norm': ..., 'epoch': 0.13}
{'loss': 0.3367, ...}
```

**`loss` 가 내려가면 학습이 되고 있는 것이다.** 0.69 근처에서 시작하는데, 이건 찍기(50:50)와
같은 값이다. 0.2~0.3까지 떨어지면 잘 배운 것이고, 0.69에서 안 움직이면 뭔가 잘못된 것이다.

총 1시간 20분쯤. 이 탭을 닫지 말고, 가끔 화면을 건드려 유휴 종료를 피할 것.

### 메모리가 터지면

`CUDA out of memory` 가 나오면 전사본 길이를 줄인다. `common.py` 의

```python
MAX_TEXT_TOKENS = 1024
```

를 `512` 로 바꾸고 다시 돌린다. 정확도는 조금 떨어지지만 확실히 돌아간다.

---

## 8. 평가 — 이번 작업의 진짜 결과

```python
!python evaluate.py --adapter /content/drive/MyDrive/Voice-Detection/adapter-gemma4 --compare-baseline
```

세 가지를 본다.

**① 온도** — 모델이 얼마나 과확신했는지

```
온도 T = 1.412   NLL 0.4821 → 0.3104
  T > 1 — 모델이 과확신하고 있었고, 확률을 눌러 보정했다.
```

**② 신뢰도** — "87점"이 정말 87%인지

```
    모델이 말한 확률     건수    평균 예측    실제 피싱 비율      차이
   0.8 ~  1.0        68       0.91          0.88      +0.03  ✓
   0.6 ~  0.8        21       0.70          0.67      +0.03  ✓
   ...
ECE 0.041   (0.05 미만이면 보정 성공)
```

`✓` 가 줄줄이 뜨고 ECE가 0.05 아래면 **점수를 그대로 사용자에게 보여줘도 된다는 뜻**이다.

**③ 정규식이 놓친 피싱을 건졌는가** — 이게 제일 중요하다

```
            id     옛 방식    새 방식
    vishing_397        9     91.2  ✓
    vishing_644       10     87.4  ✓
     vishing_38       30     94.1  ✓

  14건 중 12건이 70점 이상 — 86%
```

여기 비율이 높아야 파인튜닝을 한 의미가 있다. 낮으면 모델이 정규식을 흉내낸 것뿐이다.

---

## 9. 실제로 한 건 돌려보기

```python
!python infer.py --adapter /content/drive/MyDrive/Voice-Detection/adapter-gemma4 --id vishing_38
```

```
위험도  94.1   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░

근거
"지금 저희 만나실 직원 이름이 김주혁 대리입니다"라며 방문 수거를 예고하고,
"수고비 경비 5만 원 뺀 부분도 얘기를 해야 되고"라며 현금 전달을 전제로 말하고 있습니다.
"전화 저 좀 바꿔주세요"라고 통화 통제를 시도하는 점도 전형적인 대면편취 수법입니다.
```

위험도는 어댑터를 켠 채 확률을 읽은 값이고, 근거는 어댑터를 끈 원본 Gemma가 쓴 문장이다.
자세한 원리는 [METHOD.md](METHOD.md) 5절.

---

## 10. 결과 가져오기

학습이 끝나면 드라이브의 `Voice-Detection/adapter-gemma4/` 안에 이것들이 생긴다.

```
adapter_model.safetensors     92MB    이게 학습 결과물 전부다
adapter_config.json
prompt.json                           프롬프트 계약. 서버가 이것만 보고 재구성한다
calibration.json                      온도. infer.py 와 서버가 읽는다
```

맥으로 내려받아 저장소에 넣으면 된다. `checkpoint-*` 폴더는 재개용이라 빼도 된다.
베이스 모델(9.6GB)은 받을 필요 없다 — 서버가 허깅페이스에서 직접 받아 쓴다.

---

## 자주 막히는 곳

| 증상 | 원인 | 해결 |
|---|---|---|
| `401 Client Error` / `Gated repo` | 승인이 필요한 모델로 바꿨다 | 5번에서 `login()` 실행 |
| `CUDA out of memory` 가 모델 적재 중에 | `prepare_model_for_kbit_training` 을 되살렸다 | 그 호출을 다시 빼라 — 임베딩 fp32 승격만으로 8.75GB를 먹는다 |
| `CUDA out of memory` | 전사본이 김 | `MAX_TEXT_TOKENS` 를 512로 |
| 학습이 너무 느림 | GPU 미할당 | 1번 — 런타임 유형 확인 |
| `loss` 가 0.69에서 안 내려감 | 학습률·데이터 문제 | 정답 토큰 id가 서로 다른지 로그 확인 |
| 세션이 끊김 | 90분 유휴 | `--out` 이 드라이브인지 확인 후 8번부터 다시 |
| `No module named 'common'` | 폴더 이동 안 됨 | 6번 `%cd` 다시 실행 |
