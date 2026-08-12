# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Voice-Detection** — 통화 전사본으로 보이스피싱을 판정하는 모델의 **데이터·학습·평가**.

산출물은 LoRA 어댑터 하나(92MB)다. 추론은 이 저장소가 하지 않는다 — `../Detection-Server`가
어댑터를 받아 서빙하고, 안드로이드 앱(`../demo`)이 그 서버를 부른다. 학습은 가끔 돌리는
오프라인 배치이고 서버는 항상 떠 있어야 해서 주기가 다르므로 분리했다.

```
입력   통화 전사본 (한국어, 화자 분리 포함)
출력   P("예") = 0.93  →  위험도 93
```

베이스는 **Gemma 4 E2B**다. Gemma 3 4B에서 옮겨왔고 전 항목에서 같거나 나은데
(`docs/RESULTS.md`), 옮긴 진짜 이유는 성능이 아니라 **오디오**다. Gemma 4는 음성을 직접
받으므로 서버가 전사(STT)까지 같은 모델로 처리한다 — 외부 STT API를 버릴 수 있었다.
**이 저장소는 전사를 학습시키지 않는다.** 어댑터를 끈 원본 모델이 이미 하는 일이다.

**위험도를 우리가 계산하지 않는다.** 원본 라벨은 `0/1` 이진뿐이라 "87점" 같은 정답이 없다.
그래서 이진으로만 학습하고, 추론할 때 모델이 정답 토큰 자리에 매긴 확률을 읽어 쓴다.
이 결정이 이 저장소의 전부다 — 자세한 근거는 `docs/METHOD.md`.

## Critical Rules

- **위험도를 지어내지 말 것.** 학습 정답은 `0/1`이다. 예전 방식(`build_finetune_set.py`)은
  정규식 신호의 무게 합을 시그모이드로 눌러 위험도를 만들었고, 그 결과 피싱 586건 중 194건이
  40점 미만으로 들어갔다 — "피싱인데 안전하다"를 194번 가르치는 학습셋이었다.
- **프롬프트는 반드시 `common.py`를 통해서만 만들 것.** 학습과 추론에서 한 글자라도 다르면
  **에러 없이 확률만 조용히 틀어진다.** 검증할 방법이 없는 종류의 버그다.
- **어댑터를 저장할 때 `prompt.json`을 함께 남길 것** (`common.save_contract`). 추론 서버는
  별도 저장소라 이 코드를 볼 수 없다. 계약을 코드가 아니라 파일로 고정한다.
- **`--prefix` 실행은 `calibration.json`을 덮어쓰지 않는다.** 잘린 전사본으로 맞춘 온도가
  운영에 들어가면 안 된다. 이 가드를 제거하지 말 것.
- **데이터를 커밋하지 말 것.** 전사본은 금융감독원 공개 녹취와 AI Hub 음성에서 파생된 것이라
  재배포가 허용되지 않는다. `.gitignore`가 막고 있고, 받는 방법(`fetch.py`)만 커밋한다.
- **`build_finetune_set.py`를 지우지 말 것.** 정규식 기반 옛 방식이고, `evaluate.py
  --compare-baseline`이 이 결과와 대조해 "정규식이 놓친 피싱을 모델이 건졌는가"를 잰다.
  이 저장소가 파인튜닝을 한 이유를 증명하는 유일한 지표다.
- **`calibrate.py`의 배점을 실측값으로 되돌리지 말 것.** `FINANCIAL_INSTITUTION`(실측 4.37) 등
  세 항목은 의도적으로 1.0으로 눌러놨다 — 정상 표본에 금융 상담이 0건이라 변별력이 부풀려져
  있다. 실측값을 그대로 쓰면 문자에 "은행"만 나와도 위험도가 폭등한다.
- **모델에게 숫자를 묻지 말 것.** 확률이든 점수든 단계든, 생성으로 뽑은 숫자는 학습으로
  점검된 적이 없다. 진행 단계를 생성으로 물어봤다가 정답지 36건에서 33.3%가 나왔고
  규칙으로 바꿔 91.7%가 됐다(`docs/RESULTS.md`). 숫자는 **로짓에서 읽거나 규칙으로 계산**한다.
- **LoRA는 언어 모델 층에만 붙일 것.** Gemma 4는 오디오·비전 인코더를 함께 들고 있어
  `q_proj` 같은 이름이 인코더 안에도 있다. `common.lora_targets()`가 **전체 경로**를
  돌려주는 이유가 이것이다 — 잎 이름만 넘기면 PEFT가 접미사로 맞춰 인코더까지 학습한다.
  실제로 한 번 그랬고, 학습 파라미터가 24.2M이어야 할 자리에 32.8M이 찍혀 있었다.
- **`prepare_model_for_kbit_training()`을 쓰지 말 것.** 임베딩을 fp32로 올리는데 Gemma 4의
  큰 어휘 임베딩에서는 그것만으로 8.75GB를 요구해 T4에서 터진다. `train_lora.py`가 필요한
  것(그래디언트 체크포인팅 + 입력 그래디언트)만 손으로 켠다.
- **주석은 "왜"만 쓴다.** 코드를 읽으면 아는 "무엇"을 반복하지 말 것.

## Architecture

```markdown
Voice-Detection/
├── fetch.py                    # 전사본 sparse clone + 앱 시뮬레이션용 표본 생성
├── common.py                   # 프롬프트·모델 적재·계약. 학습/평가/추론이 공유
│                               #   python3 common.py --write-contract adapter/
├── build_binary_set.py         # 이진 학습셋 (층화 분할). 위험도를 만들지 않는다
├── train_lora.py               # Gemma + QLoRA. 손실은 정답 토큰 한 자리에만
├── evaluate.py                 # 온도 보정 · AUROC · 신뢰도 곡선 · 베이스라인 비교
├── infer.py                    # 한 건 확인용 CLI (위험도 + 근거)
│
├── build_finetune_set.py       # [베이스라인] 정규식으로 위험도를 합성하던 옛 방식
├── calibrate.py                # 신호별 로그 우도비 산출 → 앱의 다채널 갱신에 쓰임
├── eval_hard_normal.py         # 어려운 정상 통화를 서버에 걸어 오탐률을 잰다
│
├── eval/                       # 손으로 만든 평가 자료. **지표가 못 잡는 것을 잡으려고 둔다**
│   ├── stage_gold.jsonl        #   진행 단계 정답지 36건. 규칙보다 먼저 만들었다
│   └── hard_normal.jsonl       #   금융기관이 먼저 걸어온 정상 통화 20건 (지어낸 문장)
│
├── docs/
│   ├── METHOD.md               # 왜 이 방식인가 (그림·실측 숫자)
│   ├── COLAB.md                # Colab 학습 절차 (처음 쓰는 사람용)
│   └── RESULTS.md              # 측정된 성능과 그 한계
│
├── adapter-gemma4/             # 학습 산출물 (gitignore, prompt/calibration만 예외)
│   ├── adapter_model.safetensors    92MB
│   ├── prompt.json                 지시문·입력 상한·정답 후보  ← 계약
│   └── calibration.json            온도 3.136
│
├── adapter/                    # 이전 산출물 (Gemma 3 4B, 125MB, 온도 1.370)
│                               #   되돌릴 때를 위해 남겨둔다. 서버에는 하나만 넣을 것 —
│                               #   베이스가 다른 어댑터를 섞으면 확률만 조용히 틀어진다
│
├── finetune/                   # 학습셋 (gitignore)
│   ├── binary_train.jsonl          1,204건
│   ├── binary_val.jsonl              213건
│   └── train.jsonl / val.jsonl     옛 방식 산출물. 베이스라인 비교용
│
└── repo/                       # 원본 저장소 sparse clone (gitignore, 98MB)
    └── Multimodal/data/transcripts/{vishing,non_vishing}/*.json
```

## Tech Stack

- Python 3.13 (표준 라이브러리 우선 — 데이터 준비 단계는 외부 의존성 없음)
- `transformers>=4.50` / `peft` / `accelerate` — **4.x와 5.x 양쪽에서 동작해야 한다.**
  Colab이 어느 버전을 깔지 정할 수 없다 (`train_lora.supported()` 참고)
- `bitsandbytes` — 4bit 양자화. **CUDA 전용이라 맥에서는 못 쓴다**
- 베이스 모델 `google/gemma-4-E2B-it` — 오디오·비전 인코더를 포함한 멀티모달 5.13B.
  게이트되지 않아 토큰 없이 받을 수 있다 (Gemma 3은 승인이 필요했다)
- 학습 환경 Google Colab 무료 T4 16GB

## Build & Test Commands

**맥에서** — 데이터 준비와 배점 산출.

```bash
python3 fetch.py                    # 전사본 1,417건 (sparse clone, 약 82MB)
python3 fetch.py --per-class 25 --push   # 앱 시뮬레이션 표본 생성 + adb 전송
python3 build_binary_set.py --stats # 층화 분할 확인만
python3 build_binary_set.py         # binary_train / binary_val 생성
python3 calibrate.py --derive       # 신호별 로그 우도비 산출
python3 common.py --write-contract adapter-gemma4/   # 기존 어댑터에 계약 붙이기
```

**Colab에서** — GPU가 필요한 학습·평가. 처음이면 `docs/COLAB.md`를 그대로 따라간다.

```bash
python3 train_lora.py                        # 1 epoch, T4에서 약 1시간 20분
python3 evaluate.py --compare-baseline       # 온도 보정 + 정규식 대비 개선폭
python3 evaluate.py --prefix 200 --reuse     # 앞부분만 보고 판정 (실전 조건 재현)
python3 infer.py --id vishing_38             # 한 건 확인
```

**자동화된 테스트가 없다.** 검증은 `evaluate.py`가 낸 숫자로 한다. 코드를 고쳤다면
최소한 아래 둘이 이전과 같은지 확인할 것.

```bash
python3 build_binary_set.py --stats    # 1,417건 / 피싱 49.8% (train·val 모두)
python3 calibrate.py                   # 피싱 706건 중 652건(92%) 커버
```

**`evaluate.py`의 오탐률 0.0%를 개선의 지표로 삼지 말 것.** 검증셋에 어려운 정상이 없어서
나온 값이다. 실제 구멍은 따로 잰다 — 서버를 띄우고:

```bash
python3 eval_hard_normal.py            # 어려운 정상 20건. 현재 오탐률 10.0%
```

**맥에서 `evaluate.py`를 돌릴 때는 `--no-save`를 붙일 것.** 맥(bf16)과 Colab(4bit)은 로짓이
미세하게 달라 맞춰지는 온도도 다른데, 운영에 들어갈 값은 학습을 돌린 환경에서 구한 것이어야
한다. 이 옵션이 없으면 어댑터의 `calibration.json`을 조용히 덮어쓴다.

## Domain Context

- **전사본(transcript)** — 통화를 음성인식으로 옮긴 텍스트. 말더듬과 중복이 그대로 남아 있다
  (`"조사 과정에서 이제 사건이나 본인의 정보가 이제…"`). 실전 입력도 이 형태다.
- **신호(signal)** — `LAW_ENFORCEMENT`, `ACCOUNT_HANDOVER` 등 19개. 정규식으로 검출하며,
  이제 학습이 아니라 **문자 채널의 위험도 갱신**에만 쓰인다.
- **로그 우도비** — `log(피싱 출현율 / 정상 출현율)`. 신호 하나가 위험도를 얼마나 올리는지.
  로그오즈 공간에서는 덧셈이 되므로 다채널 융합이 베이즈 갱신이 된다.
- **온도 보정(temperature scaling)** — 신경망은 과확신한다. 검증셋으로 스칼라 `T` 하나를 찾아
  로짓을 나눠 "87점 = 실제로 87% 확률"이 성립하게 만든다.
- **ECE** — 구간별 예측확률과 실제 비율의 차이. 0.05 미만이면 점수를 사용자에게 보여도 된다.
- **어댑터(adapter)** — LoRA 가중치. 원본 Gemma는 얼려두고 이것만 학습한다. 별개 파일이라
  껐다 켤 수 있고, 끄면 원본 Gemma로 돌아간다(근거 생성·전사에 이용).

## Coding Conventions

- 모든 주석·docstring·출력 문자열은 **한국어**. 커밋 메시지는 영어.
- docstring은 **결정의 근거**를 적는다. 특히 "왜 이 방식이 아닌가"를 남긴다
  (예: 왜 cone 모드가 아니라 `--no-cone`인지, 왜 `warmup_ratio`가 아니라 `warmup_steps`인지).
- 스크립트는 `argparse` CLI로 만들고 `--stats` 같은 미리보기 옵션을 둔다.
- 무거운 import(`torch`, `transformers`)는 **함수 안에서** 한다. `common.py`를 데이터 준비
  단계에서 import할 때 GPU 스택 없이도 동작해야 한다.
- 실패는 `raise SystemExit("… 먼저 fetch.py 를 실행하세요")`처럼 **다음 행동을 알려주며** 끝낸다.
- 랜덤은 항상 시드를 고정한다 (`random.Random(20260811)`).

## Key Patterns

- **계약 파일** — 어댑터 폴더가 자기완결이 되도록 `prompt.json`(지시문·상한·정답 후보),
  `calibration.json`(온도), 토크나이저를 함께 저장한다. 다른 저장소가 코드를 복사하지 않고도
  같은 프롬프트를 재구성할 수 있다.
- **버전 방어막** — `train_lora.supported()`가 `TrainingArguments` 시그니처를 보고 없는 인자를
  버리며 무엇을 버렸는지 알린다. transformers 5.x가 `warmup_ratio`·`group_by_length`를 없앤
  것을 실행 중에 발견했다.
- **첫 토큰만 비교** — 정답 후보 `예`/`아니오`는 토큰 수가 다르다(`아니오`는 3토큰). 첫 토큰
  id만 다르면 한 자리에서 softmax가 곧바로 확률이 된다.
- **손실을 한 자리에만** — 프롬프트 전체를 `-100`으로 가리고 정답 토큰 위치만 남긴다.
  전체에 걸면 모델이 전사본을 따라 쓰는 법을 배워 확률이 흐려진다.
- **층화 분할** — 클래스별로 따로 섞어 나눈다. 통째로 섞으면 213건짜리 검증셋의 피싱 비율이
  우연에 맡겨지고, 그만큼 온도 보정이 휘어진다.
- **로짓 캐시** — `evaluate.py`가 검증셋 로짓을 저장해 `--reuse`로 재사용한다. 온도·지표는
  로짓만 있으면 다시 계산되므로 모델을 다시 돌릴 필요가 없다.
- **앞부분 우선 자르기** — 상한을 넘는 전사본은 뒤를 버린다. 실전에서는 통화가 진행되는 도중,
  즉 앞부분만 들어온 상태에서 판정해야 하므로 학습 조건을 거기에 맞춘다.

## Reference Docs

- `docs/METHOD.md` — 왜 위험도를 만들지 않고 읽어내는가. 로짓·softmax·온도 보정·LoRA
  어댑터·로그오즈 융합을 그림과 실측 숫자로 설명한다. **먼저 읽을 것.**
- `docs/COLAB.md` — Colab 학습 절차. 라이선스 승인부터 결과 회수까지, 처음 쓰는 사람 기준.
  자주 막히는 지점 표 포함.
- `docs/RESULTS.md` — 측정된 성능(AUROC·ECE·정규식 대비)과 **아직 검증되지 않은 것**.
- `README.md` — 데이터 출처, 표본의 한계 4가지, 사용법 요약.
- `../Detection-Server/CLAUDE.md` — 이 저장소가 만든 어댑터를 받아 쓰는 쪽.
