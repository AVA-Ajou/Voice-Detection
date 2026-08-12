#!/usr/bin/env python3
"""위험도와 근거를 뽑는다. 백엔드가 호출하게 될 두 함수가 여기 있다.

    위험도  어댑터 ON  → 정답 자리의 P("예")를 읽는다. 생성하지 않는다.
    근거    어댑터 OFF → 원본 Gemma가 원문을 인용해 설명한다. 학습시키지 않은 능력이다.

모델 파일은 하나다. 30MB짜리 어댑터만 껐다 켠다 — docs/METHOD.md 5절.

    python3 infer.py --text "여보세요 서울중앙지검..."
    python3 infer.py --id vishing_38
"""

import argparse
import json
from pathlib import Path

import common

# 스크립트는 src/ 안에 있고 데이터·산출물 폴더는 저장소 루트에 있다.
ROOT = Path(__file__).resolve().parents[1]

# 판정 결과에 따라 묻는 말이 갈려야 한다. "피싱으로 판정되었다"를 무조건 전제로 깔면
# 정상 통화에도 모델이 억지 근거를 만든다 — 배송 지연 사과를 사기 수법으로 둔갑시키는 것을
# 실제로 확인했다. 모델은 주어진 전제를 의심하지 않는다.
EXPLAIN_HIGH = """다음은 통화 전사본이다.

{text}

이 통화는 보이스피싱으로 판정되었다(위험도 {risk}점).
전사본에 **실제로 나온 발언을 직접 인용**하면서 그렇게 판정된 근거를 두세 문장으로 설명하라.
전사본에 없는 내용은 절대 지어내지 마라."""

EXPLAIN_LOW = """다음은 통화 전사본이다.

{text}

이 통화는 보이스피싱이 **아닌** 것으로 판정되었다(위험도 {risk}점).
왜 위험하지 않다고 볼 수 있는지 한두 문장으로 설명하라.
없는 위험 신호를 억지로 찾아내지 마라."""

EXPLAIN_THRESHOLD = 50


def classify(model, tokenizer, text, temperature=1.0):
    """P("예"). 이 값이 곧 위험도다 — 우리가 계산하는 게 아니라 읽어내는 값이다."""
    import torch

    yes, no = common.answer_ids(tokenizer)
    ids = torch.tensor([common.build_prompt(tokenizer, text)]).to(model.device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1]

    # 정답 후보 두 개만 남기고 softmax. 온도로 나누면 과확신이 눌린다.
    pair = torch.tensor([logits[yes], logits[no]], dtype=torch.float32) / temperature
    return torch.softmax(pair, dim=0)[0].item()


def explain(model, tokenizer, text, risk):
    """어댑터를 뗀 원본 Gemma에게 근거를 받는다."""
    import torch

    score = round(risk * 100, 1)
    template = EXPLAIN_HIGH if score >= EXPLAIN_THRESHOLD else EXPLAIN_LOW
    content = template.format(text=common.clip(tokenizer, text), risk=score)
    ids = common.chat_ids(tokenizer, content)
    prompt = torch.tensor([ids]).to(model.device)

    with model.disable_adapter():  # 이 블록 안에서만 원본 Gemma로 돌아간다
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=256, do_sample=False)
    return tokenizer.decode(out[0][len(ids):], skip_special_tokens=True).strip()


def temperature_of(adapter):
    """evaluate.py가 검증셋에서 찾아 저장해둔 온도. 없으면 보정 없이 간다."""
    path = Path(adapter) / "calibration.json"
    if not path.exists():
        print("경고 — calibration.json 이 없어 보정 없이 계산합니다. evaluate.py 를 먼저 돌리세요.")
        return 1.0
    return json.loads(path.read_text())["temperature"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=common.DEFAULT_MODEL)
    parser.add_argument("--adapter", default=str(common.DEFAULT_ADAPTER))
    parser.add_argument("--text", help="전사본 직접 입력")
    parser.add_argument("--id", help="데이터셋에서 id로 찾아 쓴다 (예: vishing_38)")
    parser.add_argument("--no-reason", action="store_true", help="위험도만 계산")
    args = parser.parse_args()

    text = args.text
    if args.id:
        rows = common.load_rows("binary_train") + common.load_rows("binary_val")
        found = next((r for r in rows if r["id"] == args.id), None)
        if not found:
            raise SystemExit(f"{args.id} 를 찾을 수 없습니다.")
        text = found["text"]
    if not text:
        raise SystemExit("--text 또는 --id 중 하나가 필요합니다.")

    tokenizer, model = common.load_model(args.model, adapter=args.adapter)
    risk = classify(model, tokenizer, text, temperature_of(args.adapter))

    filled = round(risk * 20)
    print(f"\n위험도  {risk*100:.1f}   {'▓' * filled}{'░' * (20 - filled)}")
    if not args.no_reason:
        print(f"\n근거\n{explain(model, tokenizer, text, risk)}")
