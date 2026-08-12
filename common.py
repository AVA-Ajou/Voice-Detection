#!/usr/bin/env python3
"""학습·평가·추론이 공유하는 프롬프트와 모델 적재. 한 곳에 모아둔다.

프롬프트가 학습과 추론에서 한 글자라도 다르면 그 순간 확률이 의미를 잃는다.
그래서 세 스크립트가 전부 이 파일을 통해서만 프롬프트를 만든다.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent

# Colab 무료 T4(16GB) 기준. 4bit 양자화한 4B 모델이 올라간다.
DEFAULT_MODEL = "google/gemma-3-4b-it"

# 전사본을 토큰 몇 개까지 볼 것인가. T4에서 배치 1 + 그래디언트 체크포인팅 기준 한계다.
# 상위 10%가 2,300자를 넘지만 앞부분만 남긴다 — 실전에서는 통화가 진행되는 도중,
# 즉 앞부분만 들어온 상태에서 판정해야 하므로 학습 조건을 거기에 맞춘다.
MAX_TEXT_TOKENS = 1024

# 정답 후보. 첫 토큰이 서로 다르기만 하면 되므로 두 글자 이상이어도 상관없다.
YES, NO = "예", "아니오"

INSTRUCTION = "위 통화가 보이스피싱인지 판단하라. 예 또는 아니오로만 답하라."


def contract():
    """어댑터와 함께 저장할 프롬프트 계약.

    학습 때와 추론 때 프롬프트가 한 글자라도 다르면 **에러 없이 확률만 조용히 틀어진다.**
    추론 서버는 별도 저장소라 이 파일의 코드를 참조할 수 없으므로, 계약을 코드가 아니라
    파일로 고정한다. 어댑터 폴더는 이걸로 자기완결이 된다 — chat_template.jinja(토크나이저)와
    calibration.json(온도)은 이미 같은 폴더에 저장되고 있다.
    """
    return {
        "base_model": DEFAULT_MODEL,
        "instruction": INSTRUCTION,
        "max_text_tokens": MAX_TEXT_TOKENS,
        "yes": YES,
        "no": NO,
    }


def save_contract(directory):
    path = Path(directory) / "prompt.json"
    path.write_text(json.dumps(contract(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def answer_ids(tokenizer):
    """정답 후보의 **첫 토큰** id. 확률은 이 한 자리에서만 읽는다.

    "아니오"가 여러 토큰으로 쪼개져도 상관없다. 첫 토큰만 달라도 두 답을 구분할 수 있고,
    한 자리에서 읽어야 softmax가 곧바로 확률이 되기 때문이다.
    """
    yes = tokenizer.encode(YES, add_special_tokens=False)[0]
    no = tokenizer.encode(NO, add_special_tokens=False)[0]
    if yes == no:
        raise SystemExit(f"'{YES}'와 '{NO}'의 첫 토큰이 같습니다({yes}). 후보를 바꾸세요.")
    return yes, no


def chat_ids(tokenizer, content):
    """대화 템플릿을 씌운 토큰 id 목록.

    `apply_chat_template(tokenize=True)`의 반환형이 transformers 4.x와 5.x에서 다르다
    (리스트 ↔ BatchEncoding). 문자열로 받아 직접 토큰화하면 양쪽에서 같게 동작한다.
    템플릿이 `<bos>`를 이미 넣으므로 여기서 특수 토큰을 또 붙이지 않는다.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True, tokenize=False,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def clip(tokenizer, text):
    """전사본을 앞에서부터 MAX_TEXT_TOKENS 만큼만 남긴다."""
    ids = tokenizer.encode(text, add_special_tokens=False)[:MAX_TEXT_TOKENS]
    return tokenizer.decode(ids)


def build_prompt(tokenizer, text):
    """판정용 프롬프트. 학습과 추론이 반드시 이걸 같이 써야 한다."""
    return chat_ids(tokenizer, f"{clip(tokenizer, text)}\n\n{INSTRUCTION}")


def load_rows(name):
    path = HERE / "finetune" / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} 가 없습니다. 먼저 build_binary_set.py 를 실행하세요.")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def lora_targets(model):
    """LoRA를 붙일 모듈 이름.

    Gemma 3 4B는 비전 타워가 붙은 멀티모달 모델이고, 거기에도 q_proj/v_proj가 있다.
    이름만 보고 고르면 쓰지도 않을 이미지 경로까지 학습하게 되므로 걸러낸다.
    """
    wanted = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    names = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in wanted and not re.search(r"vision|multi_modal|siglip", name):
            names.add(leaf)
    return sorted(names)


def load_model(model_id, adapter=None, four_bit=True, attn="sdpa"):
    """베이스 모델(+어댑터)을 올린다.

    Gemma 3는 크기에 따라 클래스가 갈린다 — 1B는 순수 언어 모델이고 4B 이상은 멀티모달
    래퍼다. 어느 쪽이든 텍스트만 넣으면 같은 형태의 로짓이 나오므로 되는 쪽으로 연다.

    [attn]이 기본 `sdpa`인 이유 — `eager`는 T4에서 두 배 가까이 느리다. eager 권장은
    로짓 소프트캡이 있던 Gemma 2 이야기고, Gemma 3는 그 구조를 쓰지 않는다.
    """
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    kwargs = {"attn_implementation": attn}
    if four_bit:
        # T4는 bf16을 지원하지 않아 fp16으로 계산한다. A100/L4면 bf16이 잡힌다.
        compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = torch.float32

    model = _open(model_id, kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return tokenizer, model


def _open(model_id, kwargs):
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


if __name__ == "__main__":
    # 이미 학습이 끝난 어댑터에도 계약을 붙일 수 있게 CLI를 둔다.
    #     python3 common.py --write-contract adapter/
    import argparse

    parser = argparse.ArgumentParser(description="프롬프트 계약을 어댑터 폴더에 쓴다")
    parser.add_argument("--write-contract", metavar="DIR", required=True)
    target = Path(parser.parse_args().write_contract)
    if not target.is_dir():
        raise SystemExit(f"{target} 폴더가 없습니다. 어댑터를 먼저 내려받으세요.")
    print(f"계약 저장 → {save_contract(target)}")
    print(json.dumps(contract(), ensure_ascii=False, indent=1))
