#!/usr/bin/env python3
"""Gemma에 LoRA를 붙여 이진 분류를 학습한다.

손실은 **정답 토큰 한 자리에만** 건다. 프롬프트 전체에 걸면 모델이 전사본을 따라 쓰는 법을
같이 배우게 되고, 우리가 읽어야 할 그 한 자리의 확률이 그만큼 흐려진다.

    python3 train_lora.py                          # 기본값 (Colab 무료 T4)
    python3 train_lora.py --model google/gemma-3-12b-it --epochs 2
"""

import argparse
from pathlib import Path

import common

HERE = Path(__file__).parent


class Dataset:
    """프롬프트 + 정답 토큰 하나. 정답 자리를 뺀 나머지는 -100으로 가린다."""

    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer
        self.yes, self.no = common.answer_ids(tokenizer)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        prompt = common.build_prompt(self.tokenizer, row["text"])
        answer = self.yes if row["label"] == 1 else self.no
        return {
            "input_ids": prompt + [answer],
            "labels": [-100] * len(prompt) + [answer],
        }


def collate(batch, pad_id):
    import torch
    width = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        gap = width - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * gap)
        out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * gap)
        out["labels"].append(b["labels"] + [-100] * gap)
    return {k: torch.tensor(v) for k, v in out.items()}


def supported(kwargs):
    """TrainingArguments가 받는 인자만 남긴다.

    transformers 4.x와 5.x가 인자를 주고받는다 — 5.x는 `warmup_ratio`를 없앴고
    4.x 후반은 `evaluation_strategy`를 `eval_strategy`로 바꿨다. Colab이 어느 쪽을
    깔지 우리가 정할 수 없으므로, 없는 인자는 조용히 버리는 대신 무엇을 버렸는지 알린다.
    """
    import inspect
    from transformers import TrainingArguments

    known = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(k for k in kwargs if k not in known)
    if dropped:
        print(f"이 transformers 버전이 받지 않아 무시한 설정: {', '.join(dropped)}")
    return {k: v for k, v in kwargs.items() if k in known}


def main(args):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments

    tokenizer, model = common.load_model(args.model, four_bit=True, attn=args.attn)

    yes, no = common.answer_ids(tokenizer)
    print(f"정답 토큰  {common.YES!r} → {yes}   {common.NO!r} → {no}")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # 그래디언트 체크포인팅과 함께 쓸 수 없다
    model = get_peft_model(model, LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=common.lora_targets(model),
    ))
    model.print_trainable_parameters()

    train = Dataset(common.load_rows("binary_train"), tokenizer)
    val = Dataset(common.load_rows("binary_val"), tokenizer)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # 유효 배치는 8로 고정하고, 실제 배치와 누적으로 나눠 갖는다. 배치를 키울수록
    # GPU가 덜 놀아 빨라지지만 메모리를 더 쓴다 — 터지면 --batch 를 1로 낮춘다.
    # 8을 넘기지 않는 이유는 한 바퀴만 도는 계획이라 옵티마이저 스텝 수를 벌어야 해서다.
    accum = max(1, 8 // args.batch)
    steps_per_epoch = max(1, len(train) // (args.batch * accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    bf16 = torch.cuda.is_bf16_supported()
    print(f"배치 {args.batch} × 누적 {accum} → 에폭당 {steps_per_epoch}스텝, "
          f"총 {total_steps}스텝")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**supported(dict(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=accum,
            per_device_eval_batch_size=args.batch,
            # 길이가 비슷한 것끼리 묶는다. 짧은 통화가 긴 통화에 맞춰 패딩되는 낭비를 줄인다.
            group_by_length=True,
            learning_rate=2e-4,
            # 비율 대신 스텝 수로 준다 — 버전을 안 타는 쪽이다.
            warmup_steps=max(5, int(total_steps * 0.03)),
            lr_scheduler_type="cosine",
            logging_steps=20,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            bf16=bf16,
            fp16=not bf16,
            gradient_checkpointing=True,
            report_to=[],
        ))),
        train_dataset=train,
        eval_dataset=val,
        data_collator=lambda b: collate(b, pad_id),
    )
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    # 추론 서버는 별도 저장소라 이 코드를 못 본다. 프롬프트를 파일로 함께 남긴다.
    common.save_contract(args.out)
    print(f"\n어댑터 저장 완료 → {args.out}")
    print("  prompt.json 도 함께 저장했습니다 — 서버가 이걸 읽어 프롬프트를 재구성합니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=common.DEFAULT_MODEL)
    parser.add_argument("--out", default=str(HERE / "adapter"))
    # 1,200건 이진 분류에 LoRA면 한 바퀴로 대체로 수렴한다. 검증 손실을 보고 늘릴 것.
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch", type=int, default=2, help="메모리가 터지면 1로")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument("--rank", type=int, default=16)
    main(parser.parse_args())
