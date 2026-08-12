#!/usr/bin/env python3
"""이진 학습셋을 만든다. 위험도는 만들지 않는다.

`build_finetune_set.py`와의 차이가 이 저장소의 핵심 결정이다 — 자세한 건 docs/METHOD.md.
요약하면, 위험도를 우리가 지어내는 대신 모델이 매긴 확률을 추론 시점에 읽어 쓴다.
그래서 여기서 만드는 정답은 원본 라벨 `0/1` 그대로다.

    python3 build_binary_set.py            # binary_train / binary_val 생성
    python3 build_binary_set.py --stats    # 분포만 확인
"""

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
TRANSCRIPTS = HERE / "repo" / "Multimodal" / "data" / "transcripts"
OUT_DIR = HERE / "finetune"

SEED = 20260811


def build():
    rows = []
    for kind, label in (("vishing", 1), ("non_vishing", 0)):
        for path in sorted((TRANSCRIPTS / kind).glob("*.json")):
            text = json.loads(path.read_text(encoding="utf-8")).get("text", "").strip()
            if text:
                rows.append({"id": path.stem, "label": label, "text": text})
    return rows


def split(rows, val_ratio):
    """층화 분할 — 클래스별로 따로 섞어 나눈다.

    통째로 섞어 자르면 검증셋의 피싱 비율이 우연에 맡겨진다. 213건짜리 검증셋에서
    비율이 몇 %만 틀어져도 온도 보정이 그만큼 휘어지므로, 비율을 고정한다.
    """
    rng = random.Random(SEED)
    train, val = [], []
    for label in (0, 1):
        part = [r for r in rows if r["label"] == label]
        rng.shuffle(part)
        cut = int(len(part) * (1 - val_ratio))
        train += part[:cut]
        val += part[cut:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def report(name, part):
    ph = [r for r in part if r["label"] == 1]
    lens = sorted(len(r["text"]) for r in part)
    print(f"{name:6} {len(part):5}건   피싱 {len(ph):4} ({len(ph)/len(part)*100:.1f}%)"
          f"   길이 중앙 {lens[len(lens)//2]:5} / p90 {lens[int(len(lens)*.9)]:6}"
          f" / 최대 {lens[-1]:6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true", help="분포만 출력하고 파일은 쓰지 않음")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    rows = build()
    if not rows:
        raise SystemExit(f"전사본이 없습니다. 먼저 fetch.py 를 실행하세요 — {TRANSCRIPTS}")

    train, val = split(rows, args.val_ratio)
    report("전체", rows)
    report("train", train)
    report("val", val)

    if args.stats:
        raise SystemExit

    OUT_DIR.mkdir(exist_ok=True)
    for name, part in (("binary_train", train), ("binary_val", val)):
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n{path} — {len(part)}건")
