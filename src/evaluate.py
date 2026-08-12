#!/usr/bin/env python3
"""검증셋으로 온도를 찾고, 그 확률이 믿을 만한지 검사한다.

이 스크립트가 이번 방식의 존재 이유를 증명한다. 옛 방식은 "87점"이 뭘 뜻하는지 정의가
없어 아래 표를 아예 그릴 수 없었다. 여기서는 그릴 수 있고, 대각선에 붙는지로 판정한다.

    python3 evaluate.py                      # 추론 → 보정 → 검사
    python3 evaluate.py --compare-baseline   # 옛 방식(정규식 점수)과 나란히 비교
"""

import argparse
import json
import math
from pathlib import Path

import common

# 스크립트는 src/ 안에 있고 데이터·산출물 폴더는 저장소 루트에 있다.
ROOT = Path(__file__).resolve().parents[1]


def cache_path(adapter, prefix):
    """어댑터와 자른 길이마다 캐시를 따로 둔다.

    예전에는 파일 이름이 하나뿐이라 다른 어댑터를 평가하면 앞의 것을 덮어썼고, `--reuse`가
    **엉뚱한 모델의 로짓을 조용히 되썼다.** 지표만 보고는 알아챌 방법이 없는 종류의 사고다.
    """
    tag = "" if prefix is None else f"_p{prefix}"
    return ROOT / "finetune" / f"val_logits_{Path(adapter).name}{tag}.json"


# ────────────────────────────────────────────── 로짓 뽑기

def collect(model_id, adapter, prefix=None):
    """검증셋 전체의 로짓 차이(예 - 아니오). 온도와 무관한 원본 값이라 한 번만 구한다.

    [prefix]는 전사본을 앞에서부터 몇 **글자**만 남길지다. 학습·검증은 통화 전체를 보지만
    실전에서는 통화가 진행되는 도중, 즉 앞부분만 들어온 상태에서 판정해야 한다.
    확률이 0과 1로만 갈리는 것이 "통화 전체를 봐서 명백한" 탓인지 확인하는 실험이다.
    """
    import torch

    tokenizer, model = common.load_model(model_id, adapter=adapter)
    yes, no = common.answer_ids(tokenizer)
    rows = common.load_rows("binary_val")

    out = []
    for i, row in enumerate(rows, 1):
        text = row["text"][:prefix] if prefix else row["text"]
        ids = torch.tensor([common.build_prompt(tokenizer, text)]).to(model.device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1]
        out.append({"id": row["id"], "label": row["label"],
                    "diff": float(logits[yes] - logits[no])})
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}")
    return out


# ────────────────────────────────────────────── 보정

def prob(diff, t):
    return 1 / (1 + math.exp(-diff / t))


def nll(rows, t):
    total = 0.0
    for r in rows:
        p = min(max(prob(r["diff"], t), 1e-9), 1 - 1e-9)
        total -= math.log(p) if r["label"] == 1 else math.log(1 - p)
    return total / len(rows)


def fit_temperature(rows):
    """NLL을 최소로 만드는 온도 하나. 성긴 격자로 훑고 그 주변을 좁혀 들어간다."""
    best, lo, hi = 1.0, 0.05, 20.0
    for _ in range(4):
        step = (hi - lo) / 40
        cands = [lo + step * i for i in range(41)]
        best = min(cands, key=lambda t: nll(rows, t))
        lo, hi = max(0.05, best - step), best + step
    return best


# ────────────────────────────────────────────── 지표

def auroc(rows):
    """순위 기반. 동점은 평균 순위로 처리한다."""
    order = sorted(rows, key=lambda r: r["diff"])
    ranks, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1]["diff"] == order[i]["diff"]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[id(order[k])] = avg
        i = j + 1
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    if not pos or not neg:
        return float("nan")
    s = sum(ranks[id(r)] for r in pos)
    return (s - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


def reliability(rows, t):
    """모델이 말한 확률 구간별 실제 피싱 비율. 대각선에 붙어야 보정이 성공한 것이다."""
    print(f"\n{'모델이 말한 확률':>16} {'건수':>6} {'평균 예측':>10} {'실제 피싱 비율':>14} {'차이':>8}")
    print("-" * 60)
    ece = 0.0
    for lo, hi in BINS:
        part = [r for r in rows if lo <= prob(r["diff"], t) < hi]
        if not part:
            print(f"{lo:6.1f} ~ {hi:4.1f} {0:9}")
            continue
        pred = sum(prob(r["diff"], t) for r in part) / len(part)
        real = sum(r["label"] for r in part) / len(part)
        ece += len(part) / len(rows) * abs(pred - real)
        flag = "✓" if abs(pred - real) < 0.1 else "✗"
        print(f"{lo:6.1f} ~ {hi:4.1f} {len(part):9} {pred:10.2f} {real:14.2f} "
              f"{pred-real:+8.2f}  {flag}")
    print(f"\nECE {ece:.3f}   (0.05 미만이면 보정 성공)")
    return ece


def threshold_table(rows, t):
    """미탐은 돈이 나가고 오탐은 짜증에 그친다 — calibrate.py 와 같은 기준으로 본다."""
    print(f"\n{'임계값':>7} {'재현율':>10} {'오탐률':>10} {'미탐':>6} {'오탐':>6}")
    print("-" * 46)
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tp = sum(1 for r in rows if r["label"] == 1 and prob(r["diff"], t) >= th)
        fn = sum(1 for r in rows if r["label"] == 1 and prob(r["diff"], t) < th)
        fp = sum(1 for r in rows if r["label"] == 0 and prob(r["diff"], t) >= th)
        tn = sum(1 for r in rows if r["label"] == 0 and prob(r["diff"], t) < th)
        rec = tp / (tp + fn) if tp + fn else 0
        fpr = fp / (fp + tn) if fp + tn else 0
        print(f"{th:7.1f} {rec*100:9.1f}% {fpr*100:9.1f}% {fn:6} {fp:6}")


# ────────────────────────────────────────────── 옛 방식과의 비교

def old_risks():
    """정규식이 매긴 위험도를 id로 찾을 수 있게 모은다. 분할이 달라 양쪽 파일을 다 읽는다."""
    out = {}
    for name in ("train", "val"):
        path = ROOT / "finetune" / f"{name}.jsonl"
        if not path.exists():
            return {}
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            out[row["id"]] = row["output"]["risk"]
    return out


def compare(rows, t):
    """핵심 검사 — 정규식이 놓친 피싱을 모델이 건져내는가.

    건져내지 못하면 파인튜닝이 정규식을 복사한 것뿐이라 할 이유가 없다.
    """
    old = old_risks()
    if not old:
        print("\n옛 학습셋(finetune/train.jsonl)이 없어 비교를 건너뜁니다.")
        return

    missed = [r for r in rows if r["label"] == 1 and old.get(r["id"], 100) < 40]
    if not missed:
        print("\n검증셋에 저점수 피싱이 없습니다.")
        return

    print(f"\n\n{'='*60}\n정규식이 놓친 피싱 {len(missed)}건 — 모델이 건졌는가\n{'='*60}")
    print(f"\n{'id':>16} {'옛 방식':>8} {'새 방식':>8}")
    print("-" * 36)
    saved = 0
    for r in sorted(missed, key=lambda r: old[r["id"]]):
        new = prob(r["diff"], t) * 100
        saved += new >= 70
        print(f"{r['id']:>16} {old[r['id']]:8} {new:8.1f}  {'✓' if new >= 70 else ' '}")
    print(f"\n{len(missed)}건 중 {saved}건이 70점 이상 — {saved/len(missed)*100:.0f}%")
    print("이 비율이 높을수록 모델이 정규식을 넘어 원문의 맥락을 배운 것이다.")


# ────────────────────────────────────────────── 실행

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=common.DEFAULT_MODEL)
    parser.add_argument("--adapter", default=str(common.DEFAULT_ADAPTER))
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--reuse", action="store_true", help="저장해둔 로짓을 다시 쓴다")
    parser.add_argument("--prefix", type=int, metavar="N",
                        help="전사본 앞 N글자만 보고 판정 (실전 조건 재현)")
    parser.add_argument("--no-save", action="store_true",
                        help="온도를 저장하지 않는다. 맥에서 진단용으로 돌릴 때 쓴다")
    args = parser.parse_args()

    cache = cache_path(args.adapter, args.prefix)
    if args.reuse and cache.exists():
        rows = json.loads(cache.read_text())
        print(f"저장된 로짓 사용 — {cache}")
    else:
        rows = collect(args.model, args.adapter, args.prefix)
        cache.write_text(json.dumps(rows, ensure_ascii=False))

    ph = sum(r["label"] for r in rows)
    scope = f"앞 {args.prefix}글자" if args.prefix else "전체"
    print(f"\n검증셋 {len(rows)}건 (피싱 {ph} / 정상 {len(rows)-ph})   전사본 범위: {scope}")

    t = fit_temperature(rows)
    print(f"\n온도 T = {t:.3f}   NLL {nll(rows, 1.0):.4f} → {nll(rows, t):.4f}")
    if t > 1:
        print("  T > 1 — 모델이 과확신하고 있었고, 확률을 눌러 보정했다.")
    else:
        print("  T < 1 — 모델이 오히려 소심했고, 확률을 벌려 보정했다.")

    print(f"\nAUROC {auroc(rows):.4f}   (0.5 = 찍기, 1.0 = 완벽)")
    ece = reliability(rows, t)
    threshold_table(rows, t)

    if args.compare_baseline:
        compare(rows, t)

    if args.prefix:
        # 잘린 전사본으로 맞춘 온도는 실전용이 아니다. 덮어쓰면 운영 값이 오염된다.
        print(f"\n--prefix 실행이므로 온도를 저장하지 않습니다 (실험용).")
        raise SystemExit
    if args.no_save:
        # 맥(bf16)과 Colab(4bit)은 로짓이 미세하게 달라 맞춰지는 온도도 다르다. 학습을
        # 돌린 환경에서 구한 값이 운영 값이어야 하므로, 진단하러 돌릴 때는 덮어쓰지 않는다.
        print(f"\n--no-save 이므로 온도를 저장하지 않습니다 "
              f"(어댑터의 기존 값이 그대로 운영에 쓰입니다).")
        raise SystemExit

    out = Path(args.adapter) / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"temperature": t, "ece": ece, "auroc": auroc(rows)}, indent=1))
    print(f"\n온도 저장 → {out}   (infer.py 가 이 값을 읽는다)")
