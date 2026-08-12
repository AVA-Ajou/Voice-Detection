#!/usr/bin/env python3
"""**어려운 정상 통화**를 띄워둔 서버에 걸어 오탐이 얼마나 나는지 잰다.

검증셋 213건의 오탐률 0.0% 는 지표가 좋은 게 아니라 **시험이 쉬운 것**이다. 학습·검증
1,417건에서 "돈 얘기가 나오면서 기관이 먼저 건" 통화는 피싱 204건 대 정상 6건이라,
모델은 정당한 금융 아웃바운드 통화를 사실상 본 적이 없다. 그 구멍의 크기를 재는 눈금이다.

    python3 eval_hard_normal.py                      # 서버가 떠 있어야 한다
    python3 eval_hard_normal.py --url http://…       # 다른 주소

**이 점수를 성능 근거로 쓰지 말 것.** 모델이 무엇으로 학습했는지 아는 쪽이 문장을 지었다.
용도는 두 가지뿐이다 — 구멍의 크기를 눈으로 보는 것, 그리고 나중에 학습셋을 보강했을 때
나아졌는지 비교할 기준선.
"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

# 스크립트는 src/ 안에 있고 데이터·산출물 폴더는 저장소 루트에 있다.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SET = ROOT / "eval" / "hard_normal.jsonl"


def analyze(url, text, timeout):
    req = urllib.request.Request(
        f"{url}/analyze",
        data=json.dumps({"text": text, "task": "voice"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))["risk"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument("--threshold", type=float, default=70.0,
                        help="앱이 피싱으로 보는 선. 이걸 넘으면 오탐이다")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.set).open(encoding="utf-8")]
    try:
        scored = [(r, analyze(args.url, r["text"], args.timeout)) for r in rows]
    except urllib.error.URLError as e:
        raise SystemExit(f"서버에 닿지 못했습니다 ({args.url}) — 먼저 띄우세요.\n  {e}")

    scored.sort(key=lambda rv: -rv[1])
    print(f"{'id':>16} {'유형':<18} {'위험도':>7}")
    print("-" * 46)
    for row, risk in scored:
        mark = "  ← 오탐" if risk >= args.threshold else ""
        print(f"{row['id']:>16} {row['kind']:<18} {risk:7.1f}{mark}")

    risks = [v for _, v in scored]
    over = sum(1 for v in risks if v >= args.threshold)
    print(f"\n{len(risks)}건 중 {over}건이 {args.threshold:.0f}점 초과 — "
          f"오탐률 {over/len(risks)*100:.1f}%")
    print(f"중앙값 {sorted(risks)[len(risks)//2]:.1f}   최고 {max(risks):.1f}   최저 {min(risks):.1f}")
    print("\n검증셋 213건에서는 오탐률 0.0% 였다. 두 숫자의 차이가 구멍의 크기다.")
