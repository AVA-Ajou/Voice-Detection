#!/usr/bin/env python3
"""Gemma 파인튜닝용 학습셋을 만든다.

모델에게 **위험도와 근거를 직접 말하게** 하는 것이 목표라, 입력은 통화 전사본이고
출력은 `{"risk": 0~100, "reason": "...", "signals": [...]}` 형태다.

데이터셋이 주지 않는 두 가지를 여기서 만든다.

  위험도 — 원본 라벨은 0/1뿐이다. 그대로 쓰면 모델이 0 아니면 100만 뱉어 등급이 안 생긴다.
           그래서 관찰된 신호의 무게 합을 0~100으로 눌러 등급을 만든다. 정상 통화도
           신호가 하나쯤 잡히면 낮은 점수를 받게 되어, "무해함"과 "약한 의심"이 구분된다.
  근거   — 아예 없다. 신호 이름과 전사본에서 실제로 인용한 구절을 엮어 문장을 만든다.
           지어낸 설명이 아니라 원문에 있는 표현이라 모델이 환각을 배우지 않는다.

라벨 품질의 상한은 아래 패턴이다. 모델은 이 패턴을 문맥으로 일반화할 수는 있어도,
패턴이 통째로 놓치는 유형은 배울 수 없다. 표본을 늘리거나 사람이 손보면 그만큼 좋아진다.

    python3 data/korccvi/build_finetune_set.py            # train/val 생성
    python3 data/korccvi/build_finetune_set.py --stats    # 분포만 확인
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
TRANSCRIPTS = HERE / "repo" / "Multimodal" / "data" / "transcripts"
OUT_DIR = HERE / "finetune"

# 위험도를 만들 때 쓰는 신호와 무게. calibrate.py 에서 실측한 값을 그대로 가져온다.
# (그룹, 무게, 패턴, 사람이 읽을 이름)
SIGNALS = {
    "LAW_ENFORCEMENT":       ("IMPERSONATION", 3.7, r"검찰|검사님|수사관|지검|사건번호|영장|조사받|경찰서", "수사기관 사칭"),
    "FINANCIAL_INSTITUTION": ("IMPERSONATION", 1.0, r"금융감독원|금감원|캐피탈|저축은행|카드사에서|은행입니다", "금융기관 사칭"),
    "GOVERNMENT_PROGRAM":    ("IMPERSONATION", 2.0, r"정부지원|서민금융|정책자금|햇살론|버팀목", "정부지원 사칭"),
    "ACQUAINTANCE":          ("IMPERSONATION", 0.5, r"엄마|아빠|아들인데|딸인데|친구인데", "지인 사칭"),

    "CRIMINAL_INVOLVEMENT":  ("PRETEXT", 5.4, r"명의도용|명의가 도용|피해자|도용되|대포통장|공범|연루", "범죄 연루 주장"),
    "FEAR_PRESSURE":         ("PRETEXT", 4.7, r"구속|영장|처벌|동결|출석|압수", "공포 조성"),
    "LOAN_OFFER":            ("PRETEXT", 3.3, r"대출|대환|한도|저금리|신용등급", "대출 미끼"),
    "URGENCY":               ("PRETEXT", 0.3, r"지금 바로|오늘까지|즉시|서둘|마감", "시간 압박"),

    "TRANSFER_DEMAND":       ("DEMAND", 1.0, r"이체|송금|입금해|계좌로 보내", "이체 요구"),
    "EXISTING_LOAN_REPAY":   ("DEMAND", 4.4, r"기존.{0,10}대출|먼저.{0,6}상환|중도상환|완납", "기존 대출 선상환 요구"),
    "UPFRONT_FEE":           ("DEMAND", 1.2, r"수수료|보증금|보증보험|예치금|선입금", "선입금 요구"),
    "APP_INSTALL":           ("DEMAND", 1.0, r"앱\s*설치|어플|다운로드|원격|링크|주소로 들어", "앱 설치 유도"),
    "PERSONAL_INFO":         ("DEMAND", 2.5, r"주민등록번호|주민번호|비밀번호|카드번호|보안카드|OTP", "개인정보 요구"),
    "ACCOUNT_HANDOVER":      ("DEMAND", 3.4, r"계좌를? 빌려|통장을? 빌려|계좌 임대|체크카드를? 보내|양도", "계좌 양도 요구"),
    "ARS_KEYPAD":            ("DEMAND", 0.4, r"번호를? 눌러|1번|2번을? 눌러|버튼을? 눌러", "ARS 번호 입력 유도"),
    "ACCOUNT_NUMBER":        ("DEMAND", 0.0, r"\d{2,3}-\d{2,6}-\d{4,8}", "계좌번호 언급"),

    "SECRECY":               ("CONTROL", 3.2, r"말하지 마|알리지 마|비밀로|말씀하지 마|얘기하지 마", "비밀 유지 요구"),
    "STAY_ON_LINE":          ("CONTROL", 0.7, r"끊지 마|끊으시면|전화 끊|계속 통화", "통화 유지 요구"),
    "STAFF_ASSIGNMENT":      ("DEMAND", 2.4, r"법무사|배정|대리입니다|과장입니다|팀장입니다|담당자입니다", "담당자 배정 연출"),
}

# 상관된 신호가 함께 켜져도 무한정 더해지지 않게 그룹마다 상한을 둔다.
GROUP_CAPS = {"IMPERSONATION": 3.0, "PRETEXT": 4.0, "DEMAND": 6.0, "CONTROL": 3.0}

# 점수 합을 0~100으로 누르는 기울기. 합이 이 값일 때 50점이 되도록 잡았다.
MIDPOINT = 5.0
STEEPNESS = 0.55

# 인용은 앞뒤로 이만큼만 잘라 붙인다. 통째로 넣으면 출력이 전사본만큼 길어진다.
QUOTE_PAD = 18


def detect(text):
    """켜진 신호와, 그 근거가 된 원문 구절."""
    found = {}
    for name, (_, _, pattern, _) in SIGNALS.items():
        m = re.search(pattern, text)
        if not m:
            continue
        start = max(0, m.start() - QUOTE_PAD)
        end = min(len(text), m.end() + QUOTE_PAD)
        found[name] = " ".join(text[start:end].split())
    return found


def risk_of(fired):
    """신호 무게를 그룹 상한과 함께 더한 뒤 0~100으로 누른다."""
    total = 0.0
    for group, cap in GROUP_CAPS.items():
        s = sum(SIGNALS[f][1] for f in fired if SIGNALS[f][0] == group)
        total += min(cap, s)
    return round(100 / (1 + math.exp(-STEEPNESS * (total - MIDPOINT))))


def reason_of(fired, risk):
    """근거 문장. 무게가 큰 신호부터 최대 셋을 인용과 함께 엮는다."""
    if not fired:
        return "위험 신호가 관찰되지 않았습니다."
    top = sorted(fired, key=lambda f: -SIGNALS[f][1])[:3]
    parts = [f"{SIGNALS[f][3]}(\"{fired[f]}\")" for f in top]
    lead = "위험" if risk >= 70 else "주의" if risk >= 40 else "약한 의심"
    return f"{lead} — " + ", ".join(parts) + " 정황이 확인됩니다."


def build():
    rows = []
    for kind, label in (("vishing", 1), ("non_vishing", 0)):
        for path in sorted((TRANSCRIPTS / kind).glob("*.json")):
            text = json.loads(path.read_text(encoding="utf-8")).get("text", "").strip()
            if not text:
                continue
            fired = detect(text)
            risk = risk_of(fired)
            rows.append({
                "id": path.stem,
                "label": label,
                "input": text,
                "output": {
                    "risk": risk,
                    "signals": [{"name": n, "quote": q} for n, q in fired.items()],
                    "reason": reason_of(fired, risk),
                },
            })
    return rows


def report(rows):
    ph = [r for r in rows if r["label"] == 1]
    nm = [r for r in rows if r["label"] == 0]
    print(f"전체 {len(rows)}건 (피싱 {len(ph)} / 정상 {len(nm)})\n")
    print(f"{'구간':>10} {'피싱':>8} {'정상':>8}")
    print("-" * 30)
    for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]:
        a = sum(1 for r in ph if lo <= r["output"]["risk"] < hi)
        b = sum(1 for r in nm if lo <= r["output"]["risk"] < hi)
        print(f"{lo:3}~{hi-1:<6} {a:8} {b:8}")
    print()
    print(f"피싱 평균 위험도 {sum(r['output']['risk'] for r in ph)/len(ph):.1f}")
    print(f"정상 평균 위험도 {sum(r['output']['risk'] for r in nm)/len(nm):.1f}")
    print(f"신호 0개인 피싱 {sum(1 for r in ph if not r['output']['signals'])}건")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true", help="분포만 출력하고 파일은 쓰지 않음")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    rows = build()
    report(rows)
    if args.stats:
        raise SystemExit

    # 통화 단위로 섞어 나눈다. 같은 통화가 학습과 검증에 함께 들어가면 성능이 부풀려진다.
    random.Random(20260811).shuffle(rows)
    cut = int(len(rows) * (1 - args.val_ratio))
    OUT_DIR.mkdir(exist_ok=True)
    for name, part in (("train", rows[:cut]), ("val", rows[cut:])):
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n{path} — {len(part)}건")
