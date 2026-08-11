#!/usr/bin/env python3
"""스코어링 로직을 LLM 없이 데이터셋 전체로 검증하고 배점·임계값을 산출한다.

왜 LLM을 빼는가 — 지금 두 가지가 엉켜 있어서 어느 쪽이 문제인지 가려지지 않는다.

    ① 항목 추출이 제대로 되나   ← LLM 몫
    ② 점수 계산·임계값이 맞나   ← 우리 코드 몫

②만 떼어내면 730건 전부를 몇 초 만에, 호출 한도 없이 돌릴 수 있다. 여기서는 정규식을
항목 추출기의 대역으로 쓴다. 대역이므로 **여기서 나온 배점을 그대로 앱에 넣으면 안 된다** —
LLM 추출기는 출현율이 다르다. 이 스크립트가 답하는 것은 "구조가 작동하는가"와
"임계값을 어디에 그어야 하는가"이고, 최종 배점은 같은 절차를 LLM 추출 결과에 다시 적용해 얻는다.

    python3 data/korccvi/calibrate.py            # 시드 배점으로 평가
    python3 data/korccvi/calibrate.py --derive   # 데이터에서 배점을 다시 뽑아 평가
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent

# RiskScorer.kt 와 같은 값. 한쪽만 고치면 검증이 무의미해지므로 바꿀 때 함께 고칠 것.
BIAS = 0.0
GROUP_CAPS = {"IMPERSONATION": 3.0, "PRETEXT": 4.0, "DEMAND": 6.0, "CONTROL": 3.0}

# (그룹, 시드 배점, 탐지 패턴) — 패턴은 LLM 추출기의 대역이다.
FEATURES = {
    "LAW_ENFORCEMENT":       ("IMPERSONATION", 3.7, r"검찰|검사님|수사관|지검|사건번호|영장|조사받|경찰서"),
    "FINANCIAL_INSTITUTION": ("IMPERSONATION", 1.0, r"금융감독원|금감원|캐피탈|저축은행|카드사에서|은행입니다"),
    "GOVERNMENT_PROGRAM":    ("IMPERSONATION", 2.0, r"정부지원|서민금융|정책자금|햇살론|버팀목"),
    "ACQUAINTANCE":          ("IMPERSONATION", 0.5, r"엄마|아빠|아들인데|딸인데|친구인데"),

    "CRIMINAL_INVOLVEMENT":  ("PRETEXT", 5.4, r"명의도용|명의가 도용|피해자|도용되|대포통장|공범|연루"),
    "FEAR_PRESSURE":         ("PRETEXT", 4.7, r"구속|영장|처벌|동결|출석|압수"),
    "LOAN_OFFER":            ("PRETEXT", 3.3, r"대출|대환|한도|저금리|신용등급"),
    "URGENCY":               ("PRETEXT", 0.3, r"지금 바로|오늘까지|즉시|서둘|마감"),

    "TRANSFER_DEMAND":       ("DEMAND", 1.0, r"이체|송금|입금해|계좌로 보내"),
    "EXISTING_LOAN_REPAY":   ("DEMAND", 4.4, r"기존.{0,10}대출|먼저.{0,6}상환|중도상환|완납"),
    "UPFRONT_FEE":           ("DEMAND", 1.2, r"수수료|보증금|보증보험|예치금|선입금"),
    "APP_INSTALL":           ("DEMAND", 1.0, r"앱\s*설치|어플|다운로드|원격|링크|주소로 들어"),
    "PERSONAL_INFO":         ("DEMAND", 2.5, r"주민등록번호|주민번호|비밀번호|카드번호|보안카드|OTP"),
    "ACCOUNT_HANDOVER":      ("DEMAND", 3.4, r"계좌를? 빌려|통장을? 빌려|계좌 임대|체크카드를? 보내|양도"),
    "ARS_KEYPAD":            ("DEMAND", 0.4, r"번호를? 눌러|1번|2번을? 눌러|버튼을? 눌러"),
    "ACCOUNT_NUMBER":        ("DEMAND", 0.0, r"\d{2,3}-\d{2,6}-\d{4,8}"),

    "SECRECY":               ("CONTROL", 3.2, r"말하지 마|알리지 마|비밀로|말씀하지 마|얘기하지 마"),
    "STAY_ON_LINE":          ("CONTROL", 0.7, r"끊지 마|끊으시면|전화 끊|계속 통화"),
    "STAFF_ASSIGNMENT":      ("CONTROL", 2.4, r"법무사|배정|대리입니다|과장입니다|팀장입니다|담당자입니다"),
}


TRANSCRIPTS = HERE / "repo" / "Multimodal" / "data" / "transcripts"


def load(speaker_only=False):
    """전사본을 (텍스트, 피싱여부)로 읽는다.

    통화별 JSON(1,417건)이 있으면 그쪽을 쓴다. v1.3 CSV(730건)보다 피싱이 5.8배 많고
    화자 분리가 붙어 있어서, 사용자 본인의 되묻는 말을 걷어낼 수 있다.

    [speaker_only]가 참이면 **가장 많이 말한 화자**의 발화만 남긴다. 화자 라벨에는 역할이
    없으므로 "사기범이 더 많이 말한다"는 경험칙으로 대신한다 — 설득하는 쪽이 말을 길게
    하기 때문이고, 실제로 피싱 통화의 화자 수 분포도 그 가정과 어긋나지 않는다.
    """
    if TRANSCRIPTS.is_dir():
        out = []
        for kind, is_phishing in (("vishing", True), ("non_vishing", False)):
            for path in sorted((TRANSCRIPTS / kind).glob("*.json")):
                d = json.loads(path.read_text(encoding="utf-8"))
                if not speaker_only:
                    out.append((d.get("text", ""), is_phishing))
                    continue
                by_speaker = {}
                for s in d.get("segments", []):
                    name = (s.get("speaker") or {}).get("name") or "?"
                    by_speaker[name] = by_speaker.get(name, "") + " " + s.get("text", "")
                dominant = max(by_speaker.values(), key=len) if by_speaker else d.get("text", "")
                out.append((dominant, is_phishing))
        return out

    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader((HERE / "korccvi_v1.3.csv").open(encoding="utf-8")))
    return [(r["Transcript"], r["Label"] == "1") for r in rows]


def extract(text):
    return {name for name, (_, _, pat) in FEATURES.items() if re.search(pat, text)}


def score(fired, weights):
    """RiskScorer.scoreContent 와 같은 규칙 — 그룹별 상한을 걸고 합산한다."""
    total = 0.0
    for group, cap in GROUP_CAPS.items():
        s = sum(weights[f] for f in fired if FEATURES[f][0] == group)
        total += min(cap, s)
    return BIAS + total


def derive_weights(data):
    """log(피싱 출현율 / 정상 출현율). 0건이 발산하지 않게 라플라스 보정을 넣는다."""
    ph = [t for t, y in data if y]
    nm = [t for t, y in data if not y]
    out = {}
    for name in FEATURES:
        a = sum(1 for t in ph if re.search(FEATURES[name][2], t))
        b = sum(1 for t in nm if re.search(FEATURES[name][2], t))
        pa = (a + 0.5) / (len(ph) + 1)
        pb = (b + 0.5) / (len(nm) + 1)
        # 음수 배점(정상에 더 흔한 항목)은 0으로 막는다. 없는 게 무해함의 증거는 아니다.
        out[name] = max(0.0, math.log(pa / pb))
    return out


def roc(scored):
    """임계값 후보별 재현율·오탐률. 선을 어디에 그을지는 이 표를 보고 정한다."""
    print(f"\n{'임계값':>7} {'재현율':>10} {'오탐률':>10} {'미탐':>6} {'오탐':>6}")
    print("-" * 46)
    best = None
    for th in [x * 0.5 for x in range(-8, 13)]:
        tp = sum(1 for z, y in scored if y and z >= th)
        fn = sum(1 for z, y in scored if y and z < th)
        fp = sum(1 for z, y in scored if not y and z >= th)
        tn = sum(1 for z, y in scored if not y and z < th)
        rec = tp / (tp + fn) if tp + fn else 0
        fpr = fp / (fp + tn) if fp + tn else 0
        mark = ""
        # 미탐은 돈이 나가고 오탐은 짜증에 그치므로 재현율을 우선하되 오탐 5%를 상한으로 둔다.
        if fpr <= 0.05 and (best is None or rec > best[1]):
            best, mark = (th, rec), "  ←"
        print(f"{th:7.1f} {rec*100:9.1f}% {fpr*100:9.1f}% {fn:6} {fp:6}{mark}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--derive", action="store_true", help="배점을 데이터에서 산출")
    parser.add_argument("--speaker-only", action="store_true", help="가장 많이 말한 화자의 발화만 사용")
    args = parser.parse_args()

    data = load(speaker_only=args.speaker_only)
    ph = sum(1 for _, y in data if y)
    print(f"전체 {len(data)}건 (피싱 {ph} / 정상 {len(data)-ph})")

    weights = {n: w for n, (_, w, _) in FEATURES.items()}
    if args.derive:
        weights = derive_weights(data)
        print("\n산출 배점 (log 우도비, 라플라스 보정)")
        print("-" * 46)
        for n, w in sorted(weights.items(), key=lambda kv: -kv[1]):
            print(f"  {n:24} {w:5.2f}   (시드 {FEATURES[n][1]:.1f})")

    scored = [(score(extract(t), weights), y) for t, y in data]
    covered = sum(1 for t, y in data if y and extract(t))
    print(f"\n피싱 {ph}건 중 항목이 하나라도 잡힌 건: {covered}건 ({covered/ph*100:.0f}%)")

    best = roc(scored)
    if best:
        print(f"\n권장 임계값 {best[0]:.1f} — 재현율 {best[1]*100:.1f}% (오탐률 5% 이하 조건)")
