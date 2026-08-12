#!/usr/bin/env python3
"""KorCCViD 전사본을 내려받고, 앱 시뮬레이션용 표본을 만든다.

데이터 자체는 저장소에 커밋하지 않는다(재배포 불가). 이 스크립트가 커밋되는 쪽이고,
받는 사람이 같은 표본을 다시 만들 수 있어야 측정을 재현할 수 있다.

    python3 fetch.py            # 전사본 받고 표본 생성
    python3 fetch.py --push     # 생성 후 연결된 기기로 밀어넣기

표본이 균형(피싱 N + 정상 N)인 이유는 항목별 출현율을 양쪽에서 같은 크기로 세야
log(피싱 출현율 / 정상 출현율)이 한쪽 표본 크기에 휘둘리지 않기 때문이다.
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

ORIGIN = "https://github.com/selfcontrol7/Korean_Voice_Phishing_Detection.git"

# 스크립트는 src/ 안에 있고 데이터·산출물 폴더는 저장소 루트에 있다.
ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "korccvi_sample.json"
REPO = ROOT / "repo"
TRANSCRIPTS = REPO / "Multimodal" / "data" / "transcripts"

# 기기 내 경로. 앱이 내부 저장소를 먼저 보고 없으면 외부를 본다
# (TranscriptReplayer.sampleFile 참고). 내부 쪽이 adb로 확실히 들어간다.
PACKAGE = "com.ava.proto"
DEVICE_TMP = "/data/local/tmp/korccvi_sample.json"

# 아주 긴 전사본은 분류 API의 분당 토큰 한도를 혼자 다 써버려서 뺀다.
MIN_LEN, MAX_LEN = 200, 1500


def clone() -> None:
    """전사본 1,417건을 받는다. 학습·평가가 전부 이 폴더를 본다.

    저장소 전체는 190MB인데 그중 우리가 쓰는 건 전사본 82MB뿐이다. 나머지는 음성
    멀티모달 연구용 세그먼트 매니페스트라 오디오 없이는 쓸 데가 없다. 그래서 blob:none
    부분 클론에 sparse-checkout을 걸어 필요한 폴더만 내려받는다.

    `--no-cone`을 쓰는 이유 — 기본값인 cone 모드는 지정한 폴더의 **상위 폴더에 있는 파일**도
    함께 받는다. `Multimodal/data/` 바로 아래에 95MB짜리 매니페스트들이 있어서, cone 모드로는
    전혀 줄지 않는다.
    """
    if TRANSCRIPTS.is_dir():
        print(f"이미 있음: {TRANSCRIPTS.relative_to(ROOT)}")
        return
    print("전사본 받는 중 (약 82MB)")
    subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", ORIGIN, str(REPO)],
                   check=True)
    subprocess.run(["git", "-C", str(REPO), "sparse-checkout", "set", "--no-cone",
                    "/Multimodal/data/transcripts/**"], check=True)
    n = len(list(TRANSCRIPTS.glob("*/*.json")))
    print(f"전사본 {n}건 → {TRANSCRIPTS.relative_to(ROOT)}")


def build_sample(per_class: int) -> None:
    """앱 시뮬레이션용 표본. **학습·평가와 같은 전사본에서 뽑는다.**

    원본 id(`vishing_38` 등)를 그대로 실어 보내는 것이 중요하다. 그래야 기기에서 나온
    판정을 `evaluate.py` 결과와 같은 id로 대조할 수 있다. 임의 번호를 붙이면 앱이 틀렸는지
    모델이 틀렸는지 가릴 수가 없다.
    """
    def pick(kind, label):
        out = []
        for path in sorted((TRANSCRIPTS / kind).glob("*.json")):
            text = json.loads(path.read_text(encoding="utf-8")).get("text", "").strip()
            if MIN_LEN <= len(text) <= MAX_LEN:
                out.append({"id": path.stem, "label": label, "text": text})
        # 파일명 순으로 앞에서 자르면 특정 시기 녹취에 쏠린다. 뽑는 자리는 고정한다.
        random.Random(20260811).shuffle(out)
        return out

    phishing, normal = pick("vishing", 1), pick("non_vishing", 0)
    items = phishing[:per_class] + normal[:per_class]

    SAMPLE.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"샘플 생성: 피싱 {min(per_class, len(phishing))}건 + "
          f"정상 {min(per_class, len(normal))}건 → {SAMPLE}")


def push() -> None:
    subprocess.run(["adb", "push", str(SAMPLE), DEVICE_TMP], check=True)
    subprocess.run(
        ["adb", "shell", f"run-as {PACKAGE} sh -c "
         f"'cat {DEVICE_TMP} > /data/data/{PACKAGE}/files/korccvi_sample.json'"],
        check=True,
    )
    print("기기 전송 완료 — 앱의 시뮬레이션 탭에서 '전사본 리플레이'를 실행하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=25, help="클래스당 표본 수")
    parser.add_argument("--push", action="store_true", help="생성 후 기기로 전송")
    args = parser.parse_args()

    clone()
    build_sample(args.per_class)
    if args.push:
        push()
