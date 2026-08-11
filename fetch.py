#!/usr/bin/env python3
"""KorCCViD 전사본을 내려받아 평가용 샘플을 만든다.

데이터 자체는 저장소에 커밋하지 않는다(재배포 불가). 이 스크립트가 커밋되는 쪽이고,
받는 사람이 같은 샘플을 다시 만들 수 있어야 측정을 재현할 수 있다.

    python3 data/korccvi/fetch.py            # 받고 샘플 생성
    python3 data/korccvi/fetch.py --push     # 생성 후 연결된 기기로 밀어넣기

샘플이 균형 표본(피싱 N + 정상 N)인 이유는 항목별 출현율을 양쪽에서 같은 크기로
세야 log(피싱 출현율 / 정상 출현율)이 한쪽 표본 크기에 휘둘리지 않기 때문이다.
"""

import argparse
import csv
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/selfcontrol7/Korean_Voice_Phishing_Detection/main"
SOURCES = {
    "korccvi_v1.3.csv": f"{RAW}/Data_Collection_Preprocessing/KorCCViD_v1.3.csv",
    "vishing_raw.csv": f"{RAW}/Data_Collection_Preprocessing/df_data_vishing.csv",
}

HERE = Path(__file__).parent
SAMPLE = HERE / "korccvi_sample.json"

# 기기 내 경로. 앱이 내부 저장소를 먼저 보고 없으면 외부를 본다
# (TranscriptReplayer.sampleFile 참고). 내부 쪽이 adb로 확실히 들어간다.
PACKAGE = "com.ava.proto"
DEVICE_TMP = "/data/local/tmp/korccvi_sample.json"

# 아주 긴 전사본은 분류 API의 분당 토큰 한도를 혼자 다 써버려서 뺀다.
MIN_LEN, MAX_LEN = 200, 1500


def download() -> None:
    for name, url in SOURCES.items():
        target = HERE / name
        if target.exists():
            print(f"이미 있음: {name}")
            continue
        print(f"받는 중: {name}")
        urllib.request.urlretrieve(url, target)


def build_sample(per_class: int) -> None:
    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader((HERE / "korccvi_v1.3.csv").open(encoding="utf-8")))

    phishing = [r["Transcript"].strip() for r in rows
                if r["Label"] == "1" and MIN_LEN <= len(r["Transcript"]) <= MAX_LEN]
    normal = [r["Transcript"].strip() for r in rows if r["Label"] == "0"]

    items = [{"id": f"p{i:03d}", "label": 1, "text": t}
             for i, t in enumerate(phishing[:per_class])]
    items += [{"id": f"n{i:03d}", "label": 0, "text": t}
              for i, t in enumerate(normal[:per_class])]

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

    download()
    build_sample(args.per_class)
    if args.push:
        push()
