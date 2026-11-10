
import csv
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from serial_utils import (
    show_ports,
    connect_arduino,
    read_sensor_data
)


# ==========================================
# FSR 4개 자세별 데이터 수집 및 라벨링
# 센서 통신 오류 감지 및 수집 상태 확인
# ==========================================

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_FILE = DATA_DIR / "posture_labeled.csv"

# 자세별 수집 시간 (초)
COLLECTION_SECONDS = 60

# 자세를 잡기 위한 준비 시간 (초)
PREPARE_SECONDS = 5

# 데이터 미수신 경고 기준 (초)
NO_DATA_WARNING_SECONDS = 3

POSTURE_LABELS = {
    "1": ("correct", "정자세"),
    "2": ("forward", "앞으로 숙인 자세"),
    "3": ("left", "왼쪽으로 기울어진 자세"),
    "4": ("right", "오른쪽으로 기울어진 자세")
}

FIELDNAMES = [
    "timestamp",
    "session_id",
    "fsr1",
    "fsr2",
    "fsr3",
    "fsr4",
    "total",
    "posture_label"
]


def collect_posture(arduino, label, name):
    """선택한 자세의 센서 데이터를 수집"""

    print(f"\n선택한 자세: {name}")
    print(f"{PREPARE_SECONDS}초 후 수집을 시작합니다.")

    for remaining in range(PREPARE_SECONDS, 0, -1):
        print(f"{remaining}초...")
        time.sleep(1)

    # 이전 수신 데이터 제거
    arduino.reset_input_buffer()

    session_id = uuid4().hex[:12]
    count = 0

    start_time = time.monotonic()
    last_data_time = start_time
    last_progress_time = start_time
    warning_printed = False

    print(f"\n[{name}] 데이터 수집 중...")

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        if file.tell() == 0:
            writer.writeheader()

        while time.monotonic() - start_time < COLLECTION_SECONDS:

            data = read_sensor_data(arduino)
            current_time = time.monotonic()

            # 센서 데이터가 없는 경우
            if data is None:

                no_data_seconds = current_time - last_data_time

                if (
                    no_data_seconds >= NO_DATA_WARNING_SECONDS
                    and not warning_printed
                ):
                    print("\n[경고] 센서 데이터 수신 중단!")
                    print(
                        f"{no_data_seconds:.1f}초 동안 "
                        "데이터가 수신되지 않았습니다."
                    )
                    print(
                        "Arduino 연결 및 "
                        "시리얼 통신을 확인하세요."
                    )

                    warning_printed = True

                continue

            # 정상 데이터 수신
            last_data_time = current_time

            if warning_printed:
                print("\n[복구] 센서 데이터 수신 재개!")
                warning_printed = False

            row = {
                "timestamp": datetime.now().isoformat(
                    timespec="milliseconds"
                ),
                "session_id": session_id,
                "fsr1": data["fsr1"],
                "fsr2": data["fsr2"],
                "fsr3": data["fsr3"],
                "fsr4": data["fsr4"],
                "total": data["total"],
                "posture_label": label
            }

            writer.writerow(row)
            count += 1

            # 5초마다 수집 진행 상황 표시
            if current_time - last_progress_time >= 5:
                elapsed = current_time - start_time

                print(
                    f"[진행] {elapsed:.0f}/{COLLECTION_SECONDS}초 "
                    f"| 저장된 데이터: {count}개"
                )

                last_progress_time = current_time

        file.flush()

    # 수집 결과 확인
    print("\n" + "=" * 35)

    if count == 0:
        print("[수집 실패]")
        print("센서 데이터가 저장되지 않았습니다.")
        print("이번 세션은 학습 데이터로 사용할 수 없습니다.")

    else:
        print("[수집 완료]")
        print(f"저장된 데이터: {count}개")
        print(f"자세 라벨: {label}")
        print(f"수집 세션: {session_id}")

        if count < 40:
            print(
                "[주의] 수집 데이터가 적습니다. "
                "센서 통신 상태를 확인하세요."
            )

    print("=" * 35)

    return count


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 45)
    print("AI 자세 학습용 데이터 수집 프로그램")
    print("=" * 45)

    show_ports()

    port = input(
        "\nArduino COM 포트 입력 (예: COM5): "
    ).strip()

    if not port:
        print("COM 포트가 입력되지 않았습니다.")
        return

    arduino = connect_arduino(port)

    if arduino is None:
        return

    try:
        while True:
            print("\n" + "=" * 35)
            print("수집할 자세를 선택하세요.")
            print("1. 정자세")
            print("2. 앞으로 숙인 자세")
            print("3. 왼쪽으로 기울어진 자세")
            print("4. 오른쪽으로 기울어진 자세")
            print("0. 종료")

            choice = input("\n번호 입력: ").strip()

            if choice == "0":
                print("데이터 수집 프로그램을 종료합니다.")
                break

            if choice not in POSTURE_LABELS:
                print("올바른 번호를 입력하세요.")
                continue

            label, name = POSTURE_LABELS[choice]

            count = collect_posture(
                arduino,
                label,
                name
            )

            if count == 0:
                print(
                    "\n[안내] 데이터가 수집되지 않았습니다."
                )
                print(
                    "연속으로 수집하지 말고 "
                    "센서 통신을 먼저 확인하세요."
                )

    except KeyboardInterrupt:
        print("\n사용자가 수집을 중단했습니다.")

    except Exception as error:
        print(f"\n[오류] {error}")

    finally:
        if arduino.is_open:
            arduino.close()

        print("Arduino 연결을 종료했습니다.")


if __name__ == "__main__":
    main()
