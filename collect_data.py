
import csv
import time
from datetime import datetime
from pathlib import Path

from serial_utils import show_ports, connect_arduino, read_sensor_data


# ==========================================
# FSR 압력센서 4개 데이터 수집 프로그램
# ==========================================

# CSV 저장 경로
DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_FILE = DATA_DIR / "real_fsr_log.csv"

# CSV 열 이름
FIELDNAMES = [
    "timestamp",
    "fsr1",
    "fsr2",
    "fsr3",
    "fsr4",
    "total"
]


def collect_data():
    print("=" * 45)
    print("FSR 압력센서 4개 데이터 수집 프로그램")
    print("=" * 45)

    # 1. 연결 가능한 COM 포트 확인
    print("\n[1] 연결 가능한 시리얼 포트")
    show_ports()

    # 2. Arduino COM 포트 입력
    port = input("\nArduino COM 포트 입력 (예: COM3): ").strip()

    if not port:
        print("COM 포트를 입력하지 않았습니다.")
        return

    # 3. Arduino 연결
    arduino = connect_arduino(port)

    if arduino is None:
        return

    # 4. 데이터 폴더 생성
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 기존 CSV가 있다면 이어서 저장
    write_header = not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0

    count = 0

    try:
        with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)

            if write_header:
                writer.writeheader()

            print("\n[2] 데이터 수집 시작")
            print("종료하려면 Ctrl + C를 누르세요.")
            print(f"저장 위치: {CSV_FILE}\n")

            while True:
                data = read_sensor_data(arduino)

                # 센서 데이터가 아닌 상태 메시지는 건너뛰기
                if data is None:
                    continue

                # 현재 시간 추가
                row = {
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "fsr1": data["fsr1"],
                    "fsr2": data["fsr2"],
                    "fsr3": data["fsr3"],
                    "fsr4": data["fsr4"],
                    "total": data["total"]
                }

                # CSV 저장
                writer.writerow(row)
                file.flush()

                count += 1

                print(
                    f"[{count}] "
                    f"FSR1={row['fsr1']} | "
                    f"FSR2={row['fsr2']} | "
                    f"FSR3={row['fsr3']} | "
                    f"FSR4={row['fsr4']} | "
                    f"TOTAL={row['total']}"
                )

    except KeyboardInterrupt:
        print("\n데이터 수집을 종료했습니다.")

    except OSError as error:
        print(f"\n파일 저장 오류: {error}")

    finally:
        arduino.close()
        print(f"총 저장 데이터: {count}개")


if __name__ == "__main__":
    collect_data()