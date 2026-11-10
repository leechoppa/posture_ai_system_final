# ==========================================
# Arduino UNO 시리얼 통신
# FSR 압력센서 4개
# ==========================================

import time
import serial

from serial.tools import list_ports


BAUD_RATE = 9600


def available_ports():

    return [
        (p.device, p.description)
        for p in list_ports.comports()
    ]


def connect_arduino(
    port,
    baudrate=BAUD_RATE
):

    ser = serial.Serial(
        port,
        baudrate=baudrate,
        timeout=1,
        write_timeout=1
    )

    # Arduino UNO 시리얼 연결 후 재시작 대기
    time.sleep(2)

    ser.reset_input_buffer()

    return ser


def parse_sensor_line(line):

    try:

        fields = line.strip().split(",")

        if (
            len(fields) != 6
            or fields[0] != "DATA"
        ):
            return None

        nums = [
            int(v.strip())
            for v in fields[1:]
        ]

        # 센서값 범위 확인
        if any(
            not 0 <= v <= 1023
            for v in nums[:4]
        ):
            return None

        # 전체 합계 검증
        if nums[4] != sum(nums[:4]):
            return None

        return dict(
            zip(
                [
                    "fsr1",
                    "fsr2",
                    "fsr3",
                    "fsr4",
                    "total"
                ],
                nums
            )
        )

    except (
        ValueError,
        AttributeError
    ):
        return None


def read_sensor_data(ser):

    raw = ser.readline().decode(
        "utf-8",
        errors="replace"
    )

    return parse_sensor_line(raw)

# ==========================================
# 연결 가능한 Arduino COM 포트 출력
# ==========================================

def show_ports():

    ports = available_ports()

    print("\n[연결 가능한 COM 포트]")

    if not ports:
        print("연결된 시리얼 장치가 없습니다.")
        return

    for device, description in ports:
        print(f"{device} - {description}")

def read_arduino_event(ser):
    line = ser.readline().decode(
        "utf-8", errors="replace"
    ).strip()

    if line.startswith("STATUS,"):
        return {"status": line.split(",", 1)[1]}

    return parse_sensor_line(line)