# AI 기반 개인 맞춤 자세 인식 시스템
# Arduino UNO / FSR 압력센서 4개 / COM5 기본값
# AI 피드백 요청 F 전송 (Arduino 펌웨어에서 모터 구동 잠금 해제)

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from serial_utils import (
    available_ports,
    connect_arduino,
    read_arduino_event,
)
from predict import PosturePredictor
from feedback_logic import FeedbackController

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "data" / "posture_log_v2.csv"
LOG_COLUMNS = [
    "timestamp", "fsr1", "fsr2", "fsr3", "fsr4", "total",
    "posture_label", "confidence", "posture_duration_seconds",
    "consecutive_count", "feedback_command", "alert"
]
POSTURE_NAMES = {
    "correct": "정상", "forward": "구부정", "left": "왼쪽 기울임",
    "right": "오른쪽 기울임", "no_pressure": "압력 없음"
}
CONSECUTIVE_THRESHOLD = 3
NO_PRESSURE_THRESHOLD = 10
SEND_TEST_COMMAND = True
SEPARATOR = "-" * 80


def send_feedback_command(ser):
    if not SEND_TEST_COMMAND:
        return False
    ser.write(b"F")
    ser.flush()
    return True


def main():
    parser = argparse.ArgumentParser(description="AI 자세 인식 시스템")
    parser.add_argument("--port", default="COM5", help="Arduino 포트 (기본 COM5)")
    args = parser.parse_args()

    predictor = PosturePredictor()
    controller = FeedbackController(consecutive_threshold=CONSECUTIVE_THRESHOLD)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
        with LOG_FILE.open("r", encoding="utf-8-sig", newline="") as existing:
            existing_header = next(csv.reader(existing), [])
        if existing_header != LOG_COLUMNS:
            raise ValueError("기존 CSV 로그의 열 구성이 다릅니다. 로그 파일을 백업하거나 이름을 변경하세요.")

    write_header = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0

    print("=" * 80)
    print("AI 기반 개인 맞춤 자세 인식 시스템")
    print("=" * 80)
    print(f"Arduino 포트: {args.port}")
    print("압력 없음: TOTAL 0~10 / 동일한 잘못된 자세 연속 3회")
    print(f"F 피드백 요청 전송: {'활성화' if SEND_TEST_COMMAND else '비활성화'}")
    print("실제 모터 동작: Arduino 펌웨어의 잠금 설정에 따름")
    print(f"로그: {LOG_FILE} / 종료: Ctrl+C")
    print("연결 가능한 포트:", available_ports())

    with connect_arduino(args.port) as ser, LOG_FILE.open(
        "a", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
            f.flush()

        last_data = time.monotonic()
        missing_data_warned = False
        feedback_in_progress = False
        feedback_started = 0.0
        try:
            while True:
                event = read_arduino_event(ser)
                now = time.monotonic()

                # 완료 신호가 없는데 임의로 인식을 재개하지 않음.
                if (
                    feedback_in_progress
                    and now - feedback_started > 20
                ):
                    ser.write(b"S")
                    ser.flush()
                    raise RuntimeError(
                        "피드백 완료 신호 미수신: "
                        "정지 명령을 전송했습니다. Arduino를 확인하세요."
                    )

                if event is not None and "status" in event:
                    status = event["status"]

                    if status == "COMPLETE" and feedback_in_progress:
                        feedback_in_progress = False
                        controller = FeedbackController(
                            consecutive_threshold=CONSECUTIVE_THRESHOLD
                        )
                        last_data = now
                        missing_data_warned = False

                    elif status in {
                        "FEEDBACK_RECEIVED_MOTION_LOCKED",
                        "MOTION_LOCKED",
                        "STOPPED",
                        "EMERGENCY_STOP",
                    } and feedback_in_progress:
                        raise RuntimeError(
                            f"피드백 실행 중단 또는 거부: {status}"
                        )

                    elif status == "READY" and feedback_in_progress:
                        raise RuntimeError(
                            "피드백 중 Arduino 재시작 또는 상태 초기화 감지"
                        )

                    continue

                # 피드백 중에는 예측·CSV 저장·측정값 출력을 생략.
                if feedback_in_progress:
                    continue

                sample = event
                if sample is None:
                    if now - last_data >= 5 and not missing_data_warned:
                        print("[경고] 5초 이상 센서 데이터 없음")
                        missing_data_warned = True
                        controller = FeedbackController(
                            consecutive_threshold=CONSECUTIVE_THRESHOLD
                        )
                    continue

                last_data = now
                missing_data_warned = False
                total_pressure = sum(sample[f"fsr{i}"] for i in range(1, 5))
                sample["total"] = total_pressure

                if 0 <= total_pressure <= NO_PRESSURE_THRESHOLD:
                    posture_label, confidence = "no_pressure", 0.0
                else:
                    result = predictor.predict(sample)
                    posture_label = result["label"]
                    confidence = result["confidence"]

                posture_name = POSTURE_NAMES.get(posture_label, "알 수 없음")
                feedback = controller.update(posture_label, now)
                consecutive_count = feedback["consecutive_count"]
                feedback_requested = bool(feedback["feedback_command"])
                posture_duration = feedback["posture_duration_seconds"]

                sent = False
                if feedback_requested:
                    sent = send_feedback_command(ser)

                    if sent:
                        feedback_in_progress = True
                        feedback_started = time.monotonic()

                timestamp = datetime.now().isoformat(timespec="milliseconds")
                row = {
                    "timestamp": timestamp,
                    **sample,
                    "posture_label": posture_label,
                    "confidence": round(confidence, 4),
                    "posture_duration_seconds": posture_duration,
                    "consecutive_count": consecutive_count,
                    "feedback_command": int(sent),
                    "alert": int(feedback["alert"]),
                }
                writer.writerow(row)
                f.flush()
                if sent:
                    print(
                        "피드백 동작 중 "
                        "(5초 돌출 → 2초 정지 → "
                        "5초 복귀 → 3초 정지 / 자세 인식 일시 중단)"
                    )
                    continue

                sensor_text = ", ".join(
                    str(sample[f"fsr{i}"]) for i in range(1, 5)
                )

                print(
                    f"{timestamp[11:23]} | "
                    f"FSR: {sensor_text} | "
                    f"합계: {total_pressure} | "
                    f"예측: {posture_label} | "
                    f"신뢰도: {confidence:.1%} | "
                    f"연속: {consecutive_count}/{CONSECUTIVE_THRESHOLD} | "
                    f"F전송: {int(sent)}"
                )
        finally:
            try:
                ser.write(b"S")
                ser.flush()
                print("[안전] Arduino 정지 명령 S 전송")
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n실시간 자세 모니터링 종료")
    except Exception as error:
        print(f"\n[실행 오류] {error}")