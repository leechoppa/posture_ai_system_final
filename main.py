# AI 기반 개인 맞춤 자세 인식 시스템
# Arduino UNO / FSR 압력센서 4개 / COM5 기본값

import argparse
import csv
import json
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
STATUS_FILE = BASE_DIR / "data" / "system_status.json"

STATUS_INTERVAL = 1.0
FEEDBACK_TIMEOUT = 20.0

LOG_COLUMNS = [
    "timestamp",
    "fsr1",
    "fsr2",
    "fsr3",
    "fsr4",
    "total",
    "posture_label",
    "confidence",
    "posture_duration_seconds",
    "consecutive_count",
    "feedback_command",
    "alert",
]

CONSECUTIVE_THRESHOLD = 5
NO_PRESSURE_THRESHOLD = 10

# True: Arduino에 F 명령 전송
# False: 센서 확인만 진행하며 F 명령은 전송하지 않음
SEND_TEST_COMMAND = True

FEEDBACK_MESSAGE = (
    "5초 돌출 → 2초 정지 → 5초 복귀 → 3초 정지"
)


def write_system_status(state, feedback_active, message):
    """대시보드에서 읽을 시스템 상태를 저장한다."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state": state,
        "feedback_active": feedback_active,
        "message": message,
        "updated_at": datetime.now().isoformat(
            timespec="milliseconds"
        ),
    }

    # 작성 중인 JSON을 웹이 읽지 않도록
    # 임시 파일을 완성한 다음 교체한다.
    temporary = STATUS_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(STATUS_FILE)


def send_feedback_command(ser):
    if not SEND_TEST_COMMAND:
        return False

    ser.write(b"F")
    ser.flush()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="AI 자세 인식 시스템"
    )
    parser.add_argument(
        "--port",
        default="COM5",
        help="Arduino 포트 (기본 COM5)",
    )
    args = parser.parse_args()

    write_system_status(
        "STARTING",
        False,
        "시스템 연결 준비 중",
    )

    predictor = PosturePredictor()
    controller = FeedbackController(
        consecutive_threshold=CONSECUTIVE_THRESHOLD
    )

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
        with LOG_FILE.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as existing:
            existing_header = next(csv.reader(existing), [])

        if existing_header != LOG_COLUMNS:
            raise ValueError(
                "기존 CSV 로그의 열 구성이 다릅니다. "
                "로그 파일을 백업하거나 이름을 변경하세요."
            )

    write_header = (
        not LOG_FILE.exists()
        or LOG_FILE.stat().st_size == 0
    )

    print("=" * 80)
    print("AI 기반 개인 맞춤 자세 인식 시스템")
    print("=" * 80)
    print(f"Arduino 포트: {args.port}")
    print("측정 간격: Arduino의 SENSOR_INTERVAL 설정에 따름")
    print("압력 없음: TOTAL 0~10")
    print("피드백 조건: 잘못된 자세 연속 5회")
    print(f"피드백 순서: {FEEDBACK_MESSAGE}")
    print(
        "F 피드백 요청 전송: "
        f"{'활성화' if SEND_TEST_COMMAND else '비활성화'}"
    )
    print("실제 모터 동작: Arduino 설정에 따름")
    print(f"로그: {LOG_FILE}")
    print(f"웹 상태 파일: {STATUS_FILE}")
    print("종료: Ctrl+C")
    print("연결 가능한 포트:", available_ports())

    with connect_arduino(args.port) as ser, LOG_FILE.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=LOG_COLUMNS,
        )

        if write_header:
            writer.writeheader()
            file.flush()

        last_data = time.monotonic()
        last_status_write = 0.0

        missing_data_warned = False
        feedback_in_progress = False
        feedback_started = 0.0

        exit_state = "STOPPED"
        exit_message = "프로그램 종료 — 자세 인식 중단"

        try:
            while True:
                event = read_arduino_event(ser)
                now = time.monotonic()

                # 피드백 중 센서 데이터가 없어도
                # 웹 상태 파일은 계속 갱신한다.
                if now - last_status_write >= STATUS_INTERVAL:
                    if feedback_in_progress:
                        state = "FEEDBACK"
                        message = FEEDBACK_MESSAGE

                    elif now - last_data >= 5:
                        state = "NO_DATA"
                        message = "센서 데이터 수신 대기 중"

                    else:
                        state = "MONITORING"
                        message = "자세 인식 중"

                    write_system_status(
                        state,
                        feedback_in_progress,
                        message,
                    )
                    last_status_write = now

                # 완료 신호 없이 임의로 인식을 재개하지 않는다.
                if (
                    feedback_in_progress
                    and now - feedback_started > FEEDBACK_TIMEOUT
                ):
                    ser.write(b"S")
                    ser.flush()

                    raise RuntimeError(
                        "피드백 완료 신호 미수신: "
                        "정지 명령을 전송했습니다. "
                        "Arduino를 확인하세요."
                    )

                # Arduino 상태 메시지 처리
                if event is not None and "status" in event:
                    status = event["status"]

                    if (
                        status == "COMPLETE"
                        and feedback_in_progress
                    ):
                        feedback_in_progress = False

                        controller = FeedbackController(
                            consecutive_threshold=(
                                CONSECUTIVE_THRESHOLD
                            )
                        )

                        last_data = now
                        missing_data_warned = False

                        write_system_status(
                            "MONITORING",
                            False,
                            "피드백 완료 — 자세 인식 재개",
                        )

                    elif status in {
                        "FEEDBACK_RECEIVED_MOTION_LOCKED",
                        "MOTION_LOCKED",
                        "STOPPED",
                        "EMERGENCY_STOP",
                    } and feedback_in_progress:
                        raise RuntimeError(
                            "피드백 실행 중단 또는 거부: "
                            f"{status}"
                        )

                    elif (
                        status == "READY"
                        and feedback_in_progress
                    ):
                        raise RuntimeError(
                            "피드백 중 Arduino 재시작 "
                            "또는 상태 초기화 감지"
                        )

                    continue

                # 피드백 중에는 자세 예측,
                # 측정 CSV 저장 및 센서값 출력을 하지 않는다.
                if feedback_in_progress:
                    continue

                sample = event

                if sample is None:
                    if (
                        now - last_data >= 5
                        and not missing_data_warned
                    ):
                        print(
                            "[경고] 5초 이상 센서 데이터 없음"
                        )
                        missing_data_warned = True

                        controller = FeedbackController(
                            consecutive_threshold=(
                                CONSECUTIVE_THRESHOLD
                            )
                        )

                    continue

                last_data = now
                missing_data_warned = False

                total_pressure = sum(
                    sample[f"fsr{i}"]
                    for i in range(1, 5)
                )
                sample["total"] = total_pressure

                if (
                    0 <= total_pressure
                    <= NO_PRESSURE_THRESHOLD
                ):
                    posture_label = "no_pressure"
                    confidence = 0.0

                else:
                    result = predictor.predict(sample)
                    posture_label = result["label"]
                    confidence = result["confidence"]

                feedback = controller.update(
                    posture_label,
                    now,
                )

                consecutive_count = feedback[
                    "consecutive_count"
                ]
                feedback_requested = bool(
                    feedback["feedback_command"]
                )
                posture_duration = feedback[
                    "posture_duration_seconds"
                ]

                sent = False

                if feedback_requested:
                    sent = send_feedback_command(ser)

                    if sent:
                        feedback_in_progress = True
                        feedback_started = time.monotonic()

                        write_system_status(
                            "FEEDBACK",
                            True,
                            FEEDBACK_MESSAGE,
                        )

                timestamp = datetime.now().isoformat(
                    timespec="milliseconds"
                )

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

                # 피드백을 발생시킨 측정 행까지 저장한다.
                writer.writerow(row)
                file.flush()

                if sent:
                    # 피드백 한 주기당 한 번만 표시한다.
                    print(
                        "피드백 동작 중 — 자세 인식 일시 중지"
                        f" | {FEEDBACK_MESSAGE}"
                    )
                    continue

                sensor_text = ", ".join(
                    str(sample[f"fsr{i}"])
                    for i in range(1, 5)
                )

                print(
                    f"{timestamp[11:23]} | "
                    f"FSR: {sensor_text} | "
                    f"합계: {total_pressure} | "
                    f"예측: {posture_label} | "
                    f"신뢰도: {confidence:.1%} | "
                    f"연속: {consecutive_count}/"
                    f"{CONSECUTIVE_THRESHOLD} | "
                    f"F전송: {int(sent)}"
                )

        except Exception as error:
            exit_state = "ERROR"
            exit_message = str(error)
            raise

        finally:
            # 웹 상태 저장보다 정지 명령을 먼저 시도한다.
            try:
                ser.write(b"S")
                ser.flush()
                print("[안전] Arduino 정지 명령 S 전송")
            except Exception:
                pass

            try:
                write_system_status(
                    exit_state,
                    False,
                    exit_message,
                )
            except Exception as error:
                print(
                    f"[경고] 종료 상태 저장 실패: {error}"
                )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n실시간 자세 모니터링 종료")

    except Exception as error:
        print(f"\n[실행 오류] {error}")

        try:
            write_system_status(
                "ERROR",
                False,
                str(error),
            )
        except Exception:
            pass