# ==========================================
# AI 자세 인식 시스템 - 피드백 로직
#
# 구부정·왼쪽·오른쪽 구분 없이 연속 5회 감지
# → 피드백 요청 후 횟수 초기화
#
# 정상·압력 없음·알 수 없는 자세에서는 초기화
# ==========================================

from dataclasses import dataclass


BAD_POSTURES = {"forward", "left", "right"}


@dataclass
class FeedbackState:
    current_label: str = "unknown"
    consecutive_count: int = 0
    stable_since: float = 0.0


class FeedbackController:
    def __init__(self, consecutive_threshold=5):
        self.consecutive_threshold = consecutive_threshold
        self.state = FeedbackState()

    def update(self, label, now):
        # 정상·압력 없음·알 수 없는 자세이면 초기화
        if label not in BAD_POSTURES:
            self.state.current_label = label
            self.state.consecutive_count = 0
            self.state.stable_since = now

            return {
                "label": label,
                "consecutive_count": 0,
                "posture_duration_seconds": 0.0,
                "alert": False,
                "feedback_command": False,
                "actuator_enabled": False,
            }

        # 잘못된 자세 종류와 관계없이 횟수 누적
        self.state.consecutive_count += 1

        # 현재 자세 유지 시간은 자세가 바뀌면 초기화
        if label != self.state.current_label:
            self.state.current_label = label
            self.state.stable_since = now

        duration = max(
            0.0,
            now - self.state.stable_since,
        )

        # 설정된 횟수에 도달하면 피드백 요청
        feedback_command = (
            self.state.consecutive_count
            >= self.consecutive_threshold
        )

        if feedback_command:
            self.state.consecutive_count = 0

        return {
            "label": label,
            "consecutive_count": self.state.consecutive_count,
            "posture_duration_seconds": round(duration, 1),
            "alert": feedback_command,
            "feedback_command": feedback_command,
            # 실제 명령 전송은 main.py에서 처리
            "actuator_enabled": False,
        }