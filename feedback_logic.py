
# ==========================================
# AI 자세 인식 시스템 - 최종 피드백 로직
#
# 동일한 잘못된 자세 연속 3회 감지
# -> 피드백 요청
# -> 감지 횟수 0으로 초기화
# -> 다시 1회부터 누적
#
# 정상 자세 감지 시 즉시 초기화
# ==========================================

from dataclasses import dataclass


BAD_POSTURES = {
    "forward",
    "left",
    "right"
}


@dataclass
class FeedbackState:

    current_label: str = "unknown"

    consecutive_count: int = 0

    stable_since: float = 0.0


class FeedbackController:

    def __init__(
        self,
        consecutive_threshold=3
    ):

        self.consecutive_threshold = (
            consecutive_threshold
        )

        self.state = FeedbackState()

    def update(self, label, now):

        # ==================================
        # 1. 정상 자세 감지
        # ==================================

        if label == "correct":

            self.state.current_label = "correct"

            self.state.consecutive_count = 0

            self.state.stable_since = now

            return {
                "label": label,
                "consecutive_count": 0,
                "posture_duration_seconds": 0.0,
                "alert": False,
                "feedback_command": False,
                "actuator_enabled": False
            }

        # ==================================
        # 2. 알 수 없는 자세
        # ==================================

        if label not in BAD_POSTURES:

            self.state.current_label = "unknown"

            self.state.consecutive_count = 0

            self.state.stable_since = now

            return {
                "label": label,
                "consecutive_count": 0,
                "posture_duration_seconds": 0.0,
                "alert": False,
                "feedback_command": False,
                "actuator_enabled": False
            }

        # ==================================
        # 3. 잘못된 자세 연속 감지
        # ==================================

        if label == self.state.current_label:

            self.state.consecutive_count += 1

        else:

            self.state.current_label = label

            self.state.consecutive_count = 1

            self.state.stable_since = now

        duration = max(
            0.0,
            now - self.state.stable_since
        )

        feedback_command = False

        # ==================================
        # 4. 연속 3회 감지 시 피드백 요청
        # ==================================

        if (
            self.state.consecutive_count
            >= self.consecutive_threshold
        ):

            feedback_command = True

            # 피드백 요청 후 즉시 횟수 초기화
            self.state.consecutive_count = 0

        # ==================================
        # 5. 결과 반환
        # ==================================

        return {
            "label": label,

            "consecutive_count": (
                self.state.consecutive_count
            ),

            "posture_duration_seconds": round(
                duration, 1
            ),

            "alert": feedback_command,

            "feedback_command": feedback_command,

            # 실제 액추에이터 구동은 안전 검증 후 연동
            "actuator_enabled": False
        }
