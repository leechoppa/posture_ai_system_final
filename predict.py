# ==========================================
# AI 자세 인식 시스템 - 실시간 AI 예측
# ==========================================

from pathlib import Path

import joblib

from preprocess import (
    FEATURE_COLUMNS,
    prepare_prediction_data
)


MODEL_FILE = (
    Path(__file__).resolve().parent
    / "model"
    / "posture_model.pkl"
)


class PosturePredictor:

    def __init__(self, model_path=MODEL_FILE):

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"AI 모델이 없습니다: {model_path}\n"
                "python train_model.py를 먼저 실행하세요."
            )

        # 직접 학습한 신뢰할 수 있는 모델만 불러오기
        payload = joblib.load(model_path)

        self.model = payload["model"]

        self.labels = payload["posture_labels"]

        self.feature_columns = list(
            payload["feature_columns"]
        )

        # 학습과 예측의 특징값 일치 확인
        if self.feature_columns != FEATURE_COLUMNS:
            raise ValueError(
                "AI 모델과 전처리 특징값이 다릅니다.\n"
                "python train_model.py로 재학습하세요."
            )

        self.validation_accuracy = payload.get(
            "validation_accuracy"
        )

    def predict(self, sensor_data):

        X = prepare_prediction_data(
            sensor_data
        )[self.feature_columns]

        # 자세 예측
        label = str(
            self.model.predict(X)[0]
        )

        # 자세별 예측 확률
        probabilities = dict(
            zip(
                self.model.classes_,
                map(
                    float,
                    self.model.predict_proba(X)[0]
                )
            )
        )

        confidence = probabilities[label]

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities
        }