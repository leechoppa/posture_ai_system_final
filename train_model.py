# ==========================================
# AI 자세 인식 시스템 - 최종 AI 학습
# Random Forest + 세션 균형
# ==========================================

from pathlib import Path
from datetime import datetime

import json
import shutil
import joblib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    GridSearchCV
)

from preprocess import (
    FEATURE_COLUMNS,
    extract_features
)


BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR / "data" / "posture_labeled.csv"
)

MODEL_DIR = BASE_DIR / "model"

MODEL_FILE = (
    MODEL_DIR / "posture_model_candidate.pkl"
)

LABELS = [
    "correct",
    "forward",
    "left",
    "right"
]

SEED = 42


def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"CSV 없음: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        encoding="utf-8-sig"
    )

    df.columns = df.columns.str.strip()

    needed = [
        "session_id",
        "posture_label",
        "fsr1",
        "fsr2",
        "fsr3",
        "fsr4"
    ]

    missing = [
        c for c in needed
        if c not in df
    ]

    if missing:
        raise ValueError(
            f"필수 열 누락: {missing}"
        )

    df["posture_label"] = (
        df["posture_label"]
        .astype("string")
        .str.strip()
    )

    df["session_id"] = (
        df["session_id"]
        .astype("string")
        .str.strip()
    )

    df = df[
        df["posture_label"].isin(LABELS)
        & df["session_id"].notna()
        & (df["session_id"] != "")
    ].copy()

    df = extract_features(df)

    if (
        df.empty
        or set(df["posture_label"]) != set(LABELS)
    ):
        raise ValueError(
            "4개 자세의 유효한 데이터가 필요합니다."
        )

    # 한 세션에 여러 자세 라벨이 섞였는지 확인
    mixed = df.groupby(
        "session_id"
    )["posture_label"].nunique()

    if (mixed > 1).any():
        raise ValueError(
            "한 세션에 여러 자세 라벨이 존재합니다: "
            + str(mixed[mixed > 1].index.tolist())
        )

    return df


def find_holdout(X, y, groups):

    splitter = GroupShuffleSplit(
        n_splits=100,
        test_size=0.2,
        random_state=SEED
    )

    for train_idx, test_idx in splitter.split(
        X, y, groups
    ):

        train_labels = set(
            y.iloc[train_idx]
        )

        test_labels = set(
            y.iloc[test_idx]
        )

        if (
            train_labels == set(LABELS)
            and test_labels == set(LABELS)
        ):
            return train_idx, test_idx

    raise ValueError(
        "모든 자세를 포함하는 평가 세션 분할 실패"
    )


def session_weights(groups):

    # 각 수집 세션의 총 가중치를 동일하게 설정
    sizes = groups.map(
        groups.value_counts()
    ).astype(float)

    weights = (
        1.0 / sizes
    ).to_numpy()

    return weights * (
        len(weights) / weights.sum()
    )


def train_model():

    df = load_data()

    print("=" * 52)
    print("FSR 4개 Random Forest 최종 AI 학습")
    print("=" * 52)

    print("전체 데이터:", len(df))
    print(
        "수집 세션:",
        df["session_id"].nunique()
    )
    print(
        "특징값:",
        len(FEATURE_COLUMNS)
    )

    print("\n[자세별 데이터 및 세션]")
    print(
        df.groupby("posture_label").agg(
            rows=("session_id", "size"),
            sessions=("session_id", "nunique")
        ).to_string()
    )

    X = df[FEATURE_COLUMNS]
    y = df["posture_label"]
    groups = df["session_id"]

    # 학습 / 평가 데이터 분리
    train_idx, test_idx = find_holdout(
        X, y, groups
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    print("\n[학습 / 평가 데이터]")
    print(
        f"학습: {len(X_train)}개 "
        f"({groups_train.nunique()}세션)"
    )

    print(
        f"평가: {len(X_test)}개 "
        f"({groups_test.nunique()}세션)"
    )

    print(
        "평가 세션:",
        ", ".join(
            groups_test.unique().tolist()
        )
    )

    # 세션 단위 교차검증
    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED
    )

    # 모델 최적화 설정
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [8, 12, None],
        "min_samples_leaf": [2, 5],
        "class_weight": [
            None,
            "balanced"
        ]
    }

    model = RandomForestClassifier(
        random_state=SEED,
        n_jobs=1
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        error_score="raise",
        refit=True,
        verbose=1
    )

    weights = session_weights(
        groups_train
    )

    print("\nAI 모델 최적화 진행 중...")

    search.fit(
        X_train,
        y_train,
        groups=groups_train,
        sample_weight=weights
    )

    # 최종 평가
    y_pred = search.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            y_pred
        )
    )

    print("\n[최적 모델 설정]")
    print(search.best_params_)

    print(
        "교차검증 균형 정확도: "
        f"{search.best_score_ * 100:.2f}%"
    )

    print("\n[최종 평가 결과]")
    print(
        f"정확도: {accuracy * 100:.2f}%"
    )

    print(
        "균형 정확도: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print("\n[자세별 평가 결과]")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=LABELS,
            zero_division=0
        )
    )

    
    # ==========================================
    # 세션별 오분류 분석
    # ==========================================

    session_results = pd.DataFrame({
        "session_id": groups_test.to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": y_pred
    })

    session_results["correct_prediction"] = (
        session_results["actual"]
        == session_results["predicted"]
    )

    print("\n[세션별 AI 정확도]")

    session_summary = (
        session_results
        .groupby(["actual", "session_id"])
        .agg(
            samples=("correct_prediction", "size"),
            correct=("correct_prediction", "sum"),
            accuracy=("correct_prediction", "mean")
        )
        .reset_index()
    )

    session_summary["accuracy"] = (
        session_summary["accuracy"] * 100
    ).round(2)

    print(session_summary.to_string(index=False))

    print("\n[세션별 오분류 상세]")

    for (actual, session_id), group in (
        session_results.groupby(["actual", "session_id"])
    ):
        print(f"\n자세: {actual}")
        print(f"세션: {session_id}")

        counts = (
            group["predicted"]
            .value_counts()
            .reindex(LABELS, fill_value=0)
        )

        print(counts.to_string())

    # CSV 파일로 분석 결과 저장
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    session_summary.to_csv(
        MODEL_DIR / "session_accuracy.csv",
        index=False,
        encoding="utf-8-sig"
    )

    session_results.to_csv(
        MODEL_DIR / "session_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n세션별 분석 결과 저장 완료!")


    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 기존 모델 백업
    if MODEL_FILE.exists():

        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        backup = MODEL_DIR / (
            f"posture_model_backup_{stamp}.pkl"
        )

        shutil.copy2(
            MODEL_FILE,
            backup
        )

        print(
            "\n기존 모델 백업:",
            backup
        )

    # 혼동행렬 저장
    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=LABELS,
        display_labels=LABELS,
        cmap="Blues",
        ax=ax,
        xticks_rotation=45
    )

    fig.tight_layout()

    fig.savefig(
        MODEL_DIR / "confusion_matrix.png",
        dpi=150
    )

    plt.close(fig)

    # 모델 저장
    model_data = {
        "model": search.best_estimator_,
        "feature_columns": list(FEATURE_COLUMNS),
        "posture_labels": LABELS,
        "sensor_count": 4,
        "sensor_layout": [
            ["fsr1", "fsr4"],
            ["fsr2", "fsr3"]
        ],
        "validation_accuracy": float(accuracy),
        "validation_balanced_accuracy": float(
            balanced_accuracy
        ),
        "cv_balanced_accuracy": float(
            search.best_score_
        ),
        "model_version": 2
    }

    joblib.dump(
        model_data,
        MODEL_FILE
    )

    # 평가 결과 저장
    metrics = {
        "rows": int(len(df)),
        "sessions": int(groups.nunique()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_session_ids": (
            groups_test.unique().tolist()
        ),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "cv_balanced_accuracy": float(
            search.best_score_
        ),
        "best_params": search.best_params_
    }

    metrics_file = (
        MODEL_DIR / "metrics.json"
    )

    metrics_file.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nAI 모델 학습 완료!")
    print("모델 저장:", MODEL_FILE)
    print("평가 결과 저장:", metrics_file)

    print(
        "\n주의: 정확도 검증 전에는 "
        "액추에이터 자동 동작을 켜지 마세요."
    )


if __name__ == "__main__":

    try:
        train_model()

    except (
        ValueError,
        FileNotFoundError
    ) as error:

        print(
            "학습 중단:",
            error
        )