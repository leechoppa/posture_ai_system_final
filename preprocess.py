# ==========================================
# AI 자세 인식 시스템 - 데이터 전처리 최종
# FSR 압력센서 4개
# ==========================================

from itertools import combinations

import numpy as np
import pandas as pd


# 실제 센서 배치
# FSR1   FSR4
# FSR2   FSR3

SENSOR_COLUMNS = [
    "fsr1", "fsr2", "fsr3", "fsr4"
]

SENSOR_PAIRS = list(
    combinations(SENSOR_COLUMNS, 2)
)

BASE_FEATURES = (
    SENSOR_COLUMNS
    + ["total"]
    + [f"{s}_ratio" for s in SENSOR_COLUMNS]
    + ["fsr_max", "fsr_min", "fsr_range", "fsr_std"]
)

PAIR_FEATURES = [
    name
    for a, b in SENSOR_PAIRS
    for name in (
        f"{a}_{b}_diff",
        f"{a}_{b}_relative_diff"
    )
]

POSITION_FEATURES = [
    "left_pressure",
    "right_pressure",
    "upper_pressure",
    "lower_pressure",
    "left_right_diff",
    "upper_lower_diff",
    "left_right_ratio_diff",
    "upper_lower_ratio_diff"
]

FEATURE_COLUMNS = (
    BASE_FEATURES
    + PAIR_FEATURES
    + POSITION_FEATURES
)


def validate_sensor_data(df):

    df = df.copy()

    missing = [
        c for c in SENSOR_COLUMNS
        if c not in df
    ]

    if missing:
        raise ValueError(
            f"필수 센서 열 없음: {missing}"
        )

    for col in SENSOR_COLUMNS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(subset=SENSOR_COLUMNS)

    valid = (
        (df[SENSOR_COLUMNS] >= 0)
        & (df[SENSOR_COLUMNS] <= 1023)
        & (df[SENSOR_COLUMNS] % 1 == 0)
    ).all(axis=1)

    df = df.loc[valid].copy()
    df = df.reset_index(drop=True)

    df["total"] = (
        df[SENSOR_COLUMNS].sum(axis=1)
    )

    return df


def extract_features(df):

    df = validate_sensor_data(df)

    safe_total = df["total"].replace(
        0, np.nan
    )

    # 센서별 압력 비율
    for s in SENSOR_COLUMNS:
        df[f"{s}_ratio"] = (
            df[s] / safe_total
        ).fillna(0.0)

    # 압력 분포
    df["fsr_max"] = (
        df[SENSOR_COLUMNS].max(axis=1)
    )

    df["fsr_min"] = (
        df[SENSOR_COLUMNS].min(axis=1)
    )

    df["fsr_range"] = (
        df["fsr_max"] - df["fsr_min"]
    )

    df["fsr_std"] = (
        df[SENSOR_COLUMNS].std(axis=1, ddof=0)
    )

    # 센서 간 압력 차이
    for a, b in SENSOR_PAIRS:

        diff = f"{a}_{b}_diff"

        df[diff] = df[a] - df[b]

        df[f"{a}_{b}_relative_diff"] = (
            df[diff] / safe_total
        ).fillna(0.0)

    # 실제 센서 배치 기반 특징값
    df["left_pressure"] = (
        df["fsr1"] + df["fsr2"]
    )

    df["right_pressure"] = (
        df["fsr4"] + df["fsr3"]
    )

    df["upper_pressure"] = (
        df["fsr1"] + df["fsr4"]
    )

    df["lower_pressure"] = (
        df["fsr2"] + df["fsr3"]
    )

    # 좌우 / 상하 압력 차이
    df["left_right_diff"] = (
        df["left_pressure"]
        - df["right_pressure"]
    )

    df["upper_lower_diff"] = (
        df["upper_pressure"]
        - df["lower_pressure"]
    )

    # 상대 압력 차이
    df["left_right_ratio_diff"] = (
        df["left_right_diff"] / safe_total
    ).fillna(0.0)

    df["upper_lower_ratio_diff"] = (
        df["upper_lower_diff"] / safe_total
    ).fillna(0.0)

    return df


def prepare_training_data(df):

    if "posture_label" not in df:
        raise ValueError(
            "posture_label 열 없음"
        )

    df = df.copy()

    df["posture_label"] = (
        df["posture_label"]
        .astype("string")
        .str.strip()
    )

    df = df.dropna(
        subset=["posture_label"]
    )

    df = df.loc[
        df["posture_label"] != ""
    ]

    df = extract_features(df)

    return (
        df[FEATURE_COLUMNS],
        df["posture_label"]
    )


def prepare_prediction_data(sensor_data):

    df = pd.DataFrame([sensor_data])

    df = extract_features(df)

    if df.empty:
        raise ValueError(
            "유효한 센서 데이터 없음"
        )

    return df[FEATURE_COLUMNS]


if __name__ == "__main__":

    sample = {
        "fsr1": 600,
        "fsr2": 650,
        "fsr3": 300,
        "fsr4": 350
    }

    result = prepare_prediction_data(sample)

    print(
        "특징값 개수:",
        len(FEATURE_COLUMNS)
    )

    print(
        result[POSITION_FEATURES].to_string(
            index=False
        )
    )