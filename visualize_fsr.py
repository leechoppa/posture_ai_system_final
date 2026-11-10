
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# 1. 파일 경로 설정
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "posture_labeled.csv"
OUTPUT_DIR = BASE_DIR / "data" / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. CSV 파일 불러오기
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

# 기존 CSV에 헤더가 없는 경우에도 읽을 수 있도록 처리
required = ["fsr1", "fsr2", "fsr3", "fsr4"]


if not all(col in df.columns for col in required):
    df = pd.read_csv(CSV_PATH, header=None)

    if df.shape[1] == 8:
        df.columns = [
            "timestamp",
            "session_id",
            "fsr1",
            "fsr2",
            "fsr3",
            "fsr4",
            "total",
            "posture_label"
        ]

    elif df.shape[1] == 6:
        df.columns = [
            "timestamp",
            "fsr1",
            "fsr2",
            "fsr3",
            "fsr4",
            "total"
        ]

    elif df.shape[1] == 5:
        df.columns = [
            "fsr1",
            "fsr2",
            "fsr3",
            "fsr4",
            "total"
        ]

    else:
        raise ValueError(
            "CSV 열 구성을 확인해 주세요. "
            f"현재 열 개수: {df.shape[1]}"
        )


# 3. 센서값 숫자 변환
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required).copy()

if df.empty:
    raise ValueError("그래프로 표시할 센서 데이터가 없습니다.")

# 4. 좌우 및 상하 압력 계산
# 상단: FSR1(왼쪽), FSR4(오른쪽)
# 하단: FSR2(왼쪽), FSR3(오른쪽)

df["left"] = df["fsr1"] + df["fsr2"]
df["right"] = df["fsr4"] + df["fsr3"]

df["top"] = df["fsr1"] + df["fsr4"]
df["bottom"] = df["fsr2"] + df["fsr3"]

# 데이터 순서 (측정 샘플 번호)
x = range(len(df))

# 5. FSR 센서 4개 그래프
plt.figure(figsize=(12, 5))

for col in required:
    plt.plot(x, df[col], label=col.upper())

plt.title("FSR Sensor Values")
plt.xlabel("Sample Number")
plt.ylabel("Sensor Value (0-1023)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "fsr_4_sensors.png", dpi=150)
plt.close()

# 6. 왼쪽 / 오른쪽 압력 비교
plt.figure(figsize=(12, 5))

plt.plot(x, df["left"], label="Left (FSR1+FSR2)")
plt.plot(x, df["right"], label="Right (FSR4+FSR3)")

plt.title("Left vs Right Pressure")
plt.xlabel("Sample Number")
plt.ylabel("Sum of Sensor Values")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "left_right_pressure.png", dpi=150)
plt.close()

# 7. 위쪽 / 아래쪽 압력 비교
plt.figure(figsize=(12, 5))

plt.plot(x, df["top"], label="Top (FSR1+FSR4)")
plt.plot(x, df["bottom"], label="Bottom (FSR2+FSR3)")

plt.title("Top vs Bottom Pressure")
plt.xlabel("Sample Number")
plt.ylabel("Sum of Sensor Values")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "top_bottom_pressure.png", dpi=150)
plt.close()

print("FSR 데이터 시각화 완료!")
print(f"사용한 데이터 개수: {len(df)}")
print(f"그래프 저장 위치: {OUTPUT_DIR}")

print("\n생성된 그래프:")
print("1. fsr_4_sensors.png")
print("2. left_right_pressure.png")
print("3. top_bottom_pressure.png")
