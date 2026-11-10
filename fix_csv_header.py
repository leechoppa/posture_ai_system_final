
import csv
from pathlib import Path

CSV_FILE = Path("data/posture_labeled.csv")

FIELDNAMES = [
    "timestamp",
    "session_id",
    "fsr1",
    "fsr2",
    "fsr3",
    "fsr4",
    "total",
    "posture_label"
]

with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))

if not rows:
    raise ValueError("CSV 파일이 비어 있습니다.")

if rows[0] == FIELDNAMES:
    print("이미 정상적인 헤더가 있습니다.")
else:
    if any(len(row) != len(FIELDNAMES) for row in rows):
        raise ValueError(
            "열 개수가 8개가 아닌 행이 있습니다. "
            "원본 파일을 확인하세요."
        )

    backup = CSV_FILE.with_name(
        "posture_labeled_backup.csv"
    )

    if backup.exists():
        raise FileExistsError(
            f"백업 파일이 이미 존재합니다: {backup}"
        )

    import shutil
    shutil.copy2(CSV_FILE, backup)

    with open(
        CSV_FILE, "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(FIELDNAMES)
        writer.writerows(rows)

    print("CSV 헤더 복구 완료!")
    print(f"복구한 데이터: {len(rows)}개")
    print(f"백업 파일: {backup}")