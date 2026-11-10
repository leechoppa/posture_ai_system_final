"""AI 자세 인식 프로젝트 | Streamlit 대시보드.
실행: streamlit run app.py
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = ROOT / "data" / "posture_log_v2.csv"  #웹 대시보드에 연결할 CSV 파일명
LABELS = {"correct": "정상 자세", "forward": "구부정한 자세", "left": "왼쪽 기울임", "right": "오른쪽 기울임", "no_pressure": "압력 없음"}
COLORS = {"correct": "#34c59a", "forward": "#efb45d", "left": "#8c9bff", "right": "#f18f9c", "no_pressure": "#9ca3af"}
SENSOR_NAMES = ["fsr1", "fsr2", "fsr3", "fsr4"]
px.defaults.template = "plotly_white"

st.set_page_config(page_title="ALIGN | AI 자세 관리", page_icon="🪑", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800&display=swap');
:root { --navy:#172B4D; --teal:#159E8C; --muted:#708199; }
html, body, [data-testid="stApp"] {font-family: Pretendard, 'Malgun Gothic', sans-serif;}
[data-testid="stAppViewContainer"] {background:#F4F7FC;}
[data-testid="stSidebar"] {background:#FFFFFF;border-right:1px solid #E6EBF3;}
.block-container {padding-top:1.4rem;padding-bottom:2rem;max-width:1480px;}
h1,h2,h3 {color:#1D304B;letter-spacing:-.035em;}
h3 {font-size:1.25rem!important;}
[data-testid="stMetric"] {background:#fff;border:1px solid #E4EBF5;border-radius:18px;
 padding:19px 21px;box-shadow:0 4px 16px rgba(23,43,77,.035);}
[data-testid="stMetricLabel"] {color:#718098;font-size:.9rem;}
[data-testid="stMetricValue"] {color:#1C3050;letter-spacing:-.04em;}
[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {border-radius:18px;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-color:#E5EBF3!important;border-radius:18px!important;background:#fff;}
.stTabs [data-baseweb="tab-list"] {gap:10px;flex-wrap:wrap;border-bottom:1px solid #E5EAF3;}
.stTabs [data-baseweb="tab"] {border-radius:10px 10px 0 0;padding:10px 15px;font-weight:600;}
.hero {background:linear-gradient(110deg,#142849,#28477B 70%,#33618C);border-radius:22px;
 padding:32px 36px;margin-bottom:14px;box-shadow:0 10px 26px rgba(29,57,101,.09);}
.hero h1 {color:#fff;margin:0;font-size:2.1rem;line-height:1.5;}
.hero p {color:#D9E7F7;margin:8px 0 0;font-size:1rem;}
.hero-eyebrow {color:#9ADCCF;font-size:.75rem;font-weight:800;letter-spacing:.17em;margin-bottom:3px;}
.section-caption {color:#73839A;font-size:.9rem;margin-bottom:12px;}
[data-testid="stAlert"] {border-radius:12px;}
@media (max-width: 720px) {.hero {padding:20px 23px}.hero h1 {font-size:1.6rem;}}
</style>
""", unsafe_allow_html=True)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """수집기/기존 CSV의 기본 컬럼 및 자세 이름 정규화."""
    if df.empty:
        return pd.DataFrame(columns=["timestamp", *SENSOR_NAMES, "posture", "feedback", "feedback_event", "confidence"])
    df = df.copy()
    aliases = {"time": "timestamp", "datetime": "timestamp", "predicted_posture": "posture", "prediction": "posture", "label": "posture", "posture_label": "posture", "fsr_left": "fsr1", "fsr_center": "fsr2", "fsr_right": "fsr3", "fsr_back": "fsr4"}
    for old, new in aliases.items():
        if old in df and new not in df:
            df = df.rename(columns={old: new})
    if "timestamp" not in df:
        df["timestamp"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="s")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    for c in SENSOR_NAMES:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "posture" not in df:
        df["posture"] = "unknown"
    df["posture"] = df["posture"].astype(str).str.strip().str.lower()
    df["posture"] = df["posture"].replace({"normal": "correct", "good": "correct", "slouch": "forward", "정상": "correct", "구부정": "forward", "좌측": "left", "우측": "right"})
    # main.py의 feedback_command는 0/1 정수로 저장됨.
    # 기존 serial_bridge.py의 feedback 문자열 형식도 함께 지원.
    if "feedback_command" in df:
        df["feedback_event"] = pd.to_numeric(df["feedback_command"], errors="coerce").fillna(0).gt(0)
        df["feedback"] = np.where(df["feedback_event"], "요청 발생", "")
    else:
        if "feedback" not in df:
            df["feedback"] = ""
        df["feedback"] = df["feedback"].fillna("").astype(str)
        df["feedback_event"] = (~df["feedback"].str.strip().str.lower().isin(["", "0", "false", "none", "nan"]))
    if "posture_duration_seconds" in df:
        df["posture_duration_seconds"] = pd.to_numeric(df["posture_duration_seconds"], errors="coerce")
    if "confidence" not in df:
        df["confidence"] = np.nan
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=2, show_spinner=False)
def read_csv(path: str, mtime_ns: int) -> pd.DataFrame:
    try:
        return normalize(pd.read_csv(path, on_bad_lines="skip"))
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return normalize(pd.DataFrame())


@st.cache_data(show_spinner=False)
def demo_data(day: str) -> pd.DataFrame:
    """화면 미리보기용 합성 샘플. 실제 측정 데이터 아님."""
    rng = np.random.default_rng(2026)
    times = pd.date_range(end=pd.Timestamp(day) + pd.Timedelta(hours=15), periods=240, freq="15s")
    pattern = ["correct"] * 32 + ["forward"] * 8 + ["correct"] * 20 + ["left"] * 7 + ["correct"] * 18 + ["right"] * 6
    postures = (pattern * 4)[:len(times)]
    baselines = {"correct": [510, 525, 505, 520], "forward": [335, 385, 340, 400], "left": [700, 560, 260, 390], "right": [255, 390, 700, 565]}
    rows = []
    for t, posture in zip(times, postures):
        values = np.clip(rng.normal(baselines[posture], 25), 0, 1023).round().astype(int)
        rows.append({"timestamp": t, **dict(zip(SENSOR_NAMES, values)), "posture": posture, "confidence": round(float(rng.uniform(.83, .98)), 3), "feedback": "ACTUATOR" if posture != "correct" and rng.random() < .055 else ""})
    return normalize(pd.DataFrame(rows))


def duration_estimate(df: pd.DataFrame) -> tuple[dict, float]:
    """일정 간격 로그의 관측 시간 추정 (큰 수집 공백은 제외)."""
    if df.empty:
        return {}, 0.0
    gaps = df["timestamp"].diff().dt.total_seconds().dropna()
    positive = gaps[gaps > 0]
    cadence = float(positive.median()) if len(positive) else 1.0
    cadence = min(max(cadence, .1), 60.0)
    delta = (df["timestamp"].shift(-1) - df["timestamp"]).dt.total_seconds()
    weights = delta.where((delta > 0) & (delta <= max(3 * cadence, 10)), 0.0).fillna(0.0)
    if len(df) == 1:
        weights.iloc[0] = 0.0
    durations = weights.groupby(df["posture"]).sum().to_dict()
    return durations, float(weights.sum())


def fmt_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds >= 3600:
        return f"{seconds // 3600}시간 {(seconds % 3600) // 60}분"
    if seconds >= 60:
        return f"{seconds // 60}분 {seconds % 60}초"
    return f"{seconds}초"



BAD_POSTURES = ("forward", "left", "right")
KNOWN_POSTURES = ("correct", *BAD_POSTURES)


def analyze_posture(df: pd.DataFrame):
    """수집 공백을 구분하고 시간별 관측 시간을 추정한다.

    불균형 발생: 정상/다른 자세/압력 없음에서 잘못된 자세로 전환한 구간 1회.
    같은 자세가 계속 기록되어도 발생 횟수를 늘리지 않는다.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ordered = df.sort_values("timestamp").reset_index(drop=True).copy()
    gaps = ordered["timestamp"].diff().dt.total_seconds()
    positive = gaps[(gaps > 0) & gaps.notna()]
    cadence = float(positive.median()) if not positive.empty else 1.0
    cadence = min(max(cadence, 0.1), 60.0)
    max_gap = max(10.0, 3.0 * cadence)

    # 간격이 큰 경우 별도 측정 세션으로 간주
    session_break = gaps.isna() | (gaps <= 0) | (gaps > max_gap)
    prior = ordered["posture"].shift()
    is_event = (ordered["posture"].isin(BAD_POSTURES)
                & (session_break | ordered["posture"].ne(prior)))
    events = ordered.loc[is_event, ["timestamp", "posture"]].copy()
    events["hour"] = events["timestamp"].dt.floor("h")

    # 각 측정 행과 바로 다음 행 사이의 시간만 관측 시간으로 인정한다.
    # 다음 행이 없거나 너무 멀면 시간을 추정해 추가하지 않는다.
    next_time = ordered["timestamp"].shift(-1)
    gaps_forward = (next_time - ordered["timestamp"]).dt.total_seconds()
    valid = gaps_forward.gt(0) & gaps_forward.le(max_gap)
    spans = []
    for ts, end, posture, ok in zip(ordered["timestamp"], next_time,
                                     ordered["posture"], valid):
        if not ok or posture not in KNOWN_POSTURES:
            continue
        # 시각 경계를 넘는 측정 구간은 해당 시간대별로 분리한다.
        cursor = ts
        while cursor < end:
            hour = cursor.floor("h")
            boundary = hour + pd.Timedelta(hours=1)
            stop = min(boundary, end)
            spans.append({"hour": hour, "posture": posture,
                          "seconds": (stop - cursor).total_seconds()})
            cursor = stop

    observed = pd.DataFrame(spans, columns=["hour", "posture", "seconds"])
    if observed.empty:
        hourly = pd.DataFrame(columns=["hour", "correct_seconds", "bad_seconds", "observed_seconds", "correct_rate"])
    else:
        wide = observed.pivot_table(index="hour", columns="posture", values="seconds",
                                    aggfunc="sum", fill_value=0)
        for name in KNOWN_POSTURES:
            if name not in wide.columns:
                wide[name] = 0.0
        hourly = pd.DataFrame({"hour": wide.index,
                               "correct_seconds": wide["correct"].to_numpy(),
                               "bad_seconds": wide[list(BAD_POSTURES)].sum(axis=1).to_numpy()})
        hourly["observed_seconds"] = hourly["correct_seconds"] + hourly["bad_seconds"]
        hourly["correct_rate"] = 100 * hourly["correct_seconds"] / hourly["observed_seconds"]
        hourly = hourly.sort_values("hour").reset_index(drop=True)
    return events, observed, hourly


def render_extended_analysis(df: pd.DataFrame):
    """자세 불균형 빈도 분석만 표시한다."""
    st.markdown("### 01 · 자세 불균형 빈도")
    st.caption("연속된 같은 불균형 자세는 하나의 발생 구간으로 계산합니다.")
    events, _, _ = analyze_posture(df)
    counts = pd.DataFrame({"posture": list(BAD_POSTURES),
                           "count": [int((events["posture"] == k).sum()) if not events.empty else 0 for k in BAD_POSTURES]})
    counts["자세"] = counts["posture"].map(LABELS)
    x1, x2 = st.columns([1.15, 1], gap="large")
    with x1:
        fig = px.bar(counts, x="자세", y="count", color="자세", text="count",
                     color_discrete_map={LABELS[k]: COLORS[k] for k in BAD_POSTURES})
        fig.update_layout(showlegend=False, height=310, plot_bgcolor="white",
                          yaxis_title="발생 구간 수 (회)", xaxis_title=None)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with x2:
        st.metric("불균형 발생 구간", f"{int(counts['count'].sum())}회")
        if counts["count"].sum():
            common = counts.loc[counts["count"].idxmax(), "자세"]
            st.metric("가장 많이 발생한 자세", common)
        else:
            st.info("현재 집계된 불균형 발생이 없습니다.")
        st.caption("연속된 같은 불균형 자세는 1회로 집계하며, 자세가 바뀌거나 수집 공백이 발생하면 새 구간으로 판단합니다. 순간적 AI 오분류도 포함될 수 있습니다.")

with st.sidebar:
    st.markdown("## 🪑 ALIGN")
    st.caption("AI PERSONAL POSTURE ANALYTICS")
    st.markdown("---")
    st.markdown("**데이터 연결**")
    st.divider()
    mode = st.radio("데이터 모드", ["실제 측정 데이터", "데모 데이터"], index=0, help="실제 데이터는 별도 실행되는 센서 수집기가 CSV에 기록합니다.")
    log_path = st.text_input("실시간 로그 CSV 경로", str(DEFAULT_LOG), disabled=mode == "데모 데이터")
    auto = st.toggle("자동 새로고침 (3초)", value=True)
    st.divider()
    st.markdown("**자세 상태 안내**")
    for key, label in LABELS.items():
        st.markdown(f"<span style='color:{COLORS[key]}'>●</span> {label}", unsafe_allow_html=True)
    st.divider()
    st.caption("Arduino UNO · FSR × 4 · Random Forest")
    st.caption("CSV 조회는 하드웨어 연결 없이 사용할 수 있습니다.")


def render_dashboard():
    if mode == "데모 데이터":
        df = demo_data(datetime.now().date().isoformat())
        source = "DEMO · 예시 데이터"
    else:
        path = Path(log_path).expanduser()
        if not path.exists():
            st.warning(f"로그 파일이 없습니다: {path}\n\n`serial_bridge.py`를 실행하거나 기존 Python AI 코드에서 CSV를 기록하세요.")
            df = normalize(pd.DataFrame())
        else:
            try:
                df = read_csv(str(path), path.stat().st_mtime_ns)
            except Exception as exc:
                st.error(f"CSV를 읽는 중 오류가 발생했습니다: {exc}")
                df = normalize(pd.DataFrame())
        source = "실제 CSV 기록"

    st.markdown("<div class='hero'><div class='hero-eyebrow'>ALIGN · POSTURE INSIGHTS</div><h1>AI 자세 관리 대시보드</h1><p>실시간 자세 인식부터 자세별 통계와 불균형 분석까지 한눈에 확인하세요.</p></div>", unsafe_allow_html=True)
    left, right = st.columns([3, 1])
    with left:
        st.caption(f"● 데이터: {source}   ·   화면 갱신: {datetime.now():%Y-%m-%d %H:%M:%S}")
    with right:
        if st.button("↻ 지금 새로고침", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    if df.empty:
        st.info("표시할 데이터가 없습니다. 사이드바에서 데모 모드를 선택하여 화면을 미리 확인할 수 있습니다.")
        return

    durations, total = duration_estimate(df)
    latest = df.iloc[-1]
    current_key = latest["posture"]
    current_name = LABELS.get(current_key, "미분류")
    classified_total = sum(durations.get(k, 0.0) for k in ("correct", "forward", "left", "right"))
    correct_pct = 100 * durations.get("correct", 0.0) / classified_total if classified_total else 0.0
    feedback_count = int(df["feedback_event"].sum()) if "feedback_event" in df else 0
    # main.py 로그는 현재 자세 유지 시간을 직접 기록하므로 해당 값을 우선 사용.
    if "posture_duration_seconds" in df and pd.notna(latest["posture_duration_seconds"]):
        hold = max(0.0, float(latest["posture_duration_seconds"]))
    elif len(df) > 1:
        same = int(df["posture"].eq(current_key).iloc[::-1].cumprod().sum())
        recent_same = df.iloc[-same:]
        gaps = recent_same["timestamp"].diff().dt.total_seconds().dropna()
        # 오래된 수집 구간을 하나의 유지 시간으로 연결하지 않음.
        if not gaps.empty and (gaps > 10).any():
            recent_same = recent_same.iloc[int(np.flatnonzero((gaps > 10).to_numpy())[-1]) + 1:]
        hold = max(0.0, (recent_same.iloc[-1]["timestamp"] - recent_same.iloc[0]["timestamp"]).total_seconds())
    else:
        hold = 0.0

    st.markdown("#### 현재 자세 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("마지막 인식 자세", current_name)
    c2.metric("마지막 기록 자세 유지", fmt_time(hold))
    c3.metric("정상 자세 비율 (분류 관측 시간)", f"{correct_pct:.1f}%")
    c4.metric("피드백 요청 기록", f"{feedback_count}회")

    if mode == "실제 측정 데이터":
        age = (datetime.now() - latest["timestamp"].to_pydatetime()).total_seconds()
        if 0 <= age <= 10:
            st.success("● 최근 10초 이내에 센서 로그가 기록되었습니다.")
        else:
            st.warning(f"마지막 기록: {latest['timestamp']:%Y-%m-%d %H:%M:%S} · 현재 실시간 측정이 중단되었을 수 있습니다.")
    else:
        st.info("이 화면은 예시 데이터입니다. 실제 정확도, 실제 사용자 자세, 실시간 센서 측정값을 나타내지 않습니다.", icon="ℹ️")

    st.markdown("#### 자세 데이터 상세보기")
    tab1, tab2, tab_extra, tab3 = st.tabs(["📡 실시간 모니터링", "📊 자세별 통계", "📈 자세 불균형 분석", "📁 측정 기록"])

    with tab1:
        a, b = st.columns([1.05, 1.65], gap="large")
        with a:
            with st.container(border=True):
                st.subheader("현재 자세 상태")
                color = COLORS.get(current_key, "#8290a2")
                st.markdown(f"<div style='padding:20px;background:{color}18;border-radius:15px;border-left:6px solid {color}'><div style='font-size:13px;color:#66758b'>AI 분류 결과</div><div style='font-size:29px;font-weight:800;color:#17273d'>{current_name}</div><div style='font-size:13px;color:#7d889a;margin-top:5px'>기록 시각 {latest['timestamp']:%H:%M:%S}</div></div>", unsafe_allow_html=True)
                if pd.notna(latest["confidence"]):
                    val = float(latest["confidence"])
                    st.caption(f"모델 예측 확률(최대 클래스): {val * 100:.1f}%" if 0 <= val <= 1 else "예측 확률: 유효하지 않은 값")
                st.caption("예측 확률은 검증 데이터에서 측정한 모델 정확도와 다릅니다. 피드백은 실제 작동이 아닌 요청 기록입니다.")
            with st.container(border=True):
                st.subheader("센서별 압력 값")
                sensor = pd.DataFrame({"센서": ["FSR 1", "FSR 2", "FSR 3", "FSR 4"], "측정값": [latest[c] for c in SENSOR_NAMES]})
                fig = px.bar(sensor, x="센서", y="측정값", text="측정값", color="센서", color_discrete_sequence=["#6a85db", "#7a9df0", "#95ade6", "#b0c0ee"])
                fig.update_layout(showlegend=False, height=265, margin=dict(l=8, r=8, t=15, b=0), plot_bgcolor="white", yaxis_title="ADC 원시값", xaxis_title=None, yaxis_range=[0, 300])
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.caption("센서 순서는 회로에 연결된 FSR1~4 기준입니다. ADC 0~1023 값은 직접적인 힘(N) 측정값이 아닙니다.")
        with b:
            with st.container(border=True):
                st.subheader("압력 변화 추이")
                recent = df.tail(120).copy()
                pressure = recent.melt(id_vars="timestamp", value_vars=SENSOR_NAMES, var_name="센서", value_name="ADC 값")
                fig = px.line(pressure, x="timestamp", y="ADC 값", color="센서", color_discrete_sequence=["#5371d0", "#7f9ae0", "#34b9a1", "#efa76c"])
                fig.update_layout(height=340, margin=dict(l=8, r=8, t=18, b=0), plot_bgcolor="white", legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0), xaxis_title=None, yaxis_range=[0, 300])
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            with st.container(border=True):
                st.subheader("최근 자세 판별")
                recent_labels = df[["timestamp", "posture"]].tail(12).iloc[::-1].copy()
                recent_labels["자세"] = recent_labels["posture"].map(LABELS).fillna("미분류")
                recent_labels["시간"] = recent_labels["timestamp"].dt.strftime("%H:%M:%S")
                st.dataframe(recent_labels[["시간", "자세"]], hide_index=True, width="stretch", height=230)

    with tab2:
        a, b = st.columns(2, gap="large")
        with a:
            with st.container(border=True):
                st.subheader("자세별 관측 시간 비율 (압력 없음 제외)")
                counts = pd.DataFrame([{"자세": LABELS[k], "초": durations.get(k, 0), "색": COLORS[k]} for k in LABELS if k != "no_pressure"])
                if counts["초"].sum() > 0:
                    fig = px.pie(counts, names="자세", values="초", hole=.65, color="자세", color_discrete_map={LABELS[k]: COLORS[k] for k in LABELS})
                    fig.update_traces(textinfo="percent", hovertemplate="%{label}: %{value:.1f}초 (%{percent})<extra></extra>")
                    fig.update_layout(height=325, margin=dict(l=5, r=5, t=10, b=0), legend=dict(orientation="h", y=-.08))
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                else:
                    st.info("관측 시간 계산에 필요한 연속 기록이 없습니다.")
                st.caption(f"연속 로그로 추정한 분류 자세 관측 시간: {fmt_time(classified_total)} (압력 없음·수집 공백 제외)")
        with b:
            with st.container(border=True):
                st.subheader("자세별 유지 시간")
                duration_df = pd.DataFrame([{"자세": LABELS[k], "분": durations.get(k, 0) / 60} for k in LABELS if k != "no_pressure"])
                fig = px.bar(duration_df, x="자세", y="분", color="자세", color_discrete_map={LABELS[k]: COLORS[k] for k in LABELS})
                fig.update_layout(showlegend=False, height=325, margin=dict(l=5, r=5, t=10, b=0), plot_bgcolor="white", xaxis_title=None, yaxis_title="분")
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.caption("데이터 간격을 이용한 추정치이며, 의료적 평가를 의미하지 않습니다.")
        with st.container(border=True):
            st.subheader("자세 변화 타임라인")
            plot_df = df.tail(500).copy()
            plot_df["자세"] = plot_df["posture"].map(LABELS).fillna("미분류")
            order = list(LABELS.values()) + ["미분류"]
            fig = px.scatter(plot_df, x="timestamp", y="자세", color="자세", category_orders={"자세": order}, color_discrete_map={LABELS[k]: COLORS[k] for k in LABELS}, opacity=.8)
            fig.update_traces(marker_size=8)
            fig.update_layout(height=265, showlegend=False, margin=dict(l=5, r=5, t=12, b=0), plot_bgcolor="white", xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with tab_extra:
        render_extended_analysis(df)

    with tab3:
        st.subheader("센서 및 AI 판단 기록")
        show = df.iloc[::-1].copy()
        show["자세"] = show["posture"].map(LABELS).fillna(show["posture"])
        show = show.rename(columns={"timestamp": "기록 시간", "fsr1": "FSR1", "fsr2": "FSR2", "fsr3": "FSR3", "fsr4": "FSR4", "feedback": "피드백", "confidence": "예측 확률"})
        st.dataframe(show[["기록 시간", "자세", "FSR1", "FSR2", "FSR3", "FSR4", "예측 확률", "피드백"]], width="stretch", hide_index=True, height=365)
        st.download_button("⬇ CSV 다운로드", data=df.to_csv(index=False).encode("utf-8-sig"), file_name=f"posture_history_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")
        st.caption(f"현재 선택한 CSV 경로: `{log_path}`" if mode == "실제 측정 데이터" else "데모 데이터는 실제 측정 기록이 아닙니다.")


if auto:
    @st.fragment(run_every="3s")
    def live_fragment():
        render_dashboard()
    live_fragment()
else:
    render_dashboard()
