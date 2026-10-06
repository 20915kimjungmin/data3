import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

st.set_page_config(page_title="서울 연평균기온 선형회귀 평가", layout="wide")
st.title("🌡️ 서울 연평균기온 선형회귀 모델 평가 및 비교")

@st.cache_data
def load_yearly():
    df = pd.read_csv(DATA_URL)
    df["연도"] = pd.to_datetime(df["날짜"]).dt.year
    grouped = df.groupby("연도")["평균기온"].agg(["mean", "count"]).reset_index()
    valid = (grouped["연도"] <= 2025) & (grouped["count"] >= 300)
    return grouped[valid].rename(columns={"mean": "연평균기온"})

yearly = load_yearly()

# 1. 데이터셋 분할 (학습용 / 테스트용)
train_50 = yearly[(yearly["연도"] >= 1956) & (yearly["연도"] <= 2005)].copy()
train_100 = yearly[(yearly["연도"] >= 1906) & (yearly["연도"] <= 2005)].copy()
test_20 = yearly[(yearly["연도"] >= 2006) & (yearly["연도"] <= 2025)].copy()

# 2. 모델 학습 함수
def fit_and_evaluate(train_df, test_df, label):
    # 선형 회귀 학습 (a: 기울기, b: 절편)
    a, b = np.polyfit(train_df["연도"], train_df["연평균기온"], 1)
    
    # 테스트 데이터 예측
    y_true = test_df["연평균기온"]
    y_pred = a * test_df["연초"] if "연초" in test_df else a * test_df["연도"] + b
    
    # 성능 평가 지표 계산
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "모델": label,
        "학습 기간": f"{train_df['연도'].min()}~{train_df['연도'].max()} ({len(train_df)}년)",
        "기울기 (℃/년)": round(a, 4),
        "절편": round(b, 2),
        "MAE (℃)": round(mae, 3),
        "MSE": round(mse, 3),
        "R² Score": round(r2, 3),
        "a": a,
        "b": b
    }

# 3. 모델 구축 및 비교
res_full = fit_and_evaluate(yearly, test_20, "전체 데이터 모델")
res_50 = fit_and_evaluate(train_50, test_20, "최근 50년 모델 (1956~2005)")
res_100 = fit_and_evaluate(train_100, test_20, "최근 100년 모델 (1906~2005)")

results_df = pd.DataFrame([res_full, res_50, res_100])

# 4. 시각화 (Plotly)
fig = go.Figure()

# 전체 데이터 산점도 (학습/테스트 색상 구분)
train_all_past = yearly[yearly["연도"] <= 2005]
fig.add_trace(go.Scatter(
    x=train_all_past["연도"], y=train_all_past["연평균기온"],
    mode="markers", name="과거 관측 데이터 (~2005)", opacity=0.5, marker=dict(color="gray")
))
fig.add_trace(go.Scatter(
    x=test_20["연도"], y=test_20["연평균기온"],
    mode="markers", name="공통 테스트 데이터 (2006~2025)", marker=dict(color="red", size=8)
))

# 각 모델별 회귀선 그리기 (전체 기간 확장 표시)
x_range = np.array([yearly["연도"].min(), 2025])
colors = {"전체 데이터 모델": "green", "최근 50년 모델 (1956~2005)": "blue", "최근 100년 모델 (1906~2005)": "orange"}

for res in [res_full, res_50, res_100]:
    y_line = res["a"] * x_range + res["b"]
    fig.add_trace(go.Scatter(
        x=x_range, y=y_line, mode="lines",
        name=f"{res['모델']} (기울기: {res['기울기 (℃/년)']})",
        line=dict(color=colors[res["모델"]])
    ))

fig.update_layout(
    title="학습 기간별 회귀선과 테스트 데이터(2006~2025) 비교",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# 5. 평가 결과 수치 비교 표
st.subheader("📊 테스트 데이터(2006~2025년) 기준 예측 성능 비교")
st.dataframe(
    results_df[["모델", "학습 기간", "기울기 (℃/년)", "MAE (℃)", "MSE", "R² Score"]],
    hide_index=True,
    use_container_width=True
)

# 6. 미래 예측 인터랙티브 슬라이더
st.subheader("🔮 모델별 미래 연도 예측")
target_year = st.slider("예측할 연도를 선택하세요", 2026, 2100, 2050)

col1, col2, col3 = st.columns(3)
for col, res in zip([col1, col2, col3], [res_full, res_50, res_100]):
    pred_val = res["a"] * target_year + res["b"]
    col.metric(
        label=f"{res['모델']}",
        value=f"{pred_val:.2f} ℃",
        delta=f"기울기: {res['기울기 (℃/년)']} ℃/년"
    )
