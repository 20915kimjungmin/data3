import streamlit as st
import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

st.title("기온 예측기 및 모델 평가")


@st.cache_data
def load_yearly():
    df = pd.read_csv(DATA_URL)
    df["연도"] = pd.to_datetime(df["날짜"]).dt.year
    grouped = df.groupby("연도")["평균기온"].agg(["mean", "count"]).reset_index()
    valid = (grouped["연도"] <= 2025) & (grouped["count"] >= 300)
    return grouped[valid].rename(columns={"mean": "연평균기온"})


# 데이터 로드
yearly = load_yearly()

# 1. 데이터셋 분할 (학습용 및 테스트용)
train_50 = yearly[(yearly["연도"] >= 1956) & (yearly["연도"] <= 2005)]
train_100 = yearly[(yearly["연도"] >= 1906) & (yearly["연도"] <= 2005)]
test_20 = yearly[(yearly["연도"] >= 2006) & (yearly["연도"] <= 2025)]


# 2. 모델 학습 및 평가 함수
def evaluate_model(train_df, test_df, name):
    # 선형 회귀 (a: 기울기, b: 절편)
    a, b = np.polyfit(train_df["연도"], train_df["연평균기온"], 1)

    # 테스트 데이터 예측
    y_true = test_df["연평균기온"].values
    y_pred = a * test_df["연도"].values + b

    # MAE, MSE, R2 수동 계산
    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors**2)

    ss_res = np.sum(errors**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        "모델": name,
        "학습 기간": f"{train_df['연도'].min()}~{train_df['연df'].max()}"
        if "연df" in locals()
        else f"{train_df['연도'].min()}~{train_df['연도'].max()}",
        "기울기(℃/년)": round(a, 4),
        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "R²": round(r2, 3),
        "a": a,
        "b": b,
    }


# 각 모델 계산
res_full = evaluate_model(yearly, test_20, "전체 데이터")
res_50 = evaluate_model(train_50, test_20, "최근 50년 (1956~2005)")
res_100 = evaluate_model(train_100, test_20, "최근 100년 (1906~2005)")

# 3. 평가 결과 표 출력
st.subheader("📌 최근 20년(2006~2025) 공통 테스트 평가")
eval_df = pd.DataFrame([res_full, res_50, res_100])
st
