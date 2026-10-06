import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

st.set_page_config(page_title="서울 연평균기온 선형회귀 평가", layout="wide")
st.title("🌡️️ 서울 연평균기온 선형회귀 모델 평가 및 비교")


@st.cache_data
def load_yearly():
    df = pd.read_csv(DATA_URL)
    df["연도"] = pd.to_datetime(df["날짜"]).dt.year
    grouped = df.groupby("연도")["평균기온"].agg(["mean", "count"]).reset_index()
    valid = (grouped["연도"] <= 2025) & (grouped["count"] >= 300)
    return grouped[valid].rename(columns={"mean": "연평균기온"})


yearly = load_yearly()

# 1. 데이터셋 분할 (과거 학습용 / 최근 20년 공통 테스트용)
train_50 = yearly[(yearly["연도"] >= 1956) & (yearly["연도"] <= 2005)].copy()
train_100 = yearly[(yearly["연도"] >= 1906) & (yearly["연도"] <= 2005)].copy()
test_20 = yearly[(yearly["연도"] >= 2006) & (yearly["연도"] <= 2025)].copy()


# 2. 평가 지표 직접 계산 함수 (sklearn 미사용)
def calc_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return mae, mse, r2


# 3. 모델 학습 및 평가 함수
def fit_and_evaluate(train_df, test_df, label):
    # y = a * x + b
    a, b = np.polyfit(train_df["연도"], train_df["연평균기온"], 1)

    y_true = test_df["연평균기온"]
    y_pred = a * test_df["연도"] + b

    mae, mse, r2 = calc_metrics(y_true, y_pred)

    return {
        "모델": label,
        "학습 기간": f"{train_df['연도'].min()}~{train_df['연도'].max()} ({len
    
