import streamlit as st
import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

st.set_page_config(page_title="서울 연평균기온 선형회귀 평가", layout="wide")
st.title("🌡️ 서울 연평균기온 선형회귀 모델 평가 및 비교")


# 데이터 로드
@st.cache_data
def load_yearly():
    try:
        df = pd.read_csv(DATA_URL)
        df["연도"] = pd.to_datetime(df["날짜"]).dt.year
        grouped = (
            df.groupby("연도")["평균기온"].agg(["mean", "count"]).reset_index()
        )
        valid = (grouped["연도"] <= 2025) & (grouped["count"] >= 300)
        return grouped[valid].rename(columns={"mean": "연평균기온"})
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()


yearly = load_yearly()

if not yearly.empty:
    # 1. 데이터셋 분할 (과거 학습용 / 최근 20년 공통 테스트용)
    train_50 = yearly[
        (yearly["연도"] >= 1956) & (yearly["연도"] <= 2005)
    ].copy()
    train_100 = yearly[
        (yearly["연도"] >= 1906) & (yearly["연도"] <= 2005)
    ].copy()
    test_20 = yearly[(yearly["연도"] >= 2006) & (yearly["연도"] <= 2025)].copy()

    # 2. 평가 지표 계산 함수 (오류 방지를 위해 식을 명확히 분리)
    def calc_metrics(y_true, y_pred):
        yt = np.array(y_true)
        yp = np.array(y_pred)

        errors = yt - yp
        mae =
        
