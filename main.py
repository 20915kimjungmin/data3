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

    # 2. 평가 지표 계산 함수 (문법 에러 방지를 위해 단순화)
    def calc_metrics(y_true, y_pred):
        yt = np.array(y_true)
        yp = np.array(y_pred)

        diff = yt - yp
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff**2))

        ss_res = float(np.sum(diff**2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

        return mae, mse, r2

    # 3. 모델 학습 및 평가 함수
    def fit_and_evaluate(train_df, test_df, label):
        a, b = np.polyfit(train_df["연도"], train_df["연평균기온"], 1)

        y_true = test_df["연평균기온"]
        y_pred = a * test_df["연도"] + b

        mae, mse, r2 = calc_metrics(y_true, y_pred)

        min_yr = int(train_df["연도"].min())
        max_yr = int(train_df["연도"].max())
        cnt_yr = len(train_df)

        return {
            "모델": label,
            "학습 기간": f"{min_yr}~{max_yr} ({cnt_yr}년)",
            "기울기 (℃/년)": round(float(a), 4),
            "절편": round(float(b), 2),
            "MAE (℃)": round(mae, 3),
            "MSE": round(mse, 3),
            "R² Score": round(r2, 3),
            "a": float(a),
            "b": float(b),
        }

    # 모델 구축
    res_full = fit_and_evaluate(yearly, test_20, "전체 데이터 모델")
    res_50 = fit_and_evaluate(train_50, test_20, "최근 50년 모델 (1956~2005)")
    res_100 = fit_and_evaluate(train_100, test_20, "최근 100년 모델 (1906~2005)")

    results_df = pd.DataFrame([res_full, res_50, res_100])

    # 4. 차트 생성
    chart_df = pd.DataFrame({"연도": yearly["연도"]})
    chart_df["실제 연평균기온"] = yearly["연평균기온"]
    chart_df["전체 데이터 회귀선"] = res_full["a"] * yearly["연도"] + res_full["b"]
    chart_df["최근 50년 회귀선"] = res_50["a"] * yearly["연도"] + res_50["b"]
    chart_df["최근 100년 회귀선"] = res_100["a"] * yearly["연도"] + res_100["b"]

    st.subheader("📈 학습 기간별 회귀선 비교")
    st.line_chart(chart_df, x="연도", use_container_width=True)

    # 5. 테스트 데이터 평가 지표 출력
    st.subheader("📊 공통 테스트 데이터(2006~2025년) 기준 예측 성능 평가")
    st.dataframe(
        results_df[
            ["모델", "학습 기간", "기울기 (℃/년)", "MAE (℃)", "MSE", "R² Score"]
        ],
        hide_index=True,
        use_container_width=True,
    )

    # 6. 미래 연도 기온 예측 슬라이더
    st.subheader("
