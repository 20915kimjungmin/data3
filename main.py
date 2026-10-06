import numpy as np
import pandas as pd
import streamlit as st

URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

st.title("기온 예측기 및 모델 평가")


@st.cache_data
def load_data():
    df = pd.read_csv(URL)
    df["연도"] = pd.to_datetime(df["날짜"]).dt.year
    g = df.groupby("연도")["평균기온"].agg(["mean", "count"]).reset_index()
    v = (g["연도"] <= 2025) & (g["count"] >= 300)
    return g[v].rename(columns={"mean": "연평균기온"})


yearly = load_data()

if not yearly.empty:
    t50 = yearly[(yearly["연도"] >= 1956) & (yearly["연도"] <= 2005)]
    t100 = yearly[(yearly["연도"] >= 1906) & (yearly["연도"] <= 2005)]
    test = yearly[(yearly["연도"] >= 2006) & (yearly["연도"] <= 2025)]

    def run_model(tr, te, name):
        a, b = np.polyfit(tr["연도"], tr["연평균기온"], 1)
        yt = te["연평균기온"].values
        yp = a * te["연도"].values + b
        err = yt - yp
        mae = float(np.mean(np.abs(err)))
        mse = float(np.mean(err**2))
        s_res = float(np.sum(err**2))
        s_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = float(1 - (s_res / s_tot)) if s_tot != 0 else 0.0
        return {
            "모델": name,
            "학습기간": str(int(tr["연도"].min()))
            + "~"
            + str(int(tr["연도"].max())),
            "기울기": round(float(a), 4),
            "MAE": round(mae, 3),
            "MSE": round(mse, 3),
            "R2": round(r2, 3),
            "a": float(a),
            "b": float(b),
        }

    m_all = run_model(yearly, test, "전체")
    m_50 = run_model(t50, test, "최근50년")
    m_100 = run_model(t100, test, "최근100년")

    st.subheader("평가 지표")
    res_df = pd.DataFrame([m_all, m_50, m_100])
    st.dataframe(
        res_df[["모델", "학습기간", "기울기", "MAE", "MSE", "R2"]],
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("회귀선 비교")
    c_df = pd.DataFrame({"연도": yearly["연도"], "실제기온": yearly["연평균기온"]})
    c_df["전체"] = m_all["a"] * yearly["연도"] + m_all["b"]
    c_df["50년"] = m_50["a"] * yearly["연도"] + m_50["b"]
    c_df["100년"] = m_100["a"] * yearly["연도"] + m_100["b"]
    st.line_chart(c_df, x="연도", use_container_width=True)

    st.subheader("예측기")
    yr = st.slider("연도 선택", 1900, 2100, 2045)
    v_all = round(m_all["a"] * yr + m_all["b"], 1)
    v_50 = round(m_50["a"] * yr + m_50["b"], 1)
    v_100 = round(m_100["a"] * yr + m_100["b"], 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("전체 모델", str(v_all) + " 도")
    c2.metric("50년 모델", str(v_50) + " 도")
    c3.metric("100년 모델", str(v_100) + " 도")
    
