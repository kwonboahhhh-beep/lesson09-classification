import pandas as pd
import plotly.express as px
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = (
    "https://raw.githubusercontent.com/kwonboahhhh-beep/data/"
    "refs/heads/main/stroke.csv"
)

st.set_page_config(page_title=f"탐색 - {APP_TITLE}", page_icon="🧠", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🧠 탐색")
st.caption("데이터를 여러 방향에서 살펴보는 화면입니다.")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

df = df.copy()
df["뇌졸중"] = df["stroke"].map({0: "겪지 않음", 1: "겪음"})

# ---------- 1) 히스토그램 두 개 ----------
st.header("1. 나이와 평균 혈당의 분포")
left, right = st.columns(2)
with left:
    fig = px.histogram(df, x="age", nbins=30, title="나이 분포")
    fig.update_layout(xaxis_title="나이(age)", yaxis_title="사람 수")
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.histogram(df, x="avg_glucose_level", nbins=40, title="평균 혈당 분포")
    fig.update_layout(xaxis_title="평균 혈당(avg_glucose_level)", yaxis_title="사람 수")
    st.plotly_chart(fig, use_container_width=True)

# ---------- 2) 상자그림 두 개 + 평균 표 ----------
st.header("2. 뇌졸중 여부에 따른 나이와 평균 혈당")
order = {"뇌졸중": ["겪지 않음", "겪음"]}
left, right = st.columns(2)
with left:
    fig = px.box(df, x="뇌졸중", y="age", color="뇌졸중",
                 category_orders=order, title="나이 비교")
    fig.update_layout(xaxis_title="뇌졸중 여부", yaxis_title="나이(age)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.box(df, x="뇌졸중", y="avg_glucose_level", color="뇌졸중",
                 category_orders=order, title="평균 혈당 비교")
    fig.update_layout(xaxis_title="뇌졸중 여부", yaxis_title="평균 혈당(avg_glucose_level)",
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

mean_table = (
    df.groupby("뇌졸중")[["age", "avg_glucose_level"]]
    .mean()
    .round(2)
    .reindex(["겪지 않음", "겪음"])
    .rename(columns={"age": "나이 평균", "avg_glucose_level": "평균 혈당 평균"})
    .reset_index()
    .rename(columns={"뇌졸중": "뇌졸중 여부"})
)
st.subheader("두 그룹의 평균값")
st.dataframe(mean_table, use_container_width=True, hide_index=True)

# ---------- 3) 고혈압·심장병별 뇌졸중 비율 ----------
st.header("3. 고혈압·심장병에 따른 뇌졸중 비율")


def rate_bar(column: str, title: str, label: str):
    g = df.groupby(column)["stroke"].agg(["mean", "sum", "count"]).reset_index()
    g["뇌졸중 비율(%)"] = (g["mean"] * 100).round(2)
    g[label] = g[column].map({0: "없음", 1: "있음"})
    g = g.rename(columns={"sum": "뇌졸중 사람 수", "count": "전체 사람 수"})
    fig = px.bar(
        g, x=label, y="뇌졸중 비율(%)", text="뇌졸중 비율(%)", color=label,
        category_orders={label: ["없음", "있음"]},
        hover_data=["뇌졸중 사람 수", "전체 사람 수"], title=title,
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="뇌졸중 비율(%)")
    return fig


left, right = st.columns(2)
with left:
    st.plotly_chart(rate_bar("hypertension", "고혈압 여부별 뇌졸중 비율", "고혈압"),
                    use_container_width=True)
with right:
    st.plotly_chart(rate_bar("heart_disease", "심장병 여부별 뇌졸중 비율", "심장병"),
                    use_container_width=True)

# ---------- 4) bmi가 빈 사람들 ----------
st.header("4. 체질량지수(bmi)가 비어 있는 사람들")
missing = df[df["bmi"].isna()]


def summarize(part: pd.DataFrame) -> list:
    n = len(part)
    s = int((part["stroke"] == 1).sum())
    ratio = f"{s / n * 100:.2f}%" if n else "-"
    return [f"{n:,}명", f"{s:,}명", ratio]


bmi_table = pd.DataFrame(
    {
        "항목": ["사람 수", "뇌졸중을 겪은 사람 수", "뇌졸중 비율"],
        "bmi가 빈 사람": summarize(missing),
        "전체": summarize(df),
    }
)
st.dataframe(bmi_table, use_container_width=True, hide_index=True)

# ---------- 5) 흡연 상태별 사람 수 ----------
st.header("5. 흡연 상태별 사람 수")
smoke = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태(smoking_status)")
    .reset_index(name="사람 수")
)
st.dataframe(smoke, use_container_width=True, hide_index=True)
