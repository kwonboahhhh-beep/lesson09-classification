import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = (
    "https://raw.githubusercontent.com/kwonboahhhh-beep/data/"
    "refs/heads/main/stroke.csv"
)
RECORD_DIR = Path(__file__).parent / "내_기록"
RECORD_FILE = RECORD_DIR / "stroke_기록.json"

# 브라우저 탭 제목과 아이콘 (가장 먼저 호출해야 함)
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-size: 2.8rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- 데이터 불러오기 ----------
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


# ---------- 내 기록(json) 불러오기·저장 ----------
def load_record() -> dict:
    if RECORD_FILE.exists():
        try:
            with open(RECORD_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_record(meanings: dict, source: str) -> None:
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"meanings": meanings, "source": source},
            f,
            ensure_ascii=False,
            indent=2,
        )


def describe_values(series: pd.Series) -> str:
    """열의 값의 종류를 글로 설명한다."""
    values = series.dropna().unique()
    if series.dtype == "object" or len(values) <= 5:
        return ", ".join(sorted(str(v) for v in values))
    return f"숫자 {len(values)}가지 ({series.min():g} ~ {series.max():g})"


# ---------- 화면 ----------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("뇌졸중 데이터가 어떤 모양인지 살펴보는 첫 화면입니다.")

try:
    df = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# 1) 큰 숫자 카드 네 개
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    with st.container(border=True):
        st.metric("전체 사람 수", f"{total:,}명")
with c2:
    with st.container(border=True):
        st.metric("열 개수", f"{n_cols}개")
with c3:
    with st.container(border=True):
        st.metric("stroke가 1인 사람 수", f"{n_stroke:,}명")
with c4:
    with st.container(border=True):
        st.metric("stroke 1의 비율", f"{ratio:.2f}%")

# 저장된 기록 불러오기 (처음 한 번만)
if "record_loaded" not in st.session_state:
    record = load_record()
    saved_meanings = record.get("meanings", {})
    st.session_state["saved_meanings"] = dict(saved_meanings)
    st.session_state["saved_source"] = record.get("source", "")
    st.session_state["base_table"] = pd.DataFrame(
        {
            "열 이름": df.columns,
            "우리말 뜻": [saved_meanings.get(c, "") for c in df.columns],
            "값의 종류": [describe_values(df[c]) for c in df.columns],
            "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
        }
    )
    st.session_state["record_loaded"] = True

# 2) 열 설명 표
st.subheader("열 설명 표")
st.caption("‘우리말 뜻’ 칸을 눌러 교재를 보고 직접 적어 보세요. 적은 내용은 자동으로 저장됩니다.")

edited = st.data_editor(
    st.session_state["base_table"],
    key="column_table",
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름", width="small"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻", width="medium", help="교재를 보고 직접 채워 넣으세요."
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", width="small"),
    },
)

# 3) 처음 다섯 줄
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# 4) 데이터 출처
st.subheader("데이터 출처")
source_text = st.text_area(
    "출처를 적어 주세요 (교재에 있는 글)",
    value=st.session_state["saved_source"],
    height=120,
    key="source_text",
    placeholder="여기에 교재의 데이터 출처 글을 적어 주세요.",
)

# ---------- 바뀐 내용이 있으면 json으로 저장 ----------
new_meanings = {
    row["열 이름"]: ("" if pd.isna(row["우리말 뜻"]) else str(row["우리말 뜻"]))
    for _, row in edited.iterrows()
}
if (
    new_meanings != st.session_state["saved_meanings"]
    or source_text != st.session_state["saved_source"]
):
    try:
        save_record(new_meanings, source_text)
        st.session_state["saved_meanings"] = new_meanings
        st.session_state["saved_source"] = source_text
        st.toast("내_기록 폴더에 저장했어요.", icon="💾")
    except OSError as e:
        st.error(f"저장하지 못했습니다: {e}")

st.caption(f"저장 위치: {RECORD_FILE.parent.name}/{RECORD_FILE.name}")
