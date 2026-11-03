import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = (
    "https://raw.githubusercontent.com/kwonboahhhh-beep/data/"
    "refs/heads/main/stroke.csv"
)

st.set_page_config(page_title=f"채점표 - {APP_TITLE}", page_icon="🧠", layout="wide")

# '분류 모델' 페이지에서 처음에 골라져 있는 입력 속성(체질량지수를 뺀 넷)
FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

NAME_LOGI = "로지스틱 회귀(확률로 답하는 모델)"
NAME_TREE = "의사결정트리(질문으로 답하는 모델)"
NAME_BASE = "기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)"
NOT_AVAILABLE = "계산할 수 없음"
MAX_ROWS = 20


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def confusion(y_true, y_pred):
    """뇌졸중(1)을 양성으로 두고 TP, FN, FP, TN 사람 수를 센다. 행은 실제, 열은 예측."""
    (tp, fn), (fp, tn) = confusion_matrix(y_true, y_pred, labels=[1, 0])
    return int(tp), int(fn), int(fp), int(tn)


def confusion_table(tp, fn, fp, tn) -> pd.DataFrame:
    table = pd.DataFrame(
        [
            [f"TP · {tp:,}명", f"FN · {fn:,}명", f"{tp + fn:,}명"],
            [f"FP · {fp:,}명", f"TN · {tn:,}명", f"{fp + tn:,}명"],
            [f"{tp + fp:,}명", f"{fn + tn:,}명", f"{tp + fn + fp + tn:,}명"],
        ],
        index=["실제 뇌졸중", "실제 아님", "합계"],
        columns=["예측 뇌졸중", "예측 아님", "합계"],
    )
    table.index.name = "실제 ＼ 예측"
    return table


def percent(v):
    return NOT_AVAILABLE if v is None else f"{v * 100:.2f}%"


def four_metrics(tp, fn, fp, tn):
    """네 칸에서 정확도·재현율·정밀도·F1을 구한다. 구할 수 없으면 None."""
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else None        # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) > 0 else None     # 분모: 뇌졸중이라 예측한 사람 수
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return accuracy, recall, precision, f1


def show_people(title: str, people: pd.DataFrame) -> None:
    n = len(people)
    suffix = f" — 전체 {n:,}명" if n <= MAX_ROWS else f" — 전체 {n:,}명 중 나이가 많은 앞 {MAX_ROWS}줄"
    st.markdown(f"#### {title}{suffix}")
    if n == 0:
        st.caption("해당하는 사람이 없어요.")
        return
    shown = people.head(MAX_ROWS)
    table = pd.DataFrame(
        {
            "번호": shown["id"].astype(int).astype(str).to_numpy(),
            "나이": [f"{v:g}" for v in shown["age"]],
            "평균 혈당": [f"{v:g}" for v in shown["avg_glucose_level"]],
            "고혈압 여부": np.where(shown["hypertension"] == 1, "있음", "없음"),
            "심장병 여부": np.where(shown["heart_disease"] == 1, "있음", "없음"),
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


# ---------- 화면 시작 ----------
st.title("🧠 채점표")
st.caption(
    "'분류 모델' 페이지와 같은 방법으로 데이터를 나누고 두 모델을 학습한 뒤, 테스트 데이터로 채점해 봅니다. "
    "뇌졸중(stroke=1)이 양성이고, 입력 속성은 '분류 모델' 페이지에서 처음 골라져 있는 "
    "나이·평균 혈당·고혈압·심장병 넷이에요."
)

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# ---------- 데이터 나누기 (번호 순 정렬 → 열 명씩 묶어 앞 세 명은 테스트용) ----------
df = raw.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(df)) % 10) < 3
train_df, test_df = df[~is_test], df[is_test]

X_train = train_df[FEATURES].astype(float)
X_test = test_df[FEATURES].astype(float)
y_train = train_df["stroke"].to_numpy()
y_test = test_df["stroke"].to_numpy()

# 크기 맞추기는 훈련용 사람들로만 한다 (로지스틱 회귀에 사용)
scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)

# ---------- 두 모델과 기준 모델 학습 ----------
logi = LogisticRegression().fit(Xs_train, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(
    X_train, y_train
)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

preds = {
    NAME_LOGI: logi.predict(Xs_test),
    NAME_TREE: tree.predict(X_test),
    NAME_BASE: base.predict(X_test),
}
cells = {name: confusion(y_test, p) for name, p in preds.items()}

n_test = len(test_df)
n_test_stroke = int((y_test == 1).sum())
st.markdown(
    f"**테스트용 {n_test:,}명 중 실제 뇌졸중인 사람은 {n_test_stroke:,}명이에요.**"
)

# ---------- 1) 혼동행렬 ----------
st.header("1. 혼동행렬")
st.caption("행은 실제, 열은 예측이에요. 칸마다 사람 수를 적었고, 행과 열의 합계도 함께 적었어요.")
left, right = st.columns(2)
for col, name in ((left, NAME_LOGI), (right, NAME_TREE)):
    with col:
        st.markdown(f"**{name}**")
        st.table(confusion_table(*cells[name]))
st.caption(
    "TP: 뇌졸중인 사람을 뇌졸중이라고 맞힌 사람 · "
    "FN: 뇌졸중인데 아니라고 예측한 사람(놓친 환자) · "
    "FP: 아닌데 뇌졸중이라고 예측한 사람(헛짚은 사람) · "
    "TN: 아닌 사람을 아니라고 맞힌 사람"
)

# ---------- 2) 네 가지 점수 ----------
st.header("2. 네 가지 점수 비교")
rows = []
for name in (NAME_LOGI, NAME_TREE, NAME_BASE):
    accuracy, recall, precision, f1 = four_metrics(*cells[name])
    rows.append(
        {
            "모델": name,
            "정확도": percent(accuracy),
            "재현율": percent(recall),
            "정밀도": percent(precision),
            "F1": percent(f1),
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption(
    "정확도 = (TP + TN) ÷ 전체 사람 수 · "
    "재현율 = TP ÷ (TP + FN), 분모는 실제 뇌졸중인 사람 수 · "
    "정밀도 = TP ÷ (TP + FP), 분모는 뇌졸중이라고 예측한 사람 수 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
    "뇌졸중이라고 예측한 사람이 0명이면 정밀도와 F1은 계산할 수 없어요. "
    "정밀도와 재현율이 모두 0이면 F1은 0으로 적었어요."
)

# ---------- 3) 놓친 환자와 헛짚은 사람 ----------
st.header("3. 놓친 환자와 헛짚은 사람")
choice = st.radio(
    "어느 모델의 목록을 볼까요?", [NAME_LOGI, NAME_TREE], horizontal=True
)
pred = preds[choice]
tp, fn, fp, tn = cells[choice]

people = test_df.assign(pred=pred).sort_values(["age", "id"], ascending=[False, True])
missed = people[(people["stroke"] == 1) & (people["pred"] == 0)]
wrong = people[(people["stroke"] == 0) & (people["pred"] == 1)]

left, right = st.columns(2)
with left:
    show_people("놓친 환자 (실제 뇌졸중인데 아니라고 예측)", missed)
with right:
    show_people("헛짚은 사람 (아닌데 뇌졸중이라고 예측)", wrong)

if len(missed) == fn and len(wrong) == fp:
    st.markdown(
        f"놓친 환자 {len(missed):,}명은 혼동행렬의 FN 칸({fn:,}명)과, "
        f"헛짚은 사람 {len(wrong):,}명은 FP 칸({fp:,}명)과 같아요."
    )
else:
    st.warning(
        f"목록의 사람 수(놓친 환자 {len(missed):,}명, 헛짚은 사람 {len(wrong):,}명)가 "
        f"혼동행렬의 FN 칸({fn:,}명), FP 칸({fp:,}명)과 달라요. 확인이 필요해요."
    )
