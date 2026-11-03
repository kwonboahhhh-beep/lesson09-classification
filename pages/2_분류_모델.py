import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = (
    "https://raw.githubusercontent.com/kwonboahhhh-beep/data/"
    "refs/heads/main/stroke.csv"
)

st.set_page_config(page_title=f"분류 모델 - {APP_TITLE}", page_icon="🧠", layout="wide")

# ---------- 이름과 상수 ----------
FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
BINARY = {"hypertension", "heart_disease"}
KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
SUBJ = {  # 질문 문장에 쓰는 말 (조사 포함)
    "age": "나이가",
    "avg_glucose_level": "평균 혈당이",
    "bmi": "체질량지수가",
    "hypertension": "고혈압이",
    "heart_disease": "심장병이",
}

NAME_LOGI = "로지스틱 회귀(확률로 답하는 모델)"
NAME_TREE = "의사결정트리(질문으로 답하는 모델)"
NAME_BASE = "기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

COLOR_NO = "#1f77b4"        # 실제 아님
COLOR_YES = "#d62728"       # 실제 뇌졸중
REGION_NO = "rgba(31, 119, 180, 0.15)"
REGION_YES = "rgba(214, 39, 40, 0.18)"
COLOR_LINE = "#2ca02c"
FONT = "Noto Sans KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif"


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def value_text(feature: str, v: float) -> str:
    if feature in BINARY:
        if v == 0:
            return "0(없음)"
        if v == 1:
            return "1(있음)"
        return f"{v:g}"
    return f"{v:.1f}"


def axis_range(series: pd.Series, is_binary: bool):
    if is_binary:
        return -0.35, 1.35
    lo, hi = float(series.min()), float(series.max())
    span = (hi - lo) or 1.0
    pad = span * 0.06
    return lo - pad, hi + pad


def clip_line(a, b, c, x0, x1, y0, y1):
    """직선 a*x + b*y + c = 0 이 사각형 안에서 지나는 두 끝점을 돌려준다. 안 지나면 None."""
    tol_x = 1e-9 * (x1 - x0)
    tol_y = 1e-9 * (y1 - y0)
    pts = []
    if b != 0:
        for x in (x0, x1):
            y = -(a * x + c) / b
            if y0 - tol_y <= y <= y1 + tol_y:
                pts.append((x, y))
    if a != 0:
        for y in (y0, y1):
            x = -(b * y + c) / a
            if x0 - tol_x <= x <= x1 + tol_x:
                pts.append((x, y))
    best, best_d = None, 0.0
    for p, q in itertools.combinations(pts, 2):
        d = ((p[0] - q[0]) / (x1 - x0)) ** 2 + ((p[1] - q[1]) / (y1 - y0)) ** 2
        if d > best_d:
            best, best_d = (p, q), d
    return best if best_d > 1e-12 else None


def accuracy_card(title: str, train_acc: float, test_acc: float) -> str:
    return f"""
    <div style="border:1px solid rgba(128,128,128,0.45); border-radius:14px;
                padding:16px 18px; height:100%;">
      <div style="font-size:0.95rem; opacity:0.85; min-height:3.6em;">{title}</div>
      <div style="font-size:0.8rem; opacity:0.7; margin-top:4px;">테스트 데이터 정확도</div>
      <div style="font-size:2.9rem; font-weight:700; line-height:1.15;">{test_acc * 100:.2f}%</div>
      <div style="display:flex; gap:24px; margin-top:12px; font-size:0.82rem; opacity:0.85;">
        <div>훈련 데이터 정확도<br><b>{train_acc * 100:.2f}%</b></div>
        <div>테스트 데이터 정확도<br><b>{test_acc * 100:.2f}%</b></div>
      </div>
    </div>
    """


def build_dot(tree_model, X_train, y_train, selected) -> str:
    t = tree_model.tree_
    path = tree_model.decision_path(X_train)
    n_all = np.asarray(path.sum(axis=0)).ravel()
    n_pos = np.asarray(path[np.asarray(y_train == 1)].sum(axis=0)).ravel()

    lines = [
        "digraph Tree {",
        "rankdir=TB;",
        'bgcolor="transparent";',
        f'node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=12, '
        'margin="0.16,0.1", color="#888888", fontcolor="#111111"];',
        f'edge [color="#888888", fontcolor="#888888", fontname="{FONT}", fontsize=12];',
    ]
    for i in range(t.node_count):
        pct = n_pos[i] / n_all[i] * 100 if n_all[i] else 0.0
        stats = f"훈련용 {n_all[i]:,}명 중\\n뇌졸중 {n_pos[i]:,}명 ({pct:.1f}%)"
        if t.children_left[i] == -1:  # 답을 내는 마디
            answer = int(np.argmax(t.value[i][0]))
            fill = "#f7c9c4" if answer == 1 else "#c9def5"
            text = "뇌졸중" if answer == 1 else "아님"
            lines.append(f'n{i} [label="답: {text}\\n{stats}", fillcolor="{fill}"];')
        else:
            feat = selected[t.feature[i]]
            if feat in BINARY:
                question = f"{SUBJ[feat]} 있나요?"
            else:
                question = f"{SUBJ[feat]} {round(float(t.threshold[i]), 2):g} 이하인가요?"
            lines.append(f'n{i} [label="{question}\\n{stats}", fillcolor="#eeeeee"];')
    for i in range(t.node_count):
        left, right = t.children_left[i], t.children_right[i]
        if left == -1:
            continue
        feat = selected[t.feature[i]]
        if feat in BINARY:      # 있으면(1) 오른쪽
            left_label, right_label = "아니요", "예"
        else:                   # 기준값 이하이면 왼쪽
            left_label, right_label = "예", "아니요"
        lines.append(f'n{i} -> n{left} [label=" {left_label} "];')
        lines.append(f'n{i} -> n{right} [label=" {right_label} "];')
    lines.append("}")
    return "\n".join(lines)


# ---------- 화면 시작 ----------
st.title("🧠 분류 모델")
st.caption(
    "stroke가 1이면 뇌졸중, 0이면 아님이에요. 뇌졸중(1)을 양성으로 두고 모델을 만들어 비교해 봅니다."
)

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# ---------- 1) 입력 속성 고르기 ----------
st.header("1. 입력으로 사용할 속성 고르기")
chosen = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda f: KOR[f],
)
selected = [f for f in FEATURES if f in chosen]
if len(selected) < 2:
    st.info(f"속성을 두 개 이상 골라 주세요. 지금은 {len(selected)}개만 골라져 있어요.")
    st.stop()

# ---------- 2) 데이터 나누기와 준비 ----------
df = raw.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(df)) % 10) < 3  # 열 명씩 묶어 앞 세 명은 테스트용
train_df, test_df = df[~is_test], df[is_test]

X_train = train_df[selected].astype(float).copy()
X_test = test_df[selected].astype(float).copy()
y_train = train_df["stroke"].to_numpy()
y_test = test_df["stroke"].to_numpy()

fill_note = ""
if "bmi" in selected:
    bmi_median = float(X_train["bmi"].median())
    n_fill_train = int(X_train["bmi"].isna().sum())
    n_fill_test = int(X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    fill_note = (
        f" 체질량지수가 빈 사람(훈련용 {n_fill_train:,}명, 테스트용 {n_fill_test:,}명)은 "
        f"훈련용의 중앙값 {bmi_median:.1f}으로 채웠어요."
    )

st.caption(
    f"번호 순으로 정렬해 열 명씩 묶고, 묶음마다 앞 세 명을 테스트용으로 정했어요. "
    f"전체 {len(df):,}명 → 훈련용 {len(train_df):,}명, 테스트용 {len(test_df):,}명.{fill_note}"
)

# 크기 맞추기: 훈련용 사람들로만 평균과 퍼짐을 구한다
scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)

# ---------- 3) 모델 만들기 ----------
logi = LogisticRegression().fit(Xs_train, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(
    X_train, y_train
)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

acc = {
    "logi": (
        accuracy_score(y_train, logi.predict(Xs_train)),
        accuracy_score(y_test, logi.predict(Xs_test)),
    ),
    "tree": (
        accuracy_score(y_train, tree.predict(X_train)),
        accuracy_score(y_test, tree.predict(X_test)),
    ),
    "base": (
        accuracy_score(y_train, base.predict(X_train)),
        accuracy_score(y_test, base.predict(X_test)),
    ),
}

# ---------- 4) 정확도 카드 ----------
st.header("2. 모델의 정확도")
c1, c2, c3 = st.columns(3)
for col, key, name in (
    (c1, "logi", NAME_LOGI),
    (c2, "tree", NAME_TREE),
    (c3, "base", NAME_BASE),
):
    with col:
        st.markdown(accuracy_card(name, *acc[key]), unsafe_allow_html=True)
st.caption(
    "정확도는 맞게 답한 사람의 비율이에요. 입력을 보지 않는 기준 모델보다 높아야 입력 속성이 도움이 된 거예요. "
    "의사결정트리(질문으로 답하는 모델)는 질문의 기준값을 원래 단위로 보여 주려고 "
    "크기를 맞추지 않은 값으로 학습했어요(질문 결과는 같아요)."
)
st.caption(
    "의사결정트리(질문으로 답하는 모델) 설정: 질문은 세 번까지만, 마지막 마디에 사람이 다섯 명 미만이 되는 "
    "나누기는 하지 않고, 난수는 42로 고정했어요."
)

# ---------- 5) 산점도 ----------
st.header("3. 두 속성으로 본 테스트 데이터와 모델의 경계")
sel_x, sel_y = st.columns(2)
with sel_x:
    xf = st.selectbox("가로축 속성", selected, format_func=lambda f: KOR[f], key="axis_x")
with sel_y:
    y_options = [f for f in selected if f != xf]
    yf = st.selectbox("세로축 속성", y_options, format_func=lambda f: KOR[f], key="axis_y")

x_lo, x_hi = axis_range(X_test[xf], xf in BINARY)
y_lo, y_hi = axis_range(X_test[yf], yf in BINARY)
med = X_test.median()
others = [f for f in selected if f not in (xf, yf)]

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산한다
nx = ny = 250
xs = np.linspace(x_lo, x_hi, nx)
ys = np.linspace(y_lo, y_hi, ny)
gx, gy = np.meshgrid(xs, ys)
grid = pd.DataFrame(index=range(gx.size))
for f in selected:
    if f == xf:
        grid[f] = gx.ravel()
    elif f == yf:
        grid[f] = gy.ravel()
    else:
        grid[f] = float(med[f])
tree_region = tree.predict(grid[selected]).reshape(ny, nx)

# 로지스틱 회귀의 0.5 경계선: 원래 단위의 식으로 바꿔 직선으로 계산한다
coef = logi.coef_[0] / scaler.scale_
intercept = float(logi.intercept_[0] - np.sum(logi.coef_[0] * scaler.mean_ / scaler.scale_))
ix, iy = selected.index(xf), selected.index(yf)
const = intercept + sum(coef[selected.index(f)] * float(med[f]) for f in others)
line = clip_line(float(coef[ix]), float(coef[iy]), float(const), x_lo, x_hi, y_lo, y_hi)

# 점 (0/1 속성은 겹치지 않게 살짝 흩뿌려 그린다)
rng = np.random.RandomState(0)
jit_x = rng.uniform(-0.06, 0.06, len(X_test))
jit_y = rng.uniform(-0.06, 0.06, len(X_test))
real_x = X_test[xf].to_numpy()
real_y = X_test[yf].to_numpy()
plot_x = real_x + (jit_x if xf in BINARY else 0)
plot_y = real_y + (jit_y if yf in BINARY else 0)

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x=xs, y=ys, z=tree_region, zmin=0, zmax=1,
        colorscale=[[0, REGION_NO], [1, REGION_YES]],
        showscale=False, hoverinfo="skip", showlegend=False,
    )
)
for cls, label, color, size in ((0, "실제 아님", COLOR_NO, 5), (1, "실제 뇌졸중", COLOR_YES, 8)):
    m = y_test == cls
    fig.add_trace(
        go.Scatter(
            x=plot_x[m], y=plot_y[m], mode="markers", name=label,
            marker=dict(color=color, size=size, opacity=0.7,
                        line=dict(width=0.5, color="white")),
            customdata=np.column_stack([real_x[m], real_y[m]]),
            hovertemplate=(
                f"{KOR[xf]}: %{{customdata[0]:g}}<br>"
                f"{KOR[yf]}: %{{customdata[1]:g}}<extra>{label}</extra>"
            ),
        )
    )
for label, color in (
    ("의사결정트리가 '아님'이라고 답하는 칸", "rgba(31, 119, 180, 0.35)"),
    ("의사결정트리가 '뇌졸중'이라고 답하는 칸", "rgba(214, 39, 40, 0.35)"),
):
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers", name=label,
            marker=dict(symbol="square", size=14, color=color),
        )
    )
if line is not None:
    (lx0, ly0), (lx1, ly1) = line
    fig.add_trace(
        go.Scatter(
            x=[lx0, lx1], y=[ly0, ly1], mode="lines",
            name="로지스틱 회귀가 0.5로 가르는 선",
            line=dict(color=COLOR_LINE, width=3), hoverinfo="skip",
        )
    )
fig.update_xaxes(title_text=KOR[xf], range=[x_lo, x_hi])
fig.update_yaxes(title_text=KOR[yf], range=[y_lo, y_hi])
if xf in BINARY:
    fig.update_xaxes(tickvals=[0, 1], ticktext=["없음", "있음"])
if yf in BINARY:
    fig.update_yaxes(tickvals=[0, 1], ticktext=["없음", "있음"])
fig.update_layout(
    height=580,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)

if others:
    fixed = ", ".join(f"{KOR[f]} {value_text(f, float(med[f]))}" for f in others)
    st.markdown(
        f"**두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요:** {fixed}"
    )
else:
    st.markdown("**고른 속성이 둘뿐이라 따로 세워 둔 값은 없어요.**")

if line is not None:
    st.markdown(
        "초록 선은 로지스틱 회귀(확률로 답하는 모델)가 뇌졸중일 확률을 0.5로 보는 자리예요. "
        "선의 한쪽은 0.5보다 높아 '뇌졸중', 반대쪽은 낮아 '아님'이라고 답해요."
    )
else:
    z_center = (
        coef[ix] * (x_lo + x_hi) / 2 + coef[iy] * (y_lo + y_hi) / 2 + const
    )
    side = "높아서 모두 뇌졸중이라고" if z_center > 0 else "낮아서 모두 아님이라고"
    st.markdown(
        "**로지스틱 회귀(확률로 답하는 모델)가 0.5로 가르는 선은 이 그림의 범위 밖에 있어서 그려지지 않았어요.** "
        f"이 그림 안에서는 뇌졸중일 확률이 어디서나 0.5보다 {side} 답해요."
    )
st.caption(
    "옅게 칠한 칸은 의사결정트리(질문으로 답하는 모델)의 답이에요. "
    "고혈압·심장병 같은 0/1 속성은 점이 겹치지 않게 살짝 흩뿌려 그렸고, 마우스를 올리면 실제 값이 보여요."
)

# ---------- 6) 의사결정트리가 던진 질문 ----------
st.header("4. 의사결정트리(질문으로 답하는 모델)가 던진 질문")
st.graphviz_chart(build_dot(tree, X_train, y_train, selected), use_container_width=True)
st.caption("마디의 숫자는 그 자리에 온 훈련용 사람 수와 그중 실제 뇌졸중인 사람 수, 그 비율이에요.")

t = tree.tree_
leaf_ids = [i for i in range(t.node_count) if t.children_left[i] == -1]
n_leaf = len(leaf_ids)
n_leaf_no = sum(1 for i in leaf_ids if int(np.argmax(t.value[i][0])) == 0)
used = sorted({int(t.feature[i]) for i in range(t.node_count) if t.children_left[i] != -1})
asked = ", ".join(KOR[selected[k]] for k in used) if used else "없음(질문 없이 바로 답해요)"

st.markdown(f"- 답을 내는 마디는 모두 **{n_leaf}칸**이고, 그중 **{n_leaf_no}칸**이 '아님'이라고 답해요.")
st.markdown(f"- 고른 속성 {len(selected)}개 가운데 이 나무가 실제로 물은 것: **{asked}**")
