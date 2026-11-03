import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = (
    "https://raw.githubusercontent.com/kwonboahhhh-beep/data/"
    "refs/heads/main/stroke.csv"
)

st.set_page_config(page_title=f"성능 높이기 - {APP_TITLE}", page_icon="🧠", layout="wide")

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

NAME_LOGI = "로지스틱 회귀(확률로 답하는 모델)"
NAME_TREE = "의사결정트리(질문으로 답하는 모델)"
NOT_AVAILABLE = "계산할 수 없음"

COLOR_NO = "#1f77b4"       # 실제 아님
COLOR_YES = "#d62728"      # 실제 뇌졸중
COLOR_BEFORE = "#9467bd"   # 가중치를 맞추기 전 경계
COLOR_AFTER = "#2ca02c"    # 가중치를 맞춘 뒤 경계
LABEL_BEFORE = "가중치 맞추기 전 0.5 경계"
LABEL_AFTER = "가중치 맞춘 뒤 0.5 경계"


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


def jittered(series: pd.Series, feature: str, rng) -> np.ndarray:
    """0/1 속성은 점이 겹치지 않게 살짝 흩뿌려서 그린다."""
    v = series.to_numpy().astype(float)
    if feature in BINARY:
        v = v + rng.uniform(-0.06, 0.06, len(v))
    return v


def confusion(y_true, y_pred):
    """뇌졸중(1)을 양성으로 두고 TP, FN, FP, TN 사람 수를 센다."""
    (tp, fn), (fp, tn) = confusion_matrix(y_true, y_pred, labels=[1, 0])
    return int(tp), int(fn), int(fp), int(tn)


def four_metrics(tp, fn, fp, tn):
    """네 칸에서 정확도·재현율·정밀도·F1을 구한다. 구할 수 없으면 None."""
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else None       # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) > 0 else None    # 분모: 뇌졸중이라 예측한 사람 수
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return accuracy, recall, precision, f1


def percent(v):
    return NOT_AVAILABLE if v is None else f"{v * 100:.2f}%"


def before_after_table(before, after) -> pd.DataFrame:
    """before, after: (tp, fn, fp, tn). 네 칸과 네 지표를 전·후로 나란히 놓는다."""
    rows = [
        ("TP · 찾아낸 환자", f"{before[0]:,}명", f"{after[0]:,}명"),
        ("FN · 놓친 환자", f"{before[1]:,}명", f"{after[1]:,}명"),
        ("FP · 헛짚은 사람", f"{before[2]:,}명", f"{after[2]:,}명"),
        ("TN · 아닌 사람을 아니라고 맞힌 사람", f"{before[3]:,}명", f"{after[3]:,}명"),
    ]
    mb, ma = four_metrics(*before), four_metrics(*after)
    for name, vb, va in zip(("정확도", "재현율", "정밀도", "F1"), mb, ma):
        rows.append((name, percent(vb), percent(va)))
    table = pd.DataFrame(rows, columns=["항목", "맞추기 전", "맞춘 뒤"]).set_index("항목")
    return table


def raw_logistic(model, scaler):
    """크기를 맞춘 값으로 학습한 로지스틱 회귀를 원래 단위의 식(계수, 절편)으로 바꾼다."""
    coef = model.coef_[0] / scaler.scale_
    intercept = float(model.intercept_[0] - np.sum(model.coef_[0] * scaler.mean_ / scaler.scale_))
    return coef, intercept


def boundary(coef, intercept, axes, selected, med):
    """축이 아닌 속성을 중앙값에 세워 두고, 축 속성에 대한 계수와 상수항을 구한다."""
    others = [f for f in selected if f not in axes]
    const = intercept + sum(coef[selected.index(f)] * float(med[f]) for f in others)
    return [float(coef[selected.index(f)]) for f in axes], float(const)


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


def plane_polygon(a, b, c, d, lo, hi):
    """평면 a*x + b*y + c*z + d = 0 이 상자(lo~hi) 안에서 이루는 다각형의 꼭짓점을 차례대로 돌려준다."""
    if a == 0 and b == 0 and c == 0:
        return None
    corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    scale = max(h - l for l, h in zip(lo, hi))

    def f(p):
        return a * p[0] + b * p[1] + c * p[2] + d

    pts = []

    def add(p):
        for q in pts:
            if max(abs(p[i] - q[i]) for i in range(3)) < 1e-9 * scale:
                return
        pts.append(p)

    for p, q in itertools.combinations(corners, 2):
        if sum(1 for i in range(3) if p[i] != q[i]) != 1:  # 상자의 모서리만 본다
            continue
        fp, fq = f(p), f(q)
        if fp == 0:
            add(p)
        if fq == 0:
            add(q)
        if fp * fq < 0:
            t = fp / (fp - fq)
            add(tuple(p[i] + t * (q[i] - p[i]) for i in range(3)))
    if len(pts) < 3:
        return None

    n = np.array([a, b, c], dtype=float)
    n /= np.linalg.norm(n)
    e = np.eye(3)[int(np.argmin(np.abs(n)))]
    u = np.cross(n, e)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    center = np.mean(pts, axis=0)
    angles = [
        np.arctan2(np.dot(np.array(p) - center, v), np.dot(np.array(p) - center, u)) for p in pts
    ]
    ordered = [pts[i] for i in np.argsort(angles)]
    p0 = np.array(ordered[0])
    area = 0.5 * abs(
        sum(
            np.dot(n, np.cross(np.array(ordered[i]) - p0, np.array(ordered[i + 1]) - p0))
            for i in range(1, len(ordered) - 1)
        )
    )
    if area < 1e-9 * scale**2:  # 상자의 모서리나 꼭짓점만 스친 경우
        return None
    return ordered


def outside_text(label: str, z_center: float, shape: str, where: str) -> str:
    side = "높아서 모두 뇌졸중이라고" if z_center > 0 else "낮아서 모두 아님이라고"
    return (
        f"**{label}의 {shape}은 이 {where}의 범위 밖에 있어서 그려지지 않았어요.** "
        f"이 {where} 안에서는 뇌졸중일 확률이 어디서나 0.5보다 {side} 답해요."
    )


# ---------- 화면 시작 ----------
st.title("🧠 성능 높이기")

# ---------- 1) 입력 속성 고르기 ----------
st.header("1. 입력으로 사용할 속성 고르기")
chosen = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda f: KOR[f],
    key="perf_features",
)
selected = [f for f in FEATURES if f in chosen]
if len(selected) < 2:
    st.info(f"속성을 두 개 이상 골라 주세요. 지금은 {len(selected)}개만 골라져 있어요.")
    st.stop()

st.caption(
    "뇌졸중을 겪은 사람이 아주 드물어서, 두 모델이 거의 '아님'이라고만 답하기 쉬워요. "
    "양쪽(뇌졸중·아님)의 가중치를 맞추기 전과 맞춘 뒤의 결과를 나란히 비교해 봅니다. "
    "뇌졸중(stroke=1)이 양성이에요."
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

# 크기 맞추기는 훈련용 사람들로만 한다 (로지스틱 회귀에 사용)
scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)

n_pos_train = int((y_train == 1).sum())
n_neg_train = int((y_train == 0).sum())
ratio = n_neg_train / n_pos_train if n_pos_train else float("nan")
st.caption(
    f"번호 순으로 정렬해 열 명씩 묶고, 묶음마다 앞 세 명을 테스트용으로 정했어요. "
    f"전체 {len(df):,}명 → 훈련용 {len(train_df):,}명, 테스트용 {len(test_df):,}명.{fill_note}"
)
st.caption(
    f"훈련용에는 뇌졸중인 사람이 {n_pos_train:,}명, 아닌 사람이 {n_neg_train:,}명 있어요. "
    f"가중치를 맞추면 뇌졸중인 사람 한 명을 아닌 사람 {ratio:.1f}명만큼 무겁게 세어서 학습해요."
)

# ---------- 모델: 가중치를 맞추기 전과 맞춘 뒤 ----------
logi_before = LogisticRegression().fit(Xs_train, y_train)
logi_after = LogisticRegression(class_weight="balanced").fit(Xs_train, y_train)
tree_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(
    X_train, y_train
)
tree_after = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42, class_weight="balanced"
).fit(X_train, y_train)

cells = {
    NAME_LOGI: (
        confusion(y_test, logi_before.predict(Xs_test)),
        confusion(y_test, logi_after.predict(Xs_test)),
    ),
    NAME_TREE: (
        confusion(y_test, tree_before.predict(X_test)),
        confusion(y_test, tree_after.predict(X_test)),
    ),
}

# ---------- 2) 전·후 비교 표 ----------
st.header("2. 가중치를 맞추기 전과 맞춘 뒤")
st.markdown(
    f"**테스트용 {len(test_df):,}명 중 실제 뇌졸중인 사람은 {int((y_test == 1).sum()):,}명이에요.**"
)
left, right = st.columns(2)
for col, name in ((left, NAME_LOGI), (right, NAME_TREE)):
    before, after = cells[name]
    with col:
        st.markdown(f"**{name}**")
        st.table(before_after_table(before, after))
        st.markdown(
            f"찾아낸 환자 {before[0]:,}명 → {after[0]:,}명 · "
            f"놓친 환자 {before[1]:,}명 → {after[1]:,}명 · "
            f"헛짚은 사람 {before[2]:,}명 → {after[2]:,}명"
        )
st.caption(
    "TP는 찾아낸 환자, FN은 놓친 환자, FP는 헛짚은 사람이에요. "
    "가중치를 맞추면 보통 놓친 환자는 줄고 헛짚은 사람은 늘어요. "
    "재현율 = TP ÷ (TP + FN), 정밀도 = TP ÷ (TP + FP), "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율)이고, "
    "뇌졸중이라고 예측한 사람이 0명이면 정밀도와 F1은 계산할 수 없어요. "
    "정밀도와 재현율이 모두 0이면 F1은 0으로 적었어요."
)

# 두 로지스틱 회귀를 원래 단위의 식으로 바꿔 둔다
coef_before, intercept_before = raw_logistic(logi_before, scaler)
coef_after, intercept_after = raw_logistic(logi_after, scaler)
med = X_test.median()

# ---------- 3) 2차원 산점도 ----------
st.header("3. 두 속성으로 본 0.5 경계선")
sel_x, sel_y = st.columns(2)
with sel_x:
    xf = st.selectbox("가로축 속성", selected, format_func=lambda f: KOR[f], key="perf_axis_x")
with sel_y:
    y_options = [f for f in selected if f != xf]
    yf = st.selectbox("세로축 속성", y_options, format_func=lambda f: KOR[f], key="perf_axis_y")

x_lo, x_hi = axis_range(X_test[xf], xf in BINARY)
y_lo, y_hi = axis_range(X_test[yf], yf in BINARY)
others2 = [f for f in selected if f not in (xf, yf)]

rng = np.random.RandomState(0)
real_x, real_y = X_test[xf].to_numpy(), X_test[yf].to_numpy()
plot_x = jittered(X_test[xf], xf, rng)
plot_y = jittered(X_test[yf], yf, rng)

fig2 = go.Figure()
for cls, label, color, size in ((0, "실제 아님", COLOR_NO, 5), (1, "실제 뇌졸중", COLOR_YES, 8)):
    m = y_test == cls
    fig2.add_trace(
        go.Scatter(
            x=plot_x[m], y=plot_y[m], mode="markers", name=label,
            marker=dict(color=color, size=size, opacity=0.7, line=dict(width=0.5, color="white")),
            customdata=np.column_stack([real_x[m], real_y[m]]),
            hovertemplate=(
                f"{KOR[xf]}: %{{customdata[0]:g}}<br>"
                f"{KOR[yf]}: %{{customdata[1]:g}}<extra>{label}</extra>"
            ),
        )
    )

notes2 = []
for label, color, dash, coef, intercept in (
    (LABEL_BEFORE, COLOR_BEFORE, "dash", coef_before, intercept_before),
    (LABEL_AFTER, COLOR_AFTER, "solid", coef_after, intercept_after),
):
    (a, b), const = boundary(coef, intercept, (xf, yf), selected, med)
    line = clip_line(a, b, const, x_lo, x_hi, y_lo, y_hi)
    if line is not None:
        (lx0, ly0), (lx1, ly1) = line
        fig2.add_trace(
            go.Scatter(
                x=[lx0, lx1], y=[ly0, ly1], mode="lines", name=label,
                line=dict(color=color, width=3, dash=dash), hoverinfo="skip",
            )
        )
    else:
        z_center = a * (x_lo + x_hi) / 2 + b * (y_lo + y_hi) / 2 + const
        which = "가중치를 맞추기 전" if label == LABEL_BEFORE else "가중치를 맞춘 뒤"
        notes2.append(outside_text(f"{which} 로지스틱 회귀", z_center, "0.5 경계선", "그림"))

fig2.update_xaxes(title_text=KOR[xf], range=[x_lo, x_hi])
fig2.update_yaxes(title_text=KOR[yf], range=[y_lo, y_hi])
if xf in BINARY:
    fig2.update_xaxes(tickvals=[0, 1], ticktext=["없음", "있음"])
if yf in BINARY:
    fig2.update_yaxes(tickvals=[0, 1], ticktext=["없음", "있음"])
fig2.update_layout(
    height=560,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig2, use_container_width=True)

if others2:
    fixed = ", ".join(f"{KOR[f]} {value_text(f, float(med[f]))}" for f in others2)
    st.markdown(f"**두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요:** {fixed}")
else:
    st.markdown("**고른 속성이 둘뿐이라 따로 세워 둔 값은 없어요.**")
for note in notes2:
    st.markdown(note)
st.caption(
    f"{NAME_LOGI}이 뇌졸중일 확률을 0.5로 보는 자리예요. 선의 한쪽은 0.5보다 높아 '뇌졸중', "
    "반대쪽은 낮아 '아님'이라고 답해요. 보라색 점선은 가중치를 맞추기 전, 초록색 실선은 맞춘 뒤예요. "
    "고혈압·심장병 같은 0/1 속성은 점이 겹치지 않게 살짝 흩뿌려 그렸고, 마우스를 올리면 실제 값이 보여요."
)

# ---------- 4) 3차원 산점도 ----------
st.header("4. 세 속성으로 본 0.5 경계 평면")
if len(selected) < 3:
    st.caption("속성을 셋 이상 고르면 3차원 그림이 나와요.")
else:
    s1, s2, s3 = st.columns(3)
    with s1:
        xf3 = st.selectbox("가로축 속성", selected, format_func=lambda f: KOR[f], key="perf_axis3_x")
    with s2:
        y_opts3 = [f for f in selected if f != xf3]
        yf3 = st.selectbox("세로축 속성", y_opts3, format_func=lambda f: KOR[f], key="perf_axis3_y")
    with s3:
        z_opts3 = [f for f in selected if f not in (xf3, yf3)]
        zf3 = st.selectbox("깊이축 속성", z_opts3, format_func=lambda f: KOR[f], key="perf_axis3_z")

    axes3 = (xf3, yf3, zf3)
    ranges3 = [axis_range(X_test[f], f in BINARY) for f in axes3]
    lo3 = [r[0] for r in ranges3]
    hi3 = [r[1] for r in ranges3]
    others3 = [f for f in selected if f not in axes3]

    rng3 = np.random.RandomState(0)
    real3 = [X_test[f].to_numpy() for f in axes3]
    plot3 = [jittered(X_test[f], f, rng3) for f in axes3]

    fig3 = go.Figure()
    for cls, label, color, size in ((0, "실제 아님", COLOR_NO, 3), (1, "실제 뇌졸중", COLOR_YES, 5)):
        m = y_test == cls
        fig3.add_trace(
            go.Scatter3d(
                x=plot3[0][m], y=plot3[1][m], z=plot3[2][m], mode="markers", name=label,
                marker=dict(color=color, size=size, opacity=0.75),
                customdata=np.column_stack([real3[0][m], real3[1][m], real3[2][m]]),
                hovertemplate=(
                    f"{KOR[xf3]}: %{{customdata[0]:g}}<br>"
                    f"{KOR[yf3]}: %{{customdata[1]:g}}<br>"
                    f"{KOR[zf3]}: %{{customdata[2]:g}}<extra>{label}</extra>"
                ),
            )
        )

    notes3 = []
    center3 = [(lo + hi) / 2 for lo, hi in zip(lo3, hi3)]
    for label, color, coef, intercept in (
        (LABEL_BEFORE, COLOR_BEFORE, coef_before, intercept_before),
        (LABEL_AFTER, COLOR_AFTER, coef_after, intercept_after),
    ):
        (a, b, c), const = boundary(coef, intercept, axes3, selected, med)
        polygon = plane_polygon(a, b, c, const, lo3, hi3)
        if polygon is not None:
            k = len(polygon)
            fig3.add_trace(
                go.Mesh3d(
                    x=[p[0] for p in polygon], y=[p[1] for p in polygon], z=[p[2] for p in polygon],
                    i=[0] * (k - 2), j=list(range(1, k - 1)), k=list(range(2, k)),
                    color=color, opacity=0.4, name=label, showlegend=True, hoverinfo="skip",
                )
            )
        else:
            z_center = a * center3[0] + b * center3[1] + c * center3[2] + const
            which = "가중치를 맞추기 전" if label == LABEL_BEFORE else "가중치를 맞춘 뒤"
            notes3.append(outside_text(f"{which} 로지스틱 회귀", z_center, "0.5 평면", "3차원 그림"))

    scene = {}
    for name, f, (lo, hi) in zip(("xaxis", "yaxis", "zaxis"), axes3, ranges3):
        axis = dict(title=KOR[f], range=[lo, hi])
        if f in BINARY:
            axis.update(tickvals=[0, 1], ticktext=["없음", "있음"])
        scene[name] = axis
    scene["aspectmode"] = "cube"
    fig3.update_layout(
        height=680,
        scene=scene,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig3, use_container_width=True)

    if others3:
        fixed = ", ".join(f"{KOR[f]} {value_text(f, float(med[f]))}" for f in others3)
        st.markdown(f"**세 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요:** {fixed}")
    else:
        st.markdown("**고른 속성이 셋뿐이라 따로 세워 둔 값은 없어요.**")
    for note in notes3:
        st.markdown(note)
    st.caption(
        "마우스로 끌면 그림이 돌아가고, 휠로 크기를 바꿀 수 있어요. "
        f"반투명한 면은 {NAME_LOGI}이 뇌졸중일 확률을 0.5로 보는 평면이에요. "
        "보라색은 가중치를 맞추기 전, 초록색은 맞춘 뒤예요."
    )
