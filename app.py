from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "processed" / "reviews_processed.csv"

st.set_page_config(page_title="Customer Voice | Review Intelligence", page_icon="✦", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#162329; --muted:#6c7b7a; --mint:#c7f36b; --coral:#ff725e; --cream:#f8f6f0; }
    .stApp { background:var(--cream); color:var(--ink); font-family:'DM Sans', sans-serif; }
    [data-testid="stSidebar"] { background:#162329; }
    [data-testid="stSidebar"] * { color:#f8f6f0 !important; }
    h1,h2,h3 { font-family:'Space Grotesk', sans-serif !important; letter-spacing:0 !important; }
    h1 { font-size:2.45rem !important; line-height:1.05 !important; margin-bottom:.2rem !important; }
    .eyebrow { color:#e45c49; font-size:.75rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
    .subtitle { color:var(--muted); margin-bottom:1.7rem; }
    .metric { background:#fff; border:1px solid #e8e5dc; border-radius:8px; padding:1rem 1.1rem; min-height:112px; }
    .metric-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
    .metric-value { font-family:'Space Grotesk'; font-size:2rem; font-weight:700; margin-top:.35rem; }
    .metric-note { color:var(--muted); font-size:.8rem; margin-top:.2rem; }
    .section-title { font-family:'Space Grotesk'; font-size:1.2rem; font-weight:700; margin:1.5rem 0 .65rem; }
    .stButton > button { border-radius:6px; border:0; background:var(--mint); color:var(--ink); font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce").fillna(0).astype(int)
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    return frame


def metric(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


st.sidebar.markdown("## ✦ Customer Voice")
st.sidebar.caption("Review intelligence workspace")
data_path = st.sidebar.text_input("Đường dẫn dữ liệu", str(DEFAULT_DATA))

try:
    df = load_data(data_path)
except Exception as error:
    st.error(f"Không thể đọc dữ liệu: {error}")
    st.stop()

st.sidebar.markdown("### Bộ lọc")
platforms = sorted(df["platform"].dropna().astype(str).unique())
selected_platforms = st.sidebar.multiselect("Nền tảng", platforms, default=platforms)
products = sorted(df["product_name"].dropna().astype(str).unique())
selected_products = st.sidebar.multiselect("Sản phẩm", products, default=[])
ratings = st.sidebar.slider("Khoảng rating", 1, 5, (1, 5))
valid_only = st.sidebar.checkbox("Chỉ review hợp lệ", value=False)

view = df[df["platform"].astype(str).isin(selected_platforms) & df["rating"].between(*ratings)]
if selected_products:
    view = view[view["product_name"].astype(str).isin(selected_products)]
if valid_only:
    view = view[view["is_valid_for_analysis"]]

st.markdown('<div class="eyebrow">Seller intelligence / 01</div>', unsafe_allow_html=True)
st.title("Customer Voice")
st.markdown("<div class='subtitle'>Một góc nhìn sạch và nhanh về chất lượng phản hồi khách hàng.</div>", unsafe_allow_html=True)

total = len(view)
valid = int(view["is_valid_for_analysis"].sum()) if total else 0
spam = int(view["is_spam"].sum()) if total else 0
average = view["rating"].mean() if total else 0
cols = st.columns(4)
with cols[0]: metric("Reviews đang xem", f"{total:,}", "sau bộ lọc hiện tại")
with cols[1]: metric("Rating trung bình", f"{average:.2f} / 5", "tín hiệu hài lòng")
with cols[2]: metric("Hợp lệ cho NLP", f"{valid:,}", f"{valid / total * 100:.1f}% tập đang xem" if total else "chưa có dữ liệu")
with cols[3]: metric("Spam phát hiện", f"{spam:,}", f"{spam / total * 100:.1f}% tập đang xem" if total else "chưa có dữ liệu")

left, right = st.columns([1, 1])
with left:
    st.markdown('<div class="section-title">Phân bố rating</div>', unsafe_allow_html=True)
    rating_chart = view["rating"].value_counts().reindex(range(1, 6), fill_value=0).rename_axis("Số sao").to_frame("Reviews")
    st.bar_chart(rating_chart, color="#ff725e", height=280)
with right:
    st.markdown('<div class="section-title">Chất lượng dữ liệu</div>', unsafe_allow_html=True)
    status_chart = view["filter_status"].value_counts().rename_axis("Trạng thái").to_frame("Số lượng") if total else pd.DataFrame(columns=["Số lượng"])
    st.bar_chart(status_chart, color="#9bcf3a", height=280)

st.markdown('<div class="section-title">Review explorer</div>', unsafe_allow_html=True)
search = st.text_input("Tìm trong nội dung review", placeholder="Ví dụ: giao hàng, chất lượng, đóng gói...")
table = view.copy()
if search:
    table = table[table["comment_text_raw"].astype(str).str.contains(search, case=False, na=False)]
columns = ["created_at", "product_name", "rating", "filter_status", "comment_text_raw", "comment_text_clean"]
st.caption(f"Hiển thị {len(table):,} review")
st.dataframe(table[[column for column in columns if column in table.columns]], use_container_width=True, hide_index=True)