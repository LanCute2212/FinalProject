import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from scripts.run_eda import build_report, load_aspects
from src.features.feature_engineering import FeatureEngineeringPipeline
from src.preprocessing.pipeline import PreprocessingPipeline


ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "processed" / "reviews_processed.csv"
ASPECT_DATA = ROOT / "data" / "dictionaries" / "aspect_keywords.json"

st.set_page_config(
    page_title="Customer Voice | Seller Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#162329; --muted:#6c7b7a; --mint:#c7f36b; --coral:#ff725e; --cream:#f8f6f0; --line:#e6e2d9; }
    .stApp { background:var(--cream); color:var(--ink); font-family:'DM Sans', sans-serif; }
    [data-testid="stSidebar"] { background:#162329; border-right:1px solid #293b40; }
    [data-testid="stSidebar"] * { color:#f8f6f0 !important; }
    [data-testid="stSidebar"] .stRadio > label { color:#aebdbc !important; font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap:.28rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label { border-radius:8px; padding:.45rem .6rem; transition:background .15s ease; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#294047; }
    [data-testid="stMain"] .stRadio [role="radiogroup"] { display:flex; flex-wrap:wrap; gap:.45rem; margin-bottom:1.2rem; }
    [data-testid="stMain"] .stRadio [role="radiogroup"] label { background:#fff; border:1px solid var(--line); border-radius:999px; padding:.48rem .85rem; color:var(--muted) !important; box-shadow:0 3px 10px rgba(22,35,41,.03); transition:all .15s ease; }
    [data-testid="stMain"] .stRadio [role="radiogroup"] label:hover { border-color:#b5d65f; color:var(--ink) !important; transform:translateY(-1px); }
    [data-testid="stMain"] .stRadio [role="radiogroup"] label:has(input:checked) { background:var(--mint); border-color:#b5d65f; color:var(--ink) !important; font-weight:700; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { color:#9fb0ad !important; font-size:.72rem !important; text-transform:uppercase; letter-spacing:.12em; }
    [data-testid="stSidebar"] .stCaption { color:#9fb0ad !important; }
    [data-testid="stMain"] { padding-top:2.6rem; }
    h1,h2,h3 { font-family:'Space Grotesk', sans-serif !important; color:var(--ink) !important; }
    h1 { font-size:2.75rem !important; line-height:1.03 !important; letter-spacing:-.04em !important; }
    h2 { letter-spacing:-.025em !important; }
    .eyebrow { color:#e45c49; font-size:.7rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
    .subtitle { color:var(--muted); font-size:1.02rem; margin-bottom:1.7rem; }
    .hero { display:flex; justify-content:space-between; align-items:flex-end; background:linear-gradient(135deg,#fff 0%,#f2f5e9 100%); border:1px solid var(--line); border-radius:16px; padding:1.35rem 1.5rem; margin:.4rem 0 1.7rem; box-shadow:0 10px 30px rgba(22,35,41,.04); }
    .hero-copy { max-width:720px; }
    .hero-label { color:#e45c49; font-size:.7rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
    .hero-title { color:var(--ink); font-family:'Space Grotesk'; font-size:2.25rem; font-weight:700; letter-spacing:-.045em; margin:.35rem 0 .45rem; }
    .hero-note { color:var(--muted); font-size:.95rem; margin:0; }
    .hero-mark { display:flex; align-items:center; justify-content:center; width:68px; height:68px; background:var(--mint); border-radius:50%; color:var(--ink); font-family:'Space Grotesk'; font-size:1.8rem; font-weight:700; }
    .nav-label { color:var(--muted); font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin:1.2rem 0 .35rem; }
    .metric { background:#fff; border:1px solid var(--line); border-radius:12px; padding:1rem 1.1rem; min-height:112px; box-shadow:0 6px 18px rgba(22,35,41,.035); }
    .metric-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
    .metric-value { font-family:'Space Grotesk'; font-size:2rem; font-weight:700; margin-top:.35rem; }
    .metric-note { color:var(--muted); font-size:.8rem; margin-top:.2rem; }
    .section-title { font-family:'Space Grotesk'; font-size:1.2rem; font-weight:700; margin:1.5rem 0 .65rem; }
    .stButton > button { border-radius:8px; border:1px solid #b5d65f; background:var(--mint); color:var(--ink); font-weight:700; min-height:2.5rem; }
    .stButton > button:hover { border-color:#8eae37; background:#b9e957; color:var(--ink); }
    .stDownloadButton > button { border-radius:8px; border:1px solid var(--line); background:#fff; color:var(--ink); font-weight:700; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:10px; padding:.8rem 1rem; }
    [data-testid="stExpander"] { background:#fff; border:1px solid var(--line); border-radius:10px; }
    .stTabs [data-baseweb="tab-list"] { gap:.25rem; border-bottom:1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { height:2.7rem; padding:0 .9rem; color:var(--muted); }
    .stTabs [aria-selected="true"] { color:var(--ink); border-bottom-color:var(--coral); }
    hr { border-color:var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def load_data(path: str, mtime: float) -> pd.DataFrame:
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


def sentiment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    labels = pd.cut(
        frame["rating"],
        bins=[0, 2, 3, 5],
        labels=["Tiêu cực", "Trung tính", "Tích cực"],
    )
    return labels.value_counts().reindex(["Tiêu cực", "Trung tính", "Tích cực"], fill_value=0).rename_axis("Sentiment").to_frame("Reviews")


def valid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "is_valid_for_analysis" not in frame.columns:
        return frame
    return frame[frame["is_valid_for_analysis"].astype(bool)]


def sentiment_label(rating: pd.Series) -> pd.Series:
    return pd.cut(
        rating,
        bins=[0, 2, 3, 5],
        labels=["negative", "neutral", "positive"],
    ).astype(str)


POSITIVE_ASPECT_TERMS = ("tốt", "hay", "đẹp", "nhanh", "ổn", "hài lòng", "ưng ý", "mượt", "đáng tiền")
NEGATIVE_ASPECT_TERMS = ("lỗi", "hỏng", "chậm", "tệ", "thất vọng", "móp", "trầy", "đắt", "kém", "chập chờn")


def build_aspect_sentiment_rows(text: str, aspect_keywords: dict, meta: dict = None) -> list[dict]:
    """Create an explainable aspect sentiment baseline without extra dependencies."""
    normalized = re.sub(r"[\s_]+", " ", str(text or "").lower()).strip()
    rows = []
    for aspect, keywords in aspect_keywords.items():
        matches = [
            keyword for keyword in keywords
            if re.sub(r"[\s_]+", " ", keyword.lower()).strip() in normalized
        ]
        if not matches:
            continue
        positive = any(term in normalized for term in POSITIVE_ASPECT_TERMS)
        negative = any(term in normalized for term in NEGATIVE_ASPECT_TERMS)
        sentiment = (
            "Hỗn hợp" if positive and negative
            else "Tiêu cực" if negative
            else "Tích cực" if positive
            else "Trung tính"
        )
        row_dict = {"aspect": aspect, "sentiment": sentiment, "matched_keywords": ", ".join(matches)}
        if meta:
            row_dict.update(meta)
        rows.append(row_dict)
    return rows


def render_overview(view: pd.DataFrame, aspects: dict) -> None:
    st.markdown("### Tổng quan vận hành")
    if view.empty:
        st.info("Không có dữ liệu phù hợp với bộ lọc hiện tại.")
        return

    valid = view[view.get("is_valid_for_analysis", True).astype(bool)]
    negative = int((valid["rating"] <= 2).sum()) if not valid.empty else 0
    try:
        _, report = build_report(view, aspects)
        contradictions = report["rating_text_contradictions"]
    except (KeyError, ValueError):
        contradictions = 0

    cols = st.columns(4)
    with cols[0]:
        metric("Reviews đang xem", f"{len(view):,}", "sau bộ lọc")
    with cols[1]:
        metric("Rating trung bình", f"{view['rating'].mean():.2f} / 5", "tín hiệu hài lòng")
    with cols[2]:
        metric("Cần ưu tiên", f"{negative:,}", "rating 1–2 sao")
    with cols[3]:
        metric("Nghịch lý", f"{contradictions:,}", "rating không khớp nội dung")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Phân bố cảm xúc baseline")
        st.bar_chart(sentiment_summary(valid if not valid.empty else view), color="#ff725e", height=280)
    with right:
        st.markdown("#### Sức khỏe dữ liệu")
        status = view.get("filter_status", pd.Series(["VALID"] * len(view))).value_counts()
        st.bar_chart(status.rename_axis("Trạng thái").to_frame("Reviews"), color="#9bcf3a", height=280)

    # Thống kê phân bố theo Ngành hàng & Nền tảng
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Phân bố theo ngành hàng")
        if "category" in view.columns:
            cat_counts = view["category"].value_counts().rename_axis("Ngành hàng").to_frame("Reviews")
            st.bar_chart(cat_counts, color="#3a86ff", height=260)
        else:
            st.caption("Chưa có dữ liệu ngành hàng.")
    with c2:
        st.markdown("#### Phân bố theo nền tảng sàn")
        if "platform" in view.columns:
            plat_counts = view["platform"].value_counts().rename_axis("Nền tảng").to_frame("Reviews")
            st.bar_chart(plat_counts, color="#ffbe0b", height=260)
        else:
            st.caption("Chưa có dữ liệu nền tảng.")

    st.markdown("#### Bảng tổng hợp theo ngành hàng")
    if "category" in view.columns:
        cat_summary = (
            valid.groupby("category", dropna=False)
            .agg(
                reviews=("review_id", "count"),
                average_rating=("rating", "mean"),
                total_likes=("like_count", "sum") if "like_count" in valid.columns else ("review_id", "count"),
                low_rating_reviews=("rating", lambda s: int((s <= 2).sum()))
            )
            .sort_values(["reviews"], ascending=[False])
            .reset_index()
        )
        cat_summary["average_rating"] = cat_summary["average_rating"].round(2)
        cat_summary.rename(columns={
            "category": "Ngành hàng",
            "reviews": "Số review hợp lệ",
            "average_rating": "Rating trung bình",
            "total_likes": "Tổng lượt thích",
            "low_rating_reviews": "Review 1–2★"
        }, inplace=True)
        st.dataframe(cat_summary, use_container_width=True, hide_index=True)

    st.markdown("#### Sản phẩm cần ưu tiên theo rating")
    agg_dict = {
        "reviews": ("review_id", "count"),
        "average_rating": ("rating", "mean"),
    }
    if "category" in valid.columns:
        agg_dict["category"] = ("category", "first")
    if "platform" in valid.columns:
        agg_dict["platform"] = ("platform", "first")
    if "like_count" in valid.columns:
        agg_dict["total_likes"] = ("like_count", "sum")

    product_summary = (
        valid.groupby("product_name", dropna=False)
        .agg(**agg_dict)
        .sort_values(["average_rating", "reviews"], ascending=[True, False])
        .reset_index()
    )
    product_summary["average_rating"] = product_summary["average_rating"].round(2)
    st.dataframe(product_summary, use_container_width=True, hide_index=True)

    st.markdown("#### Xu hướng sentiment theo thời gian")
    trend = valid.dropna(subset=["created_at"]).copy()
    if not trend.empty:
        trend["sentiment"] = sentiment_label(trend["rating"]).map(
            {"negative": "Tiêu cực", "neutral": "Trung tính", "positive": "Tích cực"}
        )
        trend["tháng"] = trend["created_at"].dt.to_period("M").astype(str)
        chart = trend.pivot_table(
            index="tháng", columns="sentiment", values="review_id",
            aggfunc="count", fill_value=0,
        ).reindex(columns=["Tiêu cực", "Trung tính", "Tích cực"], fill_value=0)
        st.line_chart(chart)


def render_attention(view: pd.DataFrame, aspects: dict) -> None:
    st.markdown("### Review cần chú ý")
    st.caption("Các tín hiệu giúp seller ưu tiên kiểm tra và cải thiện sản phẩm/dịch vụ.")
    if view.empty:
        st.info("Không có dữ liệu phù hợp.")
        return

    details, _ = build_report(view, aspects)
    attention = details[
        (details["rating"] <= 2)
        | details["rating_text_contradiction"].ne("NONE")
    ].copy()
    attention["loai_tin_hieu"] = attention.apply(
        lambda row: "Nghịch lý rating" if row["rating_text_contradiction"] != "NONE" else "Rating thấp",
        axis=1,
    )
    col_c1, col_c2 = st.columns([1, 2])
    with col_c1:
        st.metric("Số review cần kiểm tra", f"{len(attention):,}")
    with col_c2:
        sig_options = ["Tất cả", "Rating thấp (1–2★)", "Nghịch lý rating"]
        sig_choice = st.selectbox("Lọc loại tín hiệu", sig_options, label_visibility="collapsed")
    
    if sig_choice == "Rating thấp (1–2★)":
        attention = attention[attention["loai_tin_hieu"] == "Rating thấp"]
    elif sig_choice == "Nghịch lý rating":
        attention = attention[attention["loai_tin_hieu"] == "Nghịch lý rating"]

    columns = [
        "loai_tin_hieu", "created_at", "platform", "category", "product_name",
        "reviewer_name", "rating", "like_count", "comment_text_raw", "rating_text_contradiction",
    ]
    st.dataframe(
        attention[[column for column in columns if column in attention.columns]],
        use_container_width=True,
        hide_index=True,
    )
    if not attention.empty:
        st.download_button(
            "Tải danh sách cần chú ý",
            attention.to_csv(index=False).encode("utf-8-sig"),
            "reviews_attention.csv",
            "text/csv",
        )


def render_product_analysis(view: pd.DataFrame, aspects: dict) -> None:
    st.markdown("### Phân tích theo sản phẩm")
    if view.empty:
        st.info("Không có dữ liệu phù hợp.")
        return
    product_names = sorted(view["product_name"].dropna().astype(str).unique())
    product = st.selectbox("Chọn sản phẩm", product_names)
    selected = view[view["product_name"].astype(str) == product]
    valid = selected[selected.get("is_valid_for_analysis", True).astype(bool)]

    # Product Metadata Badge
    cat_val = selected["category"].dropna().iloc[0] if "category" in selected.columns and not selected["category"].dropna().empty else "Chưa phân loại"
    plat_val = selected["platform"].dropna().iloc[0] if "platform" in selected.columns and not selected["platform"].dropna().empty else "N/A"
    max_likes = int(selected["like_count"].max()) if "like_count" in selected.columns and not selected["like_count"].dropna().empty else 0

    st.markdown(
        f'<div style="background:#fff;border:1px solid #e6e2d9;border-radius:10px;padding:0.6rem 0.9rem;margin-bottom:0.9rem;font-size:0.9rem;">'
        f'📁 Ngành hàng: <b>{cat_val}</b> &nbsp;|&nbsp; 🛒 Nền tảng: <b>{plat_val}</b> &nbsp;|&nbsp; 👍 Lượt thích cao nhất: <b>{max_likes:,}</b>'
        f'</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(3)
    with cols[0]:
        metric("Reviews", f"{len(selected):,}", "sản phẩm đã chọn")
    with cols[1]:
        metric("Rating", f"{selected['rating'].mean():.2f} / 5", "trung bình")
    with cols[2]:
        metric("Review thấp", f"{int((selected['rating'] <= 2).sum()):,}", "cần xem lại")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Sentiment theo rating")
        st.bar_chart(sentiment_summary(valid if not valid.empty else selected), color="#ff725e")
    with right:
        st.markdown("#### Aspect được nhắc đến")
        selected_valid = valid_rows(selected)
        text_column = "comment_text_clean" if "comment_text_clean" in selected_valid else "comment_text_raw"
        aspect_rows = []
        for aspect, keywords in aspects.items():
            count = selected_valid[text_column].fillna("").astype(str).map(
                lambda text: bool(build_aspect_sentiment_rows(text, {aspect: keywords}))
            ).sum()
            aspect_rows.append({"aspect": aspect, "reviews": int(count)})
        st.bar_chart(pd.DataFrame(aspect_rows).set_index("aspect"), color="#9bcf3a")

    st.markdown("#### Sentiment theo thời gian")
    product_trend = valid.dropna(subset=["created_at"]).copy()
    if not product_trend.empty:
        product_trend["sentiment"] = sentiment_label(product_trend["rating"]).map(
            {"negative": "Tiêu cực", "neutral": "Trung tính", "positive": "Tích cực"}
        )
        product_trend["tháng"] = product_trend["created_at"].dt.to_period("M").astype(str)
        chart = product_trend.pivot_table(
            index="tháng", columns="sentiment", values="review_id",
            aggfunc="count", fill_value=0,
        ).reindex(columns=["Tiêu cực", "Trung tính", "Tích cực"], fill_value=0)
        st.line_chart(chart)

    st.markdown("#### Review nổi bật của sản phẩm")
    col_filter_sort, _ = st.columns([2, 2])
    with col_filter_sort:
        sort_p = st.selectbox("Sắp xếp review", ["Nhiều lượt thích nhất", "Rating thấp nhất (1–2★)", "Rating cao nhất (5★)", "Mới nhất"], key="prod_sort")

    prod_reviews = selected.copy()
    if sort_p == "Nhiều lượt thích nhất" and "like_count" in prod_reviews.columns:
        prod_reviews = prod_reviews.sort_values(by="like_count", ascending=False)
    elif sort_p == "Rating thấp nhất (1–2★)":
        prod_reviews = prod_reviews.sort_values(by="rating", ascending=True)
    elif sort_p == "Rating cao nhất (5★)":
        prod_reviews = prod_reviews.sort_values(by="rating", ascending=False)
    elif sort_p == "Mới nhất" and "created_at" in prod_reviews.columns:
        prod_reviews = prod_reviews.sort_values(by="created_at", ascending=False)

    display_cols = [c for c in ["created_at", "reviewer_name", "rating", "like_count", "filter_status", "comment_text_raw"] if c in prod_reviews.columns]
    st.dataframe(prod_reviews[display_cols].head(50), use_container_width=True, hide_index=True)


def render_alerts(view: pd.DataFrame, aspects: dict) -> None:
    st.markdown("### Trung tâm cảnh báo & Insight")
    valid = valid_rows(view)
    if valid.empty:
        st.info("Chưa có review hợp lệ để tạo cảnh báo.")
        return
    details, summary = build_report(view, aspects)
    negative_rate = float((valid["rating"] <= 2).mean() * 100)
    alerts = []
    if negative_rate >= 15:
        alerts.append(("Cao", f"Tỷ lệ review 1–2 sao đang ở mức {negative_rate:.1f}%."))
    if summary["rating_text_contradictions"] > 0:
        alerts.append(("Trung bình", f"Có {summary['rating_text_contradictions']:,} review có rating không khớp nội dung (nghịch lý rating)."))
    coverage = summary["aspect_coverage"]
    if coverage:
        top_aspect = max(coverage, key=lambda name: coverage[name]["reviews"])
        alerts.append(("Theo dõi", f"Aspect được nhắc đến nhiều nhất là '{top_aspect}' ({coverage[top_aspect]['reviews']:,} lượt)."))

    # Cảnh báo theo ngành hàng
    if "category" in valid.columns:
        cat_neg = valid.groupby("category")["rating"].apply(lambda s: (s <= 2).mean() * 100)
        if not cat_neg.empty:
            worst_cat = cat_neg.idxmax()
            worst_rate = cat_neg.max()
            if worst_rate >= 8:
                alerts.append(("Cảnh báo ngành hàng", f"Ngành hàng '{worst_cat}' có tỷ lệ review 1–2 sao cao nhất: {worst_rate:.1f}%."))

    # Cảnh báo review có like cao (high impact)
    if "like_count" in valid.columns:
        high_like_neg = valid[(valid["rating"] <= 2) & (valid["like_count"] >= 5)]
        if not high_like_neg.empty:
            alerts.append(("Tác động cao", f"Có {len(high_like_neg):,} review tiêu cực có từ 5 lượt thích trở lên; cần xử lý ngay vì ảnh hưởng lớn đến quyết định mua hàng."))

    if not alerts:
        st.success("Chưa phát hiện cảnh báo nổi bật trong phạm vi đang xem.")
    for level, message in alerts:
        (st.error if level in ["Cao", "Tác động cao"] else st.warning)(f"**{level}:** {message}")

    st.markdown("#### Gợi ý hành động")
    if negative_rate >= 15:
        st.write("• **Chất lượng**: Ưu tiên đọc các review rating thấp và kiểm tra sản phẩm hoặc quy trình dịch vụ liên quan.")
    if coverage:
        st.write(f"• **Khía cạnh nóng**: Tập trung phân tích các review đề cập đến aspect **{top_aspect}** để tìm nguyên nhân lặp lại.")
    st.write("• **Phân bổ nhân lực**: Dùng bộ lọc ngành hàng và sản phẩm kết hợp nút xuất dữ liệu để giao cho bộ phận liên quan giải quyết.")


def render_labeling(view: pd.DataFrame, aspects: dict) -> None:
    st.markdown("### Gán nhãn thủ công")
    st.caption("Nhãn này tạo tập dữ liệu kiểm tra/huấn luyện; dữ liệu gốc không bị thay đổi.")
    valid = valid_rows(view)
    if valid.empty:
        st.info("Chưa có review hợp lệ.")
        return
    if "manual_labels" not in st.session_state:
        st.session_state.manual_labels = []
    review_ids = valid["review_id"].astype(str).tolist()
    review_id = st.selectbox("Chọn review", review_ids)
    row = valid[valid["review_id"].astype(str) == review_id].iloc[0]

    # Context Card
    prod_name = row.get("product_name", "N/A")
    cat_name = row.get("category", "N/A")
    star = row.get("rating", "N/A")
    user = row.get("reviewer_name", "Ẩn danh")
    likes = row.get("like_count", 0)
    st.markdown(
        f'<div style="background:#fff;border:1px solid #e6e2d9;border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.8rem;font-size:0.9rem;">'
        f'📦 <b>Sản phẩm:</b> {prod_name} &nbsp;|&nbsp; 📁 <b>Ngành hàng:</b> {cat_name} &nbsp;|&nbsp; ⭐ <b>Rating:</b> {star}★ &nbsp;|&nbsp; 👤 <b>Người viết:</b> {user} ({likes} 👍)'
        f'</div>',
        unsafe_allow_html=True
    )
    st.info(str(row.get("comment_text_raw", "")))
    with st.form("manual_label_form"):
        sentiment = st.selectbox("Nhãn sentiment", ["Tiêu cực", "Trung tính", "Tích cực"])
        selected_aspects = st.multiselect("Aspect", list(aspects), format_func=lambda value: value.title())
        submitted = st.form_submit_button("Lưu nhãn")
    if submitted:
        record = {
            "review_id": review_id,
            "product_name": prod_name,
            "category": cat_name,
            "rating": star,
            "text": str(row.get("comment_text_raw", "")),
            "sentiment_manual": sentiment,
            "aspects_manual": ", ".join(selected_aspects),
        }
        st.session_state.manual_labels = [
            item for item in st.session_state.manual_labels if item["review_id"] != review_id
        ]
        st.session_state.manual_labels.append(record)
        st.success("Đã lưu nhãn trong phiên làm việc.")
    labels = pd.DataFrame(st.session_state.manual_labels)
    if not labels.empty:
        st.dataframe(labels, use_container_width=True, hide_index=True)
        st.download_button(
            "Xuất nhãn thủ công",
            labels.to_csv(index=False).encode("utf-8-sig"),
            "manual_labels.csv",
            "text/csv",
        )


def render_model_lab(view: pd.DataFrame) -> None:
    st.markdown("### Model Lab · Sentiment")
    st.caption("Baseline minh họa: TF-IDF kết hợp Logistic Regression hoặc Linear SVM.")
    data = valid_rows(view).copy()
    data["text"] = data.get("comment_text_segmented", data["comment_text_raw"]).fillna("").astype(str)
    data["label"] = sentiment_label(data["rating"])
    data = data[data["label"].isin(["negative", "neutral", "positive"]) & data["text"].str.strip().ne("")]
    counts = data["label"].value_counts()
    if len(data) < 12 or counts.min() < 2:
        st.warning("Cần ít nhất 12 review hợp lệ và tối thiểu 2 review cho mỗi lớp để huấn luyện.")
        return
    model_name = st.selectbox("Mô hình", ["Logistic Regression", "Linear SVM"])
    test_size = st.slider("Tỷ lệ test", 0.2, 0.4, 0.25, 0.05)
    if st.button("Huấn luyện và đánh giá", type="primary"):
        x_train, x_test, y_train, y_test = train_test_split(
            data["text"], data["label"], test_size=test_size, random_state=42, stratify=data["label"]
        )
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced") if model_name == "Logistic Regression" else LinearSVC(class_weight="balanced")
        model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", classifier),
        ])
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        metrics = st.columns(4)
        with metrics[0]:
            metric("Accuracy", f"{accuracy_score(y_test, predictions):.3f}", "tập test")
        with metrics[1]:
            metric("Precision", f"{report['macro avg']['precision']:.3f}", "macro average")
        with metrics[2]:
            metric("Recall", f"{report['macro avg']['recall']:.3f}", "macro average")
        with metrics[3]:
            metric("Macro-F1", f"{report['macro avg']['f1-score']:.3f}", "chỉ số chính")
        labels = ["negative", "neutral", "positive"]
        matrix = pd.DataFrame(confusion_matrix(y_test, predictions, labels=labels), index=labels, columns=labels)
        st.markdown("#### Confusion Matrix")
        st.dataframe(matrix, use_container_width=True)
        st.json({label: report[label] for label in labels if label in report})
        
        # Lưu mô hình bền vững ra đĩa
        os.makedirs("models", exist_ok=True)
        import joblib
        joblib.dump(model, "models/sentiment_model.joblib")
        st.session_state["trained_model"] = model
        st.success("Mô hình đã được huấn luyện và lưu an toàn tại `models/sentiment_model.joblib`. Bạn có thể thử nghiệm ngay tại trang Mô phỏng!")
    elif os.path.exists("models/sentiment_model.joblib") and "trained_model" not in st.session_state:
        try:
            import joblib
            st.session_state["trained_model"] = joblib.load("models/sentiment_model.joblib")
        except Exception:
            pass
    if "trained_model" in st.session_state:
        st.info("Mô hình đã sẵn sàng. Bạn có thể sang trang 'Mô phỏng bình luận' để dự đoán câu review bất kỳ.")


def render_aspect_analysis(view: pd.DataFrame, aspects: dict) -> None:
    st.markdown("### Sentiment theo khía cạnh")
    st.caption("Baseline giải thích được: phát hiện aspect bằng lexicon và suy ra sắc thái từ tín hiệu trong câu.")
    valid = valid_rows(view)
    if valid.empty:
        st.info("Chưa có review hợp lệ.")
        return
    text_column = "comment_text_clean" if "comment_text_clean" in valid else "comment_text_raw"
    rows = []
    for _, row in valid.iterrows():
        meta = {
            "product_name": row.get("product_name", ""),
            "category": row.get("category", ""),
            "rating": row.get("rating", 0),
            "reviewer_name": row.get("reviewer_name", ""),
            "like_count": row.get("like_count", 0),
            "comment_text": row.get(text_column, ""),
        }
        rows.extend(build_aspect_sentiment_rows(str(row[text_column]), aspects, meta=meta))
    if not rows:
        st.info("Chưa phát hiện aspect trong phạm vi đang xem.")
        return
    result = pd.DataFrame(rows)
    counts = result.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
    st.bar_chart(
        counts.reindex(columns=["Tiêu cực", "Trung tính", "Hỗn hợp", "Tích cực"], fill_value=0)
    )

    st.markdown("#### Chi tiết các nhận xét theo khía cạnh")
    c_asp, c_sent = st.columns(2)
    with c_asp:
        selected_aspects = st.multiselect("Lọc theo Aspect", list(aspects.keys()), default=[])
    with c_sent:
        selected_sent = st.multiselect("Lọc theo Sắc thái", ["Tiêu cực", "Trung tính", "Hỗn hợp", "Tích cực"], default=[])

    filtered_res = result.copy()
    if selected_aspects:
        filtered_res = filtered_res[filtered_res["aspect"].isin(selected_aspects)]
    if selected_sent:
        filtered_res = filtered_res[filtered_res["sentiment"].isin(selected_sent)]

    show_cols = [c for c in ["aspect", "sentiment", "product_name", "category", "rating", "like_count", "matched_keywords", "comment_text"] if c in filtered_res.columns]
    st.dataframe(filtered_res[show_cols], use_container_width=True, hide_index=True)


def render_report(view: pd.DataFrame, aspects: dict) -> None:
    st.markdown("### Báo cáo seller tổng hợp")
    if view.empty:
        st.info("Không có dữ liệu để tạo báo cáo.")
        return
    valid = valid_rows(view)
    _, summary = build_report(view, aspects)

    st.markdown("#### 1. Chỉ số tổng thể")
    report_rows = [
        {"Chỉ số": "Tổng số review thu thập", "Giá trị": len(view)},
        {"Chỉ số": "Review hợp lệ cho phân tích", "Giá trị": len(valid)},
        {"Chỉ số": "Review bị lọc (Spam / Trùng lặp)", "Giá trị": len(view) - len(valid)},
        {"Chỉ số": "Rating trung bình toàn sàn", "Giá trị": f"{round(float(view['rating'].mean()), 2)} / 5"},
        {"Chỉ số": "Số lượng review tiêu cực (1–2 sao)", "Giá trị": int((view["rating"] <= 2).sum())},
        {"Chỉ số": "Tỷ lệ review tiêu cực", "Giá trị": f"{(view['rating'] <= 2).mean() * 100:.1f}%"},
        {"Chỉ số": "Số review nghịch lý rating (text trái ngược sao)", "Giá trị": summary["rating_text_contradictions"]},
    ]
    report_df = pd.DataFrame(report_rows)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    if "category" in valid.columns:
        st.markdown("#### 2. Thống kê theo ngành hàng")
        cat_stats = valid.groupby("category").agg(
            total_reviews=("review_id", "count"),
            avg_rating=("rating", "mean"),
            negative_reviews=("rating", lambda s: int((s <= 2).sum())),
            negative_rate=("rating", lambda s: f"{(s <= 2).mean() * 100:.1f}%"),
            total_likes=("like_count", "sum") if "like_count" in valid.columns else ("review_id", "count")
        ).reset_index().rename(columns={
            "category": "Ngành hàng",
            "total_reviews": "Số review",
            "avg_rating": "Rating TB",
            "negative_reviews": "Review 1–2★",
            "negative_rate": "Tỷ lệ tiêu cực",
            "total_likes": "Tổng like"
        })
        cat_stats["Rating TB"] = cat_stats["Rating TB"].round(2)
        st.dataframe(cat_stats, use_container_width=True, hide_index=True)

    if "platform" in valid.columns:
        st.markdown("#### 3. Thống kê theo nền tảng")
        plat_stats = valid.groupby("platform").agg(
            total_reviews=("review_id", "count"),
            avg_rating=("rating", "mean"),
            negative_rate=("rating", lambda s: f"{(s <= 2).mean() * 100:.1f}%")
        ).reset_index().rename(columns={
            "platform": "Nền tảng",
            "total_reviews": "Số review",
            "avg_rating": "Rating TB",
            "negative_rate": "Tỷ lệ tiêu cực"
        })
        plat_stats["Rating TB"] = plat_stats["Rating TB"].round(2)
        st.dataframe(plat_stats, use_container_width=True, hide_index=True)

    st.markdown("#### 4. Phủ sóng theo Aspect (Khía cạnh)")
    coverage_rows = [
        {"Khía cạnh": asp, "Số review đề cập": data["reviews"], "Tỷ lệ (%)": f"{data['rate_percent']}%"}
        for asp, data in summary.get("aspect_coverage", {}).items()
    ]
    st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)

    st.download_button(
        "Xuất toàn bộ báo cáo CSV",
        report_df.to_csv(index=False).encode("utf-8-sig"),
        "seller_summary_report.csv",
        "text/csv",
    )


def render_review_explorer(view: pd.DataFrame) -> None:
    st.markdown("### Tra cứu review")
    col_search, col_sort = st.columns([3, 1])
    with col_search:
        search = st.text_input("Tìm trong nội dung", placeholder="Ví dụ: giao hàng, chất lượng, đóng gói...", label_visibility="collapsed")
    with col_sort:
        sort_by = st.selectbox(
            "Sắp xếp theo",
            ["Mới nhất", "Lượt thích nhiều nhất (Helpful)", "Rating thấp nhất (1★)", "Rating cao nhất (5★)"],
            label_visibility="collapsed"
        )

    status_options = sorted(view["filter_status"].dropna().astype(str).unique()) if "filter_status" in view else []
    statuses = st.multiselect("Trạng thái", status_options, default=status_options)
    table = view.copy()
    if search:
        table = table[table["comment_text_raw"].astype(str).str.contains(search, case=False, na=False)]
    if statuses and "filter_status" in table:
        table = table[table["filter_status"].astype(str).isin(statuses)]

    if sort_by == "Lượt thích nhiều nhất (Helpful)" and "like_count" in table:
        table = table.sort_values(by="like_count", ascending=False)
    elif sort_by == "Rating thấp nhất (1★)":
        table = table.sort_values(by="rating", ascending=True)
    elif sort_by == "Rating cao nhất (5★)":
        table = table.sort_values(by="rating", ascending=False)
    elif sort_by == "Mới nhất" and "created_at" in table:
        table = table.sort_values(by="created_at", ascending=False)

    columns = [
        "created_at", "platform", "category", "product_name",
        "reviewer_name", "rating", "like_count", "filter_status",
        "comment_text_raw", "comment_text_clean"
    ]
    st.caption(f"Hiển thị {len(table):,} review")
    st.dataframe(table[[column for column in columns if column in table.columns]], use_container_width=True, hide_index=True)
    if not table.empty:
        st.download_button("Xuất kết quả lọc", table.to_csv(index=False).encode("utf-8-sig"), "reviews_filtered.csv", "text/csv")


def render_simulator(aspects: dict) -> None:
    st.markdown("### Mô phỏng bình luận")
    st.caption("Dùng để kiểm thử pipeline; dữ liệu mô phỏng không được trộn vào thống kê review thật.")
    examples = {
        "Tích cực": "Sản phẩm dùng rất tốt, đóng gói cẩn thận và giao hàng nhanh.",
        "Tiêu cực": "Hàng bị lỗi, giao hàng chậm và shop phản hồi rất lâu.",
        "Nhiều khía cạnh": "Sản phẩm tốt nhưng giao hàng chậm, hộp bị móp và giá hơi đắt.",
        "Teencode + emoji": "sp okela nhưng ship hơi lâu nha 😐",
        "Spam": "hình ảnh mang tính chất nhận xu",
    }
    selected_example = st.selectbox("Kịch bản mẫu", ["Tự nhập", *examples])
    default_text = "" if selected_example == "Tự nhập" else examples[selected_example]
    text = st.text_area("Nội dung bình luận", value=default_text, height=120)
    rating = st.slider("Rating giả lập", 1, 5, 4)
    if st.button("Phân tích bình luận", type="primary"):
        if not text.strip():
            st.warning("Hãy nhập nội dung bình luận.")
            return
        pipeline = PreprocessingPipeline()
        processed = pipeline.process_single_text(text)
        sample = pd.DataFrame([{
            "review_id": "SIMULATED_001",
            "rating": rating,
            "comment_text_raw": text,
            **processed,
        }])
        feature_pipeline = FeatureEngineeringPipeline(str(ASPECT_DATA))
        labels, _ = feature_pipeline.build_labels(sample)
        labels = labels.iloc[0]

        model = st.session_state.get("trained_model")
        if model is None and os.path.exists("models/sentiment_model.joblib"):
            try:
                import joblib
                model = joblib.load("models/sentiment_model.joblib")
                st.session_state["trained_model"] = model
            except Exception:
                model = None

        ai_sentiment = None
        if model is not None:
            try:
                seg_text = processed.get("comment_text_segmented") or processed.get("comment_text_clean") or text
                pred = model.predict([seg_text])[0]
                vn_map = {"negative": "Tiêu cực", "neutral": "Trung tính", "positive": "Tích cực"}
                ai_sentiment = vn_map.get(pred, str(pred))
            except Exception:
                ai_sentiment = None

        st.markdown("#### Kết quả mô phỏng")
        if ai_sentiment:
            cols = st.columns(4)
            with cols[0]:
                metric("Sentiment baseline", str(labels["sentiment_name"]), f"theo rating {rating} sao")
            with cols[1]:
                metric("Sentiment AI", ai_sentiment, "mô hình ML đã train")
            with cols[2]:
                metric("Trạng thái", "SPAM" if processed["is_spam"] else "VALID", processed["spam_reason"])
            with cols[3]:
                matched = sum(int(labels[f"aspect_{name}"]) for name in aspects)
                metric("Aspect phát hiện", str(matched), "theo lexicon")
        else:
            cols = st.columns(3)
            with cols[0]:
                metric("Sentiment baseline", str(labels["sentiment_name"]), f"theo rating {rating} sao")
            with cols[1]:
                metric("Trạng thái", "SPAM" if processed["is_spam"] else "VALID", processed["spam_reason"])
            with cols[2]:
                matched = sum(int(labels[f"aspect_{name}"]) for name in aspects)
                metric("Aspect phát hiện", str(matched), "theo lexicon")
        aspect_result = pd.DataFrame(
            [{
                "aspect": name,
                "phát hiện": "Có" if labels[f"aspect_{name}"] else "Không",
                "từ khóa khớp": labels[f"aspect_{name}_matches"],
            } for name in aspects]
        )
        st.dataframe(aspect_result, use_container_width=True, hide_index=True)
        with st.expander("Xem từng bước xử lý"):
            st.write({"clean": processed["comment_text_clean"], "segmented": processed["comment_text_segmented"]})


def main() -> None:
    st.sidebar.markdown(
        '<div style="font-family:Space Grotesk;font-size:1.35rem;font-weight:700;letter-spacing:-.04em">✦ Customer Voice</div>'
        '<div style="color:#9fb0ad;font-size:.78rem;margin:.25rem 0 1.4rem">Seller intelligence workspace</div>',
        unsafe_allow_html=True,
    )
    data_path = st.sidebar.text_input("Đường dẫn dữ liệu", str(DEFAULT_DATA))
    file_mtime = get_file_mtime(data_path)
    if st.sidebar.button("Làm mới dữ liệu", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    try:
        df = load_data(data_path, file_mtime)
    except Exception as error:
        st.error(f"Không thể đọc dữ liệu: {error}")
        st.stop()

    st.sidebar.caption(f"Cập nhật: {datetime.fromtimestamp(file_mtime).strftime('%d/%m/%Y %H:%M') if file_mtime else 'chưa có'}")
    st.sidebar.caption(f"Đang nạp **{len(df):,}** review")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Bộ lọc dữ liệu")
    platforms = sorted(df["platform"].dropna().astype(str).unique())
    selected_platforms = st.sidebar.multiselect("Nền tảng", platforms, default=platforms)
    
    categories = sorted([str(c) for c in df["category"].dropna().unique() if str(c).strip()]) if "category" in df.columns else []
    selected_categories = st.sidebar.multiselect("Ngành hàng", categories, default=categories) if categories else []

    cat_df = df[df["category"].astype(str).isin(selected_categories)] if (categories and selected_categories) else df
    products = sorted(cat_df["product_name"].dropna().astype(str).unique()) if "product_name" in cat_df.columns else []
    selected_products = st.sidebar.multiselect("Sản phẩm", products, default=[])
    ratings = st.sidebar.slider("Khoảng rating", 1, 5, (1, 5))
    valid_only = st.sidebar.checkbox("Chỉ review hợp lệ", value=False)
    
    view = df[df["platform"].astype(str).isin(selected_platforms) & df["rating"].between(*ratings)]
    if categories:
        view = view[view["category"].astype(str).isin(selected_categories)]
    if selected_products:
        view = view[view["product_name"].astype(str).isin(selected_products)]
    if valid_only and "is_valid_for_analysis" in view:
        view = view[view["is_valid_for_analysis"].astype(bool)]

    st.markdown(
        '<div class="hero"><div class="hero-copy"><div class="hero-label">Seller intelligence / workspace</div>'
        '<div class="hero-title">Hiểu khách hàng. Cải thiện đúng chỗ.</div>'
        '<p class="hero-note">Biến hàng nghìn review thành tín hiệu rõ ràng về sản phẩm, dịch vụ và trải nghiệm mua hàng.</p>'
        '</div><div class="hero-mark">✦</div></div>',
        unsafe_allow_html=True,
    )
    aspects = load_aspects(str(ASPECT_DATA))
    st.markdown('<div class="nav-label">Không gian làm việc</div>', unsafe_allow_html=True)
    page = st.radio(
        "Đi tới",
        [
            "Tổng quan",
            "Theo sản phẩm",
            "Sentiment theo aspect",
            "Review cần chú ý",
            "Cảnh báo & insight",
            "Tra cứu review",
            "Gán nhãn thủ công",
            "Model Lab",
            "Báo cáo",
            "Mô phỏng bình luận",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )
    if page == "Tổng quan":
        render_overview(view, aspects)
    elif page == "Theo sản phẩm":
        render_product_analysis(view, aspects)
    elif page == "Sentiment theo aspect":
        render_aspect_analysis(view, aspects)
    elif page == "Review cần chú ý":
        render_attention(view, aspects)
    elif page == "Cảnh báo & insight":
        render_alerts(view, aspects)
    elif page == "Tra cứu review":
        render_review_explorer(view)
    elif page == "Gán nhãn thủ công":
        render_labeling(view, aspects)
    elif page == "Model Lab":
        render_model_lab(view)
    elif page == "Báo cáo":
        render_report(view, aspects)
    else:
        render_simulator(aspects)


if __name__ == "__main__":
    main()
