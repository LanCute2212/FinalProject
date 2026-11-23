import os
import re
from datetime import datetime
from pathlib import Path

import joblib
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
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"

st.set_page_config(
    page_title="Customer Voice AI | Hệ Thống Phân Tích Đánh Giá",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Scoped: Giao diện chuẩn Enterprise SaaS, không phá vỡ widget mặc định
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1320px;
    }
    
    /* Header chính */
    .app-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        color: #ffffff;
        margin-bottom: 1.25rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    }
    .app-header h1 {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Thẻ tóm tắt điều hành (Executive Summary) */
    .exec-summary {
        background: #ffffff;
        border: 1px solid #bfdbfe;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 1rem 1.35rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.04);
    }
    .exec-summary-title {
        color: #1e40af;
        font-size: 0.92rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.45rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .exec-summary-content {
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.55;
        margin: 0;
    }
    
    /* Thẻ Metric hiện đại */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        min-height: 102px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        margin-bottom: 0.5rem;
    }
    .metric-card-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-card-value {
        color: #0f172a;
        font-size: 1.85rem;
        font-weight: 700;
        margin-top: 0.25rem;
        line-height: 1.1;
    }
    .metric-card-note {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    
    /* Thẻ giải thích học thuật */
    .theory-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin: 0.85rem 0;
        font-size: 0.86rem;
        color: #334155;
        line-height: 1.5;
    }
    
    /* Radio nhóm phân hệ lớn */
    [data-testid="stMain"] .stRadio [role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.2rem;
    }
    [data-testid="stMain"] .stRadio [role="radiogroup"] label {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 0.55rem 1.1rem;
        color: #334155 !important;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.15s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    [data-testid="stMain"] .stRadio [role="radiogroup"] label:hover {
        border-color: #2563eb;
        color: #2563eb !important;
    }
    [data-testid="stMain"] .stRadio [role="radiogroup"] label:has(input:checked) {
        background: #2563eb;
        border-color: #2563eb;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    
    /* Định dạng tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 2.7rem;
        font-weight: 600;
        font-size: 0.92rem;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom-color: #2563eb !important;
    }
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
    del mtime
    frame = pd.read_csv(path)
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce").fillna(0).astype(int)
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    return frame


def render_metric(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-card-label">{label}</div>'
        f'<div class="metric-card-value">{value}</div>'
        f'<div class="metric-card-note">{note}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def valid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "is_valid_for_analysis" not in frame.columns:
        return frame
    return frame[frame["is_valid_for_analysis"].astype(bool)]


def sentiment_label_vn(rating: pd.Series) -> pd.Series:
    return pd.cut(
        rating,
        bins=[0, 2, 3, 5],
        labels=["Tiêu cực", "Trung tính", "Tích cực"],
    ).astype(str)


def sentiment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    labels = pd.cut(
        frame["rating"],
        bins=[0, 2, 3, 5],
        labels=["Tiêu cực", "Trung tính", "Tích cực"],
    )
    return labels.value_counts().reindex(["Tiêu cực", "Trung tính", "Tích cực"], fill_value=0).rename_axis("Sắc thái").to_frame("Số lượng đánh giá")


POSITIVE_ASPECT_TERMS = ("tốt", "hay", "đẹp", "nhanh", "ổn", "hài lòng", "ưng ý", "mượt", "đáng tiền", "chuẩn", "chính hãng")
NEGATIVE_ASPECT_TERMS = ("lỗi", "hỏng", "chậm", "tệ", "thất vọng", "móp", "trầy", "đắt", "kém", "chập chờn", "kém chất lượng")


def build_aspect_sentiment_rows(text: str, aspect_keywords: dict, meta: dict = None) -> list[dict]:
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
        row_dict = {
            "aspect": aspect,
            "sentiment": sentiment,
            "matched_keywords": ", ".join(matches),
        }
        if meta:
            row_dict.update(meta)
        rows.append(row_dict)
    return rows


# ==========================================
# PHÂN HỆ 1: BÁO CÁO THỊ TRƯỜNG & TỔNG QUAN
# ==========================================

def render_overview_landscape(view: pd.DataFrame, valid: pd.DataFrame, aspects: dict) -> None:
    st.markdown("#### Bức tranh toàn cảnh về phản hồi khách hàng")
    
    negative = int((valid["rating"] <= 2).sum()) if not valid.empty else 0
    try:
        _, report = build_report(view, aspects)
        contradictions = report["rating_text_contradictions"]
    except Exception:
        contradictions = 0

    cols = st.columns(4)
    with cols[0]:
        render_metric("Tổng đánh giá đang xem", f"{len(view):,}", "Sau khi lọc dữ liệu")
    with cols[1]:
        render_metric("Điểm hài lòng trung bình", f"{view['rating'].mean():.2f} / 5", "Tín hiệu tổng thể")
    with cols[2]:
        render_metric("Đánh giá cần ưu tiên (1–2★)", f"{negative:,}", f"Chiếm {(negative/len(valid)*100 if len(valid) else 0):.1f}%")
    with cols[3]:
        render_metric("Nghịch lý đánh giá", f"{contradictions:,}", "Chấm sao trái ngược lời văn")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Phân bố cảm xúc tổng thể (Theo điểm sao)")
        st.bar_chart(sentiment_summary(valid if not valid.empty else view), color="#2563eb", height=270)
    with c2:
        st.markdown("##### Trạng thái kiểm định & Lọc rác (Tiền xử lý)")
        status = view.get("filter_status", pd.Series(["VALID"] * len(view))).value_counts()
        status_map = {"VALID": "Hợp lệ", "SPAM_PROMO": "Spam quảng cáo", "SPAM_EMOJI_ONLY": "Spam biểu tượng/rỗng", "DUPLICATE": "Trùng lặp"}
        status.index = [status_map.get(k, str(k)) for k in status.index]
        st.bar_chart(status.rename_axis("Trạng thái").to_frame("Số lượng"), color="#10b981", height=270)

    st.markdown("##### Xu hướng sắc thái cảm xúc theo thời gian")
    trend = valid.dropna(subset=["created_at"]).copy()
    if not trend.empty:
        trend["sentiment"] = sentiment_label_vn(trend["rating"])
        trend["tháng"] = trend["created_at"].dt.to_period("M").astype(str)
        chart = trend.pivot_table(
            index="tháng", columns="sentiment", values="review_id",
            aggfunc="count", fill_value=0,
        ).reindex(columns=["Tiêu cực", "Trung tính", "Tích cực"], fill_value=0)
        st.line_chart(chart, height=260)

    with st.expander("📘 Cơ sở học thuật & Giải thích chỉ số"):
        st.markdown(
            """
            - **Cảm xúc cơ sở (Baseline Sentiment)**: Phân loại theo số sao (1–2★: Tiêu cực, 3★: Trung tính, 4–5★: Tích cực).
            - **Nghịch lý đánh giá (Rating-Text Contradiction)**: Các bình luận chấm 4–5★ nhưng chứa từ khóa tiêu cực nghiêm trọng (hoặc chấm 1–2★ nhưng khen ngợi), phản ánh thói quen chấm sao cảm tính hoặc nhầm lẫn của người mua.
            - **Quy tắc khử rác (Spam Filter)**: Loại bỏ các bình luận xin xu, ký tự vô nghĩa lặp lại hoặc bình luận trùng lặp để bảo vệ độ sạch của mô hình AI.
            """
        )


def render_overview_category(view: pd.DataFrame, valid: pd.DataFrame) -> None:
    st.markdown("#### Phân tích & So sánh theo Ngành hàng")
    if "category" not in valid.columns:
        st.info("Dữ liệu hiện tại chưa có thông tin ngành hàng.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Số lượng đánh giá theo ngành hàng")
        cat_counts = valid["category"].value_counts().rename_axis("Ngành hàng").to_frame("Số lượng đánh giá")
        st.bar_chart(cat_counts, color="#3b82f6", height=280)
    with c2:
        st.markdown("##### Điểm số hài lòng trung bình theo ngành hàng")
        cat_rating = valid.groupby("category")["rating"].mean().round(2).rename_axis("Ngành hàng").to_frame("Điểm TB (1-5★)")
        st.bar_chart(cat_rating, color="#f59e0b", height=280)

    st.markdown("##### Bảng chỉ số chi tiết từng ngành hàng")
    cat_summary = (
        valid.groupby("category", dropna=False)
        .agg(
            total_reviews=("review_id", "count"),
            avg_rating=("rating", "mean"),
            low_rating=("rating", lambda s: int((s <= 2).sum())),
            low_rating_pct=("rating", lambda s: f"{(s <= 2).mean() * 100:.1f}%"),
            total_likes=("like_count", "sum") if "like_count" in valid.columns else ("review_id", "count"),
        )
        .sort_values(["total_reviews"], ascending=[False])
        .reset_index()
    )
    cat_summary["avg_rating"] = cat_summary["avg_rating"].round(2)
    cat_summary.rename(columns={
        "category": "Ngành hàng",
        "total_reviews": "Số đánh giá hợp lệ",
        "avg_rating": "Điểm hài lòng TB",
        "low_rating": "Đánh giá 1–2★",
        "low_rating_pct": "Tỷ lệ tiêu cực",
        "total_likes": "Tổng lượt hữu ích (Like)"
    }, inplace=True)
    st.dataframe(cat_summary, use_container_width=True, hide_index=True)


def render_overview_platform(view: pd.DataFrame, valid: pd.DataFrame) -> None:
    st.markdown("#### So sánh Nền tảng: Tiki vs. Shopee")
    if "platform" not in valid.columns:
        st.info("Dữ liệu hiện tại chưa có thông tin nền tảng sàn.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Phân bổ thị phần dữ liệu")
        plat_counts = valid["platform"].value_counts().rename_axis("Nền tảng").to_frame("Số lượng đánh giá")
        st.bar_chart(plat_counts, color="#6366f1", height=260)
    with c2:
        st.markdown("##### Điểm đánh giá trung bình theo sàn")
        plat_rating = valid.groupby("platform")["rating"].mean().round(2).rename_axis("Nền tảng").to_frame("Điểm TB (1-5★)")
        st.bar_chart(plat_rating, color="#14b8a6", height=260)

    st.markdown("##### Bảng đối chiếu chất lượng dữ liệu giữa các sàn")
    plat_table = (
        valid.groupby("platform")
        .agg(
            total=("review_id", "count"),
            avg_star=("rating", "mean"),
            negative_rate=("rating", lambda s: f"{(s <= 2).mean() * 100:.1f}%"),
            avg_likes=("like_count", "mean") if "like_count" in valid.columns else ("review_id", "count")
        )
        .reset_index()
        .rename(columns={
            "platform": "Nền tảng sàn",
            "total": "Tổng đánh giá hợp lệ",
            "avg_star": "Điểm sao trung bình",
            "negative_rate": "Tỷ lệ không hài lòng (1–2★)",
            "avg_likes": "Lượt thích TB / Review"
        })
    )
    plat_table["Điểm sao trung bình"] = plat_table["Điểm sao trung bình"].round(2)
    if "Lượt thích TB / Review" in plat_table:
        plat_table["Lượt thích TB / Review"] = plat_table["Lượt thích TB / Review"].round(1)
    st.dataframe(plat_table, use_container_width=True, hide_index=True)


# ==========================================
# PHÂN HỆ 2: PHÂN TÍCH KHÍA CẠNH (ABSA)
# ==========================================

def render_absa_product(view: pd.DataFrame, aspects: dict) -> None:
    st.markdown("#### Phân tích chi tiết theo Sản phẩm")
    if view.empty:
        st.info("Không có dữ liệu phù hợp với bộ lọc hiện tại.")
        return

    product_names = sorted(view["product_name"].dropna().astype(str).unique())
    selected_product = st.selectbox("Chọn sản phẩm cần phân tích:", product_names)
    selected = view[view["product_name"].astype(str) == selected_product]
    valid = selected[selected.get("is_valid_for_analysis", True).astype(bool)]

    cat_val = selected["category"].dropna().iloc[0] if "category" in selected.columns and not selected["category"].dropna().empty else "Chưa phân loại"
    plat_val = selected["platform"].dropna().iloc[0] if "platform" in selected.columns and not selected["platform"].dropna().empty else "N/A"
    max_likes = int(selected["like_count"].max()) if "like_count" in selected.columns and not selected["like_count"].dropna().empty else 0

    st.markdown(
        f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:0.75rem 1rem;margin-bottom:1rem;font-size:0.92rem;">'
        f'📦 <b>Sản phẩm:</b> <span style="color:#2563eb;font-weight:600;">{selected_product}</span> &nbsp;|&nbsp; '
        f'📁 <b>Ngành hàng:</b> {cat_val} &nbsp;|&nbsp; '
        f'🛒 <b>Nền tảng:</b> {plat_val} &nbsp;|&nbsp; '
        f'👍 <b>Lượt thích cao nhất:</b> {max_likes:,}'
        f'</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(3)
    with cols[0]:
        render_metric("Tổng số đánh giá", f"{len(selected):,}", "Của riêng sản phẩm này")
    with cols[1]:
        render_metric("Điểm hài lòng trung bình", f"{selected['rating'].mean():.2f} / 5", "Mức độ hài lòng")
    with cols[2]:
        render_metric("Đánh giá tiêu cực (1–2★)", f"{int((selected['rating'] <= 2).sum()):,}", "Cần bộ phận SP kiểm tra")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Phân bố cảm xúc theo điểm sao")
        st.bar_chart(sentiment_summary(valid if not valid.empty else selected), color="#2563eb", height=260)
    with c2:
        st.markdown("##### Khía cạnh được khách hàng nhắc tới nhiều nhất")
        selected_valid = valid_rows(selected)
        text_column = "comment_text_clean" if "comment_text_clean" in selected_valid else "comment_text_raw"
        aspect_rows = []
        for aspect, keywords in aspects.items():
            count = selected_valid[text_column].fillna("").astype(str).map(
                lambda text: bool(build_aspect_sentiment_rows(text, {aspect: keywords}))
            ).sum()
            aspect_rows.append({"Khía cạnh": aspect.title(), "Số lượt nhắc": int(count)})
        st.bar_chart(pd.DataFrame(aspect_rows).set_index("Khía cạnh"), color="#10b981", height=260)

    st.markdown("##### Danh sách đánh giá tiêu biểu của sản phẩm")
    c_sort, _ = st.columns([2, 2])
    with c_sort:
        sort_p = st.selectbox("Sắp xếp đánh giá theo:", ["Nhiều lượt thích nhất (Hữu ích)", "Rating thấp nhất (1–2★)", "Rating cao nhất (5★)", "Mới nhất"])

    prod_reviews = selected.copy()
    if sort_p == "Nhiều lượt thích nhất (Hữu ích)" and "like_count" in prod_reviews.columns:
        prod_reviews = prod_reviews.sort_values(by="like_count", ascending=False)
    elif sort_p == "Rating thấp nhất (1–2★)":
        prod_reviews = prod_reviews.sort_values(by="rating", ascending=True)
    elif sort_p == "Rating cao nhất (5★)":
        prod_reviews = prod_reviews.sort_values(by="rating", ascending=False)
    elif sort_p == "Mới nhất" and "created_at" in prod_reviews.columns:
        prod_reviews = prod_reviews.sort_values(by="created_at", ascending=False)

    display_cols = [c for c in ["created_at", "reviewer_name", "rating", "like_count", "comment_text_raw"] if c in prod_reviews.columns]
    rename_cols = {
        "created_at": "Ngày đánh giá",
        "reviewer_name": "Người mua",
        "rating": "Số sao",
        "like_count": "Lượt thích",
        "comment_text_raw": "Nội dung nhận xét"
    }
    st.dataframe(prod_reviews[display_cols].rename(columns=rename_cols).head(50), use_container_width=True, hide_index=True)


def render_absa_matrix(view: pd.DataFrame, aspects: dict) -> None:
    st.markdown("#### Ma trận Khía cạnh & Sắc thái cảm xúc (ABSA Matrix)")
    valid = valid_rows(view)
    if valid.empty:
        st.info("Chưa có dữ liệu đánh giá hợp lệ.")
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
        st.info("Chưa phát hiện khía cạnh nào trong phạm vi dữ liệu đang lọc.")
        return

    result = pd.DataFrame(rows)
    result["Khía cạnh"] = result["aspect"].str.title()
    counts = result.groupby(["Khía cạnh", "sentiment"]).size().unstack(fill_value=0)
    
    st.markdown("##### Tương quan Cảm xúc trên từng Khía cạnh")
    st.bar_chart(
        counts.reindex(columns=["Tiêu cực", "Trung tính", "Hỗn hợp", "Tích cực"], fill_value=0),
        height=300
    )

    st.markdown("##### Khám phá chi tiết các ý kiến theo khía cạnh")
    c_asp, c_sent = st.columns(2)
    with c_asp:
        selected_aspects = st.multiselect("Lọc khía cạnh:", list(counts.index), default=[])
    with c_sent:
        selected_sent = st.multiselect("Lọc sắc thái:", ["Tiêu cực", "Trung tính", "Hỗn hợp", "Tích cực"], default=[])

    filtered_res = result.copy()
    if selected_aspects:
        filtered_res = filtered_res[filtered_res["Khía cạnh"].isin(selected_aspects)]
    if selected_sent:
        filtered_res = filtered_res[filtered_res["sentiment"].isin(selected_sent)]

    show_cols = [c for c in ["Khía cạnh", "sentiment", "product_name", "category", "rating", "like_count", "matched_keywords", "comment_text"] if c in filtered_res.columns]
    rename_dict = {
        "sentiment": "Sắc thái",
        "product_name": "Sản phẩm",
        "category": "Ngành hàng",
        "rating": "Số sao",
        "like_count": "Lượt thích",
        "matched_keywords": "Từ khóa nhận diện",
        "comment_text": "Nội dung trích đoạn"
    }
    st.dataframe(filtered_res[show_cols].rename(columns=rename_dict), use_container_width=True, hide_index=True)

    with st.expander("📘 Lý thuyết Phân tích Khía cạnh (Aspect-Based Sentiment Analysis - ABSA)"):
        st.markdown(
            """
            - **ABSA là gì?**: Thay vì chỉ gán 1 nhãn cảm xúc duy nhất cho toàn bộ bài viết, ABSA bóc tách cảm xúc cụ thể cho từng thuộc tính của sản phẩm (ví dụ: *'chất lượng tốt'* [Tích cực] nhưng *'giao hàng chậm'* [Tiêu cực]).
            - **Phương pháp Lexicon-Based**: Hệ thống đối soát văn bản với từ điển chuyên gia định nghĩa sẵn các từ khóa khía cạnh (Keywords) kết hợp quy tắc xác định sắc thái xung quanh để giải thích rõ ràng lý do đưa ra kết luận.
            """
        )


def render_absa_simulator(aspects: dict) -> None:
    st.markdown("#### Trình Mô phỏng & Kiểm thử Đánh giá Thời gian thực")
    st.caption("Công cụ thử nghiệm cho phép kiểm tra trực tiếp quy trình tiền xử lý, trích xuất khía cạnh và dự đoán cảm xúc AI.")
    
    examples = {
        "Tích cực hoàn hảo": "Sản phẩm dùng rất tốt, đóng gói cẩn thận và giao hàng siêu nhanh.",
        "Tiêu cực gay gắt": "Hàng bị lỗi nặng, giao hàng quá chậm và shop phản hồi rất thiếu trách nhiệm.",
        "Nhiều khía cạnh trái chiều": "Sản phẩm xài ổn áp nhưng giao hàng chậm rì, hộp bị móp méo và giá hơi đắt.",
        "Tiếng lóng & Emoji (Teencode)": "sp okela lắm nha mn, ship nhanh cực ưng bụng luôn ❤️",
        "Đánh giá rác / Xin xu": "hình ảnh và video chỉ mang tính chất nhận xu ạ",
    }
    selected_example = st.selectbox("Chọn kịch bản mẫu để kiểm tra:", ["Tự nhập nhận xét", *examples])
    default_text = "" if selected_example == "Tự nhập nhận xét" else examples[selected_example]
    text = st.text_area("Nội dung đánh giá cần kiểm thử:", value=default_text, height=105)
    rating = st.slider("Số sao giả lập:", 1, 5, 4)

    if st.button("Phân tích đánh giá ngay", type="primary"):
        if not text.strip():
            st.warning("Vui lòng nhập nội dung đánh giá.")
            return

        pipeline = PreprocessingPipeline()
        processed = pipeline.process_single_text(text)
        sample = pd.DataFrame([{
            "review_id": "SIMULATED_TEST",
            "rating": rating,
            "comment_text_raw": text,
            **processed,
        }])
        feature_pipeline = FeatureEngineeringPipeline(str(ASPECT_DATA))
        labels, _ = feature_pipeline.build_labels(sample)
        labels = labels.iloc[0]

        # Dự đoán từ Model AI đã train
        ai_sentiment = None
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                seg_text = processed.get("comment_text_segmented") or processed.get("comment_text_clean") or text
                pred = model.predict([seg_text])[0]
                vn_map = {"negative": "Tiêu cực", "neutral": "Trung tính", "positive": "Tích cực"}
                ai_sentiment = vn_map.get(pred, str(pred))
            except Exception:
                ai_sentiment = None

        st.markdown("##### Kết quả phân tích chi tiết")
        cols = st.columns(4)
        with cols[0]:
            render_metric("Cảm xúc cơ sở", str(labels["sentiment_name"]), f"Dựa trên {rating} sao")
        with cols[1]:
            render_metric("Dự đoán AI (Máy học)", ai_sentiment if ai_sentiment else "Chưa nạp model", "TF-IDF + Phân loại")
        with cols[2]:
            render_metric("Kiểm tra rác / Spam", "RÁC (SPAM)" if processed["is_spam"] else "HỢP LỆ", processed.get("spam_reason", "Đạt chuẩn"))
        with cols[3]:
            matched = sum(int(labels[f"aspect_{name}"]) for name in aspects)
            render_metric("Khía cạnh tìm thấy", f"{matched} khía cạnh", "Theo từ điển ngôn ngữ")

        aspect_result = pd.DataFrame(
            [{
                "Khía cạnh": name.title(),
                "Xuất hiện trong câu": "Có" if labels[f"aspect_{name}"] else "Không",
                "Từ khóa kích hoạt": labels[f"aspect_{name}_matches"] if labels[f"aspect_{name}_matches"] else "—",
            } for name in aspects]
        )
        st.dataframe(aspect_result, use_container_width=True, hide_index=True)

        with st.expander("🔍 Chi tiết các bước tiền xử lý ngôn ngữ tự nhiên (NLP Pipeline)"):
            st.json({
                "1. Văn bản gốc": text,
                "2. Khử nhiễu & Chuẩn hóa (Cleaned)": processed.get("comment_text_clean", ""),
                "3. Tách từ tiếng Việt (Word Segmented)": processed.get("comment_text_segmented", ""),
                "4. Số từ (Word count)": processed.get("word_count", 0),
                "5. Lý do nếu bị đánh dấu rác": processed.get("spam_reason", "Không")
            })


# ==========================================
# PHÂN HỆ 3: CẢNH BÁO RỦI RO & KHÁCH HÀNG
# ==========================================

def render_risk_attention(view: pd.DataFrame, aspects: dict) -> None:
    st.markdown("#### Đánh giá Cần chú ý Khẩn cấp")
    st.caption("Tổng hợp các đánh giá 1–2 sao hoặc có nghịch lý giữa lời nhận xét và số sao, hỗ trợ bộ phận CSKH xử lý khiếu nại.")
    if view.empty:
        st.info("Không có dữ liệu phù hợp với bộ lọc hiện tại.")
        return

    details, _ = build_report(view, aspects)
    attention = details[
        (details["rating"] <= 2)
        | details["rating_text_contradiction"].ne("NONE")
    ].copy()
    attention["loai_tin_hieu"] = attention.apply(
        lambda row: "Nghịch lý đánh giá" if row["rating_text_contradiction"] != "NONE" else "Đánh giá thấp (1–2★)",
        axis=1,
    )

    col_c1, col_c2 = st.columns([1, 2])
    with col_c1:
        st.metric("Tổng số đánh giá cần rà soát", f"{len(attention):,} nhận xét")
    with col_c2:
        sig_choice = st.selectbox("Lọc theo loại tín hiệu cảnh báo:", ["Tất cả", "Chỉ đánh giá thấp (1–2★)", "Chỉ trường hợp nghịch lý đánh giá"])

    if sig_choice == "Chỉ đánh giá thấp (1–2★)":
        attention = attention[attention["loai_tin_hieu"] == "Đánh giá thấp (1–2★)"]
    elif sig_choice == "Chỉ trường hợp nghịch lý đánh giá":
        attention = attention[attention["loai_tin_hieu"] == "Nghịch lý đánh giá"]

    columns = [
        "loai_tin_hieu", "created_at", "platform", "category", "product_name",
        "reviewer_name", "rating", "like_count", "comment_text_raw",
    ]
    rename_cols = {
        "loai_tin_hieu": "Phân loại cảnh báo",
        "created_at": "Thời gian",
        "platform": "Sàn",
        "category": "Ngành hàng",
        "product_name": "Sản phẩm",
        "reviewer_name": "Khách hàng",
        "rating": "Số sao",
        "like_count": "Lượt thích",
        "comment_text_raw": "Nội dung phản ánh"
    }
    st.dataframe(attention[columns].rename(columns=rename_cols), use_container_width=True, hide_index=True)
    if not attention.empty:
        st.download_button(
            "Tải danh sách cần xử lý (CSV)",
            attention.to_csv(index=False).encode("utf-8-sig"),
            "danh_sach_danh_gia_can_chu_y.csv",
            "text/csv",
        )


def render_risk_alerts(view: pd.DataFrame, aspects: dict) -> None:
    st.markdown("#### Trung tâm Cảnh báo Sớm & Đề xuất Hành động")
    valid = valid_rows(view)
    if valid.empty:
        st.info("Chưa có dữ liệu hợp lệ để trích xuất cảnh báo.")
        return

    details, summary = build_report(view, aspects)
    negative_rate = float((valid["rating"] <= 2).mean() * 100)
    alerts = []

    if negative_rate >= 15:
        alerts.append(("Mức độ nghiêm trọng", f"Tỷ lệ đánh giá 1–2 sao toàn hệ thống đang ở mức cao ({negative_rate:.1f}%)."))
    if summary["rating_text_contradictions"] > 0:
        alerts.append(("Độ tin cậy dữ liệu", f"Phát hiện {summary['rating_text_contradictions']:,} đánh giá có số sao và lời văn trái ngược nhau."))

    coverage = summary["aspect_coverage"]
    if coverage:
        top_aspect = max(coverage, key=lambda name: coverage[name]["reviews"])
        alerts.append(("Khía cạnh nóng", f"Khía cạnh nhận được nhiều quan tâm nhất là '{top_aspect.title()}' với {coverage[top_aspect]['reviews']:,} lượt phản ánh."))

    if "category" in valid.columns:
        cat_neg = valid.groupby("category")["rating"].apply(lambda s: (s <= 2).mean() * 100)
        if not cat_neg.empty:
            worst_cat = cat_neg.idxmax()
            worst_rate = cat_neg.max()
            if worst_rate >= 8:
                alerts.append(("Cảnh báo ngành hàng", f"Ngành hàng '{worst_cat}' có tỷ lệ không hài lòng cao nhất ({worst_rate:.1f}%)."))

    if "like_count" in valid.columns:
        high_like_neg = valid[(valid["rating"] <= 2) & (valid["like_count"] >= 5)]
        if not high_like_neg.empty:
            alerts.append(("Tác động lan tỏa cao", f"Có {len(high_like_neg):,} đánh giá tiêu cực nhận được từ 5 lượt hữu ích trở lên; có nguy cơ ảnh hưởng trực tiếp đến tỷ lệ chuyển đổi mua hàng."))

    if not alerts:
        st.success("Tình trạng ổn định: Chưa phát hiện cảnh báo rủi ro nào vượt ngưỡng cho phép.")
    for level, message in alerts:
        if level in ["Mức độ nghiêm trọng", "Tác động lan tỏa cao"]:
            st.error(f"🚨 **{level}:** {message}")
        else:
            st.warning(f"⚠️ **{level}:** {message}")

    st.markdown("##### Khuyến nghị giải pháp cho doanh nghiệp / Người bán")
    st.markdown(
        """
        - 🛠️ **Cải tiến sản phẩm**: Ưu tiên kiểm định chất lượng các mặt hàng thuộc ngành hàng có tỷ lệ tiêu cực cao nhất.
        - 🚚 **Tối ưu quy trình vận hành**: Nếu khía cạnh *Giao hàng* hoặc *Đóng gói* bị phàn nàn, cần xem xét lại đối tác vận chuyển và quy chuẩn bọc hàng.
        - 💬 **Chăm sóc khách hàng chủ động**: Liên hệ trực tiếp với các khách hàng có đánh giá tiêu cực nhận được nhiều like để giải quyết khiếu nại trước khi khủng hoảng truyền thông lan rộng.
        """
    )


def render_risk_explorer(view: pd.DataFrame) -> None:
    st.markdown("#### Tra cứu & Trích xuất Dữ liệu Đánh giá Toàn văn")
    col_search, col_sort = st.columns([3, 1])
    with col_search:
        search = st.text_input("Tìm kiếm theo từ khóa trong nội dung:", placeholder="Ví dụ: giao hàng chậm, chất lượng kém, đóng gói móp...")
    with col_sort:
        sort_by = st.selectbox(
            "Sắp xếp theo:",
            ["Nhiều lượt thích nhất (Hữu ích)", "Mới nhất", "Rating thấp nhất (1★)", "Rating cao nhất (5★)"],
        )

    status_options = sorted(view["filter_status"].dropna().astype(str).unique()) if "filter_status" in view else []
    statuses = st.multiselect("Lọc trạng thái dữ liệu:", status_options, default=status_options)
    table = view.copy()
    if search:
        table = table[table["comment_text_raw"].astype(str).str.contains(search, case=False, na=False)]
    if statuses and "filter_status" in table:
        table = table[table["filter_status"].astype(str).isin(statuses)]

    if sort_by == "Nhiều lượt thích nhất (Hữu ích)" and "like_count" in table:
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
        "comment_text_raw"
    ]
    rename_cols = {
        "created_at": "Ngày đánh giá",
        "platform": "Sàn",
        "category": "Ngành hàng",
        "product_name": "Sản phẩm",
        "reviewer_name": "Người mua",
        "rating": "Số sao",
        "like_count": "Lượt thích",
        "filter_status": "Trạng thái",
        "comment_text_raw": "Nội dung nhận xét"
    }
    st.caption(f"Tìm thấy **{len(table):,}** đánh giá phù hợp")
    st.dataframe(table[[c for c in columns if c in table.columns]].rename(columns=rename_cols), use_container_width=True, hide_index=True)
    if not table.empty:
        st.download_button(
            "Xuất kết quả tra cứu ra CSV",
            table.to_csv(index=False).encode("utf-8-sig"),
            "ket_qua_tra_cuu_danh_gia.csv",
            "text/csv"
        )


# ==========================================
# PHÂN HỆ 4: PHÒNG NGHIÊN CỨU & MÔ HÌNH AI
# ==========================================

def render_research_lab(view: pd.DataFrame) -> None:
    st.markdown("#### Phòng Thử nghiệm & Đánh giá Mô hình AI (Model Lab)")
    st.caption("Huấn luyện và so sánh hiệu năng các mô hình phân loại cảm xúc văn bản dựa trên trích xuất đặc trưng TF-IDF.")
    
    data = valid_rows(view).copy()
    data["text"] = data.get("comment_text_segmented", data["comment_text_raw"]).fillna("").astype(str)
    data["label"] = pd.cut(data["rating"], bins=[0, 2, 3, 5], labels=["negative", "neutral", "positive"]).astype(str)
    data = data[data["label"].isin(["negative", "neutral", "positive"]) & data["text"].str.strip().ne("")]

    counts = data["label"].value_counts()
    if len(data) < 12 or counts.min() < 2:
        st.warning("Cần tối thiểu 12 đánh giá hợp lệ và ít nhất 2 mẫu cho mỗi lớp cảm xúc để huấn luyện.")
        return

    c_m1, c_m2 = st.columns(2)
    with c_m1:
        model_name = st.selectbox("Chọn giải thuật học máy:", ["Hồi quy Logistic (Logistic Regression)", "Máy véc-tơ hỗ trợ (Linear SVM)"])
    with c_m2:
        test_size = st.slider("Tỷ lệ tập kiểm thử (Test split):", 0.15, 0.40, 0.25, 0.05)

    if st.button("Bắt đầu Huấn luyện & Đánh giá", type="primary"):
        x_train, x_test, y_train, y_test = train_test_split(
            data["text"], data["label"], test_size=test_size, random_state=42, stratify=data["label"]
        )
        classifier = (
            LogisticRegression(max_iter=1000, class_weight="balanced")
            if "Logistic" in model_name
            else LinearSVC(class_weight="balanced")
        )
        model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", classifier),
        ])
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

        st.markdown("##### Các chỉ số đánh giá chất lượng mô hình")
        metrics = st.columns(4)
        with metrics[0]:
            render_metric("Độ chính xác (Accuracy)", f"{accuracy_score(y_test, predictions):.3f}", "Tỷ lệ đoán đúng toàn bộ")
        with metrics[1]:
            render_metric("Độ chuẩn xác (Precision)", f"{report['macro avg']['precision']:.3f}", "Độ tin cậy khi gắn nhãn")
        with metrics[2]:
            render_metric("Độ bao phủ (Recall)", f"{report['macro avg']['recall']:.3f}", "Khả năng bắt trúng mẫu")
        with metrics[3]:
            render_metric("F1-Score trung bình", f"{report['macro avg']['f1-score']:.3f}", "Thước đo tổng hòa chính")

        labels = ["negative", "neutral", "positive"]
        labels_vn = ["Tiêu cực", "Trung tính", "Tích cực"]
        matrix = pd.DataFrame(confusion_matrix(y_test, predictions, labels=labels), index=labels_vn, columns=labels_vn)
        
        st.markdown("##### Ma trận nhầm lẫn (Confusion Matrix)")
        st.dataframe(matrix, use_container_width=True)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        st.session_state["trained_model"] = model
        st.success(f"Mô hình đã được lưu thành công tại `{MODEL_PATH}`! Bạn có thể sang phân hệ 'Phân tích Khía cạnh' -> Tab 'Mô phỏng' để thử nghiệm dự đoán ngay.")

    with st.expander("📘 Giải thích học thuật về các thước đo đánh giá mô hình"):
        st.markdown(
            """
            - **Tại sao dùng Macro-F1 thay vì Accuracy?**: Trong dữ liệu review thương mại điện tử, số lượng review 5 sao thường áp đảo (mất cân bằng dữ liệu - Class Imbalance). Nếu chỉ dùng Accuracy, mô hình chỉ cần đoán bừa 'Tích cực' cũng có thể đạt 80% nhưng hoàn toàn vô dụng với lớp 'Tiêu cực'. **Macro-F1** đối xử bình đẳng giữa các lớp, phản ánh trung thực năng lực nhận diện của AI.
            - **Đặc trưng TF-IDF N-gram (1, 2)**: Trích xuất cả từ đơn (*tốt*, *chậm*) và cặp từ liền kề (*rất tốt*, *quá chậm*, *không thích*) để lưu giữ sắc thái ngữ nghĩa tiếng Việt.
            """
        )


def render_research_labeling(view: pd.DataFrame, aspects: dict) -> None:
    st.markdown("#### Công cụ Gán nhãn Dữ liệu Thủ công (Active Labeling Tool)")
    st.caption("Cho phép nhóm nghiên cứu gán nhãn thủ công để tạo lập tập kiểm thử vàng (Gold Standard Dataset) phục vụ đánh giá mô hình.")
    
    valid = valid_rows(view)
    if valid.empty:
        st.info("Chưa có đánh giá hợp lệ.")
        return

    if "manual_labels" not in st.session_state:
        st.session_state.manual_labels = []

    review_ids = valid["review_id"].astype(str).tolist()
    review_id = st.selectbox("Chọn mã đánh giá cần gán nhãn:", review_ids)
    row = valid[valid["review_id"].astype(str) == review_id].iloc[0]

    prod_name = row.get("product_name", "N/A")
    cat_name = row.get("category", "N/A")
    star = row.get("rating", "N/A")
    user = row.get("reviewer_name", "Khách hàng ẩn danh")
    likes = row.get("like_count", 0)

    st.markdown(
        f'<div style="background:#ffffff;border:1px solid #cbd5e1;border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.8rem;font-size:0.92rem;">'
        f'📦 <b>Sản phẩm:</b> {prod_name} &nbsp;|&nbsp; 📁 <b>Ngành hàng:</b> {cat_name} &nbsp;|&nbsp; ⭐ <b>Số sao:</b> {star}★ &nbsp;|&nbsp; 👤 <b>Khách hàng:</b> {user} ({likes} 👍)'
        f'</div>',
        unsafe_allow_html=True
    )
    st.info(f"Nội dung nhận xét: \"{row.get('comment_text_raw', '')}\"")

    with st.form("manual_label_form"):
        sentiment = st.selectbox("Nhãn cảm xúc chuẩn xác:", ["Tiêu cực", "Trung tính", "Tích cực"])
        selected_aspects = st.multiselect("Các khía cạnh đề cập trong câu:", list(aspects.keys()), format_func=lambda v: v.title())
        submitted = st.form_submit_button("Lưu nhãn đánh giá này")

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
        st.success(f"Đã lưu nhãn cho đánh giá {review_id}!")

    labels_df = pd.DataFrame(st.session_state.manual_labels)
    if not labels_df.empty:
        st.markdown("##### Danh sách các mẫu đã gán nhãn trong phiên làm việc")
        st.dataframe(labels_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Xuất tập dữ liệu nhãn vàng (CSV)",
            labels_df.to_csv(index=False).encode("utf-8-sig"),
            "tap_nhan_vang_thu_cong.csv",
            "text/csv",
        )


def render_research_report(view: pd.DataFrame, aspects: dict) -> None:
    st.markdown("#### Báo cáo Khoa học Tổng hợp Đề tài")
    if view.empty:
        st.info("Không có dữ liệu để lập báo cáo.")
        return

    valid = valid_rows(view)
    _, summary = build_report(view, aspects)

    st.markdown("##### 1. Thống kê mô tả tổng thể dữ liệu")
    report_rows = [
        {"Chỉ số nghiên cứu": "Tổng số lượng đánh giá thu thập được", "Giá trị": f"{len(view):,} dòng"},
        {"Chỉ số nghiên cứu": "Số lượng đánh giá hợp chuẩn phân tích", "Giá trị": f"{len(valid):,} dòng ({len(valid)/len(view)*100:.1f}%)"},
        {"Chỉ số nghiên cứu": "Số lượng đánh giá rác / spam bị lọc bỏ", "Giá trị": f"{len(view) - len(valid):,} dòng"},
        {"Chỉ số nghiên cứu": "Điểm số hài lòng trung bình toàn sàn", "Giá trị": f"{round(float(view['rating'].mean()), 2)} / 5"},
        {"Chỉ số nghiên cứu": "Số lượng đánh giá tiêu cực (1–2 sao)", "Giá trị": f"{int((view['rating'] <= 2).sum()):,} dòng"},
        {"Chỉ số nghiên cứu": "Tỷ lệ đánh giá không hài lòng", "Giá trị": f"{(view['rating'] <= 2).mean() * 100:.1f}%"},
        {"Chỉ số nghiên cứu": "Số trường hợp phát hiện nghịch lý đánh giá", "Giá trị": f"{summary['rating_text_contradictions']:,} trường hợp"},
    ]
    report_df = pd.DataFrame(report_rows)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    if "category" in valid.columns:
        st.markdown("##### 2. Tổng kết hiệu suất theo từng Ngành hàng")
        cat_stats = valid.groupby("category").agg(
            total_reviews=("review_id", "count"),
            avg_rating=("rating", "mean"),
            negative_reviews=("rating", lambda s: int((s <= 2).sum())),
            negative_rate=("rating", lambda s: f"{(s <= 2).mean() * 100:.1f}%"),
            total_likes=("like_count", "sum") if "like_count" in valid.columns else ("review_id", "count")
        ).reset_index().rename(columns={
            "category": "Ngành hàng",
            "total_reviews": "Số đánh giá",
            "avg_rating": "Điểm TB",
            "negative_reviews": "Đánh giá 1–2★",
            "negative_rate": "Tỷ lệ tiêu cực",
            "total_likes": "Tổng Like"
        })
        cat_stats["Điểm TB"] = cat_stats["Điểm TB"].round(2)
        st.dataframe(cat_stats, use_container_width=True, hide_index=True)

    st.markdown("##### 3. Mức độ bao phủ của Từ điển Khía cạnh (Aspect Coverage)")
    coverage_rows = [
        {"Khía cạnh sản phẩm": asp.title(), "Số lượt xuất hiện": data["reviews"], "Tỷ lệ bao phủ (%)": f"{data['rate_percent']}%"}
        for asp, data in summary.get("aspect_coverage", {}).items()
    ]
    st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)

    st.download_button(
        "Tải toàn văn báo cáo tổng hợp (CSV)",
        report_df.to_csv(index=False).encode("utf-8-sig"),
        "bao_cao_khoa_hoc_customer_voice.csv",
        "text/csv",
    )


# ==========================================
# CHƯƠNG TRÌNH CHÍNH (MAIN APPLICATION)
# ==========================================

def main() -> None:
    # 1. Thanh Sidebar điều khiển dữ liệu
    st.sidebar.markdown(
        '<div style="font-size:1.2rem;font-weight:700;color:#0f172a;letter-spacing:-0.02em;">✦ Customer Voice AI</div>'
        '<div style="color:#64748b;font-size:0.8rem;margin-bottom:1rem;">Nền tảng phân tích ý kiến khách hàng</div>',
        unsafe_allow_html=True,
    )
    
    data_path = st.sidebar.text_input("Đường dẫn tệp dữ liệu:", str(DEFAULT_DATA))
    file_mtime = get_file_mtime(data_path)
    if st.sidebar.button("🔄 Làm mới dữ liệu", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    try:
        df = load_data(data_path, file_mtime)
    except Exception as error:
        st.error(f"Không thể đọc tệp dữ liệu: {error}")
        st.stop()

    st.sidebar.caption(f"Trạng thái: Đang nạp **{len(df):,}** đánh giá")
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 🎯 BỘ LỌC DỮ LIỆU")

    platforms = sorted(df["platform"].dropna().astype(str).unique())
    selected_platforms = st.sidebar.multiselect("Nền tảng sàn:", platforms, default=platforms)

    categories = sorted([str(c) for c in df["category"].dropna().unique() if str(c).strip()]) if "category" in df.columns else []
    selected_categories = st.sidebar.multiselect("Ngành hàng:", categories, default=categories) if categories else []

    cat_df = df[df["category"].astype(str).isin(selected_categories)] if (categories and selected_categories) else df
    products = sorted(cat_df["product_name"].dropna().astype(str).unique()) if "product_name" in cat_df.columns else []
    selected_products = st.sidebar.multiselect("Sản phẩm cụ thể:", products, default=[])

    ratings = st.sidebar.slider("Khoảng số sao:", 1, 5, (1, 5))
    valid_only = st.sidebar.checkbox("Chỉ phân tích đánh giá hợp lệ (Lọc rác)", value=True)

    # Lọc dữ liệu view
    view = df[df["platform"].astype(str).isin(selected_platforms) & df["rating"].between(*ratings)]
    if categories:
        view = view[view["category"].astype(str).isin(selected_categories)]
    if selected_products:
        view = view[view["product_name"].astype(str).isin(selected_products)]
    if valid_only and "is_valid_for_analysis" in view:
        view = view[view["is_valid_for_analysis"].astype(bool)]

    valid = valid_rows(view)
    aspects = load_aspects(str(ASPECT_DATA))

    # Thông tin nhóm đề tài ở cuối sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size:0.75rem;color:#64748b;line-height:1.4;">
        <b>Đề tài</b>: Khai phá ý kiến người dùng TMĐT<br>
        <b>Nhóm sinh viên</b>: Long - Mạnh<br>
        <b>Giảng viên HD</b>: Bộ môn Khoa học Dữ liệu
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Header chính của ứng dụng
    st.markdown(
        """
        <div class="app-header">
            <h1>Hệ Thống Phân Tích Ý Kiến Khách Hàng (Customer Voice AI)</h1>
            <p>Nền tảng khai phá dữ liệu đánh giá sản phẩm thương mại điện tử đa kênh phục vụ hỗ trợ ra quyết định kinh doanh</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Bản tóm tắt điều hành (Executive Summary Card dành cho Giảng viên / Ban giám khảo)
    avg_star = view["rating"].mean() if not view.empty else 0
    neg_pct = (valid["rating"] <= 2).mean() * 100 if not valid.empty else 0
    top_cat = valid["category"].value_counts().index[0] if "category" in valid.columns and not valid.empty else "N/A"
    
    st.markdown(
        f"""
        <div class="exec-summary">
            <div class="exec-summary-title">📌 BẢN TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)</div>
            <p class="exec-summary-content">
                Hệ thống đang giám sát <b>{len(view):,}</b> đánh giá thực tế từ Tiki & Shopee. 
                Điểm chất lượng trung bình toàn hệ thống đạt <b>{avg_star:.2f}/5 sao</b>, 
                với <b>{100 - neg_pct:.1f}%</b> phản hồi tích cực và <b>{neg_pct:.1f}%</b> phản hồi tiêu cực cần xử lý. 
                Ngành hàng sôi động nhất là <b>{top_cat}</b>. Đã hoàn tất tiền xử lý NLP và sẵn sàng bóc tách khía cạnh.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Thanh điều hướng phân hệ cấp cao
    module = st.radio(
        "Chọn phân hệ làm việc:",
        [
            "📊 Báo cáo Thị trường & Tổng quan",
            "🎯 Phân tích Khía cạnh Sản phẩm (ABSA)",
            "⚠️ Cảnh báo Rủi ro & Khách hàng",
            "🧪 Phòng Nghiên cứu & Mô hình AI",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    # 5. Phân nhánh màn hình theo từng phân hệ với các Tab độc lập không cuộn
    if module == "📊 Báo cáo Thị trường & Tổng quan":
        tab1, tab2, tab3 = st.tabs(["📈 Bức tranh toàn cảnh", "📦 Phân tích theo Ngành hàng", "🛒 So sánh Sàn (Tiki vs Shopee)"])
        with tab1:
            render_overview_landscape(view, valid, aspects)
        with tab2:
            render_overview_category(view, valid)
        with tab3:
            render_overview_platform(view, valid)

    elif module == "🎯 Phân tích Khía cạnh Sản phẩm (ABSA)":
        tab1, tab2, tab3 = st.tabs(["📦 Theo từng Sản phẩm", "🏷️ Ma trận Khía cạnh & Sắc thái", "💬 Thử nghiệm Mô phỏng Đánh giá"])
        with tab1:
            render_absa_product(view, aspects)
        with tab2:
            render_absa_matrix(view, aspects)
        with tab3:
            render_absa_simulator(aspects)

    elif module == "⚠️ Cảnh báo Rủi ro & Khách hàng":
        tab1, tab2, tab3 = st.tabs(["🚨 Đánh giá Cần chú ý Khẩn cấp", "🛡️ Trung tâm Cảnh báo Sớm", "🔎 Tra cứu & Trích xuất Toàn văn"])
        with tab1:
            render_risk_attention(view, aspects)
        with tab2:
            render_risk_alerts(view, aspects)
        with tab3:
            render_risk_explorer(view)

    else:
        tab1, tab2, tab3 = st.tabs(["🤖 Huấn luyện & Đánh giá Mô hình", "✍️ Công cụ Gán nhãn Thủ công", "📄 Báo cáo Khoa học Tổng hợp"])
        with tab1:
            render_research_lab(view)
        with tab2:
            render_research_labeling(view, aspects)
        with tab3:
            render_research_report(view, aspects)


if __name__ == "__main__":
    main()
