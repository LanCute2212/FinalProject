"""
==============================================================================
CUSTOMER VOICE AI - SAAS ANALYTICS DASHBOARD
Chuẩn thiết kế: Stripe Dashboard, Linear, Shopify Admin, Metabase
TUYỆT ĐỐI KHÔNG thay đổi logic xử lý dữ liệu, tên hàm nghiệp vụ, import gốc.
==============================================================================
"""

import io
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from src.features.aspect_discovery import discover_aspect_candidates, normalize_phrase
from src.features.feature_engineering import FeatureEngineeringPipeline
from src.preprocessing.pipeline import PreprocessingPipeline
from src.storage.sql_server_store import DEFAULT_DATABASE_NAME, SQLServerStore

# Import Design System & Chart Modules
import ui.charts as charts


ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "processed" / "reviews_processed.csv"
ASPECT_DATA = ROOT / "data" / "dictionaries" / "aspect_keywords.json"
ASPECT_CANDIDATES_IGNORED_PATH = ROOT / "data" / "processed" / "aspect_candidates_ignored.json"
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"
MANUAL_LABELS_PATH = ROOT / "data" / "processed" / "manual_labels.csv"
MANUAL_LABEL_COLUMNS = [
    "review_id", "platform", "product_name", "category", "rating", "text",
    "sentiment_manual", "aspects_manual", "annotated_at",
]
CSS_PATH = ROOT / "ui" / "style.css"

# ==============================================================================
# CẤU HÌNH TRANG VÀ NẠP CSS DESIGN SYSTEM
# ==============================================================================
st.set_page_config(
    page_title="Customer Voice AI | SaaS Analytics",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ==============================================================================
# HÀM BỔ TRỢ HỆ THỐNG & CACHING DỮ LIỆU
# ==============================================================================

def get_file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False, ttl=30)
def load_data(
    path: str,
    mtime: float,
    data_source: str = "CSV",
    sql_server: str = "localhost",
    database_name: str = DEFAULT_DATABASE_NAME,
) -> pd.DataFrame:
    del mtime
    if data_source == "SQL Server":
        with SQLServerStore(
            server=sql_server,
            database_name=database_name,
        ) as store:
            store.ping()
            frame = store.load_reviews_dataframe()
    else:
        frame = pd.read_csv(path)
    if "rating" in frame.columns:
        frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce").fillna(0).astype(int)
    if "created_at" in frame.columns:
        frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    return frame


def load_manual_labels(path: Path = MANUAL_LABELS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=MANUAL_LABEL_COLUMNS)
    labels = pd.read_csv(path, dtype={"review_id": str, "platform": str})
    missing_columns = set(MANUAL_LABEL_COLUMNS) - set(labels.columns)
    if missing_columns:
        raise ValueError(f"Tập nhãn thủ công thiếu cột: {', '.join(sorted(missing_columns))}")
    return labels[MANUAL_LABEL_COLUMNS]


def save_manual_labels(labels: pd.DataFrame, path: Path = MANUAL_LABELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels[MANUAL_LABEL_COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def cached_build_report(frame: pd.DataFrame, aspects: Dict[str, List[str]]) -> Tuple[pd.DataFrame, Dict]:
    return build_report(frame, aspects)


def valid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "is_valid_for_analysis" not in frame.columns:
        return frame
    return frame[frame["is_valid_for_analysis"].astype(bool)]


def _search_normalize(value: object) -> str:
    normalized = unicodedata.normalize("NFD", str(value).casefold())
    without_marks = "".join(
        character for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return without_marks.replace("đ", "d")


def searchable_selectbox(
    label: str,
    options: list,
    key: str,
    format_func=None,
    container=None,
):
    target = container if container is not None else st
    choices = list(options)
    if not choices:
        target.info(f"Không có lựa chọn cho “{label.rstrip(':')}”.")
        return None

    search_key = f"{key}_search"
    query = st.session_state.get(search_key, "")
    query_normalized = _search_normalize(query.strip())
    filtered = [
        option for option in choices
        if query_normalized in _search_normalize(
            format_func(option) if format_func else option
        )
    ]
    if not filtered:
        target.info(f"Không tìm thấy lựa chọn phù hợp với “{query}”.")
        target.text_input(
            f"Tìm kiếm {label.rstrip(':')}",
            placeholder="Nhập để lọc lựa chọn...",
            key=search_key,
        )
        return None

    selection_key = f"{key}_selection"
    if selection_key not in st.session_state or st.session_state[selection_key] not in choices:
        st.session_state[selection_key] = choices[0]
    selected = st.session_state[selection_key]
    if selected not in filtered:
        filtered.insert(0, selected)
    widget_options = {"key": selection_key}
    if format_func is not None:
        widget_options["format_func"] = format_func
    selected = target.selectbox(label, filtered, **widget_options)
    target.text_input(
        f"Tìm kiếm {label.rstrip(':')}",
        placeholder="Nhập để lọc lựa chọn...",
        key=search_key,
    )
    return selected


def searchable_multiselect(
    label: str,
    options: list,
    default: list,
    key: str,
    format_func=None,
    container=None,
):
    target = container if container is not None else st
    choices = list(options)
    if key not in st.session_state:
        st.session_state[key] = list(default)

    search_key = f"{key}_search"
    query = st.session_state.get(search_key, "")
    query_normalized = _search_normalize(query.strip())
    filtered = [
        option for option in choices
        if query_normalized in _search_normalize(
            format_func(option) if format_func else option
        )
    ]
    selected = list(st.session_state.get(key, []))
    visible = list(filtered)
    visible.extend(option for option in selected if option not in visible)
    widget_options = {"key": key}
    if format_func is not None:
        widget_options["format_func"] = format_func
    selected_values = target.multiselect(label, visible, **widget_options)
    if not filtered:
        target.caption(f"Không tìm thấy lựa chọn phù hợp với “{query}”.")
    target.text_input(
        f"Tìm kiếm {label.rstrip(':')}",
        placeholder="Nhập để lọc lựa chọn...",
        key=search_key,
    )
    return selected_values


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


# ==============================================================================
# HỆ THỐNG GIAO DIỆN CHUẨN SAAS (COMPONENTS)
# ==============================================================================

def render_page_header(title: str, description: str, action_label: str = "", action_help: str = "") -> None:
    action_html = ""
    if action_label:
        action_html = f'<div class="page-header-actions"><span class="badge-neu" title="{action_help}">{action_label}</span></div>'
    st.markdown(
        f'<div class="page-header">'
        f'  <div>'
        f'    <h2 class="page-header-title">{title}</h2>'
        f'    <p class="page-header-desc">{description}</p>'
        f'  </div>'
        f'  {action_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_kpi(
    label: str,
    value: str,
    icon: str = "",
    delta: str = "",
    delta_type: str = "neutral",
    tooltip: str = "",
) -> None:
    tooltip_attr = f'title="{tooltip}"' if tooltip else ""
    delta_html = ""
    if delta:
        delta_class = f"kpi-delta-{delta_type}"
        delta_html = f'<span class="{delta_class}">{delta}</span>'
    icon_html = f'<span class="kpi-icon">{icon}</span>' if icon else ""

    st.markdown(
        f'<div class="kpi-card" {tooltip_attr}>'
        f'  <div class="kpi-header">'
        f'    <span class="kpi-label">{label}</span>'
        f'    {icon_html}'
        f'  </div>'
        f'  <div class="kpi-value">{value}</div>'
        f'  <div class="kpi-footer">{delta_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_alert_card(level: str, title: str, message: str) -> None:
    level_class = "alert-card-high" if level == "Cao" else "alert-card-med" if level == "Trung bình" else "alert-card-low"
    icon = "🚨" if level == "Cao" else "⚠️" if level == "Trung bình" else "ℹ️"
    st.markdown(
        f'<div class="alert-card {level_class}">'
        f'  <div style="font-size:1.15rem;line-height:1;">{icon}</div>'
        f'  <div>'
        f'    <div class="alert-title">{level}: {title}</div>'
        f'    <p class="alert-message">{message}</p>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, description: str, icon_text: str = "📭") -> None:
    st.markdown(
        f'<div class="empty-state">'
        f'  <div class="empty-state-icon">{icon_text}</div>'
        f'  <div class="empty-state-title">{title}</div>'
        f'  <p class="empty-state-desc">{description}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ==============================================================================
# PHÂN HỆ 1: BÁO CÁO THỊ TRƯỜNG & TỔNG QUAN
# ==============================================================================

def render_overview_landscape(view: pd.DataFrame, valid: pd.DataFrame, aspects: dict) -> None:
    render_page_header(
        title="Bức tranh toàn cảnh về phản hồi khách hàng",
        description="Tổng quan định lượng về sản lượng, phân bố sắc thái và sức khỏe dữ liệu sau tiền xử lý",
        action_label=f"Cập nhật: {len(view):,} đánh giá",
    )
    if view.empty:
        render_empty_state("Không có dữ liệu phù hợp", "Vui lòng điều chỉnh lại bộ lọc ở thanh bên.")
        return

    negative = int((valid["rating"] <= 2).sum()) if not valid.empty else 0
    neg_pct = (negative / len(valid) * 100) if len(valid) else 0.0
    avg_rating = view["rating"].mean() if not view.empty else 0.0

    try:
        _, report = cached_build_report(view, aspects)
        contradictions = report["rating_text_contradictions"]
    except Exception:
        contradictions = 0

    cols = st.columns(4)
    with cols[0]:
        render_kpi("Tổng đánh giá nạp", f"{len(view):,}", icon="📊", delta="Toàn bộ tập dữ liệu", tooltip="Số lượng review sau khi qua bộ lọc nền tảng và thời gian")
    with cols[1]:
        delta_type = "good" if avg_rating >= 4.2 else "bad" if avg_rating < 3.5 else "neutral"
        render_kpi("Điểm hài lòng trung bình", f"{avg_rating:.2f} / 5", icon="⭐", delta=f"{avg_rating:.2f}★ / 5.0", delta_type=delta_type, tooltip="Thang điểm 1-5 sao")
    with cols[2]:
        delta_type = "bad" if neg_pct >= 10 else "good"
        render_kpi("Đánh giá tiêu cực (1–2★)", f"{negative:,}", icon="⚠️", delta=f"{neg_pct:.1f}% tổng số", delta_type=delta_type, tooltip="Tỷ lệ đánh giá 1-2 sao cần được ưu tiên xử lý")
    with cols[3]:
        render_kpi("Nghịch lý đánh giá", f"{contradictions:,}", icon="⚡", delta=f"{contradictions} trường hợp", delta_type="bad" if contradictions > 0 else "good", tooltip="Bình luận chấm sao cao nhưng lời văn tiêu cực hoặc ngược lại")

    c1, c2 = st.columns(2)
    with c1:
        fig_sent = charts.chart_sentiment_distribution(sentiment_summary(valid if not valid.empty else view))
        charts.render_chart_card("Phân bố cảm xúc theo số sao", "Thống kê tỷ lệ đánh giá theo 3 nhóm: Tiêu cực (1-2★), Trung tính (3★), Tích cực (4-5★)", fig_sent)
    with c2:
        status_counts = view.get("filter_status", pd.Series(["VALID"] * len(view))).value_counts()
        status_map = {"VALID": "Hợp lệ", "SPAM_PROMO": "Spam quảng cáo", "SPAM_EMOJI_ONLY": "Spam biểu tượng/rỗng", "DUPLICATE": "Trùng lặp"}
        status_counts.index = [status_map.get(k, str(k)) for k in status_counts.index]
        fig_health = charts.chart_data_health(status_counts)
        charts.render_chart_card("Chất lượng dữ liệu tiền xử lý", "Phân loại trạng thái sau bộ lọc khử rác ngôn ngữ tự nhiên", fig_health)

    st.markdown("##### Xu hướng cảm xúc theo thời gian")
    trend = valid.dropna(subset=["created_at"]).copy()
    if not trend.empty:
        trend["sentiment"] = sentiment_label_vn(trend["rating"])
        trend["tháng"] = trend["created_at"].dt.to_period("M").astype(str)
        chart_data = trend.pivot_table(
            index="tháng", columns="sentiment", values="review_id",
            aggfunc="count", fill_value=0,
        ).reindex(columns=["Tiêu cực", "Trung tính", "Tích cực"], fill_value=0)
        fig_trend = charts.chart_sentiment_trend(chart_data)
        charts.render_chart_card("Biến thiên sắc thái theo tháng", "Đường diễn biến số lượng phản hồi tích cực, trung tính và tiêu cực", fig_trend)


def render_overview_category(view: pd.DataFrame, valid: pd.DataFrame) -> None:
    render_page_header(
        title="Phân tích & So sánh theo Ngành hàng",
        description="Định vị hiệu suất chất lượng và tỷ lệ không hài lòng trên từng phân khúc sản phẩm",
    )
    if "category" not in valid.columns or valid.empty:
        render_empty_state("Chưa có thông tin ngành hàng", "Dữ liệu hiện tại không chứa cột 'category'.")
        return

    c1, c2 = st.columns(2)
    with c1:
        cat_counts = valid["category"].value_counts()
        fig_vol = charts.chart_category_volume(cat_counts)
        charts.render_chart_card("Sản lượng đánh giá theo ngành hàng", "Số lượng đánh giá hợp lệ ghi nhận trên từng ngành hàng", fig_vol)
    with c2:
        cat_rating = valid.groupby("category")["rating"].mean().round(2)
        fig_rat = charts.chart_category_rating(cat_rating)
        charts.render_chart_card("Điểm số hài lòng trung bình (1–5★)", "So sánh điểm số với vạch ngưỡng chuẩn 4.0 sao", fig_rat)

    st.markdown("##### Bảng chỉ số chi tiết từng ngành hàng")
    cat_summary = (
        valid.groupby("category", dropna=False)
        .agg(
            total_reviews=("review_id", "count"),
            avg_rating=("rating", "mean"),
            low_rating=("rating", lambda s: int((s <= 2).sum())),
            low_rating_pct=("rating", lambda s: round(float((s <= 2).mean() * 100), 1)),
            total_likes=("like_count", "sum") if "like_count" in valid.columns else ("review_id", "count"),
        )
        .sort_values(["total_reviews"], ascending=[False])
        .reset_index()
    )
    cat_summary["avg_rating"] = cat_summary["avg_rating"].round(2)

    col_configs = {
        "category": st.column_config.TextColumn(label="Ngành hàng", width="medium"),
        "total_reviews": st.column_config.NumberColumn(label="Số đánh giá hợp lệ", format="%d dòng"),
        "avg_rating": st.column_config.NumberColumn(label="Điểm hài lòng TB", format="%.2f ★"),
        "low_rating": st.column_config.NumberColumn(label="Đánh giá 1–2★", format="%d"),
        "low_rating_pct": st.column_config.NumberColumn(label="Tỷ lệ tiêu cực", format="%.1f%%"),
        "total_likes": st.column_config.ProgressColumn(
            label="Lượt hữu ích",
            format="%d 👍",
            min_value=0,
            max_value=int(cat_summary["total_likes"].max()) if not cat_summary.empty else 100,
        ),
    }
    st.dataframe(cat_summary, column_config=col_configs, width="stretch", hide_index=True)


def render_overview_platform(view: pd.DataFrame, valid: pd.DataFrame) -> None:
    render_page_header(
        title="So sánh Nền tảng Sàn thương mại điện tử",
        description="Đối chiếu chất lượng dịch vụ, sản lượng và mức độ tin cậy giữa các sàn được thu thập",
    )
    if "platform" not in valid.columns or valid.empty:
        render_empty_state("Chưa có thông tin sàn", "Dữ liệu hiện tại không chứa cột 'platform'.")
        return

    c1, c2 = st.columns(2)
    with c1:
        plat_counts = valid["platform"].value_counts()
        fig_pie = charts.chart_platform_pie(plat_counts)
        charts.render_chart_card("Cơ cấu thị phần dữ liệu giữa các sàn", "Tỷ trọng số lượng đánh giá thu thập được", fig_pie)
    with c2:
        plat_rating = valid.groupby("platform")["rating"].mean().round(2)
        fig_bar = charts.chart_platform_bar(plat_rating)
        charts.render_chart_card("Điểm đánh giá trung bình theo sàn", "Điểm chất lượng được người mua chấm trên thang 5 sao", fig_bar)

    st.markdown("##### Bảng đối chiếu các chỉ số vận hành giữa các sàn")
    plat_table = (
        valid.groupby("platform")
        .agg(
            total=("review_id", "count"),
            avg_star=("rating", "mean"),
            negative_count=("rating", lambda s: int((s <= 2).sum())),
            negative_rate=("rating", lambda s: round(float((s <= 2).mean() * 100), 1)),
            avg_likes=("like_count", "mean") if "like_count" in valid.columns else ("review_id", "count")
        )
        .reset_index()
    )
    plat_table["avg_star"] = plat_table["avg_star"].round(2)
    if "avg_likes" in plat_table:
        plat_table["avg_likes"] = plat_table["avg_likes"].round(1)

    col_configs = {
        "platform": st.column_config.TextColumn(label="Nền tảng sàn", width="medium"),
        "total": st.column_config.NumberColumn(label="Tổng đánh giá hợp lệ", format="%d"),
        "avg_star": st.column_config.NumberColumn(label="Điểm sao TB", format="%.2f ★"),
        "negative_count": st.column_config.NumberColumn(label="Số review 1–2★", format="%d"),
        "negative_rate": st.column_config.NumberColumn(label="Tỷ lệ không hài lòng", format="%.1f%%"),
        "avg_likes": st.column_config.NumberColumn(label="Lượt thích TB / Review", format="%.1f 👍"),
    }
    st.dataframe(plat_table, column_config=col_configs, width="stretch", hide_index=True)


# ==========================================
# PHÂN HỆ 2: PHÂN TÍCH KHÍA CẠNH (ABSA)
# ==========================================

def render_absa_product(view: pd.DataFrame, aspects: dict) -> None:
    render_page_header(
        title="Phân tích chi tiết theo Sản phẩm",
        description="Đánh giá chuyên sâu từng sản phẩm: điểm hài lòng, khía cạnh nổi bật và các nhận xét tiêu biểu",
    )
    if view.empty:
        render_empty_state("Không có dữ liệu", "Vui lòng chọn sản phẩm khác hoặc mở rộng bộ lọc.")
        return

    product_names = sorted(view["product_name"].dropna().astype(str).unique())
    selected_product = searchable_selectbox(
        "Chọn sản phẩm cần phân tích:",
        product_names,
        key="absa_product",
    )
    if selected_product is None:
        return
    selected = view[view["product_name"].astype(str) == selected_product]

    # FIX LỖI selected.get("is_valid_for_analysis", True).astype(bool) KHI THIẾU CỘT
    valid = selected[selected["is_valid_for_analysis"].astype(bool)] if "is_valid_for_analysis" in selected.columns else selected

    cat_val = selected["category"].dropna().iloc[0] if "category" in selected.columns and not selected["category"].dropna().empty else "Chưa phân loại"
    plat_val = selected["platform"].dropna().iloc[0] if "platform" in selected.columns and not selected["platform"].dropna().empty else "N/A"
    max_likes = int(selected["like_count"].max()) if "like_count" in selected.columns and not selected["like_count"].dropna().empty else 0

    st.markdown(
        f'<div class="info-strip">'
        f'  <span>📦 <b>Sản phẩm:</b> <span style="color:var(--color-primary);font-weight:600;">{selected_product}</span></span>'
        f'  <span>📁 <b>Ngành hàng:</b> {cat_val}</span>'
        f'  <span>🛒 <b>Sàn:</b> {plat_val}</span>'
        f'  <span>👍 <b>Kỷ lục thích:</b> {max_likes:,} lượt</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    with cols[0]:
        render_kpi("Tổng số đánh giá", f"{len(selected):,}", icon="📦", delta=f"{len(valid):,} hợp lệ", tooltip="Tổng số lượng đánh giá ghi nhận cho sản phẩm này")
    with cols[1]:
        p_avg = selected["rating"].mean() if not selected.empty else 0.0
        p_delta = "good" if p_avg >= 4.0 else "bad"
        render_kpi("Điểm hài lòng TB", f"{p_avg:.2f} / 5", icon="⭐", delta=f"{p_avg:.2f}★ / 5.0", delta_type=p_delta, tooltip="Điểm trung bình của riêng sản phẩm này")
    with cols[2]:
        p_neg = int((selected["rating"] <= 2).sum())
        p_neg_pct = (p_neg / len(selected) * 100) if len(selected) else 0.0
        render_kpi("Đánh giá tiêu cực", f"{p_neg:,}", icon="⚠️", delta=f"{p_neg_pct:.1f}% tổng số", delta_type="bad" if p_neg > 0 else "good", tooltip="Số lượng phản hồi chấm 1-2 sao")

    c1, c2 = st.columns(2)
    with c1:
        fig_prod_sent = charts.chart_sentiment_distribution(sentiment_summary(valid if not valid.empty else selected))
        charts.render_chart_card("Phân bố cảm xúc sản phẩm", "Cơ cấu đánh giá theo 3 nhóm cảm xúc", fig_prod_sent)
    with c2:
        selected_valid = valid_rows(selected)
        text_column = "comment_text_clean" if "comment_text_clean" in selected_valid else "comment_text_raw"
        aspect_rows = []
        for aspect, keywords in aspects.items():
            count = selected_valid[text_column].fillna("").astype(str).map(
                lambda text: bool(build_aspect_sentiment_rows(text, {aspect: keywords}))
            ).sum()
            aspect_rows.append({"Khía cạnh": aspect.title(), "Số lượt nhắc": int(count)})
        fig_asp = charts.chart_aspect_frequency(pd.DataFrame(aspect_rows).sort_values("Số lượt nhắc", ascending=False))
        charts.render_chart_card("Tần suất các khía cạnh được nhắc tới", "Các đặc tính sản phẩm được người mua bình luận nhiều nhất", fig_asp)

    st.markdown("##### Danh sách đánh giá tiêu biểu của sản phẩm")
    c_sort, _ = st.columns([2, 2])
    with c_sort:
        sort_p = searchable_selectbox(
            "Sắp xếp đánh giá:",
            ["Nhiều lượt thích nhất (Hữu ích)", "Rating thấp nhất (1–2★)", "Rating cao nhất (5★)", "Mới nhất"],
            key="absa_review_sort",
        )
    if sort_p is None:
        return

    prod_reviews = selected.copy()
    if sort_p == "Nhiều lượt thích nhất (Hữu ích)" and "like_count" in prod_reviews.columns:
        prod_reviews = prod_reviews.sort_values(by="like_count", ascending=False)
    elif sort_p == "Rating thấp nhất (1–2★)":
        prod_reviews = prod_reviews.sort_values(by="rating", ascending=True)
    elif sort_p == "Rating cao nhất (5★)":
        prod_reviews = prod_reviews.sort_values(by="rating", ascending=False)
    elif sort_p == "Mới nhất" and "created_at" in prod_reviews.columns:
        prod_reviews = prod_reviews.sort_values(by="created_at", ascending=False)

    col_configs = {
        "created_at": st.column_config.DatetimeColumn(label="Thời gian", format="DD/MM/YYYY"),
        "reviewer_name": st.column_config.TextColumn(label="Khách hàng", width="small"),
        "rating": st.column_config.NumberColumn(label="Số sao", format="%d ★", width="small"),
        "like_count": st.column_config.ProgressColumn(
            label="Lượt thích",
            format="%d 👍",
            min_value=0,
            max_value=max(1, int(selected["like_count"].max())) if "like_count" in selected.columns and not selected["like_count"].dropna().empty else 10,
        ),
        "comment_text_raw": st.column_config.TextColumn(label="Nội dung nhận xét", width="large"),
    }
    display_cols = [c for c in ["created_at", "reviewer_name", "rating", "like_count", "comment_text_raw"] if c in prod_reviews.columns]
    st.dataframe(prod_reviews[display_cols].head(50), column_config=col_configs, width="stretch", hide_index=True)


def render_absa_matrix(view: pd.DataFrame, aspects: dict) -> None:
    render_page_header(
        title="Ma trận Khía cạnh & Sắc thái cảm xúc (ABSA Matrix)",
        description="Bóc tách sắc thái cảm xúc chi tiết trên từng thuộc tính: Giá cả, Chất lượng, Giao hàng, Đóng gói, Dịch vụ",
    )
    valid = valid_rows(view)
    if valid.empty:
        render_empty_state("Chưa có dữ liệu", "Không có đánh giá hợp lệ để phân tích ma trận khía cạnh.")
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
        render_empty_state("Chưa phát hiện khía cạnh", "Không tìm thấy từ khóa khía cạnh nào trong phạm vi dữ liệu đang lọc.")
        return

    result = pd.DataFrame(rows)
    result["Khía cạnh"] = result["aspect"].str.title()
    counts = result.groupby(["Khía cạnh", "sentiment"]).size().unstack(fill_value=0)
    for sent in ["Tiêu cực", "Trung tính", "Hỗn hợp", "Tích cực"]:
        if sent not in counts.columns:
            counts[sent] = 0

    fig_matrix = charts.chart_absa_stacked_matrix(counts)
    charts.render_chart_card("Tương quan Cảm xúc theo từng Khía cạnh", "Biểu đồ thanh xếp chồng hiển thị số lượng và cơ cấu tỷ trọng %", fig_matrix)

    st.markdown("##### Khám phá chi tiết các ý kiến theo khía cạnh")
    c_asp, c_sent = st.columns(2)
    with c_asp:
        selected_aspects = searchable_multiselect(
            "Lọc khía cạnh:",
            list(counts.index),
            default=[],
            key="absa_matrix_aspects",
        )
    with c_sent:
        selected_sent = searchable_multiselect(
            "Lọc sắc thái:",
            ["Tiêu cực", "Trung tính", "Hỗn hợp", "Tích cực"],
            default=[],
            key="absa_matrix_sentiments",
        )

    filtered_res = result.copy()
    if selected_aspects:
        filtered_res = filtered_res[filtered_res["Khía cạnh"].isin(selected_aspects)]
    if selected_sent:
        filtered_res = filtered_res[filtered_res["sentiment"].isin(selected_sent)]

    col_configs = {
        "Khía cạnh": st.column_config.TextColumn(label="Khía cạnh", width="small"),
        "sentiment": st.column_config.TextColumn(label="Sắc thái", width="small"),
        "product_name": st.column_config.TextColumn(label="Sản phẩm", width="medium"),
        "category": st.column_config.TextColumn(label="Ngành hàng", width="small"),
        "rating": st.column_config.NumberColumn(label="Số sao", format="%d ★", width="small"),
        "like_count": st.column_config.NumberColumn(label="Lượt thích", format="%d 👍", width="small"),
        "matched_keywords": st.column_config.TextColumn(label="Từ khóa kích hoạt", width="medium"),
        "comment_text": st.column_config.TextColumn(label="Nội dung trích đoạn", width="large"),
    }
    show_cols = [c for c in ["Khía cạnh", "sentiment", "product_name", "category", "rating", "like_count", "matched_keywords", "comment_text"] if c in filtered_res.columns]
    st.dataframe(filtered_res[show_cols], column_config=col_configs, width="stretch", hide_index=True)


def render_aspect_discovery(view: pd.DataFrame, aspects: dict) -> None:
    render_page_header(
        title="Khám phá từ khóa Aspect",
        description="Tìm cụm từ tiềm năng chưa có trong từ điển; bạn duyệt mẫu review và quyết định aspect trước khi thêm.",
    )
    if "aspect_discovery_notice" in st.session_state:
        st.success(st.session_state.pop("aspect_discovery_notice"))

    st.caption(
        "Candidate được tìm bằng tần suất n-gram trên review hợp lệ; đây là gợi ý từ khóa, "
        "không phải nhãn aspect/sentiment tự động."
    )
    include_mock = st.checkbox(
        "Bao gồm review mô phỏng",
        value=False,
        help="Mặc định loại review mô phỏng để candidate phản ánh dữ liệu crawl thực tế.",
    )
    min_frequency = st.slider(
        "Số review khác nhau tối thiểu",
        min_value=1,
        max_value=20,
        value=5,
        help="Tăng ngưỡng để giảm các cụm hiếm hoặc nhiễu.",
    )
    try:
        discovery_view = view
        if not include_mock and "platform" in view.columns:
            mock_mask = view["platform"].fillna("").astype(str).str.lower().str.startswith("mock_")
            discovery_view = view.loc[~mock_mask]
            if mock_mask.any():
                st.caption(f"Đã loại {int(mock_mask.sum()):,} review mô phỏng.")
        candidates = discover_aspect_candidates(
            discovery_view,
            aspects,
            min_frequency=min_frequency,
        )
        if ASPECT_CANDIDATES_IGNORED_PATH.exists():
            ignored_values = json.loads(ASPECT_CANDIDATES_IGNORED_PATH.read_text(encoding="utf-8"))
            if not isinstance(ignored_values, list):
                raise ValueError("Danh sách candidate đã bỏ qua phải có định dạng JSON array.")
            ignored = {normalize_phrase(value) for value in ignored_values}
        else:
            ignored = set()
    except (OSError, ValueError) as error:
        st.error(f"Không thể tạo danh sách candidate aspect: {error}")
        return

    if not candidates.empty:
        candidates = candidates[
            ~candidates["keyword"].map(normalize_phrase).isin(ignored)
        ].reset_index(drop=True)
    if candidates.empty:
        st.info("Không tìm thấy candidate phù hợp. Hãy mở rộng bộ lọc hoặc giảm tần suất tối thiểu.")
        return

    st.markdown("#### Candidate đang chờ duyệt")
    st.dataframe(
        candidates[["keyword", "review_count"]]
        .rename(columns={"keyword": "Cụm từ tiềm năng", "review_count": "Số review"})
        .head(100),
        width="stretch",
        hide_index=True,
    )
    selected_keyword = searchable_selectbox(
        "Chọn cụm từ cần xem:",
        candidates["keyword"].tolist(),
        key="aspect_candidate",
    )
    if selected_keyword is None:
        return
    selected = candidates[candidates["keyword"] == selected_keyword].iloc[0]

    st.markdown("#### Review mẫu")
    for example in selected["examples"]:
        st.caption(
            f"{example['platform']} · {example['product_name']} · "
            f"Review {example['review_id']} · {example['rating']} sao"
        )
        st.text(example["text"])

    aspect_options = ["-- Chọn aspect --", *aspects.keys(), "__new_aspect__"]
    selected_aspect = searchable_selectbox(
        "Gán cụm từ vào khía cạnh:",
        aspect_options,
        key="aspect_target",
        format_func=lambda value: (
            "Tạo khía cạnh mới..." if value == "__new_aspect__"
            else value.replace("_", " ").title() if value != "-- Chọn aspect --"
            else value
        ),
    )
    if selected_aspect is None:
        return
    new_aspect_key = ""
    if selected_aspect == "__new_aspect__":
        new_aspect_key = st.text_input(
            "Mã khía cạnh mới",
            placeholder="content hoặc book_format",
            help="Dùng mã tiếng Anh dạng snake_case để giữ thống nhất với các khía cạnh hiện có.",
        )

    add_col, ignore_col = st.columns(2)
    with add_col:
        add_clicked = st.button("Thêm keyword vào từ điển", type="primary")
    with ignore_col:
        ignore_clicked = st.button("Bỏ qua candidate này")

    if add_clicked:
        if selected_aspect == "-- Chọn aspect --":
            st.error("Hãy chọn khía cạnh trước khi thêm keyword.")
        else:
            target_aspect = selected_aspect
            if selected_aspect == "__new_aspect__":
                target_aspect = re.sub(r"[^a-z0-9]+", "_", new_aspect_key.lower()).strip("_")
                if not target_aspect:
                    st.error("Nhập mã khía cạnh mới, ví dụ `content` hoặc `book_format`.")
                    return

            try:
                current_aspects = load_aspects(str(ASPECT_DATA))
                normalized_keyword = normalize_phrase(selected_keyword)
                existing_aspect = next(
                    (
                        aspect for aspect, keywords in current_aspects.items()
                        if any(normalize_phrase(keyword) == normalized_keyword for keyword in keywords)
                    ),
                    None,
                )
                if existing_aspect:
                    raise ValueError(f"Keyword này đã thuộc khía cạnh `{existing_aspect}`.")
                current_aspects.setdefault(target_aspect, []).append(selected_keyword)
                temporary_path = ASPECT_DATA.with_suffix(".json.tmp")
                temporary_path.write_text(
                    json.dumps(current_aspects, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary_path.replace(ASPECT_DATA)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                st.error(f"Không thể thêm keyword vào từ điển: {error}")
            else:
                st.session_state["aspect_discovery_notice"] = (
                    f"Đã thêm “{selected_keyword}” vào khía cạnh {target_aspect.replace('_', ' ')}."
                )
                st.rerun()

    if ignore_clicked:
        ignored.add(normalize_phrase(selected_keyword))
        try:
            ASPECT_CANDIDATES_IGNORED_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = ASPECT_CANDIDATES_IGNORED_PATH.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(sorted(ignored), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(ASPECT_CANDIDATES_IGNORED_PATH)
        except OSError as error:
            st.error(f"Không thể lưu candidate đã bỏ qua: {error}")
        else:
            st.session_state["aspect_discovery_notice"] = f"Đã bỏ qua candidate “{selected_keyword}”."
            st.rerun()


def render_absa_simulator(aspects: dict) -> None:
    render_page_header(
        title="Trình Mô phỏng & Kiểm thử Đánh giá Thời gian thực",
        description="Môi trường thử nghiệm trực tiếp: Tiền xử lý NLP, nhận diện khía cạnh Lexicon và dự đoán cảm xúc bằng Machine Learning",
    )
    examples = {
        "Tích cực hoàn hảo": "Sản phẩm dùng rất tốt, đóng gói cẩn thận và giao hàng siêu nhanh.",
        "Tiêu cực gay gắt": "Hàng bị lỗi nặng, giao hàng quá chậm và shop phản hồi rất thiếu trách nhiệm.",
        "Nhiều khía cạnh trái chiều": "Sản phẩm xài ổn áp nhưng giao hàng chậm rì, hộp bị móp méo và giá hơi đắt.",
        "Tiếng lóng & Emoji (Teencode)": "sp okela lắm nha mn, ship nhanh cực ưng bụng luôn ❤️",
        "Đánh giá rác / Xin xu": "hình ảnh và video chỉ mang tính chất nhận xu ạ",
    }
    selected_example = searchable_selectbox(
        "Chọn kịch bản mẫu hoặc tự nhập:",
        ["Tự nhập nhận xét", *examples],
        key="simulator_example",
    )
    if selected_example is None:
        return
    default_text = "" if selected_example == "Tự nhập nhận xét" else examples[selected_example]
    text = st.text_area("Nội dung đánh giá cần kiểm thử:", value=default_text, height=105)
    rating = st.slider("Số sao giả lập:", 1, 5, 4)

    if st.button("Phân tích đánh giá ngay", type="primary"):
        if not text.strip():
            st.warning("Vui lòng nhập nội dung đánh giá.")
            return

        with st.spinner("Đang chạy mô hình AI và trích xuất đặc trưng..."):
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

        st.markdown("##### Kết quả phân tích")
        cols = st.columns(4)
        with cols[0]:
            render_kpi("Cảm xúc cơ sở", str(labels["sentiment_name"]), icon="⭐", delta=f"Theo {rating} sao", tooltip="Phân loại trực tiếp theo số sao")
        with cols[1]:
            render_kpi("Dự đoán AI (Máy học)", ai_sentiment if ai_sentiment else "Chưa huấn luyện", icon="🤖", delta="Mô hình TF-IDF", tooltip="Dự đoán bởi mô hình máy học đã huấn luyện")
        with cols[2]:
            status_txt = "RÁC (SPAM)" if processed["is_spam"] else "HỢP LỆ"
            render_kpi("Kiểm định lọc rác", status_txt, icon="🛡️", delta=processed.get("spam_reason", "Đạt chuẩn"), delta_type="bad" if processed["is_spam"] else "good")
        with cols[3]:
            matched = sum(int(labels[f"aspect_{name}"]) for name in aspects)
            render_kpi("Khía cạnh tìm thấy", f"{matched} khía cạnh", icon="🏷️", delta="Theo từ điển Lexicon")

        aspect_result = pd.DataFrame(
            [{
                "Khía cạnh": name.title(),
                "Xuất hiện": "Có" if labels[f"aspect_{name}"] else "Không",
                "Từ khóa kích hoạt": labels[f"aspect_{name}_matches"] if labels[f"aspect_{name}_matches"] else "—",
            } for name in aspects]
        )
        st.dataframe(aspect_result, width="stretch", hide_index=True)


# ==========================================
# PHÂN HỆ 3: CẢNH BÁO RỦI RO & KHÁCH HÀNG
# ==========================================

def render_risk_attention(view: pd.DataFrame, aspects: dict) -> None:
    render_page_header(
        title="Đánh giá Cần chú ý Khẩn cấp",
        description="Tổng hợp các phản hồi 1–2 sao hoặc có nghịch lý rating, ưu tiên hỗ trợ CSKH xử lý khiếu nại",
    )
    if view.empty:
        render_empty_state("Không có dữ liệu", "Không phát hiện đánh giá cần chú ý trong phạm vi lọc hiện tại.")
        return

    details, _ = cached_build_report(view, aspects)
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
        render_kpi("Số đánh giá cần kiểm tra", f"{len(attention):,}", icon="🚨", delta="Cần can thiệp CSKH", delta_type="bad")
    with col_c2:
        sig_choice = searchable_selectbox(
            "Lọc theo loại tín hiệu cảnh báo:",
            ["Tất cả", "Chỉ đánh giá thấp (1–2★)", "Chỉ trường hợp nghịch lý đánh giá"],
            key="risk_signal_filter",
        )
    if sig_choice is None:
        return

    if sig_choice == "Chỉ đánh giá thấp (1–2★)":
        attention = attention[attention["loai_tin_hieu"] == "Đánh giá thấp (1–2★)"]
    elif sig_choice == "Chỉ trường hợp nghịch lý đánh giá":
        attention = attention[attention["loai_tin_hieu"] == "Nghịch lý đánh giá"]

    col_configs = {
        "loai_tin_hieu": st.column_config.TextColumn(label="Tín hiệu", width="small"),
        "created_at": st.column_config.DatetimeColumn(label="Thời gian", format="DD/MM/YYYY"),
        "platform": st.column_config.TextColumn(label="Sàn", width="small"),
        "category": st.column_config.TextColumn(label="Ngành hàng", width="small"),
        "product_name": st.column_config.TextColumn(label="Sản phẩm", width="medium"),
        "reviewer_name": st.column_config.TextColumn(label="Khách hàng", width="small"),
        "rating": st.column_config.NumberColumn(label="Số sao", format="%d ★", width="small"),
        "like_count": st.column_config.NumberColumn(label="Lượt thích", format="%d 👍", width="small"),
        "comment_text_raw": st.column_config.TextColumn(label="Nội dung phản ánh", width="large"),
    }
    columns = ["loai_tin_hieu", "created_at", "platform", "category", "product_name", "reviewer_name", "rating", "like_count", "comment_text_raw"]
    st.dataframe(attention[columns], column_config=col_configs, width="stretch", hide_index=True)
    if not attention.empty:
        st.download_button(
            "Tải danh sách cần xử lý (CSV)",
            attention.to_csv(index=False).encode("utf-8-sig"),
            "danh_sach_danh_gia_can_chu_y.csv",
            "text/csv",
        )


def render_risk_alerts(view: pd.DataFrame, aspects: dict) -> None:
    render_page_header(
        title="Trung tâm Cảnh báo Sớm & Đề xuất Hành động",
        description="Hệ thống cảnh báo tự động phát hiện rủi ro về chất lượng, ngành hàng phàn nàn cao và review lan tỏa",
    )
    valid = valid_rows(view)
    if valid.empty:
        render_empty_state("Chưa có dữ liệu", "Không có đánh giá hợp lệ để trích xuất cảnh báo.")
        return

    details, summary = cached_build_report(view, aspects)
    negative_rate = float((valid["rating"] <= 2).mean() * 100)
    alerts = []

    if negative_rate >= 15:
        alerts.append(("Cao", "Tỷ lệ không hài lòng toàn sàn vượt ngưỡng", f"Tỷ lệ đánh giá 1–2 sao đang ở mức {negative_rate:.1f}%. Cần rà soát lại quy trình dịch vụ và đối tác vận chuyển."))
    if summary["rating_text_contradictions"] > 0:
        alerts.append(("Trung bình", "Phát hiện nghịch lý đánh giá", f"Có {summary['rating_text_contradictions']:,} đánh giá có số sao và lời văn trái ngược nhau. Cần chú ý làm sạch dữ liệu."))

    coverage = summary["aspect_coverage"]
    if coverage:
        top_aspect = max(coverage, key=lambda name: coverage[name]["reviews"])
        alerts.append(("Thấp", "Điểm nóng phản hồi khách hàng", f"Khía cạnh nhận được nhiều quan tâm nhất là '{top_aspect.title()}' với {coverage[top_aspect]['reviews']:,} lượt phản ánh ({coverage[top_aspect]['rate_percent']}%)."))

    if "category" in valid.columns:
        cat_neg = valid.groupby("category")["rating"].apply(lambda s: (s <= 2).mean() * 100)
        if not cat_neg.empty:
            worst_cat = cat_neg.idxmax()
            worst_rate = cat_neg.max()
            if worst_rate >= 8:
                alerts.append(("Cao", f"Cảnh báo ngành hàng: {worst_cat}", f"Ngành hàng '{worst_cat}' có tỷ lệ không hài lòng cao nhất ({worst_rate:.1f}%). Cần kiểm tra lô hàng nhập gần đây."))

    if "like_count" in valid.columns:
        high_like_neg = valid[(valid["rating"] <= 2) & (valid["like_count"] >= 5)]
        if not high_like_neg.empty:
            alerts.append(("Cao", "Đánh giá tiêu cực có mức độ lan tỏa cao", f"Có {len(high_like_neg):,} đánh giá tiêu cực nhận được từ 5 lượt hữu ích trở lên; có nguy cơ ảnh hưởng trực tiếp đến quyết định mua của khách hàng mới."))

    if not alerts:
        render_alert_card("Thấp", "Tình trạng ổn định", "Chưa phát hiện rủi ro nào vượt ngưỡng cảnh báo.")
    else:
        for level, title, message in alerts:
            render_alert_card(level, title, message)

    st.markdown("##### Khuyến nghị giải pháp cho doanh nghiệp / Người bán")
    st.markdown(
        """
        - 🛠️ **Cải tiến chất lượng sản phẩm**: Ưu tiên kiểm định các mặt hàng thuộc ngành hàng có tỷ lệ tiêu cực cao nhất.
        - 🚚 **Tối ưu quy trình vận hành**: Nếu khía cạnh *Giao hàng* hoặc *Đóng gói* bị phàn nàn, cần xem xét lại đơn vị vận chuyển và quy chuẩn bọc chống sốc.
        - 💬 **Chăm sóc khách hàng chủ động**: Liên hệ hỗ trợ các khách hàng có đánh giá nhận được nhiều like để xử lý khiếu nại trước khi khủng hoảng truyền thông lan rộng.
        """
    )


def render_risk_explorer(view: pd.DataFrame) -> None:
    render_page_header(
        title="Tra cứu & Trích xuất Dữ liệu Đánh giá Toàn văn",
        description="Tìm kiếm toàn văn theo từ khóa, lọc theo trạng thái và xuất báo cáo CSV phục vụ phòng ban liên quan",
    )
    col_search, col_sort = st.columns([3, 1])
    with col_search:
        search = st.text_input("Tìm kiếm theo từ khóa trong nội dung:", placeholder="Ví dụ: giao hàng chậm, chất lượng kém, đóng gói móp...")
    with col_sort:
        sort_by = searchable_selectbox(
            "Sắp xếp theo:",
            ["Nhiều lượt thích nhất (Hữu ích)", "Mới nhất", "Rating thấp nhất (1★)", "Rating cao nhất (5★)"],
            key="risk_review_sort",
        )
        if sort_by is None:
            return

    status_options = sorted(view["filter_status"].dropna().astype(str).unique()) if "filter_status" in view else []
    statuses = searchable_multiselect(
        "Lọc trạng thái dữ liệu:",
        status_options,
        default=status_options,
        key="risk_review_statuses",
    )
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

    col_configs = {
        "created_at": st.column_config.DatetimeColumn(label="Ngày đánh giá", format="DD/MM/YYYY"),
        "platform": st.column_config.TextColumn(label="Sàn", width="small"),
        "category": st.column_config.TextColumn(label="Ngành hàng", width="small"),
        "product_name": st.column_config.TextColumn(label="Sản phẩm", width="medium"),
        "reviewer_name": st.column_config.TextColumn(label="Khách hàng", width="small"),
        "rating": st.column_config.NumberColumn(label="Số sao", format="%d ★", width="small"),
        "like_count": st.column_config.ProgressColumn(
            label="Lượt thích",
            format="%d 👍",
            min_value=0,
            max_value=max(1, int(view["like_count"].max())) if "like_count" in view.columns and not view["like_count"].dropna().empty else 10,
        ),
        "filter_status": st.column_config.TextColumn(label="Trạng thái", width="small"),
        "comment_text_raw": st.column_config.TextColumn(label="Nội dung nhận xét", width="large"),
    }
    columns = ["created_at", "platform", "category", "product_name", "reviewer_name", "rating", "like_count", "filter_status", "comment_text_raw"]
    st.caption(f"Tìm thấy **{len(table):,}** đánh giá phù hợp")
    st.dataframe(table[[c for c in columns if c in table.columns]], column_config=col_configs, width="stretch", hide_index=True)
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
    render_page_header(
        title="Phòng Thử nghiệm & Đánh giá Mô hình AI (Model Lab)",
        description="Huấn luyện, tinh chỉnh và so sánh các giải thuật học máy phân loại cảm xúc văn bản với trích xuất đặc trưng TF-IDF",
    )
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
        model_name = searchable_selectbox(
            "Chọn giải thuật học máy:",
            ["Hồi quy Logistic (Logistic Regression)", "Máy véc-tơ hỗ trợ (Linear SVM)"],
            key="research_model",
        )
    if model_name is None:
        return
    with c_m2:
        test_size = st.slider("Tỷ lệ tập kiểm thử (Test split):", 0.15, 0.40, 0.25, 0.05)

    if st.button("Bắt đầu Huấn luyện & Đánh giá", type="primary"):
        with st.spinner("Đang huấn luyện mô hình và tính toán chỉ số đánh giá..."):
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
            render_kpi("Độ chính xác (Accuracy)", f"{accuracy_score(y_test, predictions):.3f}", icon="🎯", delta="Tập kiểm thử", tooltip="Tỷ lệ đoán đúng trên toàn bộ tập test")
        with metrics[1]:
            render_kpi("Độ chuẩn xác (Precision)", f"{report['macro avg']['precision']:.3f}", icon="🔍", delta="Macro Average", tooltip="Độ tin cậy khi gán nhãn")
        with metrics[2]:
            render_kpi("Độ bao phủ (Recall)", f"{report['macro avg']['recall']:.3f}", icon="📡", delta="Macro Average", tooltip="Khả năng bắt trúng mẫu của từng lớp")
        with metrics[3]:
            f1 = report['macro avg']['f1-score']
            render_kpi("F1-Score trung bình", f"{f1:.3f}", icon="⚖️", delta="Chỉ số chính", delta_type="good" if f1 >= 0.7 else "neutral", tooltip="Trung bình điều hòa giữa Precision và Recall")

        labels = ["negative", "neutral", "positive"]
        labels_vn = ["Tiêu cực", "Trung tính", "Tích cực"]
        matrix = pd.DataFrame(confusion_matrix(y_test, predictions, labels=labels), index=labels_vn, columns=labels_vn)
        
        st.markdown("##### Ma trận nhầm lẫn (Confusion Matrix)")
        st.dataframe(matrix, width="stretch")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        st.session_state["trained_model"] = model
        st.success(f"Mô hình đã được huấn luyện và lưu thành công tại `{MODEL_PATH}`! Bạn có thể sang 'Phân tích Khía cạnh' -> Tab 'Mô phỏng' để thử nghiệm dự đoán ngay.")


def render_research_labeling(view: pd.DataFrame, aspects: dict) -> None:
    render_page_header(
        title="Công cụ Gán nhãn Dữ liệu Thủ công (Active Labeling)",
        description="Gán nhãn thủ công và lưu thành tập đánh giá độc lập với nhãn suy ra từ rating",
    )
    valid = valid_rows(view)
    if valid.empty:
        render_empty_state("Chưa có review hợp lệ để gán nhãn", "Mở rộng bộ lọc hoặc nạp thêm review đã qua tiền xử lý.")
        return

    try:
        labels_df = load_manual_labels()
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Không thể đọc tập nhãn thủ công `{MANUAL_LABELS_PATH}`: {error}")
        return

    valid = valid.copy()
    valid["review_id"] = valid["review_id"].astype(str)
    valid["platform"] = valid.get("platform", pd.Series("", index=valid.index)).fillna("").astype(str)
    valid["_label_key"] = valid["platform"] + "::" + valid["review_id"]

    if not labels_df.empty:
        labels_df["platform"] = labels_df["platform"].fillna("").astype(str)
        labels_df["review_id"] = labels_df["review_id"].astype(str)
        labeled_keys = set(labels_df["platform"] + "::" + labels_df["review_id"])
    else:
        labeled_keys = set()

    selection_mode = st.radio(
        "Mẫu cần gán nhãn:",
        ["Chưa gán nhãn", "Tất cả mẫu"],
        horizontal=True,
    )
    candidates = valid if selection_mode == "Tất cả mẫu" else valid[~valid["_label_key"].isin(labeled_keys)]

    if candidates.empty:
        st.info("Không còn mẫu chưa gán nhãn trong phạm vi bộ lọc hiện tại.")
    else:
        selected_index = searchable_selectbox(
            "Chọn review cần gán nhãn:",
            candidates.index.tolist(),
            key="label_review",
            format_func=lambda index: (
                f"{candidates.at[index, 'platform']} | "
                f"{candidates.at[index, 'product_name']} | "
                f"{candidates.at[index, 'review_id']}"
            ),
        )
        if selected_index is None:
            return
        row = candidates.loc[selected_index]
        review_id = str(row["review_id"])
        platform = str(row["platform"])
        prod_name = row.get("product_name", "N/A")
        cat_name = row.get("category", "N/A")
        star = row.get("rating", "N/A")
        user = row.get("reviewer_name", "Khách hàng ẩn danh")
        likes = row.get("like_count", 0)

        st.markdown(
            f'<div class="info-strip">'
            f'  <span>📦 <b>Sản phẩm:</b> {prod_name}</span>'
            f'  <span>📁 <b>Ngành hàng:</b> {cat_name}</span>'
            f'  <span>🛒 <b>Sàn:</b> {platform}</span>'
            f'  <span>⭐ <b>Số sao:</b> {star}★</span>'
            f'  <span>👤 <b>Khách hàng:</b> {user} ({likes} 👍)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.info(f"Nội dung nhận xét: \"{row.get('comment_text_raw', '')}\"")

        label_widget_key = f"manual_label_{platform}_{review_id}"
        sentiment = searchable_selectbox(
            "Nhãn cảm xúc thủ công:",
            ["— Chọn nhãn —", "Tiêu cực", "Trung tính", "Tích cực"],
            key=f"{label_widget_key}_sentiment",
        )
        selected_aspects = searchable_multiselect(
            "Các khía cạnh đề cập trong câu:",
            list(aspects.keys()),
            default=[],
            key=f"{label_widget_key}_aspects",
            format_func=lambda value: value.title(),
        )
        submitted = st.button("Lưu nhãn đánh giá này")

        if submitted:
            if sentiment == "— Chọn nhãn —":
                st.error("Hãy chọn nhãn cảm xúc trước khi lưu.")
            else:
                record = {
                    "review_id": review_id,
                    "platform": platform,
                    "product_name": prod_name,
                    "category": cat_name,
                    "rating": star,
                    "text": str(row.get("comment_text_raw", "")),
                    "sentiment_manual": sentiment,
                    "aspects_manual": ", ".join(selected_aspects),
                    "annotated_at": datetime.now().isoformat(timespec="seconds"),
                }
                label_key = f"{platform}::{review_id}"
                if not labels_df.empty:
                    existing_keys = labels_df["platform"] + "::" + labels_df["review_id"]
                    labels_df = labels_df.loc[existing_keys != label_key]
                updated_labels = pd.concat([labels_df, pd.DataFrame([record])], ignore_index=True)
                try:
                    save_manual_labels(updated_labels)
                except OSError as error:
                    st.error(f"Không thể lưu nhãn thủ công: {error}")
                else:
                    labels_df = updated_labels
                    st.success(f"Đã lưu nhãn cho review {review_id} vào `{MANUAL_LABELS_PATH}`.")

    st.markdown("##### Tập nhãn thủ công đã lưu")
    st.caption(f"Hiện có {len(labels_df):,} review đã gán nhãn.")
    if not labels_df.empty:
        class_counts = labels_df["sentiment_manual"].value_counts()
        st.caption(
            "Phân bố nhãn: "
            + " · ".join(f"{label}: {int(class_counts.get(label, 0))}" for label in ["Tiêu cực", "Trung tính", "Tích cực"])
        )
        st.dataframe(labels_df, width="stretch", hide_index=True)
        st.download_button(
            "Xuất tập nhãn vàng (CSV)",
            labels_df.to_csv(index=False).encode("utf-8-sig"),
            "manual_labels.csv",
            "text/csv",
        )


def render_research_report(view: pd.DataFrame, aspects: dict) -> None:
    render_page_header(
        title="Báo cáo Khoa học Tổng hợp Đề tài",
        description="Tổng kết toàn bộ chỉ số định lượng, mức độ bao phủ khía cạnh và thống kê mô tả phục vụ báo cáo đồ án",
    )
    if view.empty:
        render_empty_state("Không có dữ liệu", "Không có dữ liệu để lập báo cáo.")
        return

    valid = valid_rows(view)
    _, summary = cached_build_report(view, aspects)

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
    st.dataframe(report_df, width="stretch", hide_index=True)

    if "category" in valid.columns:
        st.markdown("##### 2. Tổng kết hiệu suất theo từng Ngành hàng")
        cat_stats = valid.groupby("category").agg(
            total_reviews=("review_id", "count"),
            avg_rating=("rating", "mean"),
            negative_reviews=("rating", lambda s: int((s <= 2).sum())),
            negative_rate=("rating", lambda s: round(float((s <= 2).mean() * 100), 1)),
            total_likes=("like_count", "sum") if "like_count" in valid.columns else ("review_id", "count")
        ).reset_index()
        cat_stats["avg_rating"] = cat_stats["avg_rating"].round(2)

        col_configs = {
            "category": st.column_config.TextColumn(label="Ngành hàng", width="medium"),
            "total_reviews": st.column_config.NumberColumn(label="Số đánh giá", format="%d"),
            "avg_rating": st.column_config.NumberColumn(label="Điểm TB", format="%.2f ★"),
            "negative_reviews": st.column_config.NumberColumn(label="Đánh giá 1–2★", format="%d"),
            "negative_rate": st.column_config.NumberColumn(label="Tỷ lệ tiêu cực", format="%.1f%%"),
            "total_likes": st.column_config.NumberColumn(label="Tổng Like", format="%d 👍"),
        }
        st.dataframe(cat_stats, column_config=col_configs, width="stretch", hide_index=True)

    st.markdown("##### 3. Mức độ bao phủ của Từ điển Khía cạnh (Aspect Coverage)")
    coverage_rows = [
        {"Khía cạnh sản phẩm": asp.title(), "Số lượt xuất hiện": data["reviews"], "Tỷ lệ bao phủ (%)": data["rate_percent"]}
        for asp, data in summary.get("aspect_coverage", {}).items()
    ]
    cov_df = pd.DataFrame(coverage_rows)
    col_configs = {
        "Khía cạnh sản phẩm": st.column_config.TextColumn(label="Khía cạnh sản phẩm", width="medium"),
        "Số lượt xuất hiện": st.column_config.NumberColumn(label="Số lượt xuất hiện", format="%d"),
        "Tỷ lệ bao phủ (%)": st.column_config.NumberColumn(label="Tỷ lệ bao phủ", format="%.2f%%"),
    }
    st.dataframe(cov_df, column_config=col_configs, width="stretch", hide_index=True)

    st.download_button(
        "Tải toàn văn báo cáo tổng hợp (CSV)",
        report_df.to_csv(index=False).encode("utf-8-sig"),
        "bao_cao_khoa_hoc_customer_voice.csv",
        "text/csv",
    )


# ==============================================================================
# MODULE 5: THU THẬP DỮ LIỆU (CRAWLER UI)
# ==============================================================================

def _crawl_worker(
    platform: str,
    product_id: str,
    shop_id: str,
    product_name: str,
    category: str,
    max_reviews: int,
    output_dir: str,
    run_preprocess: bool,
    log_queue: queue.Queue,
) -> None:
    """Chạy crawler trong luồng riêng, ghi log vào queue."""
    try:
        log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ Khởi động crawler cho sàn {platform.upper()}…")

        if platform == "tiki":
            from src.crawler.tiki_crawler import TikiCrawler
            crawler = TikiCrawler()
            crawler.fetch_product_reviews(
                shop_id=shop_id,
                product_id=product_id,
                product_name=product_name or None,
                max_reviews=max_reviews,
                category=category or None,
            )
        else:
            from src.crawler.shopee_crawler import ShopeeCrawler
            crawler = ShopeeCrawler()
            crawler.fetch_product_reviews(
                shop_id=shop_id,
                product_id=product_id,
                product_name=product_name or "Sản phẩm Shopee",
                max_reviews=max_reviews,
                category=category or "Thiết bị điện tử",
            )

        saved = crawler.save_raw_data(output_dir=output_dir)
        n_saved = len(crawler.reviews)
        log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Thu thập xong {n_saved} đánh giá.")

        if saved:
            log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Đã lưu: {saved.get('csv', 'N/A')}")

        if run_preprocess and saved:
            log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ Đang chạy tiền xử lý dữ liệu…")
            try:
                from scripts.run_preprocess import run_full_preprocessing
                run_full_preprocessing(raw_dir=output_dir)
                log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Tiền xử lý hoàn tất! Dashboard sẽ được cập nhật khi bạn làm mới.")
            except Exception as exc:
                log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Lỗi tiền xử lý: {exc}")

        log_queue.put("__DONE__")

    except Exception as exc:
        log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Lỗi crawler: {exc}")
        log_queue.put("__DONE__")


def render_crawler() -> None:
    """Module Thu thập Dữ liệu – giao diện Crawler trong Dashboard."""
    render_page_header(
        ":material/cloud_download: Thu thập Dữ liệu",
        "Cào đánh giá từ Tiki / Shopee, lưu raw data và tự động chạy tiền xử lý để cập nhật Dashboard.",
    )

    with st.expander("Hướng dẫn sử dụng Crawler", expanded=False):
        st.markdown(
            """
            1. **Lấy mã sản phẩm từ URL:**
               - Tiki: URL dạng `...-p185076264.html` → Product ID là `185076264`.
               - Shopee: URL dạng `...-i.98765432.1234567890` → Shop ID là `98765432`, Item ID là `1234567890`.
            2. Nhập các mã bắt buộc, chọn số review cần lấy (10–500), rồi nhấn **Bắt đầu Crawl**.
            3. Theo dõi log tại tab của sàn. File CSV raw được lưu trong `data/raw/`; có thể xem và tải ở tab **Lịch sử & Tiến độ**.
            4. Bật **Tự động chạy tiền xử lý** để cập nhật dữ liệu processed. Sau đó nhấn **Làm mới bộ nhớ đệm** trong **Cài đặt nâng cao** để dashboard nạp dữ liệu mới.
            """
        )
        st.info(
            "Crawl thật có thể bị giới hạn bởi sàn hoặc trả về ít review hơn số lượng yêu cầu. "
            "Hãy kiểm tra log và dữ liệu raw sau mỗi lần chạy."
        )

    tab_tiki, tab_shopee, tab_history = st.tabs([
        ":material/store: Crawl Tiki",
        ":material/shopping_bag: Crawl Shopee",
        ":material/history: Lịch sử & Tiến độ",
    ])

    # ── TAB CRAWL TIKI ──────────────────────────────────────────────────────
    with tab_tiki:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Cấu hình sản phẩm Tiki")
        st.caption("Nhập **Product ID** từ URL sản phẩm Tiki (vd: `https://tiki.vn/…-p**185076264**.html`).")

        col1, col2 = st.columns(2)
        with col1:
            tiki_pid  = st.text_input("Product ID (*bắt buộc*)", placeholder="185076264", key="tiki_pid")
            tiki_name = st.text_input("Tên sản phẩm (tuỳ chọn)", placeholder="Tự động lấy từ Tiki", key="tiki_name")
        with col2:
            tiki_cat  = st.text_input("Danh mục (tuỳ chọn)", placeholder="Tự động phát hiện", key="tiki_cat")
            tiki_max  = st.slider("Số đánh giá tối đa cần cào", 10, 500, 100, 10, key="tiki_max")

        tiki_preprocess = st.checkbox("Tự động chạy tiền xử lý sau khi cào xong", value=True, key="tiki_prep")
        tiki_output     = str(ROOT / "data" / "raw")

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button(":material/play_arrow: Bắt đầu Crawl Tiki", type="primary", key="btn_tiki"):
            if not tiki_pid.strip():
                st.error("Vui lòng nhập **Product ID** của sản phẩm Tiki.")
            else:
                # Khởi tạo session state cho Tiki
                st.session_state["crawl_tiki_logs"]   = []
                st.session_state["crawl_tiki_done"]   = False
                st.session_state["crawl_tiki_queue"]  = queue.Queue()

                t = threading.Thread(
                    target=_crawl_worker,
                    kwargs=dict(
                        platform="tiki",
                        product_id=tiki_pid.strip(),
                        shop_id="auto",
                        product_name=tiki_name.strip(),
                        category=tiki_cat.strip(),
                        max_reviews=tiki_max,
                        output_dir=tiki_output,
                        run_preprocess=tiki_preprocess,
                        log_queue=st.session_state["crawl_tiki_queue"],
                    ),
                    daemon=True,
                )
                t.start()
                st.session_state["crawl_tiki_thread"] = t
                st.rerun()

        # Hiển thị log live cho Tiki
        if "crawl_tiki_queue" in st.session_state and not st.session_state.get("crawl_tiki_done", True):
            q: queue.Queue = st.session_state["crawl_tiki_queue"]
            while not q.empty():
                msg = q.get_nowait()
                if msg == "__DONE__":
                    st.session_state["crawl_tiki_done"] = True
                    st.cache_data.clear()
                else:
                    st.session_state["crawl_tiki_logs"].append(msg)

            log_text = "\n".join(st.session_state.get("crawl_tiki_logs", []))
            st.code(log_text or "Đang khởi động…", language="text")

            if not st.session_state.get("crawl_tiki_done", True):
                time.sleep(1)
                st.rerun()

        if st.session_state.get("crawl_tiki_done"):
            log_text = "\n".join(st.session_state.get("crawl_tiki_logs", []))
            st.code(log_text, language="text")
            render_alert_card("low", "Crawl hoàn tất", "Nhấn **Làm mới bộ nhớ đệm** trong phần Cài đặt nâng cao để Dashboard hiển thị dữ liệu mới.")

            # Hiển thị preview dữ liệu raw vừa cào
            raw_csv = Path(tiki_output) / "tiki_raw.csv"
            if raw_csv.exists():
                st.markdown("**Xem trước dữ liệu raw (50 dòng mới nhất):**")
                preview_df = pd.read_csv(raw_csv, encoding="utf-8-sig").tail(50)
                st.dataframe(preview_df, width="stretch")

                csv_bytes = preview_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    ":material/download: Tải xuống raw CSV",
                    csv_bytes,
                    file_name="tiki_raw_preview.csv",
                    mime="text/csv",
                )

    # ── TAB CRAWL SHOPEE ─────────────────────────────────────────────────────
    with tab_shopee:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Cấu hình sản phẩm Shopee")
        st.caption("Shopee yêu cầu cả **Shop ID** lẫn **Item ID** (xem trong URL sản phẩm: `…i.**shopId**.**itemId**`).")

        col1, col2 = st.columns(2)
        with col1:
            sp_pid   = st.text_input("Item ID (*bắt buộc*)",  placeholder="1234567890", key="sp_pid")
            sp_sid   = st.text_input("Shop ID (*bắt buộc*)",  placeholder="98765432",   key="sp_sid")
        with col2:
            sp_name  = st.text_input("Tên sản phẩm (tuỳ chọn)", placeholder="Sản phẩm Shopee", key="sp_name")
            sp_cat   = st.text_input("Danh mục (tuỳ chọn)",      placeholder="Thiết bị điện tử", key="sp_cat")

        sp_max = st.slider("Số đánh giá tối đa cần cào", 10, 500, 100, 10, key="sp_max")
        sp_preprocess = st.checkbox("Tự động chạy tiền xử lý sau khi cào xong", value=True, key="sp_prep")
        sp_output = str(ROOT / "data" / "raw")

        st.markdown('</div>', unsafe_allow_html=True)

        render_alert_card(
            "medium",
            "Lưu ý về Shopee API",
            "Shopee thường bật WAF / rate-limit cho các request không có cookie phiên. "
            "Nếu gặp lỗi 403 / dữ liệu trống, hãy thử lại sau vài phút hoặc giảm số lượng review.",
        )

        if st.button(":material/play_arrow: Bắt đầu Crawl Shopee", type="primary", key="btn_shopee"):
            if not sp_pid.strip() or not sp_sid.strip():
                st.error("Vui lòng nhập đủ **Item ID** và **Shop ID** của sản phẩm Shopee.")
            else:
                st.session_state["crawl_sp_logs"]   = []
                st.session_state["crawl_sp_done"]   = False
                st.session_state["crawl_sp_queue"]  = queue.Queue()

                t = threading.Thread(
                    target=_crawl_worker,
                    kwargs=dict(
                        platform="shopee",
                        product_id=sp_pid.strip(),
                        shop_id=sp_sid.strip(),
                        product_name=sp_name.strip(),
                        category=sp_cat.strip(),
                        max_reviews=sp_max,
                        output_dir=sp_output,
                        run_preprocess=sp_preprocess,
                        log_queue=st.session_state["crawl_sp_queue"],
                    ),
                    daemon=True,
                )
                t.start()
                st.session_state["crawl_sp_thread"] = t
                st.rerun()

        # Hiển thị log live cho Shopee
        if "crawl_sp_queue" in st.session_state and not st.session_state.get("crawl_sp_done", True):
            q: queue.Queue = st.session_state["crawl_sp_queue"]
            while not q.empty():
                msg = q.get_nowait()
                if msg == "__DONE__":
                    st.session_state["crawl_sp_done"] = True
                    st.cache_data.clear()
                else:
                    st.session_state["crawl_sp_logs"].append(msg)

            log_text = "\n".join(st.session_state.get("crawl_sp_logs", []))
            st.code(log_text or "Đang khởi động…", language="text")

            if not st.session_state.get("crawl_sp_done", True):
                time.sleep(1)
                st.rerun()

        if st.session_state.get("crawl_sp_done"):
            log_text = "\n".join(st.session_state.get("crawl_sp_logs", []))
            st.code(log_text, language="text")
            render_alert_card("low", "Crawl hoàn tất", "Nhấn **Làm mới bộ nhớ đệm** trong phần Cài đặt nâng cao để Dashboard hiển thị dữ liệu mới.")

            raw_csv = Path(sp_output) / "shopee_raw.csv"
            if raw_csv.exists():
                st.markdown("**Xem trước dữ liệu raw (50 dòng mới nhất):**")
                preview_df = pd.read_csv(raw_csv, encoding="utf-8-sig").tail(50)
                st.dataframe(preview_df, width="stretch")

                csv_bytes = preview_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    ":material/download: Tải xuống raw CSV",
                    csv_bytes,
                    file_name="shopee_raw_preview.csv",
                    mime="text/csv",
                )

    # ── TAB LỊCH SỬ & TIẾN ĐỘ ────────────────────────────────────────────────
    with tab_history:
        st.markdown("#### Dữ liệu raw hiện tại trong hệ thống")
        raw_dir = ROOT / "data" / "raw"
        raw_files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []

        if not raw_files:
            render_empty_state("Chưa có dữ liệu raw", "Hãy chạy Crawl Tiki hoặc Crawl Shopee để bắt đầu thu thập.", ":material/inbox:")
        else:
            overview_rows = []
            for f in raw_files:
                try:
                    tmp = pd.read_csv(f, encoding="utf-8-sig")
                    overview_rows.append({
                        "Tệp"               : f.name,
                        "Số đánh giá"       : len(tmp),
                        "Sàn"               : tmp["platform"].iloc[0] if "platform" in tmp.columns else "—",
                        "Lần cập nhật cuối" : datetime.fromtimestamp(f.stat().st_mtime).strftime("%d/%m/%Y %H:%M"),
                        "Kích thước"        : f"{f.stat().st_size / 1024:.1f} KB",
                    })
                except Exception:
                    pass

            if overview_rows:
                st.dataframe(pd.DataFrame(overview_rows), width="stretch")

            # Nút tải từng file
            st.markdown("**Tải xuống file raw:**")
            dl_cols = st.columns(min(len(raw_files), 4))
            for idx, rf in enumerate(raw_files):
                with dl_cols[idx % 4]:
                    raw_bytes = rf.read_bytes()
                    st.download_button(
                        f":material/download: {rf.name}",
                        raw_bytes,
                        file_name=rf.name,
                        mime="text/csv",
                        key=f"dl_raw_{rf.name}",
                    )

        st.divider()
        st.markdown("#### Dữ liệu đã xử lý (Processed)")
        proc_csv = ROOT / "data" / "processed" / "reviews_processed.csv"
        if proc_csv.exists():
            proc_df = pd.read_csv(proc_csv, encoding="utf-8-sig")
            col1, col2, col3 = st.columns(3)
            with col1:
                render_kpi("Tổng đánh giá đã xử lý", f"{len(proc_df):,}", ":material/check_circle:", "#10b981")
            with col2:
                plats = proc_df["platform"].nunique() if "platform" in proc_df.columns else 0
                render_kpi("Số sàn TMĐT", str(plats), ":material/storefront:", "#2563eb")
            with col3:
                last_mod = datetime.fromtimestamp(proc_csv.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
                render_kpi("Cập nhật lần cuối", last_mod, ":material/update:", "#64748b")

            st.markdown("**50 dòng cuối cùng của tập processed:**")
            st.dataframe(proc_df.tail(50), width="stretch")
        else:
            render_empty_state(
                "Chưa có dữ liệu đã xử lý",
                "Sau khi crawl, hãy bật tùy chọn **Tự động chạy tiền xử lý** hoặc chạy `python scripts/run_preprocess.py` thủ công.",
                ":material/hourglass_empty:",
            )


# ==============================================================================
# HÀM TẠO 3-4 THẺ INSIGHT ĐỘNG CHO EXECUTIVE SUMMARY (KHÔNG HARDCODE)
# ==============================================================================

def render_dynamic_executive_summary(view: pd.DataFrame, valid: pd.DataFrame, summary: dict) -> None:
    if view.empty or valid.empty:
        return

    # 1. Khía cạnh bị phàn nàn nhiều nhất (Tính toán động)
    coverage = summary.get("aspect_coverage", {})
    if coverage:
        top_aspect_name = max(coverage, key=lambda k: coverage[k]["reviews"])
        top_aspect_cnt = coverage[top_aspect_name]["reviews"]
        top_aspect_pct = coverage[top_aspect_name]["rate_percent"]
        card1_val = f"{top_aspect_name.title()}"
        card1_desc = f"Nhận được <b>{top_aspect_cnt:,}</b> phản ánh ({top_aspect_pct}% tổng số), cần ưu tiên tối ưu vận hành."
    else:
        card1_val = "Chưa rõ"
        card1_desc = "Chưa phát hiện khía cạnh nào nổi bật."

    # 2. Ngành hàng rủi ro nhất (Tính toán động từ tỷ lệ tiêu cực)
    if "category" in valid.columns:
        cat_neg = valid.groupby("category")["rating"].apply(lambda s: (s <= 2).mean() * 100)
        worst_cat = cat_neg.idxmax()
        worst_rate = cat_neg.max()
        card2_val = f"{worst_rate:.1f}% tiêu cực"
        card2_desc = f"Ngành hàng <b>{worst_cat}</b> có tỷ lệ phàn nàn cao nhất toàn sàn."
    else:
        card2_val = "N/A"
        card2_desc = "Không có thông tin ngành hàng."

    # 3. Sàn hài lòng hơn (Tính toán động từ dữ liệu nền tảng)
    if "platform" in valid.columns and valid["platform"].nunique() >= 2:
        plat_means = valid.groupby("platform")["rating"].mean()
        best_plat = plat_means.idxmax()
        worst_plat = plat_means.idxmin()
        diff = plat_means.max() - plat_means.min()
        card3_val = f"{best_plat.upper()}"
        card3_desc = f"Đạt <b>{plat_means.max():.2f}★</b> (cao hơn {worst_plat.upper()} <b>+{diff:.2f}★</b>)."
    elif "platform" in valid.columns and valid["platform"].nunique() == 1:
        single_plat = valid["platform"].iloc[0]
        p_mean = valid["rating"].mean()
        card3_val = f"{single_plat.upper()}"
        card3_desc = f"Điểm trung bình nền tảng đạt <b>{p_mean:.2f}★</b> / 5.0."
    else:
        card3_val = "N/A"
        card3_desc = "Chưa có thông tin nền tảng."

    # 4. Số đánh giá cần xử lý khẩn cấp (1-2 sao hoặc có nghịch lý rating)
    neg_reviews = int((valid["rating"] <= 2).sum())
    contra_reviews = summary.get("rating_text_contradictions", 0)
    need_action = neg_reviews + contra_reviews
    card4_val = f"{need_action:,} phản hồi"
    card4_desc = f"Gồm <b>{neg_reviews:,}</b> đánh giá 1–2★ và <b>{contra_reviews:,}</b> trường hợp nghịch lý sao."

    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f'<div class="insight-card insight-card-warning">'
            f'  <div class="insight-label">Điểm nóng khía cạnh</div>'
            f'  <div class="insight-value">{card1_val}</div>'
            f'  <p class="insight-desc">{card1_desc}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f'<div class="insight-card insight-card-danger">'
            f'  <div class="insight-label">Ngành hàng rủi ro nhất</div>'
            f'  <div class="insight-value">{card2_val}</div>'
            f'  <p class="insight-desc">{card2_desc}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f'<div class="insight-card insight-card-success">'
            f'  <div class="insight-label">Sàn có điểm số cao hơn</div>'
            f'  <div class="insight-value">{card3_val}</div>'
            f'  <p class="insight-desc">{card3_desc}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            f'<div class="insight-card insight-card-primary">'
            f'  <div class="insight-label">Cần xử lý khẩn cấp</div>'
            f'  <div class="insight-value">{card4_val}</div>'
            f'  <p class="insight-desc">{card4_desc}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ==============================================================================
# CHƯƠNG TRÌNH CHÍNH (MAIN APPLICATION)
# ==============================================================================

def main() -> None:
    # 1. Thanh điều hướng một tầng và thông tin Sidebar gọn gàng
    st.sidebar.markdown(
        '<div style="padding:0.25rem 0 0.85rem 0;">'
        '<div style="font-size:1.15rem;font-weight:700;color:var(--color-text-main);letter-spacing:-0.03em;">✦ Customer Voice AI</div>'
        '<div style="font-size:0.775rem;color:var(--color-text-muted);">SaaS Analytics Platform</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ĐIỀU HƯỚNG MỘT TẦNG Ở SIDEBAR KHÔNG DÙNG EMOJI (MATERIAL ICONS)
    nav_page = st.sidebar.radio(
        "Điều hướng hệ thống",
        [
            ":material/dashboard: Báo cáo Thị trường & Tổng quan",
            ":material/category: Phân tích Khía cạnh Sản phẩm (ABSA)",
            ":material/warning: Cảnh báo Rủi ro & Khách hàng",
            ":material/science: Phòng Nghiên cứu & Mô hình AI",
            ":material/cloud_download: Thu thập Dữ liệu (Crawler)",
        ],
        label_visibility="collapsed",
    )

    # Cài đặt nâng cao gom trong Expander
    advanced_settings = st.sidebar.expander(":material/settings: Cài đặt nâng cao")
    with advanced_settings:
        data_source = searchable_selectbox(
            "Nguồn dữ liệu:",
            ["CSV", "SQL Server"],
            key="dashboard_data_source",
            container=advanced_settings,
        )
        if data_source == "SQL Server":
            data_path = str(DEFAULT_DATA)
            sql_server = st.text_input(
                "SQL Server instance:",
                os.getenv("SQLSERVER_SERVER", "localhost"),
            )
            database_name = st.text_input(
                "Tên database:",
                os.getenv("SQLSERVER_DATABASE", DEFAULT_DATABASE_NAME),
            )
        else:
            data_path = st.text_input("Đường dẫn tệp dữ liệu:", str(DEFAULT_DATA))
            sql_server = "localhost"
            database_name = DEFAULT_DATABASE_NAME
        file_mtime = get_file_mtime(data_path) if data_source == "CSV" else 0.0
        if st.button("Làm mới bộ nhớ đệm", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Nạp dữ liệu
    try:
        df = load_data(
            data_path,
            file_mtime,
            data_source=data_source,
            sql_server=sql_server,
            database_name=database_name,
        )
    except Exception as error:
        st.sidebar.error(f"Không thể nạp dữ liệu từ {data_source}: {error}")
        st.stop()

    if df.empty:
        st.info("Nguồn dữ liệu chưa có review. Hãy import CSV hoặc chạy crawler trước.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div style="font-size:0.8rem;font-weight:700;color:var(--color-text-muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.4rem;">Bộ lọc dữ liệu</div>', unsafe_allow_html=True)

    # Bộ lọc nền tảng
    platforms = sorted(df["platform"].dropna().astype(str).unique())
    selected_platforms = searchable_multiselect(
        "Nền tảng sàn:",
        platforms,
        default=platforms,
        key="sidebar_platforms",
        container=st.sidebar,
    )

    # Bộ lọc ngành hàng
    categories = sorted([str(c) for c in df["category"].dropna().unique() if str(c).strip()]) if "category" in df.columns else []
    selected_categories = (
        searchable_multiselect(
            "Ngành hàng:",
            categories,
            default=categories,
            key="sidebar_categories",
            container=st.sidebar,
        )
        if categories else []
    )

    # Cascade sản phẩm theo ngành hàng
    cat_df = df[df["category"].astype(str).isin(selected_categories)] if (categories and selected_categories) else df
    products = sorted(cat_df["product_name"].dropna().astype(str).unique()) if "product_name" in cat_df.columns else []
    selected_products = searchable_multiselect(
        "Sản phẩm cụ thể:",
        products,
        default=[],
        key="sidebar_products",
        container=st.sidebar,
    )

    # Khoảng sao và lọc rác
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

    # Thông tin nhóm ở chân Sidebar
    st.sidebar.markdown(
        """
        <div class="sidebar-footer">
        <b>Đề tài</b>: Khai phá ý kiến người dùng TMĐT<br>
        <b>Nhóm sinh viên</b>: Long - Mạnh<br>
        <b>Giảng viên HD</b>: Bộ môn Khoa học Dữ liệu
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Executive Summary Insight Cards (Hiển thị đầu trang)
    try:
        _, summary = cached_build_report(view, aspects)
        render_dynamic_executive_summary(view, valid, summary)
    except Exception:
        pass

    # 3. Phân nhánh điều hướng theo các phân hệ với Sub-Tabs sạch sẽ (Không dùng emoji)
    if "Thị trường" in nav_page:
        tab1, tab2, tab3 = st.tabs([
            ":material/insights: Bức tranh toàn cảnh",
            ":material/category: Phân tích theo Ngành hàng",
            ":material/compare: So sánh Nền tảng Sàn",
        ])
        with tab1:
            render_overview_landscape(view, valid, aspects)
        with tab2:
            render_overview_category(view, valid)
        with tab3:
            render_overview_platform(view, valid)

    elif "Khía cạnh" in nav_page:
        tab1, tab2, tab3, tab4 = st.tabs([
            ":material/inventory_2: Theo từng Sản phẩm",
            ":material/grid_view: Ma trận Khía cạnh & Sắc thái",
            ":material/terminal: Thử nghiệm Mô phỏng Đánh giá",
            ":material/quick_reference: Khám phá từ khóa Aspect",
        ])
        with tab1:
            render_absa_product(view, aspects)
        with tab2:
            render_absa_matrix(view, aspects)
        with tab3:
            render_absa_simulator(aspects)
        with tab4:
            render_aspect_discovery(view, aspects)

    elif "Cảnh báo" in nav_page:
        tab1, tab2, tab3 = st.tabs([
            ":material/notification_important: Đánh giá Cần chú ý Khẩn cấp",
            ":material/shield: Trung tâm Cảnh báo Sớm",
            ":material/search: Tra cứu & Trích xuất Toàn văn",
        ])
        with tab1:
            render_risk_attention(view, aspects)
        with tab2:
            render_risk_alerts(view, aspects)
        with tab3:
            render_risk_explorer(view)

    elif "Nghiên cứu" in nav_page:
        tab1, tab2, tab3 = st.tabs([
            ":material/psychology: Huấn luyện & Đánh giá Mô hình",
            ":material/edit_note: Công cụ Gán nhãn Thủ công",
            ":material/description: Báo cáo Khoa học Tổng hợp",
        ])
        with tab1:
            render_research_lab(view)
        with tab2:
            render_research_labeling(view, aspects)
        with tab3:
            render_research_report(view, aspects)

    elif "Crawler" in nav_page:
        render_crawler()

    else:
        tab1, tab2, tab3 = st.tabs([
            ":material/psychology: Huấn luyện & Đánh giá Mô hình",
            ":material/edit_note: Công cụ Gán nhãn Thủ công",
            ":material/description: Báo cáo Khoa học Tổng hợp",
        ])
        with tab1:
            render_research_lab(view)
        with tab2:
            render_research_labeling(view, aspects)
        with tab3:
            render_research_report(view, aspects)


if __name__ == "__main__":
    main()
