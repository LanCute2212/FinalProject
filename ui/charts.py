"""
CUSTOMER VOICE AI - SAAS CHART MODULE (Plotly)
Chuẩn thiết kế: Stripe Dashboard, Linear, Metabase
Nền trong suốt, không viền, lưới mờ, tooltip tiếng Việt, bảng màu cảm xúc chuẩn.
"""

from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ==============================================================================
# HỆ THỐNG MÀU CHUẨN CẢM XÚC & THƯƠNG HIỆU
# ==============================================================================
SENTIMENT_COLORS = {
    "Tích cực": "#10b981",    # Xanh lá ngọc (Emerald 500)
    "Trung tính": "#64748b",   # Xám Slate (Slate 500)
    "Hỗn hợp": "#f59e0b",     # Vàng cam (Amber 500)
    "Tiêu cực": "#ef4444",     # Đỏ san hô (Red 500)
}

BRAND_PRIMARY = "#2563eb"     # Xanh hoàng gia (Blue 600)
BRAND_SECONDARY = "#0284c7"   # Xanh cyan (Sky 600)
CATEGORY_PALETTE = ["#2563eb", "#0284c7", "#0d9488", "#10b981", "#f59e0b", "#6366f1", "#8b5cf6"]


def apply_saas_theme(fig: go.Figure, height: int = 300) -> go.Figure:
    """Áp dụng thiết kế chuẩn SaaS tối giản: trong suốt, không viền cứng, lưới xám mờ."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#334155"),
        margin=dict(t=25, r=20, l=20, b=25),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#e2e8f0",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#0f172a",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
            font=dict(size=11, color="#64748b"),
        ),
    )
    fig.update_xaxes(
        showline=False,
        zeroline=False,
        gridcolor="#f1f5f9",
        tickfont=dict(size=11, color="#64748b"),
        title_font=dict(size=12, color="#475569"),
    )
    fig.update_yaxes(
        showline=False,
        zeroline=False,
        gridcolor="#f1f5f9",
        tickfont=dict(size=11, color="#64748b"),
        title_font=dict(size=12, color="#475569"),
    )
    return fig


def render_chart_card(title: str, subtitle: str, fig: go.Figure) -> None:
    """Đóng gói biểu đồ trong thẻ Card giao diện SaaS."""
    st.markdown(
        f'<div class="chart-card">'
        f'<div class="chart-header">'
        f'<div class="chart-title">{title}</div>'
        f'<div class="chart-subtitle">{subtitle}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ==============================================================================
# CÁC HÀM XÂY DỰNG BIỂU ĐỒ NGHIỆP VỤ
# ==============================================================================

def chart_sentiment_distribution(summary_df: pd.DataFrame, height: int = 280) -> go.Figure:
    """Biểu đồ phân bố cảm xúc theo 3 mức chuẩn (Tiêu cực, Trung tính, Tích cực)."""
    labels = ["Tiêu cực", "Trung tính", "Tích cực"]
    values = [int(summary_df.loc[lab, "Số lượng đánh giá"]) if lab in summary_df.index else 0 for lab in labels]
    total = sum(values) if sum(values) > 0 else 1
    pcts = [v / total * 100 for v in values]
    colors = [SENTIMENT_COLORS[lab] for lab in labels]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color=colors, line=dict(width=0)),
                customdata=pcts,
                hovertemplate="<b>%{x}</b><br>Số lượng: %{y:,} đánh giá<br>Tỷ lệ: %{customdata:.1f}%<extra></extra>",
                text=[f"{v:,} ({p:.1f}%)" if v > 0 else "0" for v, p in zip(values, pcts)],
                textposition="auto",
                textfont=dict(size=11, color="#ffffff"),
            )
        ]
    )
    fig.update_layout(yaxis_title="Số đánh giá", xaxis_title="")
    return apply_saas_theme(fig, height=height)


def chart_data_health(status_counts: pd.Series, height: int = 280) -> go.Figure:
    """Biểu đồ kiểm định tiền xử lý và chất lượng dữ liệu."""
    labels = list(status_counts.index)
    values = list(status_counts.values)
    total = sum(values) if sum(values) > 0 else 1
    pcts = [v / total * 100 for v in values]

    fig = go.Figure(
        data=[
            go.Bar(
                y=labels,
                x=values,
                orientation="h",
                marker=dict(color="#0ea5e9", line=dict(width=0)),
                customdata=pcts,
                hovertemplate="<b>%{y}</b><br>Số lượng: %{x:,} dòng<br>Tỷ lệ: %{customdata:.1f}%<extra></extra>",
                text=[f"{v:,} ({p:.1f}%)" for v, p in zip(values, pcts)],
                textposition="auto",
                textfont=dict(size=11, color="#ffffff"),
            )
        ]
    )
    fig.update_layout(xaxis_title="Số dòng dữ liệu", yaxis_title="", yaxis=dict(autorange="reversed"))
    return apply_saas_theme(fig, height=height)


def chart_sentiment_trend(trend_df: pd.DataFrame, height: int = 280) -> go.Figure:
    """Biểu đồ xu hướng cảm xúc qua các tháng (Line & Area với màu ngữ nghĩa)."""
    fig = go.Figure()
    for sentiment in ["Tiêu cực", "Trung tính", "Tích cực"]:
        if sentiment in trend_df.columns:
            color = SENTIMENT_COLORS[sentiment]
            fig.add_trace(
                go.Scatter(
                    x=trend_df.index,
                    y=trend_df[sentiment],
                    name=sentiment,
                    mode="lines+markers",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=5, color=color),
                    hovertemplate=f"<b>{sentiment}</b> (%{{x}})<br>Số đánh giá: %{{y:,}}<extra></extra>",
                )
            )
    fig.update_layout(yaxis_title="Số đánh giá / Tháng", xaxis_title="Thời gian (Tháng)")
    return apply_saas_theme(fig, height=height)


def chart_category_volume(cat_counts: pd.Series, height: int = 280) -> go.Figure:
    """Biểu đồ so sánh sản lượng đánh giá theo từng ngành hàng."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=cat_counts.values,
                y=cat_counts.index,
                orientation="h",
                marker=dict(color=CATEGORY_PALETTE[:len(cat_counts)], line=dict(width=0)),
                hovertemplate="<b>%{y}</b><br>Số đánh giá: %{x:,}<extra></extra>",
                text=[f"{v:,}" for v in cat_counts.values],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(xaxis_title="Số đánh giá hợp lệ", yaxis_title="", yaxis=dict(autorange="reversed"))
    return apply_saas_theme(fig, height=height)


def chart_category_rating(cat_rating: pd.Series, height: int = 280) -> go.Figure:
    """Biểu đồ điểm số hài lòng trung bình theo ngành hàng kèm vạch tham chiếu 4.0."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=cat_rating.index,
                y=cat_rating.values,
                marker=dict(color="#f59e0b", line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>Điểm TB: %{y:.2f} / 5★<extra></extra>",
                text=[f"{v:.2f}★" for v in cat_rating.values],
                textposition="outside",
            )
        ]
    )
    fig.add_hline(y=4.0, line_dash="dash", line_color="#94a3b8", annotation_text="Ngưỡng chuẩn (4.0★)", annotation_position="top left")
    fig.update_layout(yaxis_title="Điểm số TB (1–5★)", xaxis_title="", yaxis=dict(range=[1, 5.2]))
    return apply_saas_theme(fig, height=height)


def chart_platform_pie(plat_counts: pd.Series, height: int = 270) -> go.Figure:
    """Biểu đồ hình khuyên (Donut) so sánh thị phần dữ liệu các sàn."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=plat_counts.index,
                values=plat_counts.values,
                hole=0.55,
                marker=dict(colors=["#6366f1", "#0ea5e9", "#14b8a6"]),
                hovertemplate="<b>%{label}</b><br>Số đánh giá: %{value:,}<br>Tỷ lệ: %{percent}<extra></extra>",
                textinfo="label+percent",
                textfont=dict(size=12),
            )
        ]
    )
    fig.update_layout(showlegend=False)
    return apply_saas_theme(fig, height=height)


def chart_platform_bar(plat_rating: pd.Series, height: int = 270) -> go.Figure:
    """Biểu đồ cột so sánh điểm đánh giá giữa các sàn."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=plat_rating.index,
                y=plat_rating.values,
                marker=dict(color="#10b981", line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>Điểm TB: %{y:.2f} / 5★<extra></extra>",
                text=[f"{v:.2f}★" for v in plat_rating.values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(yaxis_title="Điểm số TB (1–5★)", xaxis_title="", yaxis=dict(range=[1, 5.2]))
    return apply_saas_theme(fig, height=height)


def chart_absa_stacked_matrix(counts_df: pd.DataFrame, height: int = 340) -> go.Figure:
    """Ma trận ABSA: Biểu đồ thanh xếp chồng (Horizontal Stacked Bar) Khía cạnh x Sắc thái kèm số lượng và %."""
    fig = go.Figure()
    sentiments = ["Tiêu cực", "Trung tính", "Hỗn hợp", "Tích cực"]
    
    # Tính tổng đánh giá cho mỗi khía cạnh
    row_totals = counts_df.sum(axis=1)

    for sentiment in sentiments:
        if sentiment in counts_df.columns:
            counts = counts_df[sentiment]
            pcts = [(cnt / total * 100) if total > 0 else 0 for cnt, total in zip(counts, row_totals)]
            custom = [[cnt, pct] for cnt, pct in zip(counts, pcts)]
            
            fig.add_trace(
                go.Bar(
                    y=counts_df.index,
                    x=pcts,
                    name=sentiment,
                    orientation="h",
                    marker=dict(color=SENTIMENT_COLORS[sentiment], line=dict(width=0)),
                    customdata=custom,
                    hovertemplate=f"<b>%{{y}}</b> - {sentiment}<br>Số lượng: %{{customdata[0]:,}} nhận xét<br>Tỷ trọng: %{{customdata[1]:.1f}}%<extra></extra>",
                )
            )

    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Tỷ trọng phản hồi (%)", range=[0, 100], ticksuffix="%"),
        yaxis=dict(title="", autorange="reversed"),
    )
    return apply_saas_theme(fig, height=height)


def chart_aspect_frequency(aspect_counts: pd.DataFrame, height: int = 270) -> go.Figure:
    """Biểu đồ tần suất các khía cạnh của một sản phẩm đơn lẻ."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=aspect_counts["Số lượt nhắc"],
                y=aspect_counts["Khía cạnh"],
                orientation="h",
                marker=dict(color="#2563eb", line=dict(width=0)),
                hovertemplate="<b>%{y}</b><br>Số lượt nhắc: %{x:,}<extra></extra>",
                text=aspect_counts["Số lượt nhắc"],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(xaxis_title="Số lần được đề cập", yaxis_title="", yaxis=dict(autorange="reversed"))
    return apply_saas_theme(fig, height=height)
