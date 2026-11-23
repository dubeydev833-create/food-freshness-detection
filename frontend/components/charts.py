"""Plotly Visualizations for FreshCheck AI.

Provides interactive, modern charts matching the green/neutral design system.
"""

from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from frontend.config import CLASS_COLORS, FRESHNESS_CLASSES


def create_confidence_gauge(confidence: float, predicted_class: str) -> go.Figure:
    """Create a semi-circle gauge for model confidence."""
    # Ensure confidence is in range [0, 100]
    conf_pct = round(confidence * 100 if confidence <= 1.0 else confidence, 1)
    bar_color = CLASS_COLORS.get(predicted_class, "#10B981")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=conf_pct,
            number={"suffix": "%", "font": {"size": 36, "color": "#0F172A", "family": "Plus Jakarta Sans"}},
            title={"text": "Model Confidence", "font": {"size": 15, "color": "#64748B", "family": "Plus Jakarta Sans"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#F1F5F9",
                "borderwidth": 1,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 50], "color": "#F8FAFC"},
                    {"range": [50, 75], "color": "#F1F5F9"},
                    {"range": [75, 100], "color": "#E2E8F0"},
                ],
            },
        )
    )

    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_probability_bar_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Create a horizontal bar chart displaying prediction probabilities for all 3 classes."""
    classes = list(FRESHNESS_CLASSES)
    # Extract values, ensure float format
    values = [round(float(probabilities.get(cls, 0.0)) * 100, 1) for cls in classes]
    colors = [CLASS_COLORS[cls] for cls in classes]
    text_labels = [f"<b>{val}%</b>" for val in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=classes,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=1.5)),
            text=text_labels,
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=13, family="Plus Jakarta Sans"),
            hoverinfo="x+y",
        )
    )

    fig.update_layout(
        title=dict(
            text="Class Probability Distribution",
            font=dict(size=14, color="#334155", family="Plus Jakarta Sans"),
        ),
        xaxis=dict(
            range=[0, 105],
            showgrid=True,
            gridcolor="#F1F5F9",
            title=dict(text="Probability (%)", font=dict(size=12, color="#64748B")),
            ticksuffix="%",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=13, color="#0F172A", family="Plus Jakarta Sans"),
        ),
        height=220,
        margin=dict(l=10, r=20, t=35, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_category_distribution_chart(history: List[Dict[str, Any]]) -> go.Figure:
    """Create an interactive donut chart of food categories in scan history."""
    if not history:
        fig = go.Figure()
        fig.update_layout(
            title="Category Distribution",
            annotations=[dict(text="No scan data available", showarrow=False, font=dict(size=13, color="#64748B"))],
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    df = pd.DataFrame(history)
    cat_counts = df["category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]

    color_palette = ["#10B981", "#3B82F6", "#8B5CF6", "#EC4899", "#F59E0B"]

    fig = px.pie(
        cat_counts,
        values="Count",
        names="Category",
        hole=0.55,
        color_discrete_sequence=color_palette,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        marker=dict(line=dict(color="#FFFFFF", width=2)),
    )

    fig.update_layout(
        title=dict(text="Category Distribution", font=dict(size=15, color="#0F172A", family="Plus Jakarta Sans")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=300,
        margin=dict(l=20, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_freshness_donut_chart(history: List[Dict[str, Any]]) -> go.Figure:
    """Create a donut chart showing proportions of Fresh, Moderate, and Spoiled items."""
    if not history:
        fig = go.Figure()
        fig.update_layout(
            title="Freshness Distribution",
            annotations=[dict(text="No scan data available", showarrow=False, font=dict(size=13, color="#64748B"))],
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    df = pd.DataFrame(history)
    # Ensure all classes present
    counts = {cls: 0 for cls in FRESHNESS_CLASSES}
    actual_counts = df["predicted_class"].value_counts().to_dict()
    for cls, cnt in actual_counts.items():
        if cls in counts:
            counts[cls] = cnt

    labels = [cls for cls, cnt in counts.items() if cnt > 0]
    values = [cnt for cls, cnt in counts.items() if cnt > 0]
    colors = [CLASS_COLORS[cls] for cls in labels]

    if not values:
        labels = list(FRESHNESS_CLASSES)
        values = [0, 0, 0]
        colors = [CLASS_COLORS[cls] for cls in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="percent+value",
            hoverinfo="label+value+percent",
        )
    )

    fig.update_layout(
        title=dict(text="Freshness Quality Breakdown", font=dict(size=15, color="#0F172A", family="Plus Jakarta Sans")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=300,
        margin=dict(l=20, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_timeline_chart(history: List[Dict[str, Any]]) -> go.Figure:
    """Create a chronological timeline of scans by category and freshness."""
    if not history:
        fig = go.Figure()
        fig.update_layout(
            title="Scan Activity History",
            annotations=[dict(text="No scans recorded yet", showarrow=False, font=dict(size=13, color="#64748B"))],
            height=260,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    df = pd.DataFrame(history)
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        df["index_order"] = range(1, len(df) + 1)
        x_col = "index_order"
        x_title = "Scan Index"
    else:
        try:
            df["parsed_time"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("parsed_time")
            x_col = "parsed_time"
            x_title = "Scan Timestamp"
        except Exception:
            df["index_order"] = range(1, len(df) + 1)
            x_col = "index_order"
            x_title = "Scan Index"

    # Scatter timeline with markers colored by class
    fig = px.scatter(
        df,
        x=x_col,
        y="category",
        color="predicted_class",
        color_discrete_map=CLASS_COLORS,
        size=[12] * len(df),
        hover_data=["predicted_class", "confidence", "category"],
    )

    fig.update_layout(
        title=dict(text="Scan Activity Over Time", font=dict(size=15, color="#0F172A", family="Plus Jakarta Sans")),
        xaxis=dict(title=x_title, showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(title="Category", showgrid=True, gridcolor="#F1F5F9"),
        height=260,
        margin=dict(l=20, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
