"""Metric Cards Component for FreshCheck AI."""

from typing import Any, Dict, List
import streamlit as st


def render_metrics_row(history: List[Dict[str, Any]]) -> None:
    """Render 4 responsive summary KPI cards from real scan history.

    Args:
        history: List of scan records from backend GET /history.
    """
    total = len(history)
    fresh_count = sum(1 for item in history if item.get("predicted_class") == "Fresh")
    mod_count = sum(1 for item in history if item.get("predicted_class") == "Moderate")
    spoil_count = sum(1 for item in history if item.get("predicted_class") == "Spoiled")

    fresh_pct = round((fresh_count / total * 100), 1) if total > 0 else 0.0
    mod_pct = round((mod_count / total * 100), 1) if total > 0 else 0.0
    spoil_pct = round((spoil_count / total * 100), 1) if total > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="fc-stat-card fc-card-total">
                <div class="fc-stat-header">
                    <span class="fc-stat-title">Total Scans</span>
                    <span class="fc-stat-icon">📊</span>
                </div>
                <div class="fc-stat-val">{total}</div>
                <div class="fc-stat-sub">Lifetime items evaluated</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="fc-stat-card fc-card-fresh">
                <div class="fc-stat-header">
                    <span class="fc-stat-title">Fresh</span>
                    <span class="fc-stat-icon" style="background: #ECFDF5; color: #10B981;">🟢</span>
                </div>
                <div class="fc-stat-val" style="color: #065F46;">{fresh_count}</div>
                <div class="fc-stat-sub"><strong>{fresh_pct}%</strong> of evaluated inventory</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="fc-stat-card fc-card-moderate">
                <div class="fc-stat-header">
                    <span class="fc-stat-title">Moderate</span>
                    <span class="fc-stat-icon" style="background: #FFFBEB; color: #F59E0B;">🟡</span>
                </div>
                <div class="fc-stat-val" style="color: #92400E;">{mod_count}</div>
                <div class="fc-stat-sub"><strong>{mod_pct}%</strong> inspect & consume soon</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="fc-stat-card fc-card-spoiled">
                <div class="fc-stat-header">
                    <span class="fc-stat-title">Spoiled</span>
                    <span class="fc-stat-icon" style="background: #FEF2F2; color: #EF4444;">🔴</span>
                </div>
                <div class="fc-stat-val" style="color: #991B1B;">{spoil_count}</div>
                <div class="fc-stat-sub"><strong>{spoil_pct}%</strong> critical / discard safely</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
