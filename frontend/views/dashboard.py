"""Phase 3 — Home Dashboard View for FreshCheck AI."""

from typing import Any, Callable, Dict, List
import pandas as pd
import streamlit as st

from frontend.components.metrics_cards import render_metrics_row
from frontend.components.modals import render_empty_state
from frontend.config import CATEGORY_ICONS, CLASS_COLORS, CLASS_ICONS


def render_dashboard_view(
    history: List[Dict[str, Any]],
    on_navigate_analyzer: Callable[[], None],
) -> None:
    """Render the Home Dashboard with real stats, recent scans, and quick actions."""
    # Project Introduction Card
    st.markdown(
        """
        <div class="fc-card" style="margin-top: 0.5rem; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: flex-start; justify-content: space-between; gap: 1.5rem; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 280px;">
                    <h3 style="margin-top: 0; color: #0F172A; font-weight: 700;">Welcome to FreshCheck AI</h3>
                    <p style="color: #475569; font-size: 0.95rem; line-height: 1.6; margin-bottom: 0.8rem;">
                        FreshCheck AI leverages state-of-the-art computer vision to assess food spoilage
                        and classify items into <strong>Fresh</strong>, <strong>Moderate</strong>, and <strong>Spoiled</strong> tiers.
                        Designed for fruits, vegetables, meat, and dairy products to minimize household and retail waste.
                    </p>
                    <div style="display: flex; gap: 0.8rem; flex-wrap: wrap;">
                        <span class="fc-badge" style="background: #F1F5F9; color: #334155;">🍎 Fruits</span>
                        <span class="fc-badge" style="background: #F1F5F9; color: #334155;">🥦 Vegetables</span>
                        <span class="fc-badge" style="background: #F1F5F9; color: #334155;">🥩 Meat</span>
                        <span class="fc-badge" style="background: #F1F5F9; color: #334155;">🧀 Dairy</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick Navigation Callout Button
    col_nav, _ = st.columns([1, 2])
    with col_nav:
        if st.button("🚀 Launch Food Analyzer", use_container_width=True, type="primary"):
            on_navigate_analyzer()
            st.rerun()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Real Metrics Cards (Phase 3 requirement: Use real history data from backend)
    st.markdown("<h4 style='color: #0F172A; font-weight: 700; margin-bottom: 0.8rem;'>Inventory Freshness Overview</h4>", unsafe_allow_html=True)
    render_metrics_row(history)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Recent Prediction History
    st.markdown("<h4 style='color: #0F172A; font-weight: 700; margin-bottom: 0.8rem;'>Recent Scan History</h4>", unsafe_allow_html=True)

    if not history:
        clicked = render_empty_state(
            title="No Scans Recorded Yet",
            description="Your recent scans and freshness classifications will appear here once you analyze your first item.",
            icon="🔬",
            button_label="Analyze Your First Item Now",
        )
        if clicked:
            on_navigate_analyzer()
            st.rerun()
        return

    # Recent items (latest 5)
    recent_items = history[:5]

    for item in recent_items:
        category = item.get("category", "General")
        cat_icon = CATEGORY_ICONS.get(category, "🥗")
        pred_class = item.get("predicted_class", "Unknown")
        confidence = item.get("confidence", 0.0)
        conf_pct = round(confidence * 100 if confidence <= 1.0 else confidence, 1)
        timestamp = item.get("timestamp", "Recent")
        is_mock = item.get("_is_mock", False)

        badge_class = f"fc-badge-{pred_class.lower()}" if pred_class in ("Fresh", "Moderate", "Spoiled") else ""
        class_icon = CLASS_ICONS.get(pred_class, "•")

        mock_tag = "<span style='font-size: 0.75rem; background: #FEF3C7; color: #92400E; padding: 2px 6px; border-radius: 4px; margin-left: 6px;'>TEST DATA</span>" if is_mock else ""

        temp = item.get("temperature_celsius")
        cond = item.get("storage_condition")
        meta_str = []
        if temp is not None:
            meta_str.append(f"{temp}°C")
        if cond:
            meta_str.append(cond)
        meta_display = f" · <em>{' / '.join(meta_str)}</em>" if meta_str else ""

        st.markdown(
            f"""
            <div class="fc-card" style="padding: 0.9rem 1.2rem; margin-bottom: 0.75rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.9rem;">
                    <div style="font-size: 1.8rem; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; background: #F8FAFC; border-radius: 10px; border: 1px solid #E2E8F0;">
                        {cat_icon}
                    </div>
                    <div>
                        <div style="font-weight: 700; color: #0F172A; font-size: 1rem;">
                            {category} {mock_tag}
                        </div>
                        <div style="font-size: 0.8rem; color: #64748B;">
                            {timestamp}{meta_display}
                        </div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="text-align: right;">
                        <span class="fc-badge {badge_class}">
                            {class_icon} {pred_class}
                        </span>
                        <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px; font-weight: 500;">
                            {conf_pct}% confidence
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
