"""Phase 6 — History and Analytics View for FreshCheck AI."""

from typing import Any, Callable, Dict, List, Optional
import pandas as pd
import streamlit as st

from frontend.api_client import BackendAPIError, BackendConnectionError, BackendTimeoutError, FreshCheckAPIClient
from frontend.components.charts import (
    create_category_distribution_chart,
    create_freshness_donut_chart,
    create_timeline_chart,
)
from frontend.components.metrics_cards import render_metrics_row
from frontend.components.modals import render_empty_state
from frontend.config import CATEGORY_ICONS, CLASS_ICONS, FRESHNESS_CLASSES, SUPPORTED_CATEGORIES


def render_history_view(
    api_client: FreshCheckAPIClient,
    on_navigate_analyzer: Callable[[], None],
) -> None:
    """Render the Historical Scans & Analytics dashboard."""
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
            <h2 style="color: #0F172A; font-weight: 800; margin-bottom: 0.2rem;">📊 Scan History & Analytics</h2>
            <p style="color: #64748B; font-size: 0.95rem; margin-top: 0;">
                Comprehensive record of historical food evaluations, category breakdowns, and spoilage trends retrieved from the backend database.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Fetch fresh history from backend
    try:
        with st.spinner("Fetching historical scans from backend..."):
            history = api_client.get_history()
    except BackendConnectionError:
        st.error(f"⚠️ Could not connect to backend at {api_client.base_url} to fetch history.")
        history = []
    except Exception as exc:
        st.error(f"Failed to load history: {exc}")
        history = []

    if not history:
        clicked = render_empty_state(
            title="No Previous Scans Found",
            description="The history database is currently empty. Run your first analysis in the analyzer to build historical trends.",
            icon="📭",
            button_label="Start Food Analysis Now 🚀",
        )
        if clicked:
            on_navigate_analyzer()
            st.rerun()
        return

    # Metrics Row
    render_metrics_row(history)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Analytics Visualizations Row
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        cat_fig = create_category_distribution_chart(history)
        st.plotly_chart(cat_fig, use_container_width=True, config={"displayModeBar": False})

    with col_chart2:
        fresh_fig = create_freshness_donut_chart(history)
        st.plotly_chart(fresh_fig, use_container_width=True, config={"displayModeBar": False})

    # Timeline Chart
    timeline_fig = create_timeline_chart(history)
    st.plotly_chart(timeline_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

    # Scan Records Explorer
    st.markdown("<h4 style='color: #0F172A; font-weight: 700; margin-bottom: 0.8rem;'>Scan Log Explorer</h4>", unsafe_allow_html=True)

    # Filters
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        filter_cat = st.selectbox(
            "Filter Category",
            options=["All Categories"] + list(SUPPORTED_CATEGORIES),
        )
    with col_f2:
        filter_class = st.selectbox(
            "Filter Freshness",
            options=["All Classes"] + list(FRESHNESS_CLASSES),
        )
    with col_f3:
        search_query = st.text_input("Search (ID or condition)", placeholder="e.g. Refrigerated, scan_001...")

    # Filter logic
    filtered = list(history)
    if filter_cat != "All Categories":
        filtered = [item for item in filtered if item.get("category") == filter_cat]
    if filter_class != "All Classes":
        filtered = [item for item in filtered if item.get("predicted_class") == filter_class]
    if search_query.strip():
        q = search_query.strip().lower()
        filtered = [
            item for item in filtered
            if q in str(item.get("id", "")).lower()
            or q in str(item.get("storage_condition", "")).lower()
            or q in str(item.get("category", "")).lower()
        ]

    # Convert to presentation DataFrame
    table_rows = []
    for item in filtered:
        cat = item.get("category", "General")
        p_class = item.get("predicted_class", "Unknown")
        conf = item.get("confidence", 0.0)
        conf_str = f"{conf * 100:.1f}%" if conf <= 1.0 else f"{conf:.1f}%"

        temp = item.get("temperature_celsius")
        temp_str = f"{temp}°C" if temp is not None else "-"
        cond = item.get("storage_condition") or "-"

        dur = item.get("storage_duration_days")
        dur_str = f"{dur} d" if dur is not None else "-"

        table_rows.append({
            "Scan ID": item.get("id", "N/A"),
            "Timestamp": item.get("timestamp", "-"),
            "Category": f"{CATEGORY_ICONS.get(cat, '🥗')} {cat}",
            "Class": f"{CLASS_ICONS.get(p_class, '•')} {p_class}",
            "Confidence": conf_str,
            "Temp": temp_str,
            "Storage Condition": cond,
            "Duration": dur_str,
        })

    if table_rows:
        df_display = pd.DataFrame(table_rows)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(table_rows)} of {len(history)} total records.")
    else:
        st.info("No records matched the selected filter criteria.")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Clear History Option (Phase 6 requirement)
    with st.expander("⚙️ Manage Scan Database"):
        st.markdown("<p style='color: #64748B; font-size: 0.9rem;'>Permanently clear scan history from backend records.</p>", unsafe_allow_html=True)
        confirm_clear = st.checkbox("Yes, I want to clear all history records from the backend.", key="confirm_clear_chk")
        if st.button("🗑️ Clear History Database", type="secondary", disabled=not confirm_clear):
            try:
                res = api_client.clear_history()
                st.success("✅ History cleared successfully!")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to clear history: {exc}")
