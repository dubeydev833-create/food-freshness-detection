"""Phase 5 — Results View for FreshCheck AI."""

from typing import Any, Callable, Dict, Optional
from PIL import Image
import streamlit as st

from frontend.components.charts import create_confidence_gauge, create_probability_bar_chart
from frontend.components.modals import render_safety_disclaimer
from frontend.components.recommendations import get_storage_recommendations
from frontend.config import CATEGORY_ICONS, CLASS_BADGE_TEXT_COLORS, CLASS_BG_COLORS, CLASS_COLORS, CLASS_ICONS


def render_results_view(
    result: Dict[str, Any],
    image: Optional[Image.Image],
    on_new_scan: Callable[[], None],
    on_view_history: Callable[[], None],
) -> None:
    """Render the detailed prediction results page for a completed scan."""
    predicted_class = result.get("predicted_class", "Fresh")
    confidence = float(result.get("confidence", 0.0))
    conf_pct = round(confidence * 100 if confidence <= 1.0 else confidence, 1)
    probabilities = result.get("probabilities", {predicted_class: confidence})
    category = result.get("category", "Food Item")
    metadata = result.get("metadata", {})
    is_mock = result.get("_is_mock", False)

    # Class-based styling
    color_accent = CLASS_COLORS.get(predicted_class, "#10B981")
    icon = CLASS_ICONS.get(predicted_class, "🟢")
    cat_icon = CATEGORY_ICONS.get(category, "🥗")

    banner_class = f"fc-result-{predicted_class.lower()}"

    # Top Results Hero Banner
    mock_banner = ""
    if is_mock:
        mock_banner = """
        <div style="background: #FEF3C7; color: #92400E; padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 1rem; font-weight: 600; font-size: 0.85rem; border: 1px solid #FDE68A;">
            ⚠️ TEST DATA MODE: This result was generated in isolated UI mode because Member 3's backend was offline.
        </div>
        """

    st.markdown(
        f"""
        {mock_banner}
        <div class="fc-result-banner {banner_class}">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700; color: #475569;">
                        Neural Freshness Assessment
                    </span>
                    <h2 style="margin: 0.2rem 0; color: #0F172A; font-weight: 800; font-size: 2rem;">
                        {icon} {predicted_class}
                    </h2>
                    <div style="font-size: 1rem; color: #334155;">
                        Food Category: <strong>{cat_icon} {category}</strong> · Confidence: <strong>{conf_pct}%</strong>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2.5rem; font-weight: 800; color: {color_accent};">
                        {conf_pct}%
                    </div>
                    <div style="font-size: 0.85rem; color: #64748B;">Classification Confidence</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Action Toolbar
    col_act1, col_act2, _ = st.columns([1, 1, 2])
    with col_act1:
        if st.button("🔄 Scan Another Item", type="primary", use_container_width=True):
            on_new_scan()
            st.rerun()
    with col_act2:
        if st.button("📊 View Scan History", use_container_width=True):
            on_view_history()
            st.rerun()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Main Analysis Split: Image & Telemetry vs Plotly Visualizations
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<h4 style='color: #0F172A; font-weight: 700; margin-bottom: 0.8rem;'>Analyzed Image & Telemetry</h4>", unsafe_allow_html=True)

        if image is not None:
            st.image(image, caption=f"Analyzed {category} Sample", use_container_width=True)

        # Metadata Card
        st.markdown(
            """
            <div class="fc-card" style="margin-top: 1rem; padding: 1.1rem 1.3rem;">
                <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; color: #64748B; letter-spacing: 0.05em; margin-bottom: 0.6rem;">
                    Storage Telemetry
                </div>
            """,
            unsafe_allow_html=True,
        )

        temp_c = metadata.get("temperature_celsius") if isinstance(metadata, dict) else None
        cond = metadata.get("storage_condition") if isinstance(metadata, dict) else None
        dur_days = metadata.get("storage_duration_days") if isinstance(metadata, dict) else None
        dur_hours = metadata.get("storage_duration_hours") if isinstance(metadata, dict) else None

        has_meta = any(v is not None for v in (temp_c, cond, dur_days, dur_hours))

        if has_meta:
            if temp_c is not None:
                st.markdown(f"🌡️ **Storage Temperature**: {temp_c}°C")
            if cond:
                st.markdown(f"❄️ **Storage Condition**: {cond}")
            if dur_days is not None:
                st.markdown(f"⏱️ **Storage Duration**: {dur_days} days")
            elif dur_hours is not None:
                st.markdown(f"⏱️ **Storage Duration**: {dur_hours} hours")
        else:
            st.markdown("<span style='color: #94A3B8; font-size: 0.9rem;'>No optional environmental telemetry provided for this scan.</span>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<h4 style='color: #0F172A; font-weight: 700; margin-bottom: 0.8rem;'>Class Probabilities & Confidence</h4>", unsafe_allow_html=True)

        # Plotly Horizontal Bar for Class Probabilities
        bar_fig = create_probability_bar_chart(probabilities)
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

        # Confidence Gauge
        gauge_fig = create_confidence_gauge(confidence, predicted_class)
        st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

    # Storage and Action Recommendations
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #0F172A; font-weight: 700; margin-bottom: 0.8rem;'>Actionable Storage & Safety Guidelines</h4>", unsafe_allow_html=True)

    rec_data = get_storage_recommendations(
        category=category,
        predicted_class=predicted_class,
        temperature_celsius=temp_c,
        storage_condition=cond,
    )

    if rec_data.get("temperature_alert"):
        st.warning(rec_data["temperature_alert"])

    items_html = "".join([f"<li class='fc-rec-item'><span>•</span><span>{tip}</span></li>" for tip in rec_data["action_items"]])

    st.markdown(
        f"""
        <div class="fc-recommendation-box">
            <div style="font-weight: 700; color: #0F172A; font-size: 1rem; margin-bottom: 0.6rem;">
                Recommended Actions for {category} ({predicted_class}):
            </div>
            <ul style="list-style: none; padding-left: 0; margin: 0;">
                {items_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Explicit Safety Disclaimer (Phase 5 requirement)
    render_safety_disclaimer()
