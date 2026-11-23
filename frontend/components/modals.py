"""Empty States and Notice Components for FreshCheck AI."""

import streamlit as st

from frontend.config import FOOD_SAFETY_DISCLAIMER


def render_empty_state(
    title: str = "No Scan History Found",
    description: str = "Start by uploading an image in the Food Analyzer to receive your first freshness assessment.",
    icon: str = "🥗",
    button_label: str = "Go to Analyzer 🚀",
) -> bool:
    """Render a clean, modern empty state box.

    Returns:
        bool: True if the user clicked the action button.
    """
    st.markdown(
        f"""
        <div class="fc-empty-state">
            <span class="fc-empty-icon">{icon}</span>
            <div class="fc-empty-title">{title}</div>
            <div class="fc-empty-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        return st.button(button_label, use_container_width=True, key="empty_state_btn")


def render_safety_disclaimer() -> None:
    """Render the prominent, legally and academically sound food safety disclaimer."""
    st.markdown(
        f"""
        <div class="fc-disclaimer-card">
            {FOOD_SAFETY_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_backend_offline_alert(backend_url: str) -> None:
    """Render helpful troubleshooting instructions when FastAPI backend is offline."""
    st.markdown(
        f"""
        <div style="background: #FEF2F2; border: 1px solid #FECACA; border-left: 5px solid #EF4444; border-radius: 12px; padding: 1.2rem 1.4rem; margin: 1.2rem 0;">
            <h4 style="color: #991B1B; margin: 0 0 0.5rem 0;">⚠️ FastAPI Backend Server Offline</h4>
            <p style="color: #7F1D1D; font-size: 0.9rem; margin-bottom: 0.8rem;">
                FreshCheck AI cannot connect to Member 3's backend at <code>{backend_url}</code>.
            </p>
            <div style="background: #FFFFFF; border: 1px solid #FCA5A5; border-radius: 8px; padding: 0.75rem 1rem; font-family: monospace; font-size: 0.85rem; color: #1E293B;">
                # Launch Member 3's FastAPI backend in a separate terminal:<br/>
                uvicorn backend.main:app --reload --port 8000
            </div>
            <p style="color: #7F1D1D; font-size: 0.82rem; margin-top: 0.8rem; margin-bottom: 0;">
                💡 <em>Tip: You can toggle "UI Mock Mode" in the left sidebar for isolated interface development and demo presentations.</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
