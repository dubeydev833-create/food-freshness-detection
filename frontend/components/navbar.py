"""Header and Navigation Components for FreshCheck AI."""

from typing import Any, Dict, Optional
import streamlit as st

from frontend.config import APP_NAME, APP_SUBTITLE


def render_hero_header() -> None:
    """Render top hero branding banner."""
    st.markdown(
        f"""
        <div class="fc-hero">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <h1>🥗 {APP_NAME}</h1>
                    <p>{APP_SUBTITLE}</p>
                </div>
                <div style="text-align: right; font-size: 0.85rem; color: #A7F3D0; font-weight: 500;">
                    <span>🎓 College Mini Project · Four-Member Team</span><br/>
                    <span style="color: #FFFFFF; font-weight: 600;">Member 4 — Frontend & UI/UX</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_connectivity_badge(conn_status: Dict[str, Any]) -> None:
    """Render subtle live backend status pill.

    Args:
        conn_status: Dictionary from api_client.check_health() or session status.
    """
    status = conn_status.get("status", "unknown")
    latency = conn_status.get("latency_ms", 0.0)
    url = conn_status.get("url", "")
    is_mock = conn_status.get("_is_mock", False)

    if status == "healthy":
        html = f"""
        <div class="fc-conn-pill fc-conn-online" title="Connected to {url}">
            <div class="fc-conn-dot fc-dot-online"></div>
            <span>FastAPI Backend Online · {latency}ms</span>
        </div>
        """
    elif is_mock:
        html = """
        <div class="fc-conn-pill" style="background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A;" title="Isolated UI Dev Mode">
            <div class="fc-conn-dot" style="background-color: #F59E0B;"></div>
            <span>⚠️ UI Mock Mode (Backend Offline)</span>
        </div>
        """
    else:
        html = f"""
        <div class="fc-conn-pill fc-conn-offline" title="Unreachable at {url}">
            <div class="fc-conn-dot fc-dot-offline"></div>
            <span>Backend Offline · Not Connected</span>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)
