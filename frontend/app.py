"""FreshCheck AI — Modern Streamlit Web Application (Member 4 Frontend).

Full frontend application orchestrating the Home Dashboard, Food Analyzer,
Prediction Results display, Historical Analytics, and FastAPI backend integration.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
import streamlit as st

from frontend.api_client import (
    BackendConnectionError,
    BackendTimeoutError,
    FreshCheckAPIClient,
    MockFreshCheckClient,
)
from frontend.components.navbar import render_connectivity_badge, render_hero_header
from frontend.config import (
    APP_ICON,
    APP_NAME,
    APP_SUBTITLE,
    APP_VERSION,
    DEFAULT_BACKEND_URL,
)
from frontend.views.analyzer import render_analyzer_view
from frontend.views.dashboard import render_dashboard_view
from frontend.views.history import render_history_view
from frontend.views.results import render_results_view

# Configure Streamlit page settings
st.set_page_config(
    page_title=f"{APP_NAME} — Food Freshness Detection",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    """Inject custom design system CSS."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            custom_css = f.read()
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize app-wide session variables."""
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Dashboard"
    if "backend_url" not in st.session_state:
        st.session_state["backend_url"] = DEFAULT_BACKEND_URL
    if "use_mock_mode" not in st.session_state:
        st.session_state["use_mock_mode"] = False
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_image" not in st.session_state:
        st.session_state["last_image"] = None


def get_active_client() -> FreshCheckAPIClient:
    """Instantiate appropriate API client based on user toggle or backend availability."""
    if st.session_state["use_mock_mode"]:
        if "mock_client" not in st.session_state:
            st.session_state["mock_client"] = MockFreshCheckClient()
        return st.session_state["mock_client"]
    else:
        return FreshCheckAPIClient(base_url=st.session_state["backend_url"])


def main() -> None:
    """Main application loop."""
    load_css()
    init_session_state()

    client = get_active_client()

    # Probe backend health
    conn_status: Dict[str, Any] = {}
    if st.session_state["use_mock_mode"]:
        conn_status = client.check_health()
    else:
        try:
            conn_status = client.check_health()
        except BackendConnectionError:
            conn_status = {"status": "offline", "url": client.base_url}
        except BackendTimeoutError:
            conn_status = {"status": "timeout", "url": client.base_url}
        except Exception:
            conn_status = {"status": "error", "url": client.base_url}

    # Sidebar
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.2rem;">
                <span style="font-size: 2.2rem;">🥗</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.3rem; font-weight: 800; color: #0F172A;">{APP_NAME}</h2>
                    <span style="font-size: 0.75rem; color: #64748B; font-weight: 600;">v{APP_VERSION} · College Project</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Connection badge
        render_connectivity_badge(conn_status)

        st.markdown("<hr style='margin: 1.2rem 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

        # Main Navigation
        nav_options = [
            "🏠 Home Dashboard",
            "🔬 Food Analyzer",
            "📊 History & Analytics",
            "⚙️ API Settings",
        ]

        # Determine index based on current_page
        current_page = st.session_state["current_page"]
        page_index_map = {
            "Dashboard": 0,
            "Analyzer": 1,
            "Results": 1,
            "History": 2,
            "Settings": 3,
        }
        active_idx = page_index_map.get(current_page, 0)

        selected_nav = st.radio(
            "Navigation Menu",
            options=nav_options,
            index=active_idx,
            label_visibility="collapsed",
        )

        # Update page state when sidebar changes
        if selected_nav == "🏠 Home Dashboard" and current_page != "Dashboard":
            st.session_state["current_page"] = "Dashboard"
            st.rerun()
        elif selected_nav == "🔬 Food Analyzer" and current_page not in ("Analyzer", "Results"):
            st.session_state["current_page"] = "Analyzer"
            st.rerun()
        elif selected_nav == "📊 History & Analytics" and current_page != "History":
            st.session_state["current_page"] = "History"
            st.rerun()
        elif selected_nav == "⚙️ API Settings" and current_page != "Settings":
            st.session_state["current_page"] = "Settings"
            st.rerun()

        st.markdown("<hr style='margin: 1.2rem 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

        # Backend Connection Settings
        st.markdown("<div style='font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: #64748B; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Backend Configuration</div>", unsafe_allow_html=True)

        new_backend_url = st.text_input(
            "FastAPI Backend URL",
            value=st.session_state["backend_url"],
            help="Configurable URL for Member 3's FastAPI backend service.",
        )
        if new_backend_url != st.session_state["backend_url"]:
            st.session_state["backend_url"] = new_backend_url.strip()
            st.rerun()

        # Offline Mock Mode Toggle (for isolated UI development when backend is offline)
        mock_mode_checked = st.checkbox(
            "Isolated UI Mock Mode",
            value=st.session_state["use_mock_mode"],
            help="Enable temporary mock data mode for UI design preview when Member 3's backend is not running.",
        )
        if mock_mode_checked != st.session_state["use_mock_mode"]:
            st.session_state["use_mock_mode"] = mock_mode_checked
            st.rerun()

        if st.session_state["use_mock_mode"]:
            st.info("ℹ️ Running in isolated UI development mode with test fixtures.")

        # Project Info & Attribution
        st.markdown("<hr style='margin: 1.2rem 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size: 0.78rem; color: #64748B; line-height: 1.5;">
                <strong>FreshCheck AI Project Team</strong><br/>
                • Member 1: Model Training & ML<br/>
                • Member 2: Computer Vision Pipeline<br/>
                • Member 3: FastAPI Backend & DB<br/>
                • <strong>Member 4: Streamlit Frontend & UX</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Top Hero Branding
    render_hero_header()

    # Route to Views
    page = st.session_state["current_page"]

    if page == "Dashboard":
        # Fetch real history for dashboard metrics
        history_data = []
        try:
            history_data = client.get_history()
        except Exception:
            history_data = []

        def go_to_analyzer() -> None:
            st.session_state["current_page"] = "Analyzer"

        render_dashboard_view(history=history_data, on_navigate_analyzer=go_to_analyzer)

    elif page == "Analyzer":
        def on_success(result: Dict[str, Any], img: Image.Image) -> None:
            st.session_state["last_result"] = result
            st.session_state["last_image"] = img
            st.session_state["current_page"] = "Results"

        render_analyzer_view(api_client=client, on_analysis_success=on_success)

    elif page == "Results":
        result = st.session_state.get("last_result")
        img = st.session_state.get("last_image")

        if result is None:
            st.info("No active result found. Please submit an image in the Analyzer.")
            if st.button("Go to Analyzer"):
                st.session_state["current_page"] = "Analyzer"
                st.rerun()
        else:
            def on_new_scan() -> None:
                st.session_state["last_result"] = None
                st.session_state["last_image"] = None
                st.session_state["current_page"] = "Analyzer"

            def on_view_history() -> None:
                st.session_state["current_page"] = "History"

            render_results_view(
                result=result,
                image=img,
                on_new_scan=on_new_scan,
                on_view_history=on_view_history,
            )

    elif page == "History":
        def go_to_analyzer_from_history() -> None:
            st.session_state["current_page"] = "Analyzer"

        render_history_view(api_client=client, on_navigate_analyzer=go_to_analyzer_from_history)

    elif page == "Settings":
        st.markdown("<h3>⚙️ Application & Backend Configuration</h3>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="fc-card">
                <h4>API Connectivity Diagnostics</h4>
                <p><strong>Configured Backend:</strong> <code>{client.base_url}</code></p>
                <p><strong>Connection Status:</strong> <code>{conn_status.get('status', 'unknown')}</code></p>
                <p><strong>Ping Latency:</strong> {conn_status.get('latency_ms', '-')} ms</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="fc-card">
                <h4>Quick Backend Launch Guide</h4>
                <p style="color: #475569;">When running the full stack with Member 3's FastAPI backend:</p>
                <pre style="background: #0F172A; color: #38BDF8; padding: 1rem; border-radius: 8px;">
# Terminal 1: Run Member 3's FastAPI Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Run FreshCheck AI Streamlit Frontend
streamlit run frontend/app.py --server.port 8501
                </pre>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
