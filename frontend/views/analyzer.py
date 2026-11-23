"""Phase 4 — Food Analyzer View for FreshCheck AI."""

import io
from typing import Any, Callable, Dict, Optional
from PIL import Image
import streamlit as st

from frontend.api_client import (
    BackendAPIError,
    BackendConnectionError,
    BackendTimeoutError,
    FreshCheckAPIClient,
)
from frontend.components.modals import render_backend_offline_alert, render_safety_disclaimer
from frontend.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    CATEGORY_ICONS,
    MAX_IMAGE_SIZE_MB,
    STORAGE_CONDITIONS,
    SUPPORTED_CATEGORIES,
)


def render_analyzer_view(
    api_client: FreshCheckAPIClient,
    on_analysis_success: Callable[[Dict[str, Any], Image.Image], None],
) -> None:
    """Render the Food Analyzer page with image input and optional metadata fields."""
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
            <h2 style="color: #0F172A; font-weight: 800; margin-bottom: 0.2rem;">🔬 Food Freshness Analyzer</h2>
            <p style="color: #64748B; font-size: 0.95rem; margin-top: 0;">
                Capture or upload an image of your food item, provide optional storage telemetry, and receive real-time AI classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input method selection: File Upload vs Camera
    input_method = st.radio(
        "Select Image Input Mode",
        options=["📁 Upload Image File", "📷 Live Camera Capture"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image_bytes: Optional[bytes] = None
    image_filename = "food_scan.jpg"
    pil_image: Optional[Image.Image] = None

    col_img, col_meta = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("<h4 style='color: #0F172A; font-weight: 700; margin-bottom: 0.6rem;'>1. Food Image</h4>", unsafe_allow_html=True)

        if "Upload" in input_method:
            uploaded_file = st.file_uploader(
                "Upload a high-resolution food photo",
                type=ALLOWED_IMAGE_EXTENSIONS,
                help=f"Supported formats: JPG, PNG, WEBP. Max size: {MAX_IMAGE_SIZE_MB}MB.",
            )
            if uploaded_file is not None:
                # Size validation
                file_size_mb = uploaded_file.size / (1024 * 1024)
                if file_size_mb > MAX_IMAGE_SIZE_MB:
                    st.error(f"File size ({file_size_mb:.1f}MB) exceeds the {MAX_IMAGE_SIZE_MB}MB limit.")
                else:
                    image_bytes = uploaded_file.getvalue()
                    image_filename = uploaded_file.name
                    try:
                        pil_image = Image.open(io.BytesIO(image_bytes))
                    except Exception:
                        st.error("Uploaded file could not be read as an image. Please try another file.")
        else:
            camera_file = st.camera_input("Take a photo of the food item")
            if camera_file is not None:
                image_bytes = camera_file.getvalue()
                image_filename = "camera_capture.jpg"
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                except Exception:
                    st.error("Captured frame could not be decoded. Please retry.")

        # Preview container
        if pil_image is not None:
            st.image(pil_image, caption=f"Selected: {image_filename}", use_container_width=True)
            st.success("✅ Image ready for analysis")
        else:
            st.info("💡 Please upload an image or capture via camera to continue.")

    with col_meta:
        st.markdown("<h4 style='color: #0F172A; font-weight: 700; margin-bottom: 0.6rem;'>2. Food Category & Metadata</h4>", unsafe_allow_html=True)

        # 1. Food Category Dropdown (Required)
        category_options = SUPPORTED_CATEGORIES
        selected_category = st.selectbox(
            "Food Category *",
            options=category_options,
            format_func=lambda x: f"{CATEGORY_ICONS.get(x, '🥗')} {x}",
            help="Select the general food type to calibrate freshness thresholds.",
        )

        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: #64748B; letter-spacing: 0.05em; margin-bottom: 0.6rem;'>Optional Storage Conditions</div>", unsafe_allow_html=True)

        # 2. Optional Temperature Input
        specify_temp = st.checkbox("Specify storage / ambient temperature", value=False)
        temperature_val: Optional[float] = None
        if specify_temp:
            temperature_val = st.number_input(
                "Temperature (°C)",
                min_value=-30.0,
                max_value=55.0,
                value=4.0,
                step=0.5,
                format="%.1f",
                help="Observed refrigerator or ambient storage temperature.",
            )

        # 3. Optional Storage Condition
        specify_cond = st.checkbox("Specify storage environment", value=False)
        storage_cond_val: Optional[str] = None
        if specify_cond:
            storage_cond_val = st.selectbox(
                "Storage Condition",
                options=STORAGE_CONDITIONS,
                index=0,
            )

        # 4. Optional Storage Duration
        specify_dur = st.checkbox("Specify storage duration", value=False)
        duration_days: Optional[float] = None
        duration_hours: Optional[float] = None
        if specify_dur:
            dur_col1, dur_col2 = st.columns([2, 1])
            with dur_col1:
                dur_number = st.number_input(
                    "Duration",
                    min_value=0.1,
                    max_value=365.0,
                    value=2.0,
                    step=0.5,
                )
            with dur_col2:
                dur_unit = st.selectbox("Unit", options=["Days", "Hours"])

            if dur_unit == "Days":
                duration_days = float(dur_number)
            else:
                duration_hours = float(dur_number)

    st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

    # Submission Action
    col_submit, col_space = st.columns([1, 1])
    with col_submit:
        analyze_clicked = st.button(
            "🔍 Analyze Food Freshness",
            type="primary",
            use_container_width=True,
            disabled=(image_bytes is None),
        )

    if analyze_clicked:
        if image_bytes is None or pil_image is None:
            st.warning("⚠️ Please provide an image before submitting.")
            return

        with st.spinner("🤖 Communicating with Member 3's FastAPI backend for neural inference..."):
            try:
                # Phase 4 requirement: Send image and metadata to Member 3's POST /predict
                response = api_client.predict(
                    image_bytes=image_bytes,
                    filename=image_filename,
                    category=selected_category,
                    temperature_celsius=temperature_val,
                    storage_condition=storage_cond_val,
                    storage_duration_days=duration_days,
                    storage_duration_hours=duration_hours,
                )

                st.session_state["last_image"] = pil_image
                st.session_state["last_filename"] = image_filename
                on_analysis_success(response, pil_image)
                st.rerun()

            except BackendConnectionError as exc:
                render_backend_offline_alert(api_client.base_url)
                st.error(f"Connection Failed: {exc}")
            except BackendTimeoutError as exc:
                st.error(f"⏳ Request Timed Out: {exc}. The backend server took too long to process the image.")
            except BackendAPIError as exc:
                st.error(f"❌ Backend Error (HTTP {exc.status_code}): {exc.detail}")
            except Exception as exc:
                st.error(f"⚠️ Unexpected client error: {exc}")

    # Explicit Safety Disclaimer
    render_safety_disclaimer()
