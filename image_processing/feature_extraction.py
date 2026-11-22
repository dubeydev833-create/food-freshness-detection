"""Visual feature extraction and statistical image analysis utilities for FreshCheck AI.

Provides analytical utilities for color distribution, brightness, contrast,
and texture analysis using OpenCV and NumPy.

IMPORTANT SCIENTIFIC NOTICE & DISCLAIMER:
    The statistical metrics and heuristics implemented in this module (such as
    color distribution, luminance levels, and edge texture properties) provide
    descriptive visual indicators of an image's optical properties.
    They DO NOT independently prove or diagnose food freshness, microbial safety,
    or spoilage. Food browning, color variations, and surface textural shifts can
    stem from cultivar variation, safe enzymatic oxidation (e.g. sliced apples),
    caramelization, cooking, or lighting conditions.

    These visual features are kept strictly separate from deep-learning model
    inferences and should only be used as complementary telemetry or debugging
    aids unless an empirical, validated multi-modal model is explicitly trained
    on them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from image_processing.exceptions import FreshCheckError


class FeatureExtractionError(FreshCheckError):
    """Raised when an error occurs during visual feature extraction."""
    pass


def analyze_color_distribution(
    rgb_image: np.ndarray,
    num_dominant_colors: int = 3,
) -> Dict[str, Any]:
    """Analyze color statistics, channel distributions, and dominant colors.

    Computes channel-wise statistics across RGB and HSV representations,
    along with k-means clustered dominant color centers.

    Args:
        rgb_image: Input RGB image array of shape (H, W, 3) and uint8 dtype.
        num_dominant_colors: Number of dominant color centroids to extract.

    Returns:
        Dict[str, Any]: Color distribution metrics including channel means,
            standard deviations, HSV values, and dominant color clusters.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise FeatureExtractionError(
            f"Expected 3-channel RGB image, got shape {rgb_image.shape}"
        )

    # RGB channel statistics
    r_chan = rgb_image[:, :, 0].astype(np.float64)
    g_chan = rgb_image[:, :, 1].astype(np.float64)
    b_chan = rgb_image[:, :, 2].astype(np.float64)

    # Convert to HSV for perceptual color evaluation
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    h_chan = hsv_image[:, :, 0].astype(np.float64)  # 0 - 179
    s_chan = hsv_image[:, :, 1].astype(np.float64)  # 0 - 255
    v_chan = hsv_image[:, :, 2].astype(np.float64)  # 0 - 255

    # Colorfulness metric (Hasler and Süsstrunk, 2003)
    rg = np.abs(r_chan - g_chan)
    yb = np.abs(0.5 * (r_chan + g_chan) - b_chan)
    std_rg, mean_rg = float(np.std(rg)), float(np.mean(rg))
    std_yb, mean_yb = float(np.std(yb)), float(np.mean(yb))
    colorfulness = float(
        np.sqrt(std_rg**2 + std_yb**2) + (0.3 * np.sqrt(mean_rg**2 + mean_yb**2))
    )

    # Dominant colors via k-means clustering on a subsampled grid for performance
    pixels = rgb_image.reshape(-1, 3).astype(np.float32)
    # Downsample if image is large for real-time responsiveness
    sample_size = min(10000, pixels.shape[0])
    if pixels.shape[0] > sample_size:
        indices = np.random.choice(pixels.shape[0], sample_size, replace=False)
        sub_pixels = pixels[indices]
    else:
        sub_pixels = pixels

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = max(1, min(num_dominant_colors, 8))
    _, labels, centers = cv2.kmeans(
        sub_pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
    )

    # Calculate proportion of each cluster
    _, counts = np.unique(labels, return_counts=True)
    total_samples = float(len(labels))
    proportions = (counts / total_samples).tolist()

    dominant_colors: List[Dict[str, Any]] = []
    for center, prop in sorted(zip(centers, proportions), key=lambda x: x[1], reverse=True):
        r, g, b = [int(round(c)) for c in center]
        dominant_colors.append({
            "rgb": [r, g, b],
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "proportion": round(prop, 4),
        })

    return {
        "channel_means": {
            "red": round(float(np.mean(r_chan)), 2),
            "green": round(float(np.mean(g_chan)), 2),
            "blue": round(float(np.mean(b_chan)), 2),
        },
        "channel_std": {
            "red": round(float(np.std(r_chan)), 2),
            "green": round(float(np.std(g_chan)), 2),
            "blue": round(float(np.std(b_chan)), 2),
        },
        "hsv_means": {
            "hue_degrees": round(float(np.mean(h_chan)) * 2.0, 2),  # Scale 0-179 to 0-358
            "saturation_percent": round((float(np.mean(s_chan)) / 255.0) * 100, 2),
            "value_percent": round((float(np.mean(v_chan)) / 255.0) * 100, 2),
        },
        "colorfulness_score": round(colorfulness, 2),
        "dominant_colors": dominant_colors,
    }


def analyze_brightness(
    rgb_image: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate image brightness and optical exposure condition.

    Calculates photometric luminance using the ITU-R BT.601 standard:
    Y = 0.299*R + 0.587*G + 0.114*B, along with exposure flags.

    Args:
        rgb_image: Input RGB image array of shape (H, W, 3).

    Returns:
        Dict[str, Any]: Luminance statistics, exposure assessment,
            and clipping percentages.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise FeatureExtractionError(
            f"Expected 3-channel RGB image, got shape {rgb_image.shape}"
        )

    # Convert to grayscale luminance using standard photometric weights
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float64)

    mean_luminance = float(np.mean(gray))
    std_luminance = float(np.std(gray))
    min_luminance = float(np.min(gray))
    max_luminance = float(np.max(gray))

    # Evaluate clipping / extreme exposure
    total_pixels = float(gray.size)
    underexposed_pct = float(np.sum(gray < 15)) / total_pixels * 100.0
    overexposed_pct = float(np.sum(gray > 240)) / total_pixels * 100.0

    # Categorize overall lighting quality
    if mean_luminance < 40.0:
        exposure_status = "underexposed"
    elif mean_luminance > 215.0:
        exposure_status = "overexposed"
    else:
        exposure_status = "optimal"

    return {
        "mean_luminance": round(mean_luminance, 2),
        "std_luminance": round(std_luminance, 2),
        "min_luminance": round(min_luminance, 2),
        "max_luminance": round(max_luminance, 2),
        "underexposed_pixels_pct": round(underexposed_pct, 2),
        "overexposed_pixels_pct": round(overexposed_pct, 2),
        "exposure_assessment": exposure_status,
    }


def analyze_contrast(
    rgb_image: np.ndarray,
) -> Dict[str, Any]:
    """Compute visual contrast using Root Mean Square (RMS) and dynamic range metrics.

    RMS contrast is defined as the standard deviation of normalized pixel intensities,
    which provides a reliable metric of image contrast invariant to mean luminance.

    Args:
        rgb_image: Input RGB image array of shape (H, W, 3).

    Returns:
        Dict[str, Any]: RMS contrast, Michelson contrast, and dynamic range.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise FeatureExtractionError(
            f"Expected 3-channel RGB image, got shape {rgb_image.shape}"
        )

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float64)

    # RMS Contrast: Standard deviation of normalized pixel intensities [0, 1]
    norm_gray = gray / 255.0
    rms_contrast = float(np.std(norm_gray))

    # Michelson Contrast: (I_max - I_min) / (I_max + I_min)
    i_min = float(np.min(gray))
    i_max = float(np.max(gray))
    if (i_max + i_min) > 0:
        michelson_contrast = (i_max - i_min) / (i_max + i_min)
    else:
        michelson_contrast = 0.0

    # Dynamic Range in decibels
    dynamic_range_val = i_max - i_min

    # Assessment
    if rms_contrast < 0.10:
        contrast_assessment = "low"
    elif rms_contrast > 0.35:
        contrast_assessment = "high"
    else:
        contrast_assessment = "medium"

    return {
        "rms_contrast": round(rms_contrast, 4),
        "michelson_contrast": round(float(michelson_contrast), 4),
        "intensity_range": round(dynamic_range_val, 2),
        "contrast_assessment": contrast_assessment,
    }


def analyze_texture(
    rgb_image: np.ndarray,
) -> Dict[str, Any]:
    """Perform texture analysis using OpenCV spatial filter methods.

    Extracts:
        - Laplacian Variance: High-frequency spatial detail and focus measure.
        - Sobel Gradient Energy: Edge transitions and surface roughness.
        - Canny Edge Density: Fraction of strong structural edge pixels.
        - Local Uniformity / Homogeneity: Spatial variation proxy.

    Args:
        rgb_image: Input RGB image array of shape (H, W, 3).

    Returns:
        Dict[str, Any]: Texture and spatial frequency descriptors.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise FeatureExtractionError(
            f"Expected 3-channel RGB image, got shape {rgb_image.shape}"
        )

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # 1. Laplacian variance for sharpness/roughness
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = float(laplacian.var())

    # 2. Sobel edge magnitude (horizontal and vertical gradients)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    mean_gradient_magnitude = float(np.mean(gradient_mag))
    std_gradient_magnitude = float(np.std(gradient_mag))

    # 3. Canny edge density (ratio of edge pixels)
    # Thresholds: Otsu's thresholding adaptive estimation
    high_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    edge_density = float(np.sum(edges > 0)) / float(edges.size)

    # 4. Local texture roughness assessment
    if laplacian_var < 50.0:
        sharpness_assessment = "soft/smooth"
    elif laplacian_var > 500.0:
        sharpness_assessment = "coarse/textured"
    else:
        sharpness_assessment = "moderately_textured"

    return {
        "laplacian_variance": round(laplacian_var, 2),
        "mean_gradient_magnitude": round(mean_gradient_magnitude, 2),
        "std_gradient_magnitude": round(std_gradient_magnitude, 2),
        "canny_edge_density": round(edge_density, 4),
        "texture_profile": sharpness_assessment,
    }


def extract_visual_features(
    rgb_image: np.ndarray,
) -> Dict[str, Any]:
    """Extract a complete suite of visual and statistical image features.

    Combines color distribution, brightness, contrast, and texture profiles.
    These features represent observational metadata and are strictly decoupled
    from deep-learning model freshness classification.

    Args:
        rgb_image: Decoded RGB image array of shape (H, W, 3).

    Returns:
        Dict[str, Any]: Consolidated feature dictionary containing color,
            brightness, contrast, texture sections, and scientific notice.
    """
    return {
        "color_distribution": analyze_color_distribution(rgb_image),
        "brightness": analyze_brightness(rgb_image),
        "contrast": analyze_contrast(rgb_image),
        "texture": analyze_texture(rgb_image),
        "analysis_disclaimer": (
            "Visual features are empirical optical descriptors and do not "
            "independently confirm food freshness or bacterial safety."
        ),
    }
