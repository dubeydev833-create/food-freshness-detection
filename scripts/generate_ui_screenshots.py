"""Generate visual preview mockup images for FreshCheck AI UI documentation."""

import os
from PIL import Image, ImageDraw, ImageFont


def create_mockup(filename: str, title: str, subtitle: str, content_type: str):
    os.makedirs("docs/screenshots", exist_ok=True)
    width, height = 1200, 750
    img = Image.new("RGB", (width, height), color="#F8FAFC")
    draw = ImageDraw.Draw(img)

    # Top Hero Banner (Green gradient)
    draw.rounded_rectangle([30, 25, width - 30, 140], radius=16, fill="#065F46")
    draw.text((60, 45), f"🥗 FreshCheck AI — {title}", fill="#FFFFFF")
    draw.text((60, 85), subtitle, fill="#D1FAE5")
    draw.text((width - 320, 50), "College Mini Project · Member 4", fill="#A7F3D0")
    draw.text((width - 320, 85), "FastAPI Backend Online · 12ms", fill="#FFFFFF")

    if content_type == "dashboard":
        # 4 KPI Metric Cards
        cards = [
            ("TOTAL SCANS", "48", "Lifetime evaluated", "#3B82F6"),
            ("FRESH ITEMS", "32 (66.7%)", "Safe to consume", "#10B981"),
            ("MODERATE ITEMS", "11 (22.9%)", "Inspect & use soon", "#F59E0B"),
            ("SPOILED ITEMS", "5 (10.4%)", "Critical / discard", "#EF4444"),
        ]
        card_w = (width - 60 - 3 * 20) // 4
        for i, (ctitle, cval, csub, color) in enumerate(cards):
            x0 = 30 + i * (card_w + 20)
            x1 = x0 + card_w
            draw.rounded_rectangle([x0, 165, x1, 280], radius=12, fill="#FFFFFF", outline="#E2E8F0", width=2)
            draw.line([(x0, 165), (x1, 165)], fill=color, width=5)
            draw.text((x0 + 20, 185), ctitle, fill="#64748B")
            draw.text((x0 + 20, 215), cval, fill="#0F172A")
            draw.text((x0 + 20, 250), csub, fill="#64748B")

        # Recent Scans Table Preview
        draw.rounded_rectangle([30, 310, width - 30, 710], radius=14, fill="#FFFFFF", outline="#E2E8F0", width=2)
        draw.text((55, 335), "Recent Prediction History", fill="#0F172A")
        
        sample_rows = [
            ("🍎 Fruits (Apple)", "2026-11-23 11:20", "🟢 FRESH", "94.2% Confidence", "#ECFDF5", "#065F46"),
            ("🥦 Vegetables (Broccoli)", "2026-11-23 10:45", "🟡 MODERATE", "81.5% Confidence", "#FFFBEB", "#92400E"),
            ("🥩 Meat (Beef Mince)", "2026-11-23 09:12", "🔴 SPOILED", "96.8% Confidence", "#FEF2F2", "#991B1B"),
            ("🧀 Dairy (Whole Milk)", "2026-11-23 08:30", "🟢 FRESH", "91.0% Confidence", "#ECFDF5", "#065F46"),
        ]
        for idx, (food, ts, status, conf, bg, fg) in enumerate(sample_rows):
            y = 385 + idx * 75
            draw.rounded_rectangle([50, y, width - 50, y + 60], radius=8, fill="#F8FAFC", outline="#E2E8F0", width=1)
            draw.text((70, y + 20), food, fill="#0F172A")
            draw.text((400, y + 20), ts, fill="#64748B")
            draw.rounded_rectangle([720, y + 12, 850, y + 48], radius=15, fill=bg)
            draw.text((740, y + 20), status, fill=fg)
            draw.text((920, y + 20), conf, fill="#334155")

    elif content_type == "analyzer":
        # Two columns: Image input vs Metadata
        mid_x = width // 2 - 15
        draw.rounded_rectangle([30, 165, mid_x, 620], radius=14, fill="#FFFFFF", outline="#E2E8F0", width=2)
        draw.text((55, 190), "1. Image Upload & Live Preview", fill="#0F172A")
        draw.rounded_rectangle([55, 230, mid_x - 25, 590], radius=10, fill="#F1F5F9", outline="#CBD5E1", width=2)
        draw.text((180, 400), "📷 [Drag & drop high-res food photo or capture frame]", fill="#64748B")

        draw.rounded_rectangle([mid_x + 30, 165, width - 30, 620], radius=14, fill="#FFFFFF", outline="#E2E8F0", width=2)
        draw.text((mid_x + 55, 190), "2. Food Category & Telemetry", fill="#0F172A")
        draw.text((mid_x + 55, 240), "Food Category: 🍎 Fruits (Selected)", fill="#334155")
        draw.text((mid_x + 55, 290), "Storage Temperature: 4.0 °C (Refrigerated)", fill="#334155")
        draw.text((mid_x + 55, 340), "Storage Environment: Cool & Dry", fill="#334155")
        draw.text((mid_x + 55, 390), "Storage Duration: 2.0 Days", fill="#334155")

        # Action Button
        draw.rounded_rectangle([mid_x + 55, 460, width - 55, 520], radius=8, fill="#10B981")
        draw.text((mid_x + 180, 480), "🔍 Analyze Food Freshness (POST /predict)", fill="#FFFFFF")

        # Disclaimer
        draw.rounded_rectangle([30, 640, width - 30, 720], radius=10, fill="#FFFBEB", outline="#FDE68A", width=1)
        draw.text((50, 665), "⚠️ Important Food Safety Disclaimer: Optical models cannot detect bacterial pathogens. Inspect smell & texture.", fill="#92400E")

    elif content_type == "results":
        # Results Banner
        draw.rounded_rectangle([30, 165, width - 30, 275], radius=14, fill="#ECFDF5", outline="#A7F3D0", width=2)
        draw.text((60, 185), "NEURAL FRESHNESS ASSESSMENT", fill="#065F46")
        draw.text((60, 215), "🟢 FRESH · 94.2% Classification Confidence", fill="#065F46")
        draw.text((width - 350, 210), "Confidence: 94.2%", fill="#10B981")

        # Split: Telemetry vs Plotly Visualizations
        mid_x = width // 2 - 15
        draw.rounded_rectangle([30, 295, mid_x, 560], radius=14, fill="#FFFFFF", outline="#E2E8F0", width=2)
        draw.text((55, 320), "Sample Image & Storage Telemetry", fill="#0F172A")
        draw.text((55, 360), "• Category: 🍎 Fruits", fill="#334155")
        draw.text((55, 400), "• Temperature: 4.0 °C", fill="#334155")
        draw.text((55, 440), "• Storage Condition: Refrigerated", fill="#334155")
        draw.text((55, 480), "• Storage Duration: 2.0 Days", fill="#334155")

        draw.rounded_rectangle([mid_x + 30, 295, width - 30, 560], radius=14, fill="#FFFFFF", outline="#E2E8F0", width=2)
        draw.text((mid_x + 55, 320), "Plotly Class Probabilities Distribution", fill="#0F172A")
        draw.rounded_rectangle([mid_x + 55, 360, mid_x + 55 + 450, 395], radius=6, fill="#10B981")
        draw.text((mid_x + 70, 370), "Fresh: 94.2%", fill="#FFFFFF")
        draw.rounded_rectangle([mid_x + 55, 415, mid_x + 55 + 60, 450], radius=6, fill="#F59E0B")
        draw.text((mid_x + 70, 425), "Moderate: 4.5%", fill="#FFFFFF")
        draw.rounded_rectangle([mid_x + 55, 470, mid_x + 55 + 25, 505], radius=6, fill="#EF4444")
        draw.text((mid_x + 70, 480), "Spoiled: 1.3%", fill="#FFFFFF")

        # Recommendations Box
        draw.rounded_rectangle([30, 580, width - 30, 720], radius=12, fill="#F8FAFC", outline="#E2E8F0", width=2)
        draw.text((55, 600), "Actionable Storage Guidelines:", fill="#0F172A")
        draw.text((55, 630), "• Store in crisper drawer at 2°C to 4°C to slow natural enzymatic respiration.", fill="#334155")
        draw.text((55, 660), "• Expected optimal shelf life: 5 to 7 days under maintained cold chain.", fill="#334155")
        draw.text((55, 690), "⚠️ Disclaimer: Confidence represents visual model certainty, not a bacteriological safety guarantee.", fill="#64748B")

    img.save(f"docs/screenshots/{filename}")
    print(f"Generated docs/screenshots/{filename}")


if __name__ == "__main__":
    create_mockup("dashboard_view.png", "Home Dashboard", "Live Inventory Freshness Statistics & Recent Scans", "dashboard")
    create_mockup("analyzer_view.png", "Food Analyzer", "Multi-Modal Image Capture & Storage Metadata Inputs", "analyzer")
    create_mockup("results_view.png", "Analysis Results", "Neural Spoilage Tiers, Plotly Distribution & Storage Guidelines", "results")
