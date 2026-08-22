import base64
import io
import os
import requests
import streamlit as st
from PIL import Image, ImageDraw

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="VisionInspect AD",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling
st.markdown("""
    <style>
    /* Dark Theme Accent Refinements */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Header Container Styling */
    .main-header {
        background: linear-gradient(90deg, #1f2937 0%, #111827 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        border: 1px solid #374151;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #f9fafb;
    }
    .main-header p {
        margin: 0.25rem 0 0 0;
        color: #9ca3af;
        font-size: 0.95rem;
    }

    /* Modernized Status Badges */
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .status-defect {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #ef4444;
    }
    .status-ok {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid #22c55e;
    }

    /* Custom Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #f3f4f6;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9ca3af !important;
    }

    /* Drag and Drop Box Customization */
    [data-testid="stFileUploadDropzone"] {
        border-radius: 10px;
        border: 2px dashed #4b5563;
        background-color: #1f2937;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to convert local image to base64
def get_image_as_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Read your local PNG icon
icon_base64 = get_image_as_base64("./assets/icon.png")

st.markdown(f"""
    <div class="main-header">
        <h1 style="display: flex; align-items: center; gap: 18px;">
            <img src="data:image/png;base64,{icon_base64}" style="width: 38px; height: 38px; object-fit: contain;">
            VisionInspect AD
        </h1>
        <p>A unified industrial defect inspection solution — Anomalib and YOLO26</p>
    </div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg", "bmp"])

if uploaded:
    image_bytes = uploaded.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with st.spinner("Analyzing image..."):
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded.name, image_bytes, uploaded.type)},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            st.stop()

    if result.get("recognized") is False:
        st.warning(
            f"⚠️ **Unrecognized Image**: Router classified input as "
            f"**{result['router_raw_class']}** at **{result['router_confidence']:.1%}** "
            f"confidence (below decision threshold)."
        )
        st.stop()

    def build_predicted_output(image, result):
        if result["source_model"] == "anomalib" and result.get("heatmap_overlay_base64"):
            return Image.open(
                io.BytesIO(base64.b64decode(result["heatmap_overlay_base64"]))
            ).convert("RGB")

        predicted = image.copy()
        draw = ImageDraw.Draw(predicted)

        for region in result.get("regions", []):
            x1, y1, x2, y2 = region["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            label = region.get("label") or "defect"
            draw.text((x1, max(0, y1 - 12)), f"{label} {region['score']:.2f}", fill="red")

            if region.get("mask"):
                draw.polygon([tuple(point) for point in region["mask"]], outline="yellow", width=2)

        return predicted

    predicted_output = build_predicted_output(image, result)

    # Main Dashboard Metrics Card
    with st.container():
        st.subheader("Inspection Summary")
        
        # Display Defect Status
        if result["defect_detected"]:
            st.markdown('<div class="status-badge status-defect">🚨 Defect Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-ok">✅ No Defect Detected</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Anomaly Score", f"{result['image_score']:.3f}")
        m2.metric("Router Class", result["router_class"])
        m3.metric("Router Confidence", f"{result['router_confidence']:.1%}")
        m4.metric("Backend Model", result["source_model"])

    st.markdown("---")

    # Side-by-Side Image Visual Comparison
    st.subheader("Visual Inspection Analysis")
    input_col, output_col = st.columns(2)

    with input_col:
        st.image(image, caption="Original Input", width="stretch")

    with output_col:
        st.image(
            predicted_output, 
            caption=f"Heatmap / Bounding Prediction ({result['source_model']})", 
            width="stretch"
        )

    with st.expander("🛠️ Raw API Response"):
        st.json(result)
else:
    st.info("Please upload an industrial image sample above to trigger inspection.")