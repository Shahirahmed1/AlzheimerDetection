"""
Alzheimer's MRI Detection — Streamlit Web Demo

Upload a brain MRI scan and get:
  • Predicted Alzheimer's stage with confidence
  • Probability distribution across all 4 classes
  • Grad-CAM heatmap showing model attention regions
  • Clinical-style diagnosis summary

Usage:
    streamlit run app.py
"""

import os
import sys
import io
import numpy as np
from pathlib import Path

import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model import AlzheimerHybridModel, CLASS_NAMES, NUM_CLASSES, IMG_SIZE, load_trained_model
from gradcam import GradCAM, make_gradcam, pil_to_tensor


# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Alzheimer's MRI Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #a5b4fc;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Result cards */
    .result-card {
        background: linear-gradient(145deg, #1e1b4b, #312e81);
        border: 1px solid rgba(165, 180, 252, 0.15);
        border-radius: 14px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }
    .result-card h3 {
        color: #c4b5fd;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }

    /* Prediction display */
    .prediction-text {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .confidence-text {
        font-size: 1.1rem;
        font-weight: 400;
        color: #94a3b8;
    }

    /* Severity colors */
    .severity-none { color: #34d399; }
    .severity-very-mild { color: #fbbf24; }
    .severity-mild { color: #fb923c; }
    .severity-moderate { color: #f87171; }

    /* Diagnosis box */
    .diagnosis-box {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-left: 4px solid #818cf8;
        border-radius: 0 12px 12px 0;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
    }
    .diagnosis-box h4 {
        color: #e2e8f0;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .diagnosis-box p {
        color: #94a3b8;
        line-height: 1.6;
    }

    /* Probability bars */
    .prob-bar-container {
        margin: 0.4rem 0;
    }
    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        color: #cbd5e1;
        font-size: 0.85rem;
        margin-bottom: 2px;
    }
    .prob-bar-track {
        width: 100%;
        height: 10px;
        background: rgba(255,255,255,0.08);
        border-radius: 5px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.6s ease;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #94a3b8;
    }

    /* Hide default streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ════════════════════════════════════════════════════════════════════════════
CHECKPOINTS_DIR = ROOT / "checkpoints"
SAMPLES_DIR = ROOT / "samples"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEVERITY_INFO = {
    "Non_Demented": {
        "severity": "No Dementia",
        "css_class": "severity-none",
        "color": "#34d399",
        "description": "The MRI scan shows no significant signs of Alzheimer's disease or dementia. Brain structure appears within normal limits.",
        "recommendation": "Continue routine cognitive health monitoring. Maintain a healthy lifestyle with regular physical exercise, mental stimulation, social engagement, and balanced nutrition.",
    },
    "Very_Mild_Demented": {
        "severity": "Very Mild",
        "css_class": "severity-very-mild",
        "color": "#fbbf24",
        "description": "The MRI scan shows very mild indicators that may suggest the earliest stages of cognitive decline. Changes are subtle and may not yet affect daily functioning.",
        "recommendation": "Schedule a consultation with a neurologist for comprehensive cognitive testing. Consider baseline neuropsychological assessment. Monitor for any changes in memory or daily activities.",
    },
    "Mild_Demented": {
        "severity": "Mild",
        "css_class": "severity-mild",
        "color": "#fb923c",
        "description": "The MRI scan shows mild signs of Alzheimer's-related changes. Noticeable brain atrophy patterns are detected, consistent with mild-stage Alzheimer's disease.",
        "recommendation": "Specialist referral recommended for full neurological evaluation. Discuss treatment options including cholinesterase inhibitors. Begin cognitive rehabilitation therapy and establish a care support plan.",
    },
    "Moderate_Demented": {
        "severity": "Moderate",
        "css_class": "severity-moderate",
        "color": "#f87171",
        "description": "The MRI scan shows moderate signs of Alzheimer's disease with significant brain atrophy patterns. This stage typically involves noticeable cognitive and functional impairment.",
        "recommendation": "Urgent specialist evaluation and comprehensive treatment planning required. Combination therapy should be considered. Establish a structured daily care routine and caregiver support system.",
    },
}

BAR_COLORS = {
    "Non_Demented": "#34d399",
    "Very_Mild_Demented": "#fbbf24",
    "Mild_Demented": "#fb923c",
    "Moderate_Demented": "#f87171",
}


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    """Load the trained model. Returns (model, error_message)."""
    ckpt_path = CHECKPOINTS_DIR / "best_model.pth"
    if not ckpt_path.exists():
        return None, f"No trained model found at `{ckpt_path}`. Please train the model first (see README)."
    try:
        model = load_trained_model(str(ckpt_path), device=DEVICE)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════════════════════
def predict(model, pil_img):
    """
    Run inference on a PIL image.
    Returns: (pred_class_name, pred_idx, probs_dict, confidence)
    """
    model.eval()
    tensor = pil_to_tensor(pil_img, device=DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    probs_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return pred_name, pred_idx, probs_dict, confidence


def get_gradcam_overlay(model, pil_img):
    """Generate Grad-CAM overlay image."""
    try:
        cam = make_gradcam(model)
        heatmap, pred_class, probs = cam(pil_img, device=DEVICE)
        overlay = cam.overlay_heatmap(pil_img, heatmap, alpha=0.45)
        cam.remove_hooks()
        return overlay, heatmap
    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")
        return None, None


# ════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ════════════════════════════════════════════════════════════════════════════
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Alzheimer's Disease MRI Detection</h1>
        <p>Upload a brain MRI scan for AI-powered Alzheimer's stage classification using a CNN+BiGRU+Attention hybrid model</p>
    </div>
    """, unsafe_allow_html=True)


def render_probability_bars(probs_dict, pred_name):
    """Render custom probability bar chart."""
    st.markdown('<div class="result-card"><h3>📊 Class Probabilities</h3>', unsafe_allow_html=True)

    # Sort by probability (descending)
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)

    for class_name, prob in sorted_probs:
        pct = prob * 100
        color = BAR_COLORS.get(class_name, "#818cf8")
        display_name = class_name.replace("_", " ")

        st.markdown(f"""
        <div class="prob-bar-container">
            <div class="prob-bar-label">
                <span>{"➤ " if class_name == pred_name else ""}{display_name}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width: {pct}%; background: {color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_diagnosis(pred_name, confidence):
    """Render clinical-style diagnosis summary."""
    info = SEVERITY_INFO[pred_name]

    st.markdown(f"""
    <div class="diagnosis-box">
        <h4>📋 Diagnosis Summary</h4>
        <p><strong>Assessment:</strong> {info['description']}</p>
        <p><strong>Confidence:</strong> {confidence*100:.1f}%</p>
        <p><strong>Recommendation:</strong> {info['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.3);
                border-radius: 8px; padding: 0.75rem 1rem; margin-top: 0.75rem;">
        <p style="color: #fca5a5; font-size: 0.8rem; margin: 0;">
            ⚠️ <strong>Disclaimer:</strong> This is an AI-assisted screening tool for educational purposes only.
            It is NOT a substitute for professional medical diagnosis. Always consult a qualified healthcare
            provider for clinical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════
def main():
    render_header()

    # ── Sidebar ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📁 Upload MRI Scan")
        uploaded = st.file_uploader(
            "Drop a brain MRI image here",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Upload a grayscale brain MRI scan image",
        )

        # Sample images
        sample_choice = None
        if SAMPLES_DIR.exists():
            samples = sorted([
                f.name for f in SAMPLES_DIR.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg")
            ])
            if samples:
                st.markdown("### 🖼️ Or Pick a Sample")
                sample_choice = st.selectbox(
                    "Sample images",
                    ["(none)"] + samples,
                    label_visibility="collapsed",
                )

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Model:** CNN + BiGRU + Attention Hybrid

        **Classes:**
        - 🟢 Non Demented
        - 🟡 Very Mild Demented
        - 🟠 Mild Demented
        - 🔴 Moderate Demented

        **Input:** 128×128 grayscale MRI
        """)
        st.markdown(f"**Device:** `{DEVICE}`")

    # ── Load Model ──────────────────────────────────────────────────────
    model, error = load_model()
    if error:
        st.error(error)
        st.info("Please train the model using the training script and place `best_model.pth` in the `checkpoints/` folder.")
        st.stop()

    # ── Load Image ──────────────────────────────────────────────────────
    pil_img = None
    if uploaded:
        pil_img = Image.open(io.BytesIO(uploaded.read())).convert("L")
    elif sample_choice and sample_choice != "(none)":
        pil_img = Image.open(SAMPLES_DIR / sample_choice).convert("L")

    if pil_img is None:
        # Show placeholder
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center;
                    justify-content: center; min-height: 400px; opacity: 0.5;">
            <p style="font-size: 4rem; margin: 0;">🧠</p>
            <p style="font-size: 1.2rem; color: #94a3b8; text-align: center;">
                Upload a brain MRI scan or select a sample image<br>
                to begin Alzheimer's detection analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Run Inference ───────────────────────────────────────────────────
    with st.spinner("🔍 Analyzing MRI scan..."):
        pred_name, pred_idx, probs_dict, confidence = predict(model, pil_img)
        overlay, heatmap = get_gradcam_overlay(model, pil_img)

    # ── Display Results ─────────────────────────────────────────────────
    info = SEVERITY_INFO[pred_name]

    # Prediction header
    st.markdown(f"""
    <div class="result-card" style="text-align: center;">
        <h3>Prediction Result</h3>
        <p class="prediction-text {info['css_class']}">
            {info['severity']} — {pred_name.replace('_', ' ')}
        </p>
        <p class="confidence-text">Confidence: {confidence*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Two-column layout for images
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="result-card"><h3>🖼️ Input MRI Scan</h3></div>', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)

    with col2:
        st.markdown('<div class="result-card"><h3>🔥 Grad-CAM Attention Map</h3></div>', unsafe_allow_html=True)
        if overlay is not None:
            st.image(overlay, use_container_width=True)
        else:
            st.info("Grad-CAM visualization unavailable")

    # Probabilities and Diagnosis
    col3, col4 = st.columns(2)

    with col3:
        render_probability_bars(probs_dict, pred_name)

    with col4:
        render_diagnosis(pred_name, confidence)


if __name__ == "__main__":
    main()
