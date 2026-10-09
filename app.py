import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Classifier",
    page_icon="🩻",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 16px;
    }

    /* File uploader */
    section[data-testid="stFileUploadDropzone"] {
        background-color: #1e2130;
        border: 2px dashed #3a86ff;
        border-radius: 12px;
    }

    /* Prediction result box */
    .result-box {
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1rem;
    }
    .result-pneumonia {
        background-color: rgba(255, 75, 75, 0.15);
        border: 2px solid #ff4b4b;
        color: #ff4b4b;
    }
    .result-normal {
        background-color: rgba(0, 200, 100, 0.12);
        border: 2px solid #00c864;
        color: #00c864;
    }

    /* Confidence bar track */
    .conf-track {
        background-color: #2e3250;
        border-radius: 8px;
        height: 14px;
        overflow: hidden;
        margin-top: 8px;
    }
    .conf-fill-pneumonia {
        background: linear-gradient(90deg, #ff4b4b, #ff8c00);
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }
    .conf-fill-normal {
        background: linear-gradient(90deg, #00c864, #3a86ff);
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }

    /* Disclaimer box */
    .disclaimer {
        background-color: rgba(255, 200, 0, 0.08);
        border-left: 4px solid #ffc800;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.85rem;
        color: #ccc;
        margin-top: 1.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161824; }
    .sidebar-card {
        background-color: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
        font-size: 0.88rem;
        color: #ccc;
    }
    .sidebar-card h4 { color: #3a86ff; margin: 0 0 8px 0; font-size: 0.95rem; }
    .badge {
        display: inline-block;
        background-color: #2e3250;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.78rem;
        color: #aaa;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model():
    return tf.keras.models.load_model("pneumonia_classifier_weighted.keras")

model = load_model()


# ─── Prediction Helper ─────────────────────────────────────────────────────────
def predict(image: Image.Image, model) -> float:
    """Return pneumonia probability in [0, 1]."""
    img = image.resize((224, 224))
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)       # shape: (1, 224, 224, 3)
    return float(model.predict(arr, verbose=0)[0][0])


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩻 About")
    st.markdown("""
    <div class="sidebar-card">
        <h4>🧠 Model Architecture</h4>
        3× Conv2D blocks (32 → 64 → 128 filters)<br>
        Dense (128 units) + Dropout (0.5)<br>
        Sigmoid output layer
    </div>
    <div class="sidebar-card">
        <h4>📊 Training Details</h4>
        <span class="badge">224 × 224 px input</span>
        <span class="badge">Batch size 32</span>
        <span class="badge">Adam optimizer</span>
        <span class="badge">Class-weighted loss</span>
    </div>
    <div class="sidebar-card">
        <h4>📈 Reported Performance</h4>
        Test Accuracy: <b>81%</b><br>
        Pneumonia Recall: <b>98%</b><br>
        Normal Recall: <b>52%</b>
    </div>
    <div class="sidebar-card">
        <h4>📁 Dataset</h4>
        Pediatric Chest X-Ray (Kaggle)<br>
        Normal: 1,341 &nbsp;|&nbsp; Pneumonia: 3,875
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability above this value is classified as PNEUMONIA.",
    )


# ─── Main Layout ───────────────────────────────────────────────────────────────
st.markdown("# 🩻 Pediatric Chest X-Ray Analyzer")
st.markdown(
    "Upload a pediatric chest X-ray image and the model will predict whether "
    "it shows signs of **Pneumonia** or is **Normal**."
)
st.markdown("---")

col_upload, col_results = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a Chest X-ray (JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)

with col_results:
    st.markdown("### 🔍 Analysis Results")

    if uploaded_file is None:
        st.info("Upload an X-ray image on the left to see the prediction here.")
    else:
        with st.spinner("Running inference…"):
            prob = predict(image, model)

        is_pneumonia = prob > threshold
        label        = "PNEUMONIA" if is_pneumonia else "NORMAL"
        confidence   = prob * 100 if is_pneumonia else (1 - prob) * 100
        fill_class   = "pneumonia" if is_pneumonia else "normal"
        box_class    = "result-pneumonia" if is_pneumonia else "result-normal"
        icon         = "🚨" if is_pneumonia else "✅"

        # ── Result badge ──
        st.markdown(
            f'<div class="result-box {box_class}">'
            f'{icon} {label}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Metrics row ──
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", label)
        m2.metric("Confidence", f"{confidence:.1f}%")
        m3.metric("Threshold", f"{threshold:.0%}")

        # ── Confidence bar ──
        st.markdown("**Confidence breakdown**")
        st.markdown(
            f'<div class="conf-track">'
            f'<div class="conf-fill-{fill_class}" style="width:{confidence:.1f}%"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Pneumonia probability: **{prob*100:.2f}%**  |  Normal probability: **{(1-prob)*100:.2f}%**")

        # ── Raw probability chart ──
        st.markdown("**Class probability distribution**")
        chart_data = {"Normal": round(1 - prob, 4), "Pneumonia": round(prob, 4)}
        st.bar_chart(chart_data, color="#3a86ff")

        # ── Clinical disclaimer ──
        st.markdown(
            '<div class="disclaimer">'
            '⚠️ <b>Clinical Disclaimer:</b> This tool is intended for '
            'educational and research purposes only. It is <b>not</b> a '
            'substitute for professional medical diagnosis. Always consult '
            'a qualified radiologist or physician for clinical decisions.'
            '</div>',
            unsafe_allow_html=True,
        )
