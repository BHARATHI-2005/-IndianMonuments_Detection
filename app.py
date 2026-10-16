import base64
import html
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# Must match the notebook:
# IMAGE_SIZE = (160, 160)
# Model contains Rescaling(1./255), so do NOT divide by 255 in this app.
IMAGE_SIZE = (160, 160)
MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
GUIDE_LANGUAGES = [
    ("English", "en-IN"),
    ("Assamese", "as-IN"),
    ("Bengali", "bn-IN"),
    ("Bodo", "brx-IN"),
    ("Dogri", "doi-IN"),
    ("Gujarati", "gu-IN"),
    ("Hindi", "hi-IN"),
    ("Kannada", "kn-IN"),
    ("Kashmiri", "ks-IN"),
    ("Konkani", "kok-IN"),
    ("Maithili", "mai-IN"),
    ("Malayalam", "ml-IN"),
    ("Manipuri (Meitei)", "mni-IN"),
    ("Marathi", "mr-IN"),
    ("Nepali", "ne-IN"),
    ("Odia", "or-IN"),
    ("Punjabi", "pa-IN"),
    ("Sanskrit", "sa-IN"),
    ("Santali", "sat-IN"),
    ("Sindhi", "sd-IN"),
    ("Tamil", "ta-IN"),
    ("Telugu", "te-IN"),
    ("Urdu", "ur-IN"),
]
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "indian_monuments_cnn.keras"
LABELS_PATH = PROJECT_DIR / "class_names.json"

st.set_page_config(
    page_title="Indian Monuments Classifier",
    page_icon="🏛️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --night: #071426;
        --night-soft: #0b1c34;
        --panel: rgba(15, 35, 61, 0.90);
        --panel-border: rgba(174, 199, 226, 0.17);
        --ink: #f6f2e9;
        --muted: #c1ccda;
        --saffron: #ffad4d;
        --turmeric: #ffd36c;
        --indigo: #9aaeff;
        --teal: #66dfcd;
        --rose: #ff94a9;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(31, 77, 128, 0.35), transparent 36rem),
            radial-gradient(ellipse at 95% 24%, rgba(119, 54, 116, 0.18), transparent 30rem),
            linear-gradient(145deg, #061120 0%, #0a1a32 52%, #101b3c 100%);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMain"] > div {
        max-width: 1180px;
        padding: 2rem 2rem 4rem;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"] p {
        color: var(--muted);
    }
    h1, h2, h3, h4, strong, label, [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
    }
    .stMarkdown a { color: var(--turmeric) !important; }
    [data-testid="stFileUploaderDropzone"] {
        min-height: 190px;
        border: 1px dashed rgba(255, 173, 77, 0.62);
        border-radius: 20px;
        background: rgba(7, 20, 38, 0.56);
    }
    [data-testid="stFileUploaderDropzone"] * { color: var(--ink); }
    [data-testid="stFileUploaderDropzone"] button {
        border-color: rgba(255, 211, 108, 0.55);
        color: var(--turmeric);
        background: rgba(255, 173, 77, 0.10);
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        border-color: var(--turmeric);
        color: white;
    }
    .stButton > button {
        min-height: 3rem;
        border: 1px solid rgba(255, 211, 108, 0.45);
        border-radius: 999px;
        color: #171628;
        background: linear-gradient(100deg, #ffb154, #ffdb79);
        font-weight: 750;
        transition: transform 180ms ease, box-shadow 180ms ease;
        box-shadow: 0 8px 26px rgba(255, 173, 77, 0.20);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        color: #171628;
        box-shadow: 0 12px 32px rgba(255, 173, 77, 0.34);
    }
    .stButton > button:not(:disabled) * { color: #171628 !important; }
    .stButton > button:disabled {
        color: #c4c9d3;
        background: #334155;
        border-color: #536174;
        box-shadow: none;
    }
    .stButton > button:disabled * { color: #e2e8f0 !important; }
    [data-testid="stMetric"] {
        padding: 1rem 1.1rem;
        border: 1px solid var(--panel-border);
        border-radius: 16px;
        background: rgba(13, 31, 53, 0.76);
    }
    [data-testid="stAlert"] { border-radius: 14px; }
    [data-testid="stImage"] img { border-radius: 18px; }

    .heritage-hero {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.25rem;
        min-height: 285px;
        overflow: hidden;
        margin: 0 0 1.5rem;
        padding: clamp(1.4rem, 4vw, 3rem);
        border: 1px solid var(--panel-border);
        border-radius: 28px;
        background:
            radial-gradient(ellipse at 85% 80%, rgba(255, 173, 77, 0.13), transparent 26rem),
            linear-gradient(125deg, rgba(11, 29, 52, 0.98), rgba(22, 33, 66, 0.95));
        box-shadow: 0 22px 70px rgba(0, 0, 0, 0.23);
    }
    .hero-copy { position: relative; z-index: 1; max-width: 680px; }
    .hero-eyebrow {
        margin-bottom: .7rem;
        color: var(--turmeric);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .19em;
        text-transform: uppercase;
    }
    .hero-title {
        margin: 0;
        color: var(--ink);
        font-size: clamp(2.1rem, 5vw, 3.75rem);
        font-weight: 800;
        letter-spacing: -.045em;
        line-height: 1.05;
    }
    .hero-title span {
        color: var(--saffron);
        background: linear-gradient(90deg, #ffb454, #ffd878, #ff8f80);
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        max-width: 540px;
        margin: 1rem 0 0;
        color: #d3d9e5;
        font-size: 1.02rem;
        line-height: 1.7;
    }
    .hero-chips { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.2rem; }
    .hero-chip {
        padding: .38rem .72rem;
        border: 1px solid rgba(255, 255, 255, .16);
        border-radius: 999px;
        background: rgba(255, 255, 255, .055);
        color: #e7eaf1;
        font-size: .77rem;
    }
    .hero-art {
        flex: 0 0 265px;
        animation: hero-arrive 800ms ease-out both;
    }
    .hero-art svg { display: block; width: 100%; height: auto; overflow: visible; }
    .hero-art .guide-body {
        transform-origin: 140px 172px;
        animation: guide-sway 4.5s ease-in-out infinite;
    }
    .hero-art .wave-arm { transform-origin: 166px 126px; animation: wave 1.3s ease-in-out infinite; }
    .hero-art .point-arm { transform-origin: 160px 133px; animation: point 2.6s ease-in-out infinite; }
    .guide-card .guide-body {
        transform-origin: 140px 172px;
        animation: guide-sway 4.5s ease-in-out infinite;
    }
    .guide-card .wave-arm { transform-origin: 166px 126px; animation: wave 1.3s ease-in-out infinite; }
    .guide-card .point-arm { transform-origin: 205px 100px; animation: point 2.6s ease-in-out infinite; }

    .guide-card {
        position: relative;
        overflow: hidden;
        padding: 1.25rem 1.25rem 1.15rem;
        border: 1px solid var(--panel-border);
        border-radius: 22px;
        background:
            radial-gradient(ellipse at 85% 8%, rgba(255, 173, 77, .12), transparent 17rem),
            linear-gradient(145deg, rgba(15, 35, 61, .96), rgba(12, 25, 49, .96));
        box-shadow: 0 16px 40px rgba(0, 0, 0, .18);
    }
    .guide-card:before {
        position: absolute;
        inset: 0 auto 0 0;
        width: 3px;
        background: linear-gradient(#ffad4d, #ffdb79, #66dfcd);
        content: "";
    }
    .guide-card-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: .6rem;
        color: var(--turmeric);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .guide-live {
        display: inline-flex;
        align-items: center;
        gap: .4rem;
        padding: .27rem .55rem;
        border: 1px solid rgba(102, 223, 205, .25);
        border-radius: 999px;
        background: rgba(102, 223, 205, .08);
        color: #9af0e2;
        font-size: .63rem;
        letter-spacing: .1em;
    }
    .guide-live:before {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #66dfcd;
        box-shadow: 0 0 10px #66dfcd;
        content: "";
        animation: glow 1.8s ease-in-out infinite;
    }
    .guide-card h2 {
        margin: .8rem 0 .35rem;
        color: var(--ink);
        font-size: 1.32rem;
        letter-spacing: -.02em;
    }
    .guide-card p {
        margin: 0;
        color: #cbd5e1;
        font-size: .91rem;
        line-height: 1.65;
    }
    .guide-outfit {
        display: inline-block;
        margin-top: .9rem;
        padding: .43rem .67rem;
        border: 1px solid rgba(255, 211, 108, .26);
        border-radius: 11px;
        background: rgba(255, 173, 77, .08);
        color: #ffe0a0;
        font-size: .78rem;
    }
    .section-kicker {
        margin: .2rem 0 .3rem;
        color: var(--turmeric);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .15em;
        text-transform: uppercase;
    }
    .section-heading {
        margin: 0 0 .6rem;
        color: var(--ink);
        font-size: 1.55rem;
        font-weight: 750;
        letter-spacing: -.03em;
    }
    .upload-hint { margin: 0 0 1rem; color: var(--muted); line-height: 1.65; }
    @keyframes hero-arrive {
        from { opacity: 0; transform: translateY(10px) scale(.97); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes guide-sway {
        0%, 100% { transform: rotate(-1deg) translateY(0); }
        50% { transform: rotate(1deg) translateY(-3px); }
    }
    @keyframes wave {
        0%, 100% { transform: rotate(0); }
        50% { transform: rotate(-12deg); }
    }
    @keyframes point {
        0%, 100% { transform: translateY(1px) rotate(0); }
        50% { transform: translateY(-4px) rotate(-4deg); }
    }
    @keyframes glow { 50% { opacity: .45; } }
    @media (max-width: 760px) {
        [data-testid="stMain"] > div { padding: 1rem 1rem 3rem; }
        .heritage-hero { min-height: 230px; padding: 1.5rem; }
        .hero-art { flex-basis: 145px; margin-right: -.5rem; }
        .hero-subtitle { font-size: .93rem; }
    }
    @media (max-width: 520px) {
        .heritage-hero { gap: .25rem; }
        .hero-art { flex-basis: 105px; margin-right: -.85rem; }
        .hero-title { font-size: 2rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *:before, *:after {
            scroll-behavior: auto !important;
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: .01ms !important;
        }
    }

    :root {
        color-scheme: light;
        --night: #fff8e9;
        --night-soft: #fff1d6;
        --panel: #fffdf7;
        --panel-border: rgba(117, 66, 45, .18);
        --ink: #342044;
        --muted: #675466;
        --saffron: #e88a16;
        --turmeric: #9b5b00;
        --indigo: #51429b;
        --teal: #087d78;
        --rose: #c42f63;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        color-scheme: light;
        color: #342044;
        background-color: #fff8e9 !important;
        background-image:
            radial-gradient(circle at 8px 8px, rgba(198, 47, 99, .10) 1.5px, transparent 2px),
            radial-gradient(circle at 28px 28px, rgba(8, 125, 120, .09) 1.5px, transparent 2px),
            linear-gradient(145deg, #fffaf0, #fff2d9 52%, #ffefd9) !important;
        background-size: 40px 40px, 40px 40px, 100% 100% !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"] p {
        color: #675466 !important;
    }
    h1, h2, h3, h4, strong, label,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #342044 !important;
    }
    .stMarkdown a { color: #a52a5a !important; }
    .heritage-hero {
        border: 2px solid rgba(194, 79, 56, .28);
        background:
            radial-gradient(circle at 91% 18%, rgba(255, 201, 74, .40), transparent 9rem),
            radial-gradient(circle at 82% 84%, rgba(13, 139, 125, .15), transparent 13rem),
            repeating-conic-gradient(from 0deg at 94% 50%, rgba(218, 76, 82, .06) 0deg 8deg, transparent 8deg 16deg),
            linear-gradient(120deg, #fff4d9, #ffe9c3 54%, #fce8ed) !important;
        box-shadow: 0 18px 45px rgba(127, 76, 39, .13);
    }
    .hero-eyebrow, .section-kicker, .guide-card-top { color: #a52a5a !important; }
    .hero-title { color: #342044 !important; }
    .hero-title span {
        background-image: linear-gradient(90deg, #dd593f, #d89013, #168777) !important;
    }
    .hero-subtitle { color: #59475a !important; }
    .hero-chip {
        border-color: rgba(133, 76, 45, .19);
        background: rgba(255, 255, 255, .67);
        color: #49334e;
    }
    .guide-card {
        border: 2px solid rgba(194, 79, 56, .23);
        background:
            radial-gradient(circle at 92% 10%, rgba(255, 198, 75, .34), transparent 10rem),
            radial-gradient(circle at 4% 94%, rgba(10, 144, 128, .11), transparent 11rem),
            linear-gradient(145deg, #fffaf0, #ffefd5) !important;
        box-shadow: 0 16px 36px rgba(127, 76, 39, .13);
    }
    .guide-card:before {
        width: 5px;
        background: linear-gradient(#d54552, #e7951d, #138d79, #6652a3) !important;
    }
    .guide-card h2 { color: #342044 !important; }
    .guide-card p { color: #59475a !important; }
    .guide-live {
        border-color: rgba(8, 125, 120, .28);
        background: rgba(8, 125, 120, .09);
        color: #09645e !important;
    }
    .guide-outfit {
        border-color: rgba(198, 47, 99, .23);
        background: rgba(255, 255, 255, .65);
        color: #8c3156 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed rgba(191, 89, 48, .58) !important;
        border-radius: 20px !important;
        background: rgba(255, 255, 255, .72) !important;
    }
    [data-testid="stFileUploaderDropzone"] * { color: #4b3850 !important; }
    [data-testid="stFileUploaderDropzone"] button {
        border-color: rgba(8, 125, 120, .42) !important;
        color: #086c66 !important;
        background: #e2f3e9 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        border-color: #087d78 !important;
        color: #075d58 !important;
    }
    .stButton > button {
        border: 2px solid #a8444e !important;
        color: #fffdf7 !important;
        background: linear-gradient(100deg, #c63e61, #a63c79 52%, #644a9d) !important;
        box-shadow: 0 9px 24px rgba(146, 50, 87, .25) !important;
    }
    .stButton > button:not(:disabled) * { color: #fffdf7 !important; }
    .stButton > button:hover {
        color: white !important;
        box-shadow: 0 13px 30px rgba(146, 50, 87, .32) !important;
    }
    .stButton > button:disabled {
        border-color: #c6b8c0 !important;
        color: #59475a !important;
        background: #e8dfe2 !important;
        box-shadow: none !important;
    }
    .stButton > button:disabled * { color: #59475a !important; }
    [data-testid="stMetric"] {
        border-color: rgba(117, 66, 45, .18);
        background: rgba(255, 255, 255, .75);
    }
    [data-testid="stAlert"] { color: #342044; }
    .uploaded-photo-card {
        overflow: hidden;
        margin-top: 1rem;
        border: 2px solid rgba(194, 79, 56, .23);
        border-radius: 20px;
        background: #fffdf7;
        box-shadow: 0 12px 28px rgba(107, 66, 43, .14);
    }
    .uploaded-photo-card img {
        display: block;
        width: 100%;
        max-height: 520px;
        object-fit: contain;
        background: #f2e7d1;
    }
    .uploaded-photo-caption {
        padding: .7rem .9rem;
        color: #675466;
        font-size: .85rem;
    }
    .folk-buddy {
        transform-box: fill-box;
        transform-origin: center bottom;
        animation: buddy-dance .72s ease-in-out infinite alternate;
    }
    .folk-buddy-arm {
        transform-box: fill-box;
        transform-origin: 10% 90%;
        animation: buddy-wave .48s ease-in-out infinite alternate;
    }
    @keyframes buddy-dance {
        from { transform: translateY(0) rotate(-5deg); }
        to { transform: translateY(-5px) rotate(5deg); }
    }
    @keyframes buddy-wave {
        from { transform: rotate(-16deg); }
        to { transform: rotate(17deg); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_uploaded_preview(image: Image.Image) -> None:
    preview = image.copy()
    preview.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=84, optimize=True)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
    st.markdown(
        f"""
        <div class="uploaded-photo-card">
            <img src="data:image/jpeg;base64,{encoded_image}"
                 alt="Preview of your uploaded monument photo">
            <div class="uploaded-photo-caption">Your uploaded photo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model_and_labels():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n\n"
            "Copy indian_monuments_cnn.keras from your Colab export into the models folder."
        )
    if not LABELS_PATH.is_file():
        raise FileNotFoundError(
            f"Class labels not found: {LABELS_PATH}\n\n"
            "Copy class_names.json from the same Colab export into the project folder."
        )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list) or len(class_names) < 2:
        raise ValueError("class_names.json must contain a JSON list of class names.")
    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            f"Model has {model.output_shape[-1]} outputs but class_names.json "
            f"contains {len(class_names)} labels. Use files from the same notebook run."
        )
    return model, class_names


LANDMARK_DISPLAY_NAMES = {
    "Ajanta Caves": "Ajanta Caves",
    "Charar-E- Sharif": "Charar-e-Sharief",
    "Chhota_Imambara": "Chhota Imambara",
    "Ellora Caves": "Ellora Caves",
    "Fatehpur Sikri": "Fatehpur Sikri",
    "Gateway of India": "Gateway of India",
    "Humayun_s Tomb": "Humayun's Tomb",
    "India gate pics": "India Gate",
    "Khajuraho": "Khajuraho Group of Monuments",
    "Sun Temple Konark": "Konark Sun Temple",
    "alai_darwaza": "Alai Darwaza",
    "alai_minar": "Alai Minar",
    "basilica_of_bom_jesus": "Basilica of Bom Jesus",
    "charminar": "Charminar",
    "golden temple": "Golden Temple",
    "hawa mahal pics": "Hawa Mahal",
    "iron_pillar": "Iron Pillar of Delhi",
    "jamali_kamali_tomb": "Jamali Kamali Mosque and Tomb",
    "lotus_temple": "Lotus Temple",
    "mysore_palace": "Mysore Palace",
    "qutub_minar": "Qutub Minar",
    "tajmahal": "Taj Mahal",
    "tanjavur temple": "Brihadisvara Temple, Thanjavur",
    "victoria memorial": "Victoria Memorial",
}


def get_groq_api_key() -> str:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if api_key:
        return api_key

    env_path = PROJECT_DIR / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            entry = line.strip()
            if entry.startswith("export "):
                entry = entry[7:].lstrip()
            name, separator, value = entry.partition("=")
            if separator and name.strip() == "GROQ_API_KEY":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if value:
                    return value

    raise RuntimeError(
        "GROQ_API_KEY was not found. Add it to the project .env file "
        "or set it as an environment variable."
    )


def generate_landmark_story(class_label: str) -> str:
    landmark_name = LANDMARK_DISPLAY_NAMES.get(
        class_label, class_label.replace("_", " ").strip().title()
    )
    prompt = (
        f"Write an engaging, easy-to-understand visitor's guide to {landmark_name}. "
        f"The image classifier's original label is {class_label!r}. Explain what the "
        "landmark is, its history and historical context, its distinctive architecture "
        "or important features, and why it matters in Indian heritage. Be accurate: "
        "do not invent dates, rulers, events, UNESCO status, or architectural details. "
        "If a detail is uncertain or disputed, say so briefly. Use these four short "
        "Markdown sections: '### The landmark', '### A little history', "
        "'### Look closely', and '### Why it matters'. Keep the whole answer around "
        "180-260 words. Do not claim to have searched the web or provide fabricated "
        "citations."
    )
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful, friendly guide to India's history and heritage. "
                    "Give useful, factual explanations in clear language. Never make up "
                    "historical facts when uncertain."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.35,
        "max_tokens": 700,
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {get_groq_api_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(10, 75),
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "The heritage guide took too long to respond. Please try again."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "Could not connect to Groq. Check your internet connection and try again."
        ) from exc

    if response.status_code == 401:
        raise RuntimeError(
            "Groq rejected the API key. Check that GROQ_API_KEY in .env is valid."
        )
    if response.status_code == 429:
        raise RuntimeError(
            "The Groq free-tier rate limit has been reached. Wait for the quota "
            "to reset, then classify the image again."
        )
    if not response.ok:
        raise RuntimeError(
            f"Groq could not generate the guide (HTTP {response.status_code}). "
            "Check the model's availability and try again."
        )

    try:
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Groq returned an unexpected response. Please try again."
        ) from exc

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Groq returned an empty heritage guide. Please try again.")
    return answer.strip()


def transcribe_visitor_question(audio_file) -> str:
    if audio_file.size > MAX_UPLOAD_BYTES:
        raise RuntimeError("The voice recording is over 20 MB. Please record a shorter question.")

    try:
        response = requests.post(
            GROQ_TRANSCRIPTION_URL,
            headers={"Authorization": f"Bearer {get_groq_api_key()}"},
            files={
                "file": (
                    audio_file.name or "visitor-question.wav",
                    audio_file.getvalue(),
                    audio_file.type or "audio/wav",
                )
            },
            data={
                "model": GROQ_TRANSCRIPTION_MODEL,
                "response_format": "json",
            },
            timeout=(10, 75),
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("Voice transcription took too long. Please try again.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "Could not connect to Groq to transcribe your question. Check your "
            "internet connection and try again."
        ) from exc

    if response.status_code == 401:
        raise RuntimeError("Groq rejected the API key. Check GROQ_API_KEY in .env.")
    if response.status_code == 429:
        raise RuntimeError(
            "The Groq free-tier rate limit has been reached. Wait for the quota to reset."
        )
    if not response.ok:
        raise RuntimeError(
            f"Groq could not transcribe the recording (HTTP {response.status_code}). "
            "Try recording again or type your question."
        )

    try:
        transcription = response.json()["text"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Groq returned an unexpected transcription response.") from exc

    if not isinstance(transcription, str) or not transcription.strip():
        raise RuntimeError(
            "No speech was detected. Please try again or type your question."
        )
    return transcription.strip()


def answer_visitor_question(
    class_label: str,
    story: str,
    question: str,
    language: str,
    conversation: list[dict[str, str]],
) -> str:
    landmark_name = LANDMARK_DISPLAY_NAMES.get(
        class_label, class_label.replace("_", " ").strip().title()
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a warm, accurate interactive guide for visitors to Indian "
                "heritage sites. Answer the visitor in "
                f"{language}. The identified monument is {landmark_name}. Use the "
                "provided heritage overview as context, but do not treat uncertain "
                "details as facts or invent dates, people, legends, or features. Keep "
                "each answer conversational, clear, and suitable for being read aloud "
                "(usually 2-5 sentences). If you do not know, say so. Keep the answer "
                "focused on this monument and its history, architecture, landmarks, "
                "visitor experience, or cultural context."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Heritage overview for context:\n{story or 'No overview is available.'}"
            ),
        },
    ]
    messages.extend(
        {"role": message["role"], "content": message["content"]}
        for message in conversation[-6:]
    )
    messages.append({"role": "user", "content": question})
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.35,
        "max_tokens": 450,
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {get_groq_api_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(10, 75),
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("Your guide took too long to answer. Please try again.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "Could not connect to Groq. Check your internet connection and try again."
        ) from exc

    if response.status_code == 401:
        raise RuntimeError("Groq rejected the API key. Check GROQ_API_KEY in .env.")
    if response.status_code == 429:
        raise RuntimeError(
            "The Groq free-tier rate limit has been reached. Wait for the quota to reset."
        )
    if not response.ok:
        raise RuntimeError(
            f"Groq could not answer your question (HTTP {response.status_code}). "
            "Please try again shortly."
        )

    try:
        answer = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Groq returned an unexpected guide response.") from exc

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Groq returned an empty answer. Please try again.")
    return answer.strip()


def render_spoken_answer(answer: str, language_locale: str) -> None:
    safe_answer = json.dumps(answer).replace("<", "\\u003c")
    safe_locale = json.dumps(language_locale)
    st.components.v1.html(
        f"""
        <button id="speak-answer"
                style="border:0;border-radius:999px;padding:10px 18px;
                       background:#963257;color:#fffdf7;font-weight:700;
                       cursor:pointer;font-size:14px">
            🔊 Listen to this answer
        </button>
        <span id="speech-status" style="margin-left:10px;color:#59475a;
                                        font:13px sans-serif"></span>
        <script>
        const answer = {safe_answer};
        const locale = {safe_locale};
        document.getElementById("speak-answer").addEventListener("click", () => {{
            const status = document.getElementById("speech-status");
            if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {{
                status.textContent = "Speech is not supported in this browser.";
                return;
            }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(answer);
            utterance.lang = locale;
            const voices = window.speechSynthesis.getVoices();
            const voice = voices.find(item =>
                item.lang.toLowerCase() === locale.toLowerCase()
            ) || voices.find(item =>
                item.lang.toLowerCase().startsWith(locale.slice(0, 2).toLowerCase())
            );
            if (voice) utterance.voice = voice;
            utterance.onerror = () => {{
                status.textContent = "Could not play speech. Check your device voices.";
            }};
            status.textContent = voice
                ? "Speaking in " + voice.lang + "…"
                : "Using your device's default voice; language support varies.";
            window.speechSynthesis.speak(utterance);
        }});
        </script>
        """,
        height=58,
        scrolling=False,
    )


PLACE_STYLES = {
    "ajanta caves": ("rust-red cotton kurta", "saffron scarf", "camera"),
    "charar-e- sharif": ("cream embroidered kurta", "deep-green shawl", "namaste"),
    "chhota_imambara": ("midnight-blue achkan", "golden sash", "wave"),
    "ellora caves": ("ochre travel kurta", "indigo shoulder wrap", "camera"),
    "fatehpur sikri": ("sandstone-toned angarkha", "teal dupatta", "point"),
    "gateway of india": ("indigo Nehru jacket", "saffron stole", "wave"),
    "humayun_s tomb": ("plum velvet jacket", "rose stole", "namaste"),
    "india gate pics": ("cream parade jacket", "saffron sash", "salute"),
    "khajuraho": ("terracotta silk kurta", "turquoise shawl", "namaste"),
    "sun temple konark": ("sunset-orange dhoti kurta", "marigold sash", "point"),
    "alai_darwaza": ("emerald long kurta", "gold-trimmed scarf", "point"),
    "alai_minar": ("cobalt-blue kurta", "saffron scarf", "point"),
    "basilica_of_bom_jesus": ("ivory festive jacket", "cobalt-blue sash", "wave"),
    "charminar": ("peacock-green kurta", "golden dupatta", "wave"),
    "golden temple": ("white kurta and modest head scarf", "bright-saffron wrap", "namaste"),
    "hawa mahal pics": ("rose-pink embroidered lehenga", "teal dupatta", "wave"),
    "iron_pillar": ("indigo cotton kurta", "ochre scarf", "salute"),
    "jamali_kamali_tomb": ("soft-lilac angarkha", "silver-blue stole", "namaste"),
    "lotus_temple": ("lotus-pink kurta", "leaf-green dupatta", "namaste"),
    "mysore_palace": ("ivory silk kurta", "royal-purple Nehru jacket", "wave"),
    "qutub_minar": ("saffron long kurta", "teal shoulder scarf", "point"),
    "tajmahal": ("pearl-white sherwani", "ruby-red stole", "namaste"),
    "tanjavur temple": ("marigold dhoti kurta", "deep-red angavastram", "namaste"),
    "victoria memorial": ("cream heritage jacket", "bottle-green sash", "wave"),
}


def get_place_style(place: str) -> tuple[str, str, str]:
    normalized = place.strip().casefold()
    return PLACE_STYLES.get(
        normalized,
        ("indigo travel kurta", "saffron shoulder scarf", "wave"),
    )


def render_guide(
    place: str | None = None,
    character: str = "Boy",
    action: str | None = None,
) -> str:
    """Draw an animated monument guide in the selected outfit and pose."""
    if place is None:
        place = "Your next heritage stop"
        attire = "a bright travel kurta and saffron scarf"
        gesture = "wave"
        garment_color = "#285e86"
        scarf_color = "#f3a33b"
        accessory = "saffron scarf"
    else:
        attire, accessory, gesture = get_place_style(place)
        action_text = {
            "camera": "taking a closer look at the details",
            "namaste": "greeting this landmark with a namaste",
            "point": "pointing out a detail in the architecture",
            "salute": "saluting this historic landmark",
            "wave": "waving hello from this heritage stop",
        }[gesture]
        garment_color = {
            "rust-red": "#a34543", "cream": "#d7b88c", "midnight-blue": "#31416d",
            "ochre": "#b47a36", "sandstone-toned": "#bc8251", "indigo": "#334b83",
            "plum": "#794c78", "terracotta": "#bd6046", "sunset-orange": "#ce6838",
            "emerald": "#237a68", "cobalt-blue": "#355ca4", "ivory": "#d9cba5",
            "peacock-green": "#187d70", "white": "#e8e9df", "rose-pink": "#b75679",
            "soft-lilac": "#866e9e", "lotus-pink": "#cc6386", "marigold": "#d59432",
            "pearl-white": "#e7dfd0",
        }.get(attire.split()[0], "#285e86")
        accessory_color = {
            "saffron": "#ef982e", "deep-green": "#24755e", "golden": "#e6ba53",
            "indigo": "#455ca1", "teal": "#42a99d", "rose": "#cf6f8c",
            "ruby-red": "#bd4559", "turquoise": "#3caeb1", "marigold": "#efa935",
            "cobalt-blue": "#385faf", "leaf-green": "#4d9864", "deep-red": "#a64047",
            "bottle-green": "#327363", "silver-blue": "#83a9bd", "royal-purple": "#76519a",
            "gold-trimmed": "#e2b94c", "bright-saffron": "#f1a02d",
        }
        scarf_color = next(
            (color for name, color in accessory_color.items() if accessory.startswith(name)),
            "#efa43d",
        )

    if action is not None:
        gesture = {
            "Wave hello": "wave",
            "Say namaste": "namaste",
            "Point out a detail": "point",
            "Take a photo": "camera",
            "Dance": "dance",
        }.get(action, gesture)

    action_text = {
        "camera": "taking a closer look at the details",
        "namaste": "greeting this landmark with a namaste",
        "point": "pointing out a detail in the architecture",
        "salute": "saluting this historic landmark",
        "wave": "waving hello from this heritage stop",
        "dance": "celebrating this heritage stop with a dance",
    }[gesture]

    girl_guide = character == "Girl"
    if girl_guide:
        saree_color = accessory.split()[0].replace("-", " ").title()
        attire = f"{saree_color} silk saree with a gold border"
        costume_svg = """
            <path d="M114 92 Q115 73 140 72 Q165 73 166 92 L178 145 Q162 157 140 157 Q118 157 102 145Z"
                  fill="{garment}" stroke="#f3d08a" stroke-width="2"/>
            <path d="M119 91 Q140 101 161 91 L157 111 Q140 119 123 111Z"
                  fill="{garment}" stroke="#ffd779" stroke-width="1.6"/>
            <path d="M119 102 Q137 111 159 99 L177 144 Q165 152 151 153 L132 122 L120 116Z"
                  fill="{sari}" stroke="#ffd779" stroke-width="2"/>
            <path d="M103 143 Q121 151 140 153 L135 158 Q116 158 100 149Z"
                  fill="#f7c84c" fill-opacity=".9"/>
            <path d="M137 115 165 151 M133 121 160 155 M128 126 153 157 M123 131 146 158"
                  fill="none" stroke="#ffd779" stroke-opacity=".7" stroke-width="1.5"/>
        """.format(garment=garment_color, sari=scarf_color)
        hair_svg = """
            <path d="M120 57 Q112 69 116 91 Q119 102 126 98 L128 76Z M158 62 Q169 74 163 94 L156 98 L153 75Z"
                  fill="#30243b"/>
            <path d="M119 51 Q117 30 141 29 Q164 31 161 53 Q153 43 140 43 Q128 43 119 51Z"
                  fill="#30243b"/>
            <path d="M119 50 Q140 38 161 50" fill="none" stroke="{sari}" stroke-width="3"/>
            <circle cx="140" cy="51" r="2.2" fill="#bf3858"/>
            <circle cx="117" cy="62" r="3" fill="#f0bd4e"/>
            <circle cx="163" cy="62" r="3" fill="#f0bd4e"/>
        """.format(sari=scarf_color)
    else:
        costume_svg = """
            <path d="M114 92 Q115 73 140 72 Q165 73 166 92 L179 143 Q169 156 140 156 Q111 156 101 143Z"
                  fill="{garment}" stroke="#f3d08a" stroke-opacity=".42" stroke-width="2"/>
            <path d="M119 92 Q128 103 141 108 Q154 102 161 92 L157 120 Q140 128 123 119Z"
                  fill="{scarf}"/>
            <path d="M126 118 Q140 124 154 118 L164 150 Q141 158 116 150Z"
                  fill="#132039" fill-opacity=".32"/>
        """.format(garment=garment_color, scarf=scarf_color)
        hair_svg = """
            <path d="M119 52 Q117 30 141 29 Q164 31 161 53 Q153 44 140 44 Q128 44 119 52Z"
                  fill="#29263b"/>
            <path d="M117 43 Q140 31 164 43 L160 50 Q141 44 120 51Z"
                  fill="{scarf}" stroke="#ffe0a0" stroke-width="2"/>
        """.format(scarf=scarf_color)

    if gesture == "namaste":
        arms = """
            <path d="M96 143 Q111 129 137 125 L143 151 Q124 147 113 162Z" fill="{garment}"/>
            <path d="M183 143 Q169 129 145 125 L140 151 Q158 149 169 162Z" fill="{garment}"/>
            <path d="M135 129 Q140 121 145 129 L145 145 Q140 150 135 145Z" fill="{skin}"/>
        """.format(garment=garment_color, skin="#c88461")
    elif gesture == "point":
        arms = """
            <path d="M97 143 Q110 130 128 129 L137 148 113 162Z" fill="{garment}"/>
            <path class="point-arm" d="M174 141 Q187 120 195 102 L215 78 Q220 72 223 78 L213 96 Q222 91 225 95 Q228 99 219 104 L194 151 Q182 154 174 141Z" fill="{garment}"/>
            <path d="M210 82 222 70 Q226 67 228 71 L218 89Z" fill="{skin}"/>
        """.format(garment=garment_color, skin="#c88461")
    elif gesture == "salute":
        arms = """
            <path d="M97 143 Q110 130 128 129 L137 148 113 162Z" fill="{garment}"/>
            <path d="M174 142 Q187 129 191 116 L177 102 190 93 Q208 101 206 119 L192 151Z" fill="{garment}"/>
            <path d="M183 103 Q187 85 182 75 L194 72 Q204 86 198 105Z" fill="{skin}"/>
        """.format(garment=garment_color, skin="#c88461")
    elif gesture == "camera":
        arms = """
            <path d="M97 143 Q110 130 128 129 L139 148 113 162Z" fill="{garment}"/>
            <path d="M174 142 Q164 132 157 129 L147 148 Q164 153 176 165Z" fill="{garment}"/>
            <path d="M126 137 Q131 130 137 136 L143 144 138 150 129 145Z" fill="{skin}"/>
            <path d="M157 136 Q162 131 168 137 L169 146 159 150 153 144Z" fill="{skin}"/>
            <rect x="132" y="128" width="31" height="22" rx="5" fill="#111d38" stroke="#f4c665" stroke-width="2"/>
            <circle cx="147.5" cy="139" r="6.5" fill="#73d6d0" stroke="#f5ead0" stroke-width="2"/>
            <rect x="140" y="124" width="9" height="5" rx="2" fill="#f4c665"/>
        """.format(garment=garment_color, skin="#c88461")
    elif gesture == "dance":
        arms = """
            <path d="M99 142 Q111 130 128 128 L139 148 114 160Z" fill="{garment}"/>
            <g class="wave-arm">
                <path d="M174 142 Q186 124 190 105 L181 88 194 82 Q208 97 203 119 L191 151Z" fill="{garment}"/>
                <path d="M184 89 Q186 77 179 66 Q175 60 181 58 Q187 58 191 68 L196 78 Q196 65 202 67 Q207 68 203 81 L200 96Z" fill="{skin}"/>
            </g>
            <path d="M121 158 Q112 171 103 168 M158 157 Q169 169 178 165"
                  fill="none" stroke="#f3c75b" stroke-width="3" stroke-linecap="round"/>
        """.format(garment=garment_color, skin="#c88461")
    else:
        arms = """
            <path d="M99 141 Q111 130 128 128 L139 149 114 162Z" fill="{garment}"/>
            <g class="wave-arm">
                <path d="M174 142 Q188 127 188 109 L178 93 192 86 Q208 99 202 119 L191 151Z" fill="{garment}"/>
                <path d="M183 94 Q184 82 178 72 Q173 64 179 61 Q185 59 189 68 L194 80 Q194 67 200 68 Q205 69 202 80 L199 97Z" fill="{skin}"/>
            </g>
        """.format(garment=garment_color, skin="#c88461")

    arms = "".join(line.strip() for line in arms.splitlines())
    outfit_svg = "".join(line.strip() for line in costume_svg.splitlines())
    hair_svg = "".join(line.strip() for line in hair_svg.splitlines())
    attire_text = html.escape(attire)
    place_text = html.escape(place)
    character_text = "Girl heritage guide" if girl_guide else "Boy heritage guide"
    return f"""
        <div class="guide-card">
            <div class="guide-card-top">
                <span>Heritage trail companion</span>
                <span class="guide-live">LIVE GUIDE</span>
            </div>
            <h2>{place_text}</h2>
            <p>{character_text} is {action_text}.</p>
            <span class="guide-outfit">✦ Dressed in {attire_text}</span>
            <svg viewBox="0 0 300 220" role="img" aria-label="Animated {character_text} in traditional attire">
                <defs>
                    <linearGradient id="archGlow" x1="0" x2="1" y1="0" y2="1">
                        <stop offset="0" stop-color="{scarf_color}" stop-opacity=".23"/>
                        <stop offset="1" stop-color="#7895ff" stop-opacity=".06"/>
                    </linearGradient>
                </defs>
                <path d="M39 175 V94 Q39 35 97 35 H183 Q241 35 241 94 V175Z" fill="url(#archGlow)" stroke="#e7bf75" stroke-opacity=".55" stroke-width="3"/>
                <path d="M61 175 V96 Q61 56 99 56 H181 Q219 56 219 96 V175" fill="none" stroke="#9eacc1" stroke-opacity=".35" stroke-width="2"/>
                <circle cx="213" cy="41" r="14" fill="#ffe3a0" fill-opacity=".92"/>
                <circle cx="219" cy="37" r="14" fill="#17233e"/>
                <path d="M70 69l3 7 7 3-7 3-3 7-3-7-7-3 7-3z" fill="#ffc96a"/>
                <path d="M204 105l2 5 5 2-5 2-2 5-2-5-5-2 5-2z" fill="#70ded3"/>
                <circle cx="90" cy="39" r="2" fill="#ffc96a"/>
                <circle cx="171" cy="27" r="2" fill="#aebcff"/>
                <path d="M35 177 H280" stroke="#a6b7cc" stroke-opacity=".32" stroke-width="3" stroke-linecap="round"/>
                <ellipse cx="143" cy="180" rx="46" ry="7" fill="#030916" fill-opacity=".4"/>
                <g class="guide-body">
                    {outfit_svg}
                    <path d="M124 151 L121 173 H136 L141 153 M145 153 L151 173 H166 L158 151" fill="#d0b891"/>
                    <path d="M120 173 H138 M150 173 H169" stroke="#e0b969" stroke-width="5" stroke-linecap="round"/>
                    <circle cx="140" cy="55" r="22" fill="#c88461"/>
                    {hair_svg}
                    <circle cx="132" cy="56" r="2" fill="#172033"/>
                    <circle cx="148" cy="56" r="2" fill="#172033"/>
                    <path d="M135 67 Q141 72 147 67" fill="none" stroke="#713f3b" stroke-width="2" stroke-linecap="round"/>
                    {arms}
                </g>
                <circle cx="102" cy="99" r="4" fill="#ffc96a"><animate attributeName="opacity" values=".35;1;.35" dur="2.6s" repeatCount="indefinite"/></circle>
                <circle cx="193" cy="61" r="3" fill="#70ded3"><animate attributeName="opacity" values="1;.3;1" dur="3.1s" repeatCount="indefinite"/></circle>
                <g class="folk-buddy">
                    <ellipse cx="228" cy="177" rx="15" ry="4" fill="#47325b" fill-opacity=".16"/>
                    <path d="M223 158 220 173 M232 158 236 173" stroke="#754c35" stroke-width="4" stroke-linecap="round"/>
                    <path d="M218 172h5 M234 172h5" stroke="#c8465d" stroke-width="3" stroke-linecap="round"/>
                    <path d="M217 137 Q227 131 238 138 L241 159 Q228 165 214 158Z" fill="#159384" stroke="#fff0bb" stroke-width="1.5"/>
                    <path d="M218 143 Q228 151 239 143 L238 150 Q228 157 218 150Z" fill="#f3ba43"/>
                    <circle cx="228" cy="124" r="11" fill="#bd7958"/>
                    <path d="M216 122 Q215 111 228 110 Q242 111 240 122 Q228 118 216 122Z" fill="#49314e"/>
                    <path d="M215 117 Q228 108 241 117 L238 121 Q228 118 217 121Z" fill="#db4c64" stroke="#ffd56c" stroke-width="1.5"/>
                    <circle cx="224" cy="125" r="1.2" fill="#32203c"/><circle cx="232" cy="125" r="1.2" fill="#32203c"/>
                    <path d="M225 131 Q228 134 232 131" fill="none" stroke="#723d42" stroke-width="1.3" stroke-linecap="round"/>
                    <path d="M216 141 210 151 M238 141 243 150" stroke="#159384" stroke-width="5" stroke-linecap="round"/>
                    <g class="folk-buddy-arm">
                        <path d="M243 150 248 139" stroke="#bd7958" stroke-width="4" stroke-linecap="round"/>
                        <path d="m247 140 3-8" stroke="#d58c35" stroke-width="2" stroke-linecap="round"/>
                    </g>
                    <path d="M207 132q-5-5-1-9 5-3 9 2M249 127q5-5 9-1 3 5-2 9" fill="#f5bd48" stroke="#9d543b" stroke-width="1.5"/>
                </g>
                <g class="folk-buddy" style="animation-delay:.25s">
                    <ellipse cx="267" cy="178" rx="8" ry="3" fill="#47325b" fill-opacity=".15"/>
                    <path d="M262 164Q254 154 260 146Q267 139 274 147Q279 155 270 164Z" fill="#118e84"/>
                    <path d="M267 149Q260 139 263 132Q270 136 271 147M270 149Q274 137 280 136Q280 145 274 151" fill="#199e8d" stroke="#08766e" stroke-width="1"/>
                    <circle cx="269" cy="151" r="6" fill="#13a093"/>
                    <circle cx="272" cy="150" r="1.3" fill="#fff7dc"/>
                    <circle cx="272.4" cy="150" r=".7" fill="#342044"/>
                    <path d="m275 152 7 2-7 2Z" fill="#e39330"/>
                    <path d="M266 164 265 173 M271 164 274 173" stroke="#b47535" stroke-width="1.7" stroke-linecap="round"/>
                    <circle cx="261" cy="159" r="2" fill="#f1c352"/><circle cx="276" cy="159" r="2" fill="#e64f66"/>
                </g>
            </svg>
        </div>
    """


try:
    model, class_names = load_model_and_labels()
except Exception as exc:
    st.error(f"Could not load model files: {exc}")
    st.stop()

st.markdown(
    """
    <section class="heritage-hero">
        <div class="hero-copy">
            <div class="hero-eyebrow">A little wonder, a lot of heritage</div>
            <h1 class="hero-title">India, one landmark<br>at a <span>time.</span></h1>
            <p class="hero-subtitle">
                Meet your animated travel companion. Share a monument photo and
                explore the stories, colours, and architecture of incredible India.
            </p>
            <div class="hero-chips">
                <span class="hero-chip">✦ 24 heritage landmarks</span>
                <span class="hero-chip">✦ Your AI-powered guide</span>
                <span class="hero-chip">✦ Images up to 20 MB</span>
            </div>
        </div>
        <div class="hero-art" aria-hidden="true">
            <svg viewBox="0 0 280 205">
                <path d="M39 175 V94 Q39 35 97 35 H183 Q241 35 241 94 V175Z" fill="rgba(255,173,77,.11)" stroke="#e7bf75" stroke-opacity=".55" stroke-width="3"/>
                <path d="M61 175 V96 Q61 56 99 56 H181 Q219 56 219 96 V175" fill="none" stroke="#9eacc1" stroke-opacity=".35" stroke-width="2"/>
                <circle cx="213" cy="41" r="14" fill="#ffe3a0" fill-opacity=".92"/><circle cx="219" cy="37" r="14" fill="#17233e"/>
                <path d="M70 69l3 7 7 3-7 3-3 7-3-7-7-3 7-3z" fill="#ffc96a"/>
                <path d="M204 105l2 5 5 2-5 2-2 5-2-5-5-2 5-2z" fill="#70ded3"/>
                <path d="M35 177 H247" stroke="#a6b7cc" stroke-opacity=".32" stroke-width="3" stroke-linecap="round"/>
                <ellipse cx="143" cy="180" rx="46" ry="7" fill="#030916" fill-opacity=".4"/>
                <g class="guide-body">
                    <path d="M114 92 Q115 73 140 72 Q165 73 166 92 L179 143 Q169 156 140 156 Q111 156 101 143Z" fill="#285e86" stroke="#f3d08a" stroke-opacity=".42" stroke-width="2"/>
                    <path d="M119 92 Q128 103 141 108 Q154 102 161 92 L157 120 Q140 128 123 119Z" fill="#f3a33b"/>
                    <path d="M124 151 L121 173 H136 L141 153 M145 153 L151 173 H166 L158 151" fill="#d0b891"/>
                    <path d="M120 173 H138 M150 173 H169" stroke="#e0b969" stroke-width="5" stroke-linecap="round"/>
                    <circle cx="140" cy="55" r="22" fill="#c88461"/>
                    <path d="M119 52 Q117 30 141 29 Q164 31 161 53 Q153 44 140 44 Q128 44 119 52Z" fill="#29263b"/>
                    <path d="M117 43 Q140 31 164 43 L160 50 Q141 44 120 51Z" fill="#f3a33b" stroke="#ffe0a0" stroke-width="2"/>
                    <circle cx="132" cy="56" r="2" fill="#172033"/><circle cx="148" cy="56" r="2" fill="#172033"/>
                    <path d="M135 67 Q141 72 147 67" fill="none" stroke="#713f3b" stroke-width="2" stroke-linecap="round"/>
                    <path d="M99 141 Q111 130 128 128 L139 149 114 162Z" fill="#285e86"/>
                    <g class="wave-arm">
                        <path d="M174 142 Q188 127 188 109 L178 93 192 86 Q208 99 202 119 L191 151Z" fill="#285e86"/>
                        <path d="M183 94 Q184 82 178 72 Q173 64 179 61 Q185 59 189 68 L194 80 Q194 67 200 68 Q205 69 202 80 L199 97Z" fill="#c88461"/>
                    </g>
                </g>
                <g class="folk-buddy">
                    <path d="M221 158 219 173 M231 158 235 173" stroke="#754c35" stroke-width="4" stroke-linecap="round"/>
                    <path d="M216 137 Q227 131 238 138 L241 159 Q228 165 214 158Z" fill="#159384" stroke="#fff0bb" stroke-width="1.5"/>
                    <path d="M218 143 Q228 151 239 143 L238 150 Q228 157 218 150Z" fill="#f3ba43"/>
                    <circle cx="228" cy="124" r="11" fill="#bd7958"/>
                    <path d="M216 122 Q215 111 228 110 Q242 111 240 122 Q228 118 216 122Z" fill="#49314e"/>
                    <path d="M215 117 Q228 108 241 117 L238 121 Q228 118 217 121Z" fill="#db4c64" stroke="#ffd56c" stroke-width="1.5"/>
                    <circle cx="224" cy="125" r="1.2" fill="#32203c"/><circle cx="232" cy="125" r="1.2" fill="#32203c"/>
                    <path d="M225 131 Q228 134 232 131" fill="none" stroke="#723d42" stroke-width="1.3" stroke-linecap="round"/>
                    <g class="folk-buddy-arm">
                        <path d="M243 150 248 139" stroke="#bd7958" stroke-width="4" stroke-linecap="round"/>
                        <path d="m247 140 3-8" stroke="#d58c35" stroke-width="2" stroke-linecap="round"/>
                    </g>
                    <path d="M207 132q-5-5-1-9 5-3 9 2M249 127q5-5 9-1 3 5-2 9" fill="#f5bd48" stroke="#9d543b" stroke-width="1.5"/>
                </g>
                <g class="folk-buddy" style="animation-delay:.25s">
                    <path d="M262 164 261 173 M270 164 273 173" stroke="#b47535" stroke-width="1.7" stroke-linecap="round"/>
                    <path d="M262 164Q254 154 260 146Q267 139 274 147Q279 155 270 164Z" fill="#118e84"/>
                    <path d="M267 149Q260 139 263 132Q270 136 271 147M270 149Q274 137 280 136Q280 145 274 151" fill="#199e8d" stroke="#08766e" stroke-width="1"/>
                    <circle cx="269" cy="151" r="6" fill="#13a093"/>
                    <circle cx="272" cy="150" r="1.2" fill="#fff7dc"/>
                    <path d="m275 152 7 2-7 2Z" fill="#e39330"/>
                </g>
            </svg>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

upload_column, guide_column = st.columns([1.08, 0.92], gap="large")
with upload_column:
    st.markdown(
        """
        <div class="section-kicker">Your next stop</div>
        <div class="section-heading">Identify a monument</div>
        <p class="upload-hint">Add a clear JPG, PNG, WEBP, or BMP image. Maximum size: 20 MB.</p>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose a monument image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Choose one image up to 20 MB.",
    )

    classify_clicked = st.button(
        "✦  Discover this monument",
        type="primary",
        disabled=uploaded is None,
        use_container_width=True,
    )

    image = None
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error("This image is over 20 MB. Please choose a smaller image.")
        else:
            try:
                image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
                render_uploaded_preview(image)
            except (UnidentifiedImageError, OSError):
                st.error("Could not read this image. Please try a different JPG or PNG.")
                image = None
    else:
        image = None

upload_identity = None
if uploaded is not None:
    upload_identity = (
        f"{getattr(uploaded, 'file_id', uploaded.name)}:"
        f"{uploaded.name}:{uploaded.size}"
    )

saved_result = st.session_state.get("landmark_result")
if saved_result is not None and saved_result["upload_identity"] != upload_identity:
    st.session_state.pop("landmark_result", None)
    saved_result = None

with guide_column:
    st.markdown("### Make the guide yours")
    guide_character = st.radio(
        "Choose your guide",
        options=["Boy", "Girl (saree)"],
        horizontal=True,
        key="guide_character",
    )
    action_labels = [
        "Wave hello",
        "Say namaste",
        "Point out a detail",
        "Take a photo",
        "Dance",
    ]
    if "guide_action" not in st.session_state:
        st.session_state["guide_action"] = "Wave hello"
    guide_action = st.selectbox(
        "Choose an action",
        options=action_labels,
        key="guide_action",
    )
    guide_placeholder = st.empty()
    shown_place = saved_result["place"] if saved_result is not None else None
    guide_placeholder.markdown(
        render_guide(
            shown_place,
            character="Girl" if guide_character == "Girl (saree)" else "Boy",
            action=guide_action,
        ),
        unsafe_allow_html=True,
    )

if classify_clicked and uploaded is not None and image is not None:
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("This image is over 20 MB. Please choose a smaller image.")
    else:
        try:
            # Keep the notebook's preprocessing: resize to 160x160 and leave
            # 0-255 scaling to the Rescaling layer saved inside the model.
            resized = image.resize(
                (IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.BILINEAR
            )
            array = np.asarray(resized, dtype=np.float32)
            batch = np.expand_dims(array, axis=0)

            with st.spinner("Reading the details in your photo..."):
                scores = model.predict(batch, verbose=0)[0]

            order = np.argsort(scores)[::-1]
            best = int(order[0])
            predicted_place = str(class_names[best])
            landmark_story = None
            guide_error = None
            try:
                with st.spinner(
                    f"Your guide is gathering the story of "
                    f"{LANDMARK_DISPLAY_NAMES.get(predicted_place, predicted_place)}..."
                ):
                    landmark_story = generate_landmark_story(predicted_place)
            except RuntimeError as exc:
                guide_error = str(exc)
            st.session_state["landmark_result"] = {
                "upload_identity": upload_identity,
                "place": predicted_place,
                "scores": [float(score) for score in scores],
                "story": landmark_story,
                "guide_error": guide_error,
            }
            saved_result = st.session_state["landmark_result"]
            guide_placeholder.markdown(
                render_guide(
                    predicted_place,
                    character="Girl" if guide_character == "Girl (saree)" else "Boy",
                    action=guide_action,
                ),
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

if saved_result is not None and saved_result["upload_identity"] == upload_identity:
    predicted_place = saved_result["place"]
    scores = np.asarray(saved_result["scores"], dtype=np.float32)
    order = np.argsort(scores)[::-1]
    best = int(order[0])

    st.markdown("---")
    st.markdown(
        f"""
        <div class="section-kicker">Your discovery</div>
        <div class="section-heading">{html.escape(predicted_place)}</div>
        """,
        unsafe_allow_html=True,
    )
    result_column, score_column = st.columns([1.2, 0.8])
    with result_column:
        st.success(f"Your photo looks like **{predicted_place}**.")
    with score_column:
        st.metric("Model confidence", f"{float(scores[best]) * 100:.2f}%")

    if float(scores[best]) < 0.45:
        st.warning("Low model score. Treat this prediction as uncertain.")

    st.markdown("### 📖 The story behind this landmark")
    if saved_result["story"]:
        st.markdown(saved_result["story"])
        st.caption(
            "AI-generated heritage overview. Historical details can be "
            "simplified; check trusted sources for academic or travel planning."
        )
    elif saved_result["guide_error"]:
        st.warning(
            f"The monument was identified, but the AI guide could not "
            f"generate its story: {saved_result['guide_error']}"
        )

    st.markdown("### 💬 Ask your interactive guide")
    st.caption(
        "Type a question or record one. Choose the answer language, then tap "
        "“Listen” on a reply. Speech voices come from your browser/device and "
        "may not be installed for every language."
    )
    if st.session_state.get("visitor_chat_upload_identity") != upload_identity:
        st.session_state["visitor_chat_upload_identity"] = upload_identity
        st.session_state["visitor_chat_history"] = []

    selected_language = st.selectbox(
        "Guide answer language",
        options=GUIDE_LANGUAGES,
        format_func=lambda language: language[0],
        key="visitor_guide_language",
    )
    language_name, language_locale = selected_language

    text_question = st.text_input(
        "Type your question",
        placeholder=f"Ask about {LANDMARK_DISPLAY_NAMES.get(predicted_place, predicted_place)}…",
        key="visitor_question_input",
    )
    audio_question = st.audio_input(
        "Or record your question (speech language is detected automatically)",
        sample_rate=16000,
        key="visitor_question_audio",
    )
    text_column, voice_column = st.columns(2)
    with text_column:
        ask_text = st.button(
            "Ask by text",
            key="ask_visitor_text",
            disabled=not text_question.strip(),
            use_container_width=True,
        )
    with voice_column:
        ask_voice = st.button(
            "Transcribe & ask",
            key="ask_visitor_voice",
            disabled=audio_question is None,
            use_container_width=True,
        )

    new_question = None
    if ask_text:
        new_question = text_question.strip()
    elif ask_voice and audio_question is not None:
        try:
            with st.spinner("Listening to your question…"):
                new_question = transcribe_visitor_question(audio_question)
            st.info(f"Your question: {new_question}")
        except RuntimeError as exc:
            st.error(str(exc))

    if new_question:
        chat_history = st.session_state["visitor_chat_history"]
        try:
            with st.spinner(f"Your guide is answering in {language_name}…"):
                answer = answer_visitor_question(
                    predicted_place,
                    saved_result["story"] or "",
                    new_question,
                    language_name,
                    chat_history,
                )
            chat_history.extend(
                [
                    {"role": "user", "content": new_question},
                    {
                        "role": "assistant",
                        "content": answer,
                        "locale": language_locale,
                    },
                ]
            )
            st.session_state["visitor_chat_history"] = chat_history[-12:]
        except RuntimeError as exc:
            st.error(str(exc))

    for message in st.session_state["visitor_chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_spoken_answer(
                    message["content"],
                    message.get("locale", language_locale),
                )

    st.subheader("Other possible matches")
    prediction_columns = st.columns(min(3, len(class_names)))
    for column, idx in zip(prediction_columns, order[:3]):
        idx = int(idx)
        with column:
            st.metric(
                str(class_names[idx]),
                f"{float(scores[idx]) * 100:.2f}%",
            )

    with st.expander("View scores for all 24 landmarks"):
        st.bar_chart(
            {
                str(class_names[int(i)]): float(scores[int(i)])
                for i in order
            }
        )
