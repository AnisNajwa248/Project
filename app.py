"""
Fake News Detection — Proof-of-Concept Web Application
Integrates Logistic Regression, RoBERTa, and a zero-shot LLM (openai/gpt-oss-120b via Groq)
into a single Streamlit interface, per Chapter 3, Section 3.2.8.

Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub, then deploy on streamlit.io (Streamlit Cloud)
"""

import os
import re
import time

import streamlit as st

# ── Page setup ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detection PoC", page_icon="📰", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MAX_PROMPT_CHARS = 1000  # matches the truncation used during evaluation (Section 3.2.2)
PROMPT_TEMPLATE = (
    "Classify the following news article as real or fake. "
    "Respond with exactly one word: 'real' or 'fake'. Do not explain your reasoning. "
    "Article: {text}. Classification:"
)

MODEL_META = {
    "lr": {"name": "Logistic Regression", "icon": "📊", "accent": "#5b8ac6"},
    "roberta": {"name": "RoBERTa", "icon": "🤖", "accent": "#1f3a5f"},
    "llm": {"name": "LLM · gpt-oss-120b", "icon": "🧠", "accent": "#c9922f"},
}
LABEL_STYLE = {
    "REAL": {"color": "#22c55e", "bg": "rgba(34,197,94,0.12)"},
    "FAKE": {"color": "#ef4444", "bg": "rgba(239,68,68,0.12)"},
    "UNCERTAIN": {"color": "#f59e0b", "bg": "rgba(245,158,11,0.12)"},
}

# ── Styling ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 1.08rem;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] label {
        font-size: 1.1rem !important;
    }
    textarea {
        font-size: 1.1rem !important;
    }
    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #1f3a5f 0%, #2d5691 55%, #5b8ac6 100%);
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(31,58,95,0.35);
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .hero p {
        color: rgba(255,255,255,0.85);
        font-size: 1.3rem;
        margin: 0.5rem 0 0 0;
    }
    .pill-row { margin-top: 1rem; }
    .pill {
        display: inline-block;
        padding: 0.35rem 1rem;
        margin-right: 0.5rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 600;
    }
    div[data-testid="stButton"] button {
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 0.7rem 1.8rem;
        border: none;
        background: linear-gradient(135deg, #ef4444, #c9922f);
        transition: transform 0.15s ease;
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-1px);
        filter: brightness(1.05);
    }
    .result-card {
        border-radius: 16px;
        padding: 1.3rem 1.4rem 1.5rem 1.4rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-top: 4px solid var(--accent);
    }
    .result-card .model-name {
        font-size: 1.25rem;
        font-weight: 700;
        opacity: 0.9;
        margin-bottom: 0.7rem;
    }
    .badge {
        display: inline-block;
        padding: 0.45rem 1.1rem;
        border-radius: 10px;
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: 0.03em;
        color: var(--label-color);
        background: var(--label-bg);
        margin-bottom: 0.8rem;
    }
    .meta-line {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📰 Fake News Detection</h1>
        <p>Comparing traditional NLP, transformer-based, and zero-shot LLM approaches on the same text.</p>
        <div class="pill-row">
            <span class="pill">📊 Logistic Regression</span>
            <span class="pill">🤖 RoBERTa</span>
            <span class="pill">🧠 gpt-oss-120b (zero-shot)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Shared text cleaning (matches the notebook's preprocessing, Section 3.2.2) ──────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Cached model loaders — each returns None if the artifact isn't present yet ──────────
@st.cache_resource
def load_lr_model():
    import pickle

    model_path = os.path.join(MODELS_DIR, "lr_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        return None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


@st.cache_resource
def load_roberta_model():
    roberta_path = os.path.join(MODELS_DIR, "roberta_final")
    if not os.path.isdir(roberta_path):
        return None, None
    from transformers import RobertaForSequenceClassification, RobertaTokenizer as RobertaTokenizerFast

    tokenizer = RobertaTokenizerFast.from_pretrained(roberta_path)
    model = RobertaForSequenceClassification.from_pretrained(roberta_path)
    model.eval()
    return model, tokenizer


# ── Classification functions ─────────────────────────────────────────────────────────────
def classify_lr(text, model, vectorizer):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    start = time.time()
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    elapsed = time.time() - start
    return pred, float(max(proba)), elapsed


def classify_roberta(text, model, tokenizer):
    import torch

    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
    start = time.time()
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs))
    elapsed = time.time() - start
    return pred, float(probs[pred]), elapsed


def classify_llm(text, api_key):
    from groq import Groq

    client = Groq(api_key=api_key)
    # Matches the notebook's evaluation pipeline (Section 3.2.2), where the LLM was tested
    # on the same cleaned 'combined_text' column as Logistic Regression and RoBERTa, not on
    # raw text. Cleaning here keeps this PoC's "same text, three models" comparison accurate.
    cleaned = clean_text(text)
    prompt = PROMPT_TEMPLATE.format(text=cleaned[:MAX_PROMPT_CHARS])
    start = time.time()
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
        reasoning_effort="low",
    )
    elapsed = time.time() - start
    answer = (resp.choices[0].message.content or "").strip().lower()
    pred = 1 if "real" in answer else 0 if "fake" in answer else None
    return pred, elapsed


def label_for(pred):
    if pred == 1:
        return "REAL"
    if pred == 0:
        return "FAKE"
    return "UNCERTAIN"


def render_result_card(model_key, label, confidence=None, elapsed=None, elapsed_unit="ms"):
    meta = MODEL_META[model_key]
    style = LABEL_STYLE[label]
    # NOTE: these must be single-line strings with NO leading whitespace.
    # Streamlit's markdown renderer converts Markdown -> HTML before honoring
    # unsafe_allow_html, and any line indented 4+ spaces is treated as a
    # Markdown code block (wrapped in <pre><code>) instead of raw HTML.
    conf_html = ""
    if confidence is not None:
        conf_html = f'<div class="meta-line">Confidence: <b>{confidence:.1%}</b></div>'
    time_html = ""
    if elapsed is not None:
        time_html = f'<div class="meta-line">Inference time: <b>{elapsed}</b> {elapsed_unit}</div>'

    card_html = (
        f'<div class="result-card" style="--accent:{meta["accent"]}; '
        f'--label-color:{style["color"]}; --label-bg:{style["bg"]};">'
        f'<div class="model-name">{meta["icon"]} {meta["name"]}</div>'
        f'<span class="badge">{label}</span>'
        f'{conf_html}'
        f'{time_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)
    if confidence is not None:
        st.progress(min(max(confidence, 0.0), 1.0))


# ── Sidebar: model selection + LLM key ───────────────────────────────────────────────────
st.sidebar.header("Models to run")
use_lr = st.sidebar.checkbox("📊 Logistic Regression", value=True)
use_roberta = st.sidebar.checkbox("🤖 RoBERTa", value=True)
use_llm = st.sidebar.checkbox("🧠 LLM (gpt-oss-120b)", value=True)

groq_api_key = None
if use_llm:
    # Priority: Streamlit secrets (deployment) > environment variable > manual entry.
    default_key = ""
    try:
        if "GROQ_API_KEY" in st.secrets:
            default_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not default_key and os.environ.get("GROQ_API_KEY"):
        default_key = os.environ["GROQ_API_KEY"]

    if default_key:
        groq_api_key = default_key
        if st.sidebar.checkbox("Use a different key instead"):
            groq_api_key = st.sidebar.text_input("Groq API key", type="password")
    else:
        groq_api_key = st.sidebar.text_input("Groq API key", type="password")

# ── Main input ────────────────────────────────────────────────────────────────────────────
# NOTE: LR and RoBERTa were trained/evaluated on ISOT/WELFake text built as
# "title + ' ' + body" (see notebook's load_isot()/load_welfake(), Section 3.2.1.2/3.2.2).
# There's no separate headline field here, so we ask the user to paste the headline and
# body together (headline first) in one box, which reproduces that same concatenated
# format after clean_text() runs on it. LIAR-style short claims are unaffected either way,
# since LIAR's combined_text was never built from a title in the first place.
user_text = st.text_area(
    "Paste the headline and article text together (headline first), or a short claim:",
    height=180,
    placeholder="e.g. Scientists Confirm Moon Now Visible At Night. Astronomers today announced "
    "that the moon, long a fixture of daytime skies only, will now also be observable after dark...",
)
st.caption(
    "For best results with the RoBERTa and Logistic Regression models, include the headline "
    "as the first line, followed by the body text — this matches how they were trained."
)
classify_clicked = st.button("Classify", type="primary")

if classify_clicked:
    if not user_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        selected = [m for m, on in [("lr", use_lr), ("roberta", use_roberta), ("llm", use_llm)] if on]
        if not selected:
            st.warning("Select at least one model in the sidebar.")
        else:
            st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)
            cols = st.columns(len(selected))
            col_idx = 0

            if use_lr:
                with cols[col_idx]:
                    model, vectorizer = load_lr_model()
                    if model is None:
                        st.error("Model files not found in `models/`.")
                    else:
                        pred, conf, elapsed = classify_lr(user_text, model, vectorizer)
                        render_result_card("lr", label_for(pred), confidence=conf, elapsed=f"{elapsed * 1000:.2f}")
                col_idx += 1

            if use_roberta:
                with cols[col_idx]:
                    model, tokenizer = load_roberta_model()
                    if model is None:
                        st.error("Checkpoint not found in `models/roberta_final/`.")
                    else:
                        pred, conf, elapsed = classify_roberta(user_text, model, tokenizer)
                        render_result_card("roberta", label_for(pred), confidence=conf, elapsed=f"{elapsed * 1000:.2f}")
                col_idx += 1

            if use_llm:
                with cols[col_idx]:
                    if not groq_api_key:
                        st.warning("Enter your Groq API key in the sidebar.")
                    else:
                        try:
                            pred, elapsed = classify_llm(user_text, groq_api_key)
                            render_result_card("llm", label_for(pred), confidence=None, elapsed=f"{elapsed:.2f}", elapsed_unit="s")
                        except Exception as e:
                            st.error(f"Groq API error: {e}")
                col_idx += 1
