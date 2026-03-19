"""
app.py
------
AuthentiScan — AI vs Human Caption Detector
Streamlit frontend

Usage:
    streamlit run app.py
"""

import streamlit as st
import pickle
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AuthentiScan",
    page_icon="🔍",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

* { font-family: 'DM Mono', monospace; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.stApp {
    background-color: #0a0a0a;
    color: #f0f0f0;
}

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00ff88, #00b4d8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.result-human {
    background: linear-gradient(135deg, #0d2b1a, #0a3d1f);
    border: 1px solid #00ff88;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}

.result-ai {
    background: linear-gradient(135deg, #2b0d0d, #3d0a0a);
    border: 1px solid #ff4444;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}

.result-label-human {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #00ff88;
}

.result-label-ai {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #ff4444;
}

.confidence {
    font-size: 1rem;
    color: #aaa;
    margin-top: 0.3rem;
}

.feature-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    display: flex;
    justify-content: space-between;
}

.feature-yes { color: #ff4444; }
.feature-no  { color: #00ff88; }

.ai-source-card {
    background: #111;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.stTextArea textarea {
    background-color: #111 !important;
    color: #f0f0f0 !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00b4d8) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-size: 1rem !important;
    width: 100% !important;
    cursor: pointer !important;
}

.divider {
    border: none;
    border-top: 1px solid #222;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('results/final_model.pkl', 'rb') as f:
        final_model = pickle.load(f)
    with open('results/phase2_model.pkl', 'rb') as f:
        phase2_model = pickle.load(f)
    with open('results/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('results/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    pso_weights = np.load('results/pso_weights.npy')

    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    return final_model, phase2_model, tfidf, le, pso_weights, sbert

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
BUZZWORDS = [
    "utilize", "delve", "comprehensive", "nuanced", "tapestry",
    "elevate", "embark", "foster", "intricate", "testament",
    "vibrant", "beacon", "pivotal", "seamless", "leverage",
    "boundaries", "realm", "unleash", "curated", "innovative"
]
STARTER_VERBS = [
    "embrace", "discover", "transform", "explore", "celebrate",
    "unleash", "elevate", "unlock", "dive", "ignite", "inspire",
    "capture", "find", "let", "join", "experience"
]
EMOJI_PATTERN = re.compile(
    "[\U00010000-\U0010ffff\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF]+", flags=re.UNICODE
)

def extract_features(text, tfidf, sbert, pso_weights):
    clean = text.lower()
    clean = re.sub(r'http\S+|www\S+', '', clean)
    clean = re.sub(r'@\w+', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # TF-IDF
    f1 = tfidf.transform([clean])

    # SBERT
    f2 = csr_matrix(sbert.encode([clean]))

    # Handcrafted
    has_emoji       = 1 if EMOJI_PATTERN.search(text) else 0
    has_em_dash     = 1 if ("—" in text or "–" in text) else 0
    has_ellipsis    = 1 if ("..." in text or "…" in text) else 0
    starts_w_verb   = 1 if (text.strip().split()[0].lower().rstrip(".,!?") in STARTER_VERBS if text.strip() else False) else 0
    has_buzzwords   = 1 if any(w in text.lower() for w in BUZZWORDS) else 0
    word_count      = len(text.split())
    char_count      = len(text)

    f3 = csr_matrix([[word_count, char_count, has_emoji,
                      has_em_dash, has_ellipsis, starts_w_verb, has_buzzwords]])

    X = hstack([f1 * pso_weights[0],
                f2 * pso_weights[1],
                f3 * pso_weights[2]])

    features_info = {
        "Has Emoji"        : bool(has_emoji),
        "Has Em Dash (—)"  : bool(has_em_dash),
        "Has Ellipsis (...)" : bool(has_ellipsis),
        "Starts with AI Verb": bool(starts_w_verb),
        "Has AI Buzzwords" : bool(has_buzzwords),
        "Word Count"       : word_count,
        "Char Count"       : char_count,
    }

    return X, features_info

def get_ai_source(X, phase2_model, le):
    proba = phase2_model.predict_proba(X)[0]
    classes = le.classes_
    source_probs = dict(zip(classes, proba))
    predicted = classes[np.argmax(proba)]
    return predicted, source_probs

SOURCE_DISPLAY = {
    "ai_chatgpt"   : "ChatGPT",
    "ai_gemini"    : "Gemini",
    "ai_copilot"   : "Copilot",
    "ai_perplexity": "Perplexity"
}

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">AuthentiScan</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI vs Human Caption Detector</div>', unsafe_allow_html=True)

try:
    final_model, phase2_model, tfidf, le, pso_weights, sbert = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load models: {e}\nMake sure you've run the notebook first.")
    models_loaded = False

if models_loaded:
    caption = st.text_area(
        "Paste or type a caption below:",
        placeholder="e.g. just got back from the most magical trip ever 🌅✨ honestly needed this so bad",
        height=140,
        label_visibility="visible"
    )

    analyze_btn = st.button("🔍 Analyze Caption")

    if analyze_btn:
        if not caption.strip():
            st.warning("Please enter a caption first!")
        elif len(caption.split()) < 3:
            st.warning("Caption is too short — please enter at least 3 words.")
        else:
            with st.spinner("Analyzing..."):
                X, features_info = extract_features(caption, tfidf, sbert, pso_weights)

                # Phase 1 — Human vs AI
                proba = final_model.predict_proba(X)[0]
                prediction = final_model.predict(X)[0]
                confidence = proba[prediction] * 100

                if prediction == 0:
                    st.markdown(f"""
                    <div class="result-human">
                        <div class="result-label-human">✅ HUMAN WRITTEN</div>
                        <div class="confidence">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-ai">
                        <div class="result-label-ai">🤖 AI GENERATED</div>
                        <div class="confidence">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Phase 2 — Which AI?
                    predicted_source, source_probs = get_ai_source(X, phase2_model, le)
                    st.markdown("#### 🔎 Likely AI Source")

                    sorted_sources = sorted(source_probs.items(), key=lambda x: x[1], reverse=True)
                    for source, prob in sorted_sources:
                        display_name = SOURCE_DISPLAY.get(source, source)
                        bar_pct = prob * 100
                        is_top = source == predicted_source
                        border = "border: 1px solid #00b4d8;" if is_top else ""
                        st.markdown(f"""
                        <div class="ai-source-card" style="{border}">
                            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                                <span style="font-weight:{'700' if is_top else '400'}">
                                    {'⭐ ' if is_top else ''}{display_name}
                                </span>
                                <span style="color:#aaa">{bar_pct:.1f}%</span>
                            </div>
                            <div style="background:#222; border-radius:4px; height:6px;">
                                <div style="background:{'#00b4d8' if is_top else '#444'};
                                            width:{bar_pct:.1f}%; height:6px; border-radius:4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Features detected
                st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                st.markdown("#### 🧪 Features Detected")

                col1, col2 = st.columns(2)
                items = list(features_info.items())
                half = len(items) // 2

                for i, (feat, val) in enumerate(items):
                    col = col1 if i < half + 1 else col2
                    if isinstance(val, bool):
                        indicator = "✓" if val else "✗"
                        css_class = "feature-yes" if val else "feature-no"
                        col.markdown(f"""
                        <div class="feature-card">
                            <span>{feat}</span>
                            <span class="{css_class}">{indicator}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        col.markdown(f"""
                        <div class="feature-card">
                            <span>{feat}</span>
                            <span style="color:#aaa">{val}</span>
                        </div>
                        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#444; font-size:0.75rem;">
        AuthentiScan · PSO-Optimized Hybrid Feature Fusion · 97.47% Accuracy
    </div>
    """, unsafe_allow_html=True)