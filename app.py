import streamlit as st
import importlib
import os

st.set_page_config(
    page_title="Deep Learning Toolbox",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #1a1a26;
    --accent: #6c63ff;
    --accent2: #ff6584;
    --accent3: #43e97b;
    --accent4: #f7971e;
    --text: #e8e8f0;
    --muted: #7070a0;
    --border: #2a2a40;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; }
#MainMenu, footer, header { visibility: hidden; }

.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero h1 {
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c63ff 0%, #ff6584 50%, #43e97b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin-bottom: 0.5rem;
}
.hero p { color: var(--muted); font-size: 1.1rem; font-weight: 300; }

.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1.2rem;
    padding: 1rem 0;
}
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: var(--card-color, var(--accent));
    border-radius: 16px 16px 0 0;
}
.card:hover {
    border-color: var(--card-color, var(--accent));
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(108, 99, 255, 0.15);
    background: var(--surface2);
}
.card-icon { font-size: 2.2rem; margin-bottom: 0.8rem; }
.card-title { font-size: 1.05rem; font-weight: 600; color: var(--text); margin-bottom: 0.4rem; }
.card-desc  { font-size: 0.82rem; color: var(--muted); line-height: 1.5; }
.card-tag {
    display: inline-block;
    margin-top: 0.8rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    background: rgba(108,99,255,0.15);
    color: var(--card-color, var(--accent));
    border: 1px solid var(--card-color, var(--accent));
}

.section-title    { font-size: 1.6rem; font-weight: 700; color: var(--text); margin-bottom: 0.2rem; }
.section-subtitle { color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }

div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label { color: var(--text) !important; }

.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #a78bfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.metric-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val   { font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: var(--accent); }
.metric-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

# ── Tool Registry ─────────────────────────────────────────────────────────────
TOOLS = [
    {
        "id": "perceptron",
        "icon": "⚡",
        "title": "Single-Layer Perceptron",
        "desc": "Step-by-step perceptron learning with manual weight updates, activation functions, and decision boundary visualization.",
        "tag": "Foundational",
        "color": "#6c63ff",
        "module": "models.perceptron",
    },
    {
        "id": "multilayer_nn",
        "icon": "🧠",
        "title": "Multilayer Neural Network",
        "desc": "Build MLP from scratch: choose hidden layers, neurons, activations. Watch forward pass & error calculations per epoch.",
        "tag": "Manual Implementation",
        "color": "#ff6584",
        "module": "models.multilayer_nn",
    },
    {
        "id": "backpropagation",
        "icon": "🔄",
        "title": "Backpropagation Visualizer",
        "desc": "Step through backprop manually. See gradients, weight deltas, and chain rule unfolding at every layer.",
        "tag": "Core Algorithm",
        "color": "#f7971e",
        "module": "models.backpropagation",
    },
    {
        "id": "cnn_model",
        "icon": "🔍",
        "title": "CNN – Convolutional Network",
        "desc": "Build a CNN for image classification. Visualize feature maps, filters, and pooling on uploaded images.",
        "tag": "Computer Vision",
        "color": "#43e97b",
        "module": "models.cnn_model",
    },
    {
        "id": "rnn_sequence",
        "icon": "🔁",
        "title": "RNN – Sequence Predictor",
        "desc": "Train a character-level RNN. Input a sequence and watch it predict the next token step by step.",
        "tag": "Sequential",
        "color": "#38b2f5",
        "module": "models.rnn_sequence",
    },
    {
        "id": "hopfield",
        "icon": "🧲",
        "title": "Hopfield Network",
        "desc": "Store and retrieve binary patterns using Hebbian learning. Visualise energy, weight matrix, capacity limits, and convergence.",
        "tag": "Associative Memory",
        "color": "#e040fb",
        "module": "models.hopfield",
    },
    {
        "id": "opencv_attendance",
        "icon": "📸",
        "title": "OpenCV Attendance System",
        "desc": "Face-recognition based attendance tracker. Register faces from uploads and mark attendance automatically.",
        "tag": "OpenCV",
        "color": "#f093fb",
        "module": "models.opencv_attendance",
    },
    {
        "id": "face_counter",
        "icon": "👥",
        "title": "Face Counter & Detector",
        "desc": "Detect and count faces in images using Haar cascades. Shows bounding boxes, crops, and size stats.",
        "tag": "OpenCV",
        "color": "#4facfe",
        "module": "models.face_counter",
    },
    {
        "id": "autoencoder",
        "icon": "🗜️",
        "title": "Autoencoder",
        "desc": "Train an autoencoder for dimensionality reduction and reconstruction. Visualize the latent space in 2D.",
        "tag": "Unsupervised",
        "color": "#fa709a",
        "module": "models.autoencoder",
    },
    {
        "id": "kmeans_clustering",
        "icon": "🎯",
        "title": "K-Means Clustering",
        "desc": "Interactive K-Means with manual centroid initialization. Watch clusters evolve iteration by iteration.",
        "tag": "Clustering",
        "color": "#a18cd1",
        "module": "models.kmeans_clustering",
    },
    {
        "id": "som",
        "icon": "🗺️",
        "title": "Self-Organizing Map (SOM)",
        "desc": "Visualize high-dimensional data topology using a 2D SOM grid. Manually set grid size, learning rate, epochs.",
        "tag": "Unsupervised",
        "color": "#fd7900",
        "module": "models.som",
    },
]


def render_home():
    st.markdown("""
    <div class="hero">
        <h1>🧠 Deep Learning Toolbox</h1>
        <p>Implement, visualize, and explore deep learning algorithms — built from scratch, step by step</p>
    </div>
    """, unsafe_allow_html=True)

    cards_html = '<div class="card-grid">'
    for t in TOOLS:
        cards_html += f"""
        <div class="card" style="--card-color:{t['color']};">
            <div class="card-icon">{t['icon']}</div>
            <div class="card-title">{t['title']}</div>
            <div class="card-desc">{t['desc']}</div>
            <span class="card-tag">{t['tag']}</span>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:var(--muted);font-size:0.85rem;'>Click a button to open a tool:</p>",
                unsafe_allow_html=True)

    # 4 buttons per row
    for row_start in range(0, len(TOOLS), 4):
        row_tools = TOOLS[row_start:row_start+4]
        cols = st.columns(len(row_tools))
        for col, t in zip(cols, row_tools):
            with col:
                if st.button(f"{t['icon']} {t['title']}", key=f"btn_{t['id']}",
                             use_container_width=True):
                    st.session_state["active_tool"] = t["id"]
                    st.rerun()


def render_tool(tool_id):
    tool = next((t for t in TOOLS if t["id"] == tool_id), None)
    if tool is None:
        st.error("Tool not found.")
        return

    if st.button("← Back to Toolbox"):
        st.session_state["active_tool"] = None
        st.rerun()

    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
        <div class="section-title">{tool['icon']} {tool['title']}</div>
        <div class="section-subtitle">{tool['desc']}</div>
        <div style="height:3px;width:60px;background:{tool['color']};border-radius:2px;"></div>
    </div>
    """, unsafe_allow_html=True)

    try:
        mod = importlib.import_module(tool["module"])
        mod.run()
    except ModuleNotFoundError as e:
        st.error(f"Module not found: `{tool['module']}`\n\n{e}")
    except Exception as e:
        st.exception(e)


# ── Router ────────────────────────────────────────────────────────────────────
if "active_tool" not in st.session_state:
    st.session_state["active_tool"] = None

active = st.session_state["active_tool"]
if active:
    render_tool(active)
else:
    render_home()
