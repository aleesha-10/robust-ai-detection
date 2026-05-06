"""
AI-Generated Image Detector — Streamlit Inference App
Convolutional vs Transformer Detectors: A Robustness Study Under Real-World Distribution Shifts
NUCES, Department of AI and Data Science — Aleesha Javed & Tayyaba Amanat

Usage:
    streamlit run app.py

Requirements:
    pip install streamlit torch torchvision pillow open-clip-torch

Model checkpoints expected:
    - resnet18_best.pth         (your saved ResNet18 checkpoint)
    - clip_head_best.pth        (your saved CLIP MLP head checkpoint)

If checkpoints are missing the app still runs in demo mode with random weights
so you can see the UI — just swap in your real .pth files before submission.
"""

import io
import os
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# ── try importing open_clip; graceful fallback if not installed ──────────────
try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

/* ── root variables ── */
:root {
    --bg:        #0d0f14;
    --surface:   #151820;
    --border:    #252a35;
    --accent:    #4fffb0;
    --accent2:   #ff4f7b;
    --text:      #e8eaf0;
    --muted:     #5a6070;
    --real-col:  #4fffb0;
    --fake-col:  #ff4f7b;
}

/* global font */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── hero header ── */
.hero {
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem 2.5rem 1.5rem;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #151820 0%, #1a1f2e 100%);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(79,255,176,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 4px;
    padding: 2px 8px;
    margin-top: 0.8rem;
    letter-spacing: 1px;
}

/* ── result card ── */
.result-card {
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid var(--border);
    background: var(--surface);
}
.verdict-real {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--real-col);
}
.verdict-fake {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--fake-col);
}
.confidence-label {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.3rem;
}
.confidence-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    margin-top: 0.1rem;
}

/* ── metric mini cards ── */
.metric-row {
    display: flex;
    gap: 0.6rem;
    margin-top: 1rem;
}
.mini-card {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.7rem 0.5rem;
    text-align: center;
}
.mini-label {
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}
.mini-value {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    margin-top: 0.2rem;
}

/* ── shift panel ── */
.shift-info {
    font-size: 0.78rem;
    color: var(--muted);
    border-left: 2px solid var(--border);
    padding-left: 0.8rem;
    margin-top: 0.8rem;
    line-height: 1.6;
}

/* ── prob bar ── */
.bar-wrap {
    background: var(--border);
    border-radius: 4px;
    height: 6px;
    margin-top: 0.4rem;
    overflow: hidden;
}
.bar-fill-real { background: var(--real-col); height: 6px; border-radius: 4px; }
.bar-fill-fake { background: var(--fake-col); height: 6px; border-radius: 4px; }

/* ── section heading ── */
.section-head {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

/* ── warning / info boxes ── */
.warn-box {
    background: rgba(255,79,123,0.07);
    border: 1px solid rgba(255,79,123,0.25);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.8rem;
    color: #ff8fab;
    margin-bottom: 1rem;
}
.info-box {
    background: rgba(79,255,176,0.05);
    border: 1px solid rgba(79,255,176,0.15);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.8rem;
    color: #a8ffd8;
    margin-bottom: 1rem;
}

/* ── sidebar tweaks ── */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {
    font-size: 0.82rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ── constants ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
RESNET_SIZE   = 32
CLIP_SIZE     = 224

RESNET_CKPT = "resnet18_best.pth"
CLIP_CKPT   = "clip_head_best.pth"

# ── MLP head matching your training code ────────────────────────────────────
class CLIPHead(nn.Module):
    """3-layer MLP head attached to frozen CLIP ViT-L/14 features (dim=768)."""
    def __init__(self, in_dim: int = 768, hidden: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden is None:
            hidden = [256, 128]
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── model loading (cached so it only runs once per session) ──────────────────
@st.cache_resource(show_spinner=False)
def load_resnet(ckpt_path: str, device: str):
    """Load ResNet18 with a 2-class head. Falls back to random weights if checkpoint missing."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        # handle both raw state_dict and checkpoint dicts
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        loaded = True
    else:
        loaded = False
    model.to(device).eval()
    return model, loaded


@st.cache_resource(show_spinner=False)
def load_clip(head_ckpt_path: str, device: str):
    """Load frozen CLIP ViT-L/14 + trainable MLP head."""
    if not CLIP_AVAILABLE:
        return None, None, False

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    head = CLIPHead(in_dim=768)
    if os.path.exists(head_ckpt_path):
        state = torch.load(head_ckpt_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        head.load_state_dict(state)
        loaded = True
    else:
        loaded = False
    head.to(device).eval()
    return clip_model, head, loaded


# ── preprocessing ────────────────────────────────────────────────────────────
def get_resnet_transform(jpeg_quality: int = None, resize_factor: float = None):
    ops = []
    if resize_factor and resize_factor < 1.0:
        # bilinear down then up — same as your test protocol
        ops.append(T.Lambda(lambda img: img.resize(
            (max(1, int(img.width * resize_factor)),
             max(1, int(img.height * resize_factor))),
            Image.BILINEAR
        )))
        ops.append(T.Lambda(lambda img: img.resize(
            (RESNET_SIZE, RESNET_SIZE), Image.BILINEAR
        )))
    else:
        ops.append(T.Resize((RESNET_SIZE, RESNET_SIZE)))

    if jpeg_quality and jpeg_quality < 100:
        def apply_jpeg(img):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        ops.append(T.Lambda(apply_jpeg))

    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def get_clip_transform(jpeg_quality: int = None, resize_factor: float = None):
    ops = []
    if resize_factor and resize_factor < 1.0:
        ops.append(T.Lambda(lambda img: img.resize(
            (max(1, int(img.width * resize_factor)),
             max(1, int(img.height * resize_factor))),
            Image.BILINEAR
        )))
        ops.append(T.Lambda(lambda img: img.resize(
            (CLIP_SIZE, CLIP_SIZE), Image.BILINEAR
        )))
    else:
        ops.append(T.Resize((CLIP_SIZE, CLIP_SIZE)))

    if jpeg_quality and jpeg_quality < 100:
        def apply_jpeg(img):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        ops.append(T.Lambda(apply_jpeg))

    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


# ── inference helpers ────────────────────────────────────────────────────────
@torch.no_grad()
def run_resnet(model, img: Image.Image, transform) -> dict:
    x = transform(img.convert("RGB")).unsqueeze(0)
    logits = model(x)
    probs  = torch.softmax(logits, dim=1)[0]
    fake_p = float(probs[1])
    real_p = float(probs[0])
    label  = "AI-Generated" if fake_p >= 0.5 else "Real"
    conf   = fake_p if fake_p >= 0.5 else real_p
    return {"label": label, "fake_prob": fake_p, "real_prob": real_p, "confidence": conf}


@torch.no_grad()
def run_clip(clip_model, head, img: Image.Image, transform, device: str) -> dict:
    x = transform(img.convert("RGB")).unsqueeze(0).to(device)
    features = clip_model.encode_image(x)
    features = features / features.norm(dim=-1, keepdim=True)  # L2 normalise
    logits = head(features.float())
    probs  = torch.softmax(logits, dim=1)[0]
    fake_p = float(probs[1])
    real_p = float(probs[0])
    label  = "AI-Generated" if fake_p >= 0.5 else "Real"
    conf   = fake_p if fake_p >= 0.5 else real_p
    return {"label": label, "fake_prob": fake_p, "real_prob": real_p, "confidence": conf}


def ensemble_results(r_res: dict, r_clip: dict) -> dict:
    """Soft-vote: average softmax probabilities from both models."""
    fake_p = (r_res["fake_prob"] + r_clip["fake_prob"]) / 2
    real_p = (r_res["real_prob"] + r_clip["real_prob"]) / 2
    label  = "AI-Generated" if fake_p >= 0.5 else "Real"
    conf   = fake_p if fake_p >= 0.5 else real_p
    return {"label": label, "fake_prob": fake_p, "real_prob": real_p, "confidence": conf}


# ── result card renderer ─────────────────────────────────────────────────────
def render_result(title: str, result: dict, checkpoint_loaded: bool):
    is_fake = result["label"] == "AI-Generated"
    verdict_cls = "verdict-fake" if is_fake else "verdict-real"
    icon = "⚠️" if is_fake else "✅"
    fake_w = int(result["fake_prob"] * 100)
    real_w = int(result["real_prob"] * 100)

    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)

    if not checkpoint_loaded:
        st.markdown('<div class="warn-box">⚠️ No checkpoint found — showing demo output with random weights.</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <div class="{verdict_cls}">{icon} {result["label"]}</div>
        <div class="confidence-label">confidence</div>
        <div class="confidence-value">{result["confidence"]*100:.1f}%</div>
        <div class="metric-row">
            <div class="mini-card">
                <div class="mini-label">Real prob</div>
                <div class="mini-value" style="color:var(--real-col)">{result["real_prob"]*100:.1f}%</div>
            </div>
            <div class="mini-card">
                <div class="mini-label">Fake prob</div>
                <div class="mini-value" style="color:var(--fake-col)">{result["fake_prob"]*100:.1f}%</div>
            </div>
        </div>
        <div style="margin-top:0.8rem">
            <div style="font-size:0.65rem;color:var(--muted);text-align:left;margin-bottom:2px;">REAL</div>
            <div class="bar-wrap"><div class="bar-fill-real" style="width:{real_w}%"></div></div>
            <div style="font-size:0.65rem;color:var(--muted);text-align:left;margin:4px 0 2px;">FAKE</div>
            <div class="bar-wrap"><div class="bar-fill-fake" style="width:{fake_w}%"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Inference Settings")
    st.markdown("---")

    model_choice = st.selectbox(
        "Model",
        ["ResNet18", "CLIP ViT-L/14", "Ensemble (soft-vote)"],
        help="ResNet18: CNN baseline. CLIP: transformer. Ensemble: average of both."
    )

    st.markdown("#### Distribution Shift")
    apply_jpeg = st.checkbox("Apply JPEG compression", value=False)
    jpeg_q = 95
    if apply_jpeg:
        jpeg_q = st.select_slider(
            "JPEG quality",
            options=[95, 75, 50, 30, 10],
            value=75,
        )

    apply_resize = st.checkbox("Apply spatial resize shift", value=False)
    resize_f = 1.0
    if apply_resize:
        resize_f = st.select_slider(
            "Resize factor",
            options=[0.75, 0.5, 0.25],
            value=0.75,
        )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:var(--muted);line-height:1.7">
    <b style="color:var(--text)">Expected checkpoints:</b><br>
    <code>resnet18_best.pth</code><br>
    <code>clip_head_best.pth</code><br><br>
    Place them in the same folder as <code>app.py</code>.<br><br>
    Resize shift = bilinear down + up,<br>same as test-time protocol in the paper.
    </div>
    """, unsafe_allow_html=True)

# ── main layout ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🔍 AI Image Detector</div>
    <p class="hero-sub">Convolutional vs Transformer Detectors — Robustness Study</p>
    <p class="hero-sub">NUCES · Aleesha Javed · Tayyaba Amanat · 2026</p>
    <span class="hero-badge">CIFAKE · ResNet18 · CLIP ViT-L/14</span>
</div>
""", unsafe_allow_html=True)

# ── device ───────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
st.markdown(
    f'<div class="info-box">Running on <b>{device.upper()}</b>. '
    f'{"CUDA available — CLIP inference will be fast." if device=="cuda" else "No GPU detected — CLIP ViT-L/14 may be slow on CPU (10-30s per image). Consider using ResNet18 for quick demos."}'
    f'</div>',
    unsafe_allow_html=True
)

# ── file uploader ─────────────────────────────────────────────────────────────
col_upload, col_results = st.columns([1, 1.4], gap="large")

with col_upload:
    st.markdown('<div class="section-head">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop an image here",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True, caption=f"{uploaded.name}  ·  {img.width}×{img.height}px")

        # show what shift will be applied
        shift_desc = []
        if apply_jpeg:
            shift_desc.append(f"JPEG q={jpeg_q}")
        if apply_resize:
            shift_desc.append(f"resize ×{resize_f}")
        if shift_desc:
            st.markdown(
                f'<div class="shift-info">⚙️ Shift applied at inference: {", ".join(shift_desc)}</div>',
                unsafe_allow_html=True
            )

with col_results:
    if uploaded:
        st.markdown('<div class="section-head">Results</div>', unsafe_allow_html=True)

        jpeg_arg   = jpeg_q   if apply_jpeg   else None
        resize_arg = resize_f if apply_resize else None

        need_resnet = model_choice in ["ResNet18", "Ensemble (soft-vote)"]
        need_clip   = model_choice in ["CLIP ViT-L/14", "Ensemble (soft-vote)"]

        r_resnet = None
        r_clip   = None

        with st.spinner("Running inference…"):

            # ── ResNet18 inference ──
            if need_resnet:
                resnet_model, resnet_loaded = load_resnet(RESNET_CKPT, device)
                transform_r = get_resnet_transform(
                    jpeg_quality=jpeg_arg,
                    resize_factor=resize_arg
                )
                r_resnet = run_resnet(resnet_model, img, transform_r)

            # ── CLIP inference ──
            if need_clip:
                if not CLIP_AVAILABLE:
                    st.markdown(
                        '<div class="warn-box">open_clip not installed. '
                        'Run: <code>pip install open-clip-torch</code></div>',
                        unsafe_allow_html=True
                    )
                else:
                    clip_model, clip_head, clip_loaded = load_clip(CLIP_CKPT, device)
                    transform_c = get_clip_transform(
                        jpeg_quality=jpeg_arg,
                        resize_factor=resize_arg
                    )
                    r_clip = run_clip(clip_model, clip_head, img, transform_c, device)

        # ── render results ──
        if model_choice == "ResNet18" and r_resnet:
            render_result("ResNet18", r_resnet, resnet_loaded)

        elif model_choice == "CLIP ViT-L/14" and r_clip:
            render_result("CLIP ViT-L/14", r_clip, clip_loaded)

        elif model_choice == "Ensemble (soft-vote)":
            if r_resnet and r_clip:
                r_ens = ensemble_results(r_resnet, r_clip)
                render_result("Ensemble (soft-vote)", r_ens, resnet_loaded and clip_loaded)

                # show individual model breakdown below
                st.markdown("<br>", unsafe_allow_html=True)
                sub1, sub2 = st.columns(2)
                with sub1:
                    render_result("ResNet18", r_resnet, resnet_loaded)
                with sub2:
                    if r_clip:
                        render_result("CLIP ViT-L/14", r_clip, clip_loaded)
            elif r_resnet:
                render_result("ResNet18 (CLIP unavailable)", r_resnet, resnet_loaded)

    else:
        st.markdown("""
        <div style="height:300px;display:flex;align-items:center;justify-content:center;
                    border:1px dashed var(--border);border-radius:12px;color:var(--muted);
                    font-size:0.85rem;text-align:center;padding:2rem">
            Upload an image on the left<br>to run inference
        </div>
        """, unsafe_allow_html=True)

# ── quick reference table ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📊 Expected AUC by condition (from the paper)", expanded=False):
    st.markdown("""
    | Condition | ResNet18 | ResNet18+Aug | CLIP ViT-L/14 | Ensemble |
    |-----------|----------|--------------|---------------|----------|
    | Baseline (clean) | 0.9957 | 0.9964 | 0.9929 | **0.9912** |
    | JPEG q=95 | 0.9956 | 0.9962 | 0.9928 | 0.9911 |
    | JPEG q=75 | 0.9957 | 0.9963 | 0.9927 | 0.9908 |
    | JPEG q=50 | 0.9923 | 0.9936 | 0.9855 | 0.9827 |
    | JPEG q=30 | 0.9895 | 0.9922 | 0.9728 | 0.9672 |
    | JPEG q=10 | 0.9516 | 0.9644 | 0.8152 | 0.7996 |
    | Resize ×0.75 | 0.9207 | **0.9924** | 0.9762 | 0.9475 |
    | Resize ×0.5 | 0.8117 | 0.9412 | **0.9157** | 0.8194 |
    | Resize ×0.25 | 0.7297 | 0.7677 | 0.6965 | 0.6568 |
    | Generator shift (BigGAN) | 0.7445 | — | — | — |
    | Domain+Generator (StyleGAN faces) | 0.4890 | — | — | — |

    *Multi-scale augmentation recovers 94% of resize-induced performance loss at ×0.75.*
    """)

st.markdown("""
<div style="text-align:center;color:var(--muted);font-size:0.72rem;margin-top:2rem;
            border-top:1px solid var(--border);padding-top:1rem">
    NUCES · Department of AI and Data Science · 2026 &nbsp;|&nbsp;
    Aleesha Javed &amp; Tayyaba Amanat
</div>
""", unsafe_allow_html=True)