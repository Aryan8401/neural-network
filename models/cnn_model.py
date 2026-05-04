"""
CNN — Convolutional Neural Network.
Uses PyTorch if available, otherwise shows a helpful install guide.
DLL errors are caught gracefully so the rest of the app keeps working.
"""

import streamlit as st
import numpy as np

# ── Safe PyTorch import (catches DLL / CUDA errors on Windows) ───────────────
TORCH_OK = False
TORCH_ERROR = ""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    TORCH_OK = True
except OSError as e:
    TORCH_ERROR = f"DLL Error: {e}"
except ImportError as e:
    TORCH_ERROR = f"Import Error: {e}"
except Exception as e:
    TORCH_ERROR = str(e)

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False


# ── Model ────────────────────────────────────────────────────────────────────
if TORCH_OK:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    class SimpleCNN(nn.Module):
        def __init__(self, n_classes=10, n_filters1=8, n_filters2=16,
                     kernel_size=3, use_pooling=True, dropout=0.25):
            super().__init__()
            self.use_pooling = use_pooling
            self.conv1 = nn.Conv2d(1, n_filters1, kernel_size, padding=1)
            self.conv2 = nn.Conv2d(n_filters1, n_filters2, kernel_size, padding=1)
            self.dropout = nn.Dropout(dropout)
            flat = n_filters2 * (7 * 7 if use_pooling else 28 * 28)
            self.fc1 = nn.Linear(flat, 128)
            self.fc2 = nn.Linear(128, n_classes)

        def forward(self, x, return_maps=False):
            x1 = F.relu(self.conv1(x))
            if self.use_pooling: x1 = F.max_pool2d(x1, 2)
            x2 = F.relu(self.conv2(x1))
            if self.use_pooling: x2 = F.max_pool2d(x2, 2)
            feat = x2.view(x2.size(0), -1)
            out = F.relu(self.fc1(self.dropout(feat)))
            out = self.fc2(out)
            if return_maps:
                return out, x1, x2
            return out

    def plot_feature_maps(maps, title, max_maps=8):
        maps = maps.detach().cpu().numpy()[0]
        n = min(maps.shape[0], max_maps)
        fig = make_subplots(rows=1, cols=n,
                            subplot_titles=[f"FM {i}" for i in range(n)])
        for i in range(n):
            fig.add_trace(
                go.Heatmap(z=maps[i], colorscale="Viridis", showscale=False),
                row=1, col=i+1)
        fig.update_layout(
            title=title, height=180,
            plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
            font_color="#e8e8f0", margin=dict(l=10, r=10, t=40, b=10))
        return fig

    def plot_filters(conv_layer, max_filters=8):
        w = conv_layer.weight.detach().cpu().numpy()
        n = min(w.shape[0], max_filters)
        fig = make_subplots(rows=1, cols=n,
                            subplot_titles=[f"F{i}" for i in range(n)])
        for i in range(n):
            fig.add_trace(
                go.Heatmap(z=w[i, 0], colorscale="RdBu", showscale=False),
                row=1, col=i+1)
        fig.update_layout(
            title="Conv1 Filters", height=180,
            plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
            font_color="#e8e8f0", margin=dict(l=10, r=10, t=40, b=10))
        return fig


def run():
    # ── PyTorch not available — show fix guide ────────────────────────────────
    if not TORCH_OK:
        st.error("⚠️ PyTorch could not be loaded.")
        st.markdown(f"**Error:** `{TORCH_ERROR}`")
        st.markdown("---")
        st.markdown("### 🔧 How to Fix (Windows DLL Error)")
        st.markdown("""
**Step 1 — Reinstall PyTorch (CPU-only, most compatible):**
```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Step 2 — If error persists, install Visual C++ Redistributable:**
Download and run: https://aka.ms/vs/17/release/vc_redist.x64.exe  
Then restart your terminal and try again.

**Step 3 — Restart Streamlit after fixing:**
```powershell
streamlit run app.py
```
        """)
        st.info("All other tools (Perceptron, MLP, RNN, OpenCV, etc.) work without PyTorch.")
        return

    if not PIL_OK:
        st.error("Pillow not installed. Run: `pip install Pillow`")
        return

    import plotly.graph_objects as go

    st.markdown("### 🏗️ CNN Architecture")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_filters1  = st.select_slider("Conv1 Filters", [4, 8, 16, 32], 8)
        n_filters2  = st.select_slider("Conv2 Filters", [8, 16, 32, 64], 16)
    with c2:
        kernel_size = st.selectbox("Kernel Size", [3, 5], 0)
        use_pooling = st.checkbox("Max Pooling after each Conv", True)
    with c3:
        dropout = st.slider("Dropout", 0.0, 0.5, 0.25, 0.05)
        epochs  = st.slider("Training Epochs", 1, 10, 2)

    lr = st.select_slider("Learning Rate", [0.1, 0.01, 0.001, 0.0001], 0.001)

    arch = (f"Input(1×28×28) → Conv2d(1,{n_filters1},{kernel_size}) → ReLU → MaxPool"
            f" → Conv2d({n_filters1},{n_filters2},{kernel_size}) → ReLU → MaxPool"
            f" → FC(128) → FC(10)")
    st.code(arch)

    st.markdown("### 📥 Data Source")
    data_source = st.radio("", ["Download MNIST (small subset)",
                                "Upload a 28×28 grayscale image"], horizontal=True)

    device = torch.device("cpu")

    if data_source == "Download MNIST (small subset)":
        n_train = st.slider("Training samples (subset)", 100, 2000, 500)

        if st.button("🚀 Train CNN on MNIST", use_container_width=True):
            with st.spinner("Downloading MNIST…"):
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                train_ds = torchvision.datasets.MNIST(
                    root="./data", train=True, download=True, transform=transform)
                test_ds  = torchvision.datasets.MNIST(
                    root="./data", train=False, download=True, transform=transform)

            train_ds = torch.utils.data.Subset(train_ds, list(range(n_train)))
            test_ds  = torch.utils.data.Subset(test_ds,  list(range(200)))
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
            test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=200)

            model     = SimpleCNN(10, n_filters1, n_filters2,
                                  kernel_size, use_pooling, dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            progress = st.progress(0)
            loss_log = []
            total_steps = epochs * len(train_loader)
            step = 0

            model.train()
            for ep in range(epochs):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    loss_log.append(loss.item())
                    step += 1
                    progress.progress(step / total_steps)

            progress.empty()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    pred = model(xb.to(device)).argmax(1)
                    correct += (pred == yb.to(device)).sum().item()
                    total   += yb.size(0)

            st.markdown(
                f"<div class='metric-box'><div class='metric-val'>{correct/total*100:.1f}%</div>"
                f"<div class='metric-label'>Test Accuracy</div></div>",
                unsafe_allow_html=True)

            fig = go.Figure(go.Scatter(
                y=loss_log, mode="lines", line=dict(color="#6c63ff", width=1.5)))
            fig.update_layout(
                plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
                font_color="#e8e8f0", xaxis_title="Batch", yaxis_title="Loss",
                xaxis=dict(gridcolor="#1a1a26"), yaxis=dict(gridcolor="#1a1a26"),
                margin=dict(l=20,r=20,t=30,b=20), height=280)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 🔍 Filters (Conv1)")
            st.plotly_chart(plot_filters(model.conv1, n_filters1), use_container_width=True)

            sample_x, _ = next(iter(test_loader))
            with torch.no_grad():
                _, fm1, fm2 = model(sample_x[:1].to(device), return_maps=True)

            st.markdown("#### 📡 Feature Maps — Conv1")
            st.plotly_chart(plot_feature_maps(fm1, "After Conv1 + ReLU + Pool", n_filters1),
                            use_container_width=True)
            st.markdown("#### 📡 Feature Maps — Conv2")
            st.plotly_chart(plot_feature_maps(fm2, "After Conv2 + ReLU + Pool", n_filters2),
                            use_container_width=True)

    else:
        uploaded = st.file_uploader("Upload a grayscale 28×28 PNG/JPG",
                                    type=["png","jpg","jpeg"])
        if uploaded:
            img     = Image.open(uploaded).convert("L").resize((28, 28))
            img_arr = np.array(img, dtype=np.float32) / 255.0
            st.image(img, caption="Input (28×28)", width=140)

            model = SimpleCNN(10, n_filters1, n_filters2, kernel_size, use_pooling, dropout)
            t = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                out, fm1, fm2 = model(t, return_maps=True)
                probs = torch.softmax(out, dim=1).numpy()[0]

            st.markdown("#### Output (untrained — random weights)")
            pred = probs.argmax()
            st.markdown(f"Predicted digit: `{pred}` — confidence: `{probs[pred]*100:.1f}%`")
            st.markdown("#### Feature Maps — Conv1")
            st.plotly_chart(plot_feature_maps(fm1, "Conv1 Feature Maps", n_filters1),
                            use_container_width=True)
            st.markdown("#### Feature Maps — Conv2")
            st.plotly_chart(plot_feature_maps(fm2, "Conv2 Feature Maps", n_filters2),
                            use_container_width=True)
