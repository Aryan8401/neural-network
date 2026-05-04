# 🧠 Deep Learning Toolbox

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### An interactive deep learning playground — built from scratch, no black boxes, just pure math.

</div>

---

## 👨‍💻 Developer Info

| Field | Details |
|-------|---------|
| **Name** | Aryan Jaiswal |
| **Roll No.** | 2301420023 |
| **Course** | B.Tech Data Science |
| **Repository** | [github.com/Aryan8401/neural-network](https://github.com/Aryan8401/neural-network) |

---

## 📌 About This Project

The **Deep Learning Toolbox** is an interactive web application built using **Streamlit** that implements **11 foundational deep learning and machine learning algorithms completely from scratch using NumPy**. Every forward pass, every gradient calculation, every weight update is written manually and visualized in real time — no black-box `.fit()` calls.

This project was developed as part of the **B.Tech Data Science** curriculum to demonstrate deep learning concepts taught from first principles — making abstract math tangible through live, interactive visualizations.

> 💡 Every algorithm in this toolbox can be stepped through epoch by epoch, parameter by parameter, so you can see *exactly* what the model is learning and *why*.

---

## ✨ Key Features

- 🔢 **From-scratch implementations** — Core algorithms use NumPy only, no hidden magic
- 🎛️ **Fully interactive** — Change layers, neurons, learning rate, activation functions live
- 📊 **Rich visualizations** — Decision boundaries, loss curves, weight matrices, feature maps, heatmaps
- 🔬 **Step-through mode** — Inspect weights, gradients, and states at every single epoch
- 🧩 **Modular architecture** — Add a new model by creating just one `.py` file
- 🌑 **Dark-themed UI** — Clean, modern card-based interface

---

## 🤖 Models Implemented

| # | Model | Core Concept | Implementation |
|---|-------|-------------|----------------|
| 1 | ⚡ Single-Layer Perceptron | Perceptron Learning Rule, Linear Separability | NumPy |
| 2 | 🧠 Multilayer Neural Network | Forward Pass, Backprop, Gradient Descent | NumPy |
| 3 | 🔄 Backpropagation Visualizer | Chain Rule, Gradients, Weight Updates | NumPy |
| 4 | 🔍 CNN – Convolutional Network | Filters, Feature Maps, MaxPooling | PyTorch |
| 5 | 🔁 RNN – Sequence Predictor | Hidden State, BPTT, Character-level | NumPy |
| 6 | 🧲 Hopfield Network | Hebbian Learning, Associative Memory, Energy | NumPy |
| 7 | 📸 Attendance System | LBPH Face Recognition, Haar Cascade | OpenCV |
| 8 | 👥 Face Counter & Detector | Haar Cascade, Scale Factor, Min Neighbors | OpenCV |
| 9 | 🗜️ Autoencoder | Encoder-Decoder, Latent Space, Reconstruction | NumPy |
| 10 | 🎯 K-Means Clustering | Lloyd's Algorithm, Centroids, Elbow Method | NumPy |
| 11 | 🗺️ Self-Organizing Map | Competitive Learning, U-Matrix, BMU | NumPy |

---

## 📁 Project Structure

```
deep_learning_toolbox/
│
├── app.py                        # 🚀 Main Streamlit app — card UI + model router
├── requirements.txt              # All Python dependencies
├── README.md                     # Project documentation
├── LICENSE                       # MIT License
├── .gitignore                    # Git ignore rules
│
└── models/                       # Each file = one self-contained algorithm
    ├── __init__.py
    ├── perceptron.py             # Single-layer perceptron from scratch
    ├── multilayer_nn.py          # MLP with configurable hidden layers
    ├── backpropagation.py        # Step-through chain rule visualizer
    ├── cnn_model.py              # CNN with filter & feature map visualization
    ├── rnn_sequence.py           # Char-level RNN trained with BPTT
    ├── hopfield.py               # Hopfield associative memory network
    ├── opencv_attendance.py      # LBPH face recognition attendance system
    ├── face_counter.py           # Haar cascade face detection & counting
    ├── autoencoder.py            # Symmetric autoencoder with latent space viz
    ├── kmeans_clustering.py      # K-Means with step-through & elbow method
    └── som.py                    # Self-Organizing Map with U-Matrix
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip
- Git

### Step 1 — Clone the repository
```bash
git clone https://github.com/Aryan8401/neural-network.git
cd neural-network
```

### Step 2 — Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ **Windows — PyTorch DLL error?** Use the CPU-only build:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

> ⚠️ **OpenCV face recognition** needs the contrib package:
> ```bash
> pip install opencv-contrib-python
> ```

### Step 4 — Run the application
```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`** 🎉

---

## 📖 Model Explanations

<details>
<summary><b>⚡ 1. Single-Layer Perceptron</b></summary>

<br>

The simplest neural network — inspired by biological neurons (Rosenblatt, 1958). Takes weighted inputs, computes a net sum, passes through an activation function, and updates weights based on prediction error.

**Math:**
```
net  = w₁·x₁ + w₂·x₂ + bias
ŷ    = activation(net)
Δwᵢ  = lr × (y - ŷ) × xᵢ
Δb   = lr × (y - ŷ)
```

**Datasets:** AND, OR, NAND, NOR, XOR gates

**Key insight:** XOR gate never converges — a single perceptron can only separate classes with one straight line (linear separability). This limitation historically motivated the invention of multilayer networks.

**What to explore:**
- AND gate → converges in ~6 epochs
- XOR gate → never converges, decision boundary spins forever
- Step through the weight table to see every update
- Change learning rate → see how it affects convergence speed

</details>

<details>
<summary><b>🧠 2. Multilayer Neural Network (MLP)</b></summary>

<br>

Multiple perceptron layers stacked together, enabling non-linear decision boundaries. The hidden layers learn intermediate representations of the data.

**Math:**
```
Forward:   Z⁽ˡ⁾ = A⁽ˡ⁻¹⁾·W⁽ˡ⁾ + b⁽ˡ⁾
           A⁽ˡ⁾ = activation(Z⁽ˡ⁾)

Loss:      BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

Backward:  ∂L/∂W⁽ˡ⁾ computed via chain rule
Update:    W⁽ˡ⁾ = W⁽ˡ⁾ - lr · ∂L/∂W⁽ˡ⁾
```

**What to explore:**
- Moons/Circles datasets are non-linearly separable → MLP handles them
- Add more hidden layers → decision boundary becomes more complex
- Inspect learned weight matrices layer by layer
- Compare ReLU vs Sigmoid activation — ReLU trains faster

</details>

<details>
<summary><b>🔄 3. Backpropagation Visualizer</b></summary>

<br>

Shows every single intermediate value of the chain rule for one training sample. Edit weights, inputs, and targets manually and watch gradients change in real time.

**Chain rule flow:**
```
Loss
 └─→ dL/dŷ
      └─→ dL/dZ2 = dL/dŷ · σ'(Z2)
           ├─→ dL/dW2 = A1ᵀ · dL/dZ2
           └─→ dL/dA1 = dL/dZ2 · W2ᵀ
                └─→ dL/dZ1 = dL/dA1 · activation'(Z1)
                     └─→ dL/dW1 = Xᵀ · dL/dZ1
```

**What to explore:**
- Set target=1.0 vs target=0.0 — watch gradients flip direction
- Change learning rate — see how large/small the weight deltas become
- Click the button multiple times to simulate training steps manually

</details>

<details>
<summary><b>🔍 4. CNN – Convolutional Network</b></summary>

<br>

Designed for images. Small filters (kernels) slide across the image detecting local features — edges, curves, textures — building up to complex patterns.

**Architecture:**
```
Input(1×28×28)
  → Conv2d(1, 8, 3×3) → ReLU → MaxPool(2×2)
  → Conv2d(8, 16, 3×3) → ReLU → MaxPool(2×2)
  → Flatten
  → FC(128) → FC(10)
  → Softmax
```

**What to explore:**
- Train on MNIST digits (500 samples, 2 epochs) — watch accuracy climb
- Visualize 8 Conv1 filters — each detects a different edge orientation
- Feature maps after Conv1 → low-level edges and gradients
- Feature maps after Conv2 → abstract, semantic patterns

</details>

<details>
<summary><b>🔁 5. RNN – Sequence Predictor</b></summary>

<br>

Has memory — the hidden state `h(t)` carries information from previous inputs into the current step. Trained character-by-character on custom text using Backpropagation Through Time (BPTT).

**Math:**
```
h(t) = tanh(Wxh · x(t) + Whh · h(t-1) + bh)
y(t) = Why · h(t) + by
p(t) = softmax(y(t))
Loss = -Σ log(p(t)[target])
```

**What to explore:**
- Preset "Days of Week" → after training, seed with "monday" → generates "tuesday wednesday..."
- Temperature = 0.3 → repetitive/confident output
- Temperature = 1.5 → creative/random output
- Hidden state heatmap shows which neurons activate per character

</details>

<details>
<summary><b>🧲 6. Hopfield Network</b></summary>

<br>

A biological model of associative memory. Store patterns via Hebbian learning, then give a corrupted version and the network converges back to the original by minimizing energy.

**Math:**
```
Store:    W = (1/N) Σ ξᵘ(ξᵘ)ᵀ,   Wᵢᵢ = 0   (Hebbian rule)
Retrieve: s(t+1) = sign(W · s(t))             (sync update)
Energy:   E = -½ sᵀWs                         (always decreases)
Overlap:  m = (1/N) s · ξ                     (how close to pattern)
Capacity: p_max ≈ 0.138 × N neurons
```

**What to explore:**
- Store X, O, T → corrupt X at 30% → retrieves X perfectly
- Increase noise to 45% → sometimes retrieves wrong pattern (spurious state)
- Analysis tab → capacity test → see retrieval collapse beyond ~9 patterns
- Draw tab → draw your own 8×8 pattern on a grid → store and retrieve it

</details>

<details>
<summary><b>📸 7. Attendance System</b></summary>

<br>

Uses Haar Cascade for face detection and LBPH (Local Binary Pattern Histogram) for face recognition. Register known faces, then upload a test image to automatically mark attendance.

**Pipeline:**
```
Upload photos → Detect face (Haar Cascade) → Crop & resize to 100×100
→ Train LBPH recognizer → Upload test image → Detect faces
→ Predict identity (confidence < threshold = recognized)
→ Mark Present/Absent → Log with timestamp
```

**What to explore:**
- Register 2–3 photos of a person from different angles for better accuracy
- Lower confidence threshold = stricter recognition
- Attendance log tab shows full history with timestamps

</details>

<details>
<summary><b>🗜️ 9. Autoencoder</b></summary>

<br>

Encoder compresses input data into a small latent vector. Decoder reconstructs the original from this compressed form. Forces the network to learn the most important features automatically.

**Math:**
```
Encode: z = f(X)           where z has latent_dim dimensions
Decode: X̂ = g(z)
Loss:   MSE = (1/n) Σ (X - X̂)²
```

**What to explore:**
- Latent Dim=2 → 2D latent space plot — classes separate without supervision
- Latent Dim=1 → information bottleneck → reconstruction degrades
- Compare Original vs Reconstructed scatter plots

</details>

<details>
<summary><b>🎯 10. K-Means Clustering</b></summary>

<br>

Unsupervised algorithm that partitions data into K clusters. No labels needed — it finds natural groupings by iteratively moving centroids to the mean of assigned points.

**Algorithm:**
```
1. Initialize K centroids (K-Means++ or Random)
2. Assign each point to nearest centroid
3. Move each centroid to mean of its cluster
4. Repeat steps 2–3 until convergence (centroids stop moving)
```

**What to explore:**
- Use the iteration slider to watch centroids migrate step by step
- Centroid trails show the exact path each centroid took
- Elbow method → inertia curve shows the optimal K (look for the "elbow" bend)
- Try wrong K (e.g., K=2 on 4-cluster data) → see forced bad splits

</details>

<details>
<summary><b>🗺️ 11. Self-Organizing Map (SOM)</b></summary>

<br>

An unsupervised neural network that maps high-dimensional data onto a 2D grid while preserving topology — nearby points in input space land near each other on the grid.

**Math:**
```
BMU: i* = argmin‖x - wᵢ‖        (find closest neuron)
Update: wᵢ += lr · h(i*, σ) · (x - wᵢ)
h = exp(-‖rᵢ - r*‖² / 2σ²)      (Gaussian neighborhood)
lr and σ decay over time
```

**What to explore:**
- U-Matrix: dark regions = cluster centers, bright = boundaries
- Component planes: one heatmap per input feature
- Hit map: which neurons get activated most often
- Try Iris (4D) → U-Matrix should show 3 species regions

</details>

---

## ➕ Adding a New Model

The toolbox is designed to be extended with minimal effort:

**Step 1** — Create `models/your_model.py` with a `run()` function:
```python
import streamlit as st

def run():
    st.markdown("### Your Model Name")
    
    # Add your controls
    param = st.slider("Parameter", 0, 100, 50)
    
    if st.button("Train"):
        # Your model logic here
        st.success("Training complete!")
```

**Step 2** — Add one entry to the `TOOLS` list in `app.py`:
```python
{
    "id":     "your_model",
    "icon":   "🤖",
    "title":  "Your Model Name",
    "desc":   "Short description shown on the home card.",
    "tag":    "Category Tag",
    "color":  "#ff6584",           # Card accent color (any hex)
    "module": "models.your_model",
},
```

A new card appears on the home screen automatically. No other changes needed. ✅

---

## 📦 Requirements

```
streamlit>=1.32.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.18.0
scikit-learn>=1.3.0
Pillow>=10.0.0
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
```

---

## ⚠️ Troubleshooting

| Error | Solution |
|-------|----------|
| `File does not exist: app.py` | Run `cd neural-network` first, then `streamlit run app.py` |
| PyTorch DLL error on Windows | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| `cv2.face module not found` | `pip install opencv-contrib-python` (not just `opencv-python`) |
| MNIST download fails | Check internet; data saves to `./data/` automatically |
| Git not recognized in terminal | Close and reopen VS Code after installing Git |
| Push rejected on GitHub | `git pull origin main --allow-unrelated-histories` then push again |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with 🧠 by Aryan Jaiswal**

Roll No. **2301420023** &nbsp;|&nbsp; B.Tech Data Science

<br>

*If this helped you understand deep learning better, consider giving it a ⭐*

</div>
