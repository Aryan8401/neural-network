"""
Hopfield Network — implemented fully from scratch with NumPy.

Features:
  - Store binary patterns (±1) using Hebbian learning
  - Synchronous & asynchronous update modes
  - Retrieve patterns from noisy/corrupted inputs
  - Visualise energy landscape, weight matrix, convergence
  - Draw your own 8×8 pattern to store/retrieve
  - Shows theoretical storage capacity (0.138 × N)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Hopfield Network Core ────────────────────────────────────────────────────
class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.N = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))
        self.stored_patterns = []

    def store(self, patterns):
        """Hebbian learning rule: W = (1/N) Σ ξᵘ(ξᵘ)ᵀ, diagonal = 0"""
        self.W = np.zeros((self.N, self.N))
        self.stored_patterns = [p.copy() for p in patterns]
        for p in patterns:
            p = p.reshape(-1, 1)
            self.W += p @ p.T
        self.W /= self.N
        np.fill_diagonal(self.W, 0)

    def energy(self, s):
        """E = -½ sᵀWs"""
        return -0.5 * s @ self.W @ s

    def update_async(self, s, n_steps=None):
        """Asynchronous update: one random neuron at a time."""
        s = s.copy()
        history = [s.copy()]
        energies = [self.energy(s)]
        steps = n_steps or self.N * 5
        for _ in range(steps):
            i = np.random.randint(self.N)
            net = self.W[i] @ s
            s[i] = 1 if net >= 0 else -1
            history.append(s.copy())
            energies.append(self.energy(s))
        return s, history, energies

    def update_sync(self, s, max_iter=20):
        """Synchronous update: all neurons update simultaneously."""
        s = s.copy()
        history = [s.copy()]
        energies = [self.energy(s)]
        for _ in range(max_iter):
            s_new = np.sign(self.W @ s)
            s_new[s_new == 0] = 1
            history.append(s_new.copy())
            energies.append(self.energy(s_new))
            if np.array_equal(s_new, s):
                break
            s = s_new
        return s, history, energies

    def retrieve(self, probe, mode="sync", steps=20):
        if mode == "sync":
            return self.update_sync(probe, steps)
        else:
            return self.update_async(probe, steps * self.N)

    def overlap(self, s, pattern):
        """Overlap m = (1/N) s·ξ — how close state is to pattern."""
        return (s @ pattern) / self.N

    def capacity(self):
        return int(0.138 * self.N)


# ── Built-in 8×8 Patterns ────────────────────────────────────────────────────
def letter_pattern(letter):
    """Returns a 64-dim ±1 vector shaped like a letter on an 8×8 grid."""
    grid = np.ones((8, 8)) * -1
    if letter == "X":
        for i in range(8):
            grid[i, i] = 1
            grid[i, 7-i] = 1
    elif letter == "O":
        for i in range(2, 6):
            grid[1, i] = grid[6, i] = 1
            grid[i, 1] = grid[i, 6] = 1
    elif letter == "T":
        grid[0, :] = 1
        grid[1:, 3] = grid[1:, 4] = 1
    elif letter == "H":
        grid[:, 1] = grid[:, 6] = 1
        grid[3:5, 1:7] = 1
    elif letter == "L":
        grid[:, 1] = 1
        grid[7, 1:7] = 1
    elif letter == "Z":
        grid[0, :] = grid[7, :] = 1
        for i in range(1, 7):
            grid[i, 7-i] = 1
    return grid.flatten()


BUILTIN_PATTERNS = {
    "X": letter_pattern("X"),
    "O": letter_pattern("O"),
    "T": letter_pattern("T"),
    "H": letter_pattern("H"),
    "L": letter_pattern("L"),
    "Z": letter_pattern("Z"),
}


def add_noise(pattern, flip_prob):
    """Randomly flip bits with given probability."""
    noisy = pattern.copy()
    mask  = np.random.rand(len(pattern)) < flip_prob
    noisy[mask] *= -1
    return noisy


def pattern_to_img(pattern, size=8):
    return ((pattern.reshape(size, size) + 1) / 2)   # maps ±1 → [0,1]


def plot_pattern_grid(patterns, titles, ncols=4):
    n   = len(patterns)
    nrows = (n + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=titles + [""]*(nrows*ncols - n))
    for idx, p in enumerate(patterns):
        r, c = divmod(idx, ncols)
        img = pattern_to_img(p)
        fig.add_trace(go.Heatmap(
            z=img, colorscale=[[0,"#0a0a0f"],[1,"#6c63ff"]],
            showscale=False, zmin=0, zmax=1), row=r+1, col=c+1)
    fig.update_layout(
        plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
        font_color="#e8e8f0", height=160*nrows,
        margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def plot_weight_matrix(W):
    fig = go.Figure(go.Heatmap(z=W, colorscale="RdBu", zmid=0))
    fig.update_layout(
        title="Weight Matrix W", height=400,
        plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
        font_color="#e8e8f0", margin=dict(l=20,r=20,t=50,b=20))
    return fig


def plot_energy(energies):
    fig = go.Figure(go.Scatter(
        y=energies, mode="lines+markers",
        line=dict(color="#f7971e", width=2),
        marker=dict(size=5, color="#ff6584")))
    fig.update_layout(
        title="Energy over Iterations", height=260,
        plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
        font_color="#e8e8f0", xaxis_title="Step", yaxis_title="E",
        xaxis=dict(gridcolor="#1a1a26"), yaxis=dict(gridcolor="#1a1a26"),
        margin=dict(l=20,r=20,t=50,b=20))
    return fig


# ── Streamlit UI ─────────────────────────────────────────────────────────────
def run():
    st.markdown("### ⚙️ Network Configuration")
    c1, c2 = st.columns(2)
    with c1:
        grid_size = st.selectbox("Pattern Size", ["8×8 (64 neurons)", "6×6 (36 neurons)"], 0)
        N = 64 if "8" in grid_size else 36
        G = 8  if N == 64 else 6
    with c2:
        update_mode = st.selectbox("Update Mode", ["sync", "async"])
        max_iter    = st.slider("Max Retrieval Iterations", 5, 100, 20)

    st.markdown(f"""
    <div style='background:#111118;border:1px solid #2a2a40;border-radius:10px;padding:1rem;margin-bottom:1rem;'>
    <b>Network:</b> {N} neurons &nbsp;|&nbsp;
    <b>Theoretical Capacity:</b> ~{int(0.138*N)} patterns &nbsp;|&nbsp;
    <b>Mode:</b> {update_mode}
    </div>
    """, unsafe_allow_html=True)

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📦 Store & Retrieve", "🎨 Draw Your Pattern", "📊 Analysis"])

    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Select Patterns to Store")

        if N == 64:
            available = list(BUILTIN_PATTERNS.keys())
        else:
            # Smaller patterns — generate stripes
            def stripe(n, g):
                p = np.ones(n); p[n//2:] = -1; return p
            available = ["Stripe-H", "Stripe-V", "Checkerboard"]
            custom_pats = {
                "Stripe-H":     np.array([1 if i < G//2 else -1
                                          for i in range(G) for _ in range(G)]),
                "Stripe-V":     np.array([1 if j < G//2 else -1
                                          for _ in range(G) for j in range(G)]),
                "Checkerboard": np.array([1 if (i+j)%2==0 else -1
                                          for i in range(G) for j in range(G)]),
            }
            BUILTIN_PATTERNS.update(custom_pats)

        selected = st.multiselect("Choose patterns to memorize",
                                  available, default=available[:3])
        if not selected:
            st.warning("Select at least one pattern.")
            return

        patterns = [BUILTIN_PATTERNS[k][:N] for k in selected]
        cap = int(0.138 * N)

        if len(patterns) > cap:
            st.warning(f"⚠️ Storing {len(patterns)} patterns exceeds theoretical capacity ({cap}). "
                       "Retrieval may be unreliable.")

        # Train
        net = HopfieldNetwork(N)
        net.store(patterns)

        st.markdown("#### 🗂️ Stored Patterns")
        st.plotly_chart(plot_pattern_grid(patterns, selected), use_container_width=True)

        # ── Retrieval ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🔮 Retrieval from Noisy Input")
        probe_choice = st.selectbox("Probe pattern (will be corrupted)", selected)
        noise_level  = st.slider("Noise (flip probability)", 0.0, 0.5, 0.2, 0.05)

        if st.button("🚀 Run Retrieval", use_container_width=True):
            base = BUILTIN_PATTERNS[probe_choice][:N]
            np.random.seed(None)
            noisy = add_noise(base, noise_level)
            retrieved, history, energies = net.retrieve(noisy, update_mode, max_iter)

            # Overlaps with all stored patterns
            overlaps = {k: round(net.overlap(retrieved, BUILTIN_PATTERNS[k][:N]), 3)
                        for k in selected}
            best_match = max(overlaps, key=overlaps.get)

            # Show triple: original / noisy / retrieved
            fig = plot_pattern_grid(
                [base, noisy, retrieved],
                ["Original", f"Noisy ({noise_level*100:.0f}% flip)", f"Retrieved → {best_match}"],
                ncols=3)
            st.plotly_chart(fig, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"<div class='metric-box'><div class='metric-val'>{overlaps[best_match]:.3f}</div>"
                            f"<div class='metric-label'>Best Overlap</div></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='metric-box'><div class='metric-val'>{best_match}</div>"
                            f"<div class='metric-label'>Matched Pattern</div></div>", unsafe_allow_html=True)
            with m3:
                correct = best_match == probe_choice
                st.markdown(f"<div class='metric-box'><div class='metric-val'>{'✅' if correct else '❌'}</div>"
                            f"<div class='metric-label'>{'Correct' if correct else 'Wrong'}</div></div>",
                            unsafe_allow_html=True)

            # Energy convergence
            st.markdown("#### ⚡ Energy During Retrieval")
            st.plotly_chart(plot_energy(energies[:50]), use_container_width=True)

            # Overlap bar
            st.markdown("#### 📐 Overlap with All Stored Patterns")
            fig2 = go.Figure(go.Bar(
                x=list(overlaps.keys()),
                y=list(overlaps.values()),
                marker_color=["#43e97b" if k==best_match else "#6c63ff"
                              for k in overlaps]))
            fig2.add_hline(y=0, line_color="#ff6584", line_dash="dot")
            fig2.update_layout(
                plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
                font_color="#e8e8f0", yaxis_title="Overlap m",
                xaxis_title="Stored Pattern",
                xaxis=dict(gridcolor="#1a1a26"), yaxis=dict(gridcolor="#1a1a26"),
                margin=dict(l=20,r=20,t=30,b=20), height=260)
            st.plotly_chart(fig2, use_container_width=True)

            # Step-through
            st.markdown("#### 🔬 State at Each Update Step")
            max_show = min(len(history)-1, 20)
            step = st.slider("Step", 0, max_show, 0)
            fig3 = go.Figure(go.Heatmap(
                z=pattern_to_img(history[step], G),
                colorscale=[[0,"#0a0a0f"],[1,"#6c63ff"]],
                showscale=False, zmin=0, zmax=1))
            fig3.update_layout(
                plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
                font_color="#e8e8f0", height=300,
                margin=dict(l=20,r=20,t=30,b=20))
            fig3.update_xaxes(showticklabels=False)
            fig3.update_yaxes(showticklabels=False)
            st.plotly_chart(fig3, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### 🎨 Draw an 8×8 Pattern")
        st.markdown("Toggle cells below (click = flip). Then store and retrieve it.")

        if "drawn_grid" not in st.session_state or st.button("🔄 Reset Grid"):
            st.session_state.drawn_grid = [[False]*G for _ in range(G)]

        # Render grid as toggle buttons
        grid = st.session_state.drawn_grid
        for i in range(G):
            cols = st.columns(G)
            for j, col in enumerate(cols):
                with col:
                    label = "⬛" if grid[i][j] else "⬜"
                    if st.button(label, key=f"cell_{i}_{j}", use_container_width=True):
                        grid[i][j] = not grid[i][j]
                        st.rerun()

        # Convert grid to ±1 pattern
        drawn_pattern = np.array([1 if grid[i][j] else -1
                                   for i in range(G) for j in range(G)], dtype=float)

        c1, c2 = st.columns(2)
        with c1:
            noise_draw = st.slider("Noise to apply before retrieval", 0.0, 0.5, 0.25, 0.05,
                                   key="draw_noise")
        with c2:
            st.markdown(" ")

        if st.button("💾 Store & Retrieve This Pattern", use_container_width=True):
            net2 = HopfieldNetwork(N)
            net2.store([drawn_pattern])
            noisy = add_noise(drawn_pattern, noise_draw)
            retrieved, _, energies = net2.retrieve(noisy, update_mode, max_iter)

            fig = plot_pattern_grid(
                [drawn_pattern, noisy, retrieved],
                ["Drawn (stored)", "Noisy input", "Retrieved"],
                ncols=3)
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(plot_energy(energies[:50]), use_container_width=True)

            ov = net2.overlap(retrieved, drawn_pattern)
            st.markdown(f"**Overlap with stored pattern:** `{ov:.4f}`  "
                        f"({'Perfect recall ✅' if ov > 0.9 else 'Partial recall ⚠️'})")

    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### 📊 Network Analysis")

        # Capacity experiment
        st.markdown("##### Storage Capacity Experiment")
        st.markdown("How does retrieval quality degrade as we store more patterns?")

        if st.button("▶️ Run Capacity Test", use_container_width=True):
            np.random.seed(0)
            k_range = range(1, cap + 5)
            overlaps_mean = []
            for k in k_range:
                pats = [np.sign(np.random.randn(N)) for _ in range(k)]
                net_t = HopfieldNetwork(N)
                net_t.store(pats)
                ovs = []
                for p in pats:
                    noisy = add_noise(p, 0.15)
                    ret, _, _ = net_t.retrieve(noisy, "sync", 10)
                    ovs.append(abs(net_t.overlap(ret, p)))
                overlaps_mean.append(np.mean(ovs))

            fig_cap = go.Figure()
            fig_cap.add_trace(go.Scatter(
                x=list(k_range), y=overlaps_mean, mode="lines+markers",
                line=dict(color="#43e97b", width=2),
                marker=dict(size=6, color="#f7971e"), name="Mean |overlap|"))
            fig_cap.add_vline(x=cap, line_color="#ff6584", line_dash="dash",
                              annotation_text=f"Capacity ≈{cap}", annotation_position="top right")
            fig_cap.update_layout(
                plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
                font_color="#e8e8f0", xaxis_title="Patterns Stored (k)",
                yaxis_title="Mean |Overlap|",
                xaxis=dict(gridcolor="#1a1a26"), yaxis=dict(gridcolor="#1a1a26", range=[0,1.05]),
                margin=dict(l=20,r=20,t=30,b=20), height=300)
            st.plotly_chart(fig_cap, use_container_width=True)
            st.caption(f"Overlap drops sharply past ~{cap} patterns — the theoretical capacity limit (0.138 × N).")

        # Weight matrix visualisation (requires stored patterns)
        st.markdown("##### Weight Matrix")
        store_for_w = st.multiselect("Store these patterns to see W",
                                     list(BUILTIN_PATTERNS.keys())[:6],
                                     default=list(BUILTIN_PATTERNS.keys())[:3],
                                     key="w_select")
        if store_for_w:
            net_w = HopfieldNetwork(N)
            net_w.store([BUILTIN_PATTERNS[k][:N] for k in store_for_w])
            st.plotly_chart(plot_weight_matrix(net_w.W), use_container_width=True)
            st.markdown(f"""
            **Key properties of W:**
            - Shape: `{N}×{N}`  
            - Symmetric: `W = Wᵀ` ✅  
            - Zero diagonal: `Wᵢᵢ = 0` ✅  
            - Max |weight|: `{np.abs(net_w.W).max():.4f}`  
            - Sparsity: `{(net_w.W == 0).mean()*100:.1f}%` zeros
            """)

        st.markdown("##### Theory Reference")
        st.markdown("""
        | Concept | Formula |
        |---------|---------|
        | Hebbian learning | `W = (1/N) Σ ξᵘ(ξᵘ)ᵀ,  Wᵢᵢ=0` |
        | Sync update | `s(t+1) = sign(W · s(t))` |
        | Async update | `sᵢ(t+1) = sign(Wᵢ · s(t))` |
        | Energy | `E = -½ sᵀWs` |
        | Overlap | `m = (1/N) s · ξ` |
        | Capacity | `p_max ≈ 0.138 N` |
        """)
