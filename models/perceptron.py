import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def step(x):      return np.where(x >= 0, 1, 0)
def sigmoid(x):   return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def tanh_fn(x):   return np.tanh(x)
def relu(x):      return np.maximum(0, x)

ACTIVATIONS = {"Step": step, "Sigmoid": sigmoid, "Tanh": tanh_fn, "ReLU": relu}

DATASETS = {
    "AND Gate":  (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,0,0,1])),
    "OR Gate":   (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,1,1,1])),
    "NAND Gate": (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([1,1,1,0])),
    "NOR Gate":  (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([1,0,0,0])),
    "XOR Gate":  (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,1,1,0])),
}

class Perceptron:
    def __init__(self, n_inputs, lr=0.1, activation="Step"):
        self.weights = np.zeros(n_inputs)
        self.bias = 0.0
        self.lr = lr
        self.activation = ACTIVATIONS[activation]
        self.history = []

    def predict(self, X):
        net = X @ self.weights + self.bias
        return self.activation(net), net

    def train_epoch(self, X, y):
        errors = []
        for xi, yi in zip(X, y):
            y_hat, _ = self.predict(xi.reshape(1,-1))
            e = yi - y_hat[0]
            self.weights += self.lr * e * xi
            self.bias    += self.lr * e
            errors.append(abs(e))
        return errors

    def train(self, X, y, epochs):
        self.history = []
        for ep in range(1, epochs+1):
            errs = self.train_epoch(X, y)
            self.history.append({"epoch":ep,"weights":self.weights.copy(),
                                  "bias":self.bias,"errors":errs,"total_error":sum(errs)})
            if sum(errs) == 0: break
        return self.history

def plot_decision_boundary(X, y, perc):
    fig = go.Figure()
    colors = ["#ff6584" if yi==0 else "#43e97b" for yi in y]
    fig.add_trace(go.Scatter(x=X[:,0],y=X[:,1],mode="markers",
                             marker=dict(size=16,color=colors,line=dict(width=2,color="white"))))
    if perc.weights[1] != 0:
        xr = np.linspace(-0.5,1.5,200)
        db = (-perc.weights[0]*xr - perc.bias) / perc.weights[1]
        fig.add_trace(go.Scatter(x=xr,y=db,mode="lines",
                                 line=dict(color="#6c63ff",width=2,dash="dash"),name="Decision Boundary"))
    fig.update_layout(plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",
                      font_color="#e8e8f0",xaxis=dict(range=[-0.5,1.5],gridcolor="#1a1a26"),
                      yaxis=dict(range=[-0.5,1.5],gridcolor="#1a1a26"),
                      margin=dict(l=20,r=20,t=30,b=20),height=360)
    return fig

def plot_error_curve(history):
    epochs = [h["epoch"] for h in history]
    errors = [h["total_error"] for h in history]
    fig = go.Figure(go.Scatter(x=epochs,y=errors,mode="lines+markers",
                               line=dict(color="#6c63ff",width=2),marker=dict(size=6,color="#ff6584")))
    fig.update_layout(plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",
                      font_color="#e8e8f0",xaxis_title="Epoch",yaxis_title="Total Error",
                      xaxis=dict(gridcolor="#1a1a26"),yaxis=dict(gridcolor="#1a1a26"),
                      margin=dict(l=20,r=20,t=30,b=20),height=280)
    return fig

def run():
    st.markdown("### ⚙️ Configuration")
    c1,c2,c3 = st.columns(3)
    with c1: dataset_name = st.selectbox("Dataset", list(DATASETS.keys()))
    with c2: activation = st.selectbox("Activation Function", list(ACTIVATIONS.keys()))
    with c3: lr = st.slider("Learning Rate",0.01,1.0,0.1,0.01)
    epochs = st.slider("Max Epochs",5,200,50)
    X, y = DATASETS[dataset_name]
    st.markdown("---")
    df = pd.DataFrame(X,columns=["x₁","x₂"]); df["y (target)"] = y
    st.dataframe(df, use_container_width=True, hide_index=True)
    if st.button("🚀 Train Perceptron", use_container_width=True):
        perc = Perceptron(2, lr, activation)
        history = perc.train(X, y, epochs)
        st.markdown("---")
        m1,m2,m3 = st.columns(3)
        with m1: st.markdown(f"<div class='metric-box'><div class='metric-val'>{history[-1]['epoch']}</div><div class='metric-label'>Epochs Run</div></div>",unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-box'><div class='metric-val'>{history[-1]['total_error']:.2f}</div><div class='metric-label'>Final Error</div></div>",unsafe_allow_html=True)
        with m3:
            c = history[-1]['total_error']==0
            st.markdown(f"<div class='metric-box'><div class='metric-val'>{'✅' if c else '⚠️'}</div><div class='metric-label'>{'Converged' if c else 'Not Converged'}</div></div>",unsafe_allow_html=True)
        st.plotly_chart(plot_error_curve(history), use_container_width=True)
        st.plotly_chart(plot_decision_boundary(X, y, perc), use_container_width=True)
        rows = [{"Epoch":h["epoch"],"w₁":round(h["weights"][0],4),"w₂":round(h["weights"][1],4),
                 "Bias":round(h["bias"],4),"Total Error":h["total_error"]} for h in history]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        ep_idx = st.slider("Select Epoch to Inspect",1,len(history),1)
        h = history[ep_idx-1]
        st.markdown(f"**Epoch {h['epoch']}** | w₁=`{h['weights'][0]:.4f}` w₂=`{h['weights'][1]:.4f}` bias=`{h['bias']:.4f}` | errors=`{[round(e,4) for e in h['errors']]}`")
