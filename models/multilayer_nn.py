import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def relu(x):      return np.maximum(0,x)
def relu_d(x):    return (x>0).astype(float)
def sigmoid(x):   return 1/(1+np.exp(-np.clip(x,-500,500)))
def sigmoid_d(x): s=sigmoid(x); return s*(1-s)
def tanh_fn(x):   return np.tanh(x)
def tanh_d(x):    return 1-np.tanh(x)**2
def linear(x):    return x
def linear_d(x):  return np.ones_like(x)

ACTS = {"ReLU":(relu,relu_d),"Sigmoid":(sigmoid,sigmoid_d),"Tanh":(tanh_fn,tanh_d),"Linear":(linear,linear_d)}

class MLP:
    def __init__(self, layer_sizes, hidden_act="ReLU", out_act="Sigmoid", lr=0.01):
        self.sizes=layer_sizes; self.lr=lr
        self.hidden_act,self.hidden_d=ACTS[hidden_act]
        self.out_act,self.out_d=ACTS[out_act]
        np.random.seed(42)
        self.W=[np.random.randn(layer_sizes[i],layer_sizes[i+1])*np.sqrt(2/layer_sizes[i]) for i in range(len(layer_sizes)-1)]
        self.b=[np.zeros((1,layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
        self.loss_history=[]
    def forward(self,X):
        self.A=[X]; self.Z=[]
        for i in range(len(self.W)-1):
            z=self.A[-1]@self.W[i]+self.b[i]; self.Z.append(z); self.A.append(self.hidden_act(z))
        z=self.A[-1]@self.W[-1]+self.b[-1]; self.Z.append(z); self.A.append(self.out_act(z))
        return self.A[-1]
    def compute_loss(self,y_pred,y_true):
        eps=1e-8; y_true=y_true.reshape(-1,1)
        return -np.mean(y_true*np.log(y_pred+eps)+(1-y_true)*np.log(1-y_pred+eps))
    def backward(self,y_true):
        m=y_true.shape[0]; y_true=y_true.reshape(-1,1)
        self.dW=[None]*len(self.W); self.db=[None]*len(self.b)
        dA=(self.A[-1]-y_true)/m; dZ=dA*self.out_d(self.Z[-1])
        for i in reversed(range(len(self.W))):
            self.dW[i]=self.A[i].T@dZ; self.db[i]=np.sum(dZ,axis=0,keepdims=True)
            if i>0: dA=dZ@self.W[i].T; dZ=dA*self.hidden_d(self.Z[i-1])
    def update(self):
        for i in range(len(self.W)): self.W[i]-=self.lr*self.dW[i]; self.b[i]-=self.lr*self.db[i]
    def train_epoch(self,X,y):
        y_pred=self.forward(X); loss=self.compute_loss(y_pred,y); self.backward(y); self.update(); return loss,y_pred
    def predict(self,X): return (self.forward(X)>=0.5).astype(int).flatten()
    def accuracy(self,X,y): return np.mean(self.predict(X)==y)

def make_dataset(name,n=300):
    if name=="Moons": X,y=make_moons(n,noise=0.2,random_state=42)
    elif name=="Circles": X,y=make_circles(n,noise=0.15,factor=0.4,random_state=42)
    else: X,y=make_classification(n,n_features=2,n_redundant=0,n_clusters_per_class=1,random_state=42)
    return StandardScaler().fit_transform(X),y

def plot_loss(history):
    fig=go.Figure(go.Scatter(x=list(range(1,len(history)+1)),y=history,mode="lines",line=dict(color="#6c63ff",width=2)))
    fig.update_layout(plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="#e8e8f0",
                      xaxis_title="Epoch",yaxis_title="Loss",xaxis=dict(gridcolor="#1a1a26"),
                      yaxis=dict(gridcolor="#1a1a26"),margin=dict(l=20,r=20,t=30,b=20),height=280)
    return fig

def plot_boundary(X,y,model):
    h=0.05; x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1; x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x1_min,x1_max,h),np.arange(x2_min,x2_max,h))
    Z=model.forward(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    fig=go.Figure()
    fig.add_trace(go.Contour(x=np.arange(x1_min,x1_max,h),y=np.arange(x2_min,x2_max,h),z=Z,
                             colorscale=[[0,"#ff658455"],[1,"#43e97b55"]],showscale=False,opacity=0.5))
    colors=["#ff6584" if yi==0 else "#43e97b" for yi in y]
    fig.add_trace(go.Scatter(x=X[:,0],y=X[:,1],mode="markers",
                             marker=dict(size=8,color=colors,line=dict(width=1,color="white"))))
    fig.update_layout(plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="#e8e8f0",
                      height=360,margin=dict(l=20,r=20,t=30,b=20),showlegend=False)
    return fig

def run():
    st.markdown("### 🏗️ Architecture Builder")
    c1,c2=st.columns(2)
    with c1: dataset_name=st.selectbox("Dataset",["Moons","Circles","Linear"]); n_samples=st.slider("Samples",100,1000,300)
    with c2: hidden_act=st.selectbox("Hidden Activation",list(ACTS.keys())); lr=st.slider("Learning Rate",0.001,1.0,0.01,0.001,format="%.3f")
    epochs=st.slider("Epochs",10,2000,500)
    n_hidden=st.slider("Number of Hidden Layers",1,5,2)
    layer_sizes=[]; cols=st.columns(n_hidden)
    for i,c in enumerate(cols):
        with c: sz=c.number_input(f"Layer {i+1} neurons",1,128,[4,4,8,8,16][i],key=f"hl_{i}"); layer_sizes.append(int(sz))
    full_arch=[2]+layer_sizes+[1]
    st.markdown(f"**Architecture:** `{' → '.join(map(str,full_arch))}`")
    X,y=make_dataset(dataset_name,n_samples)
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=0)
    if st.button("🚀 Train Network", use_container_width=True):
        model=MLP(full_arch,hidden_act=hidden_act,out_act="Sigmoid",lr=lr)
        progress=st.progress(0); status=st.empty(); all_losses=[]; log_every=max(1,epochs//100)
        for ep in range(1,epochs+1):
            loss,_=model.train_epoch(X_tr,y_tr); all_losses.append(loss)
            if ep%log_every==0 or ep==epochs: progress.progress(ep/epochs); status.markdown(f"Epoch **{ep}/{epochs}** — Loss: `{loss:.5f}`")
        progress.empty(); status.empty()
        acc_tr=model.accuracy(X_tr,y_tr); acc_te=model.accuracy(X_te,y_te)
        m1,m2,m3=st.columns(3)
        with m1: st.markdown(f"<div class='metric-box'><div class='metric-val'>{acc_tr*100:.1f}%</div><div class='metric-label'>Train Acc</div></div>",unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-box'><div class='metric-val'>{acc_te*100:.1f}%</div><div class='metric-label'>Test Acc</div></div>",unsafe_allow_html=True)
        with m3: st.markdown(f"<div class='metric-box'><div class='metric-val'>{all_losses[-1]:.4f}</div><div class='metric-label'>Final Loss</div></div>",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(plot_loss(all_losses),use_container_width=True)
        with c2: st.plotly_chart(plot_boundary(X,y,model),use_container_width=True)
        for i,W in enumerate(model.W):
            lbl="Output" if i==len(model.W)-1 else f"Hidden {i+1}"
            with st.expander(f"Layer {i+1} → {lbl}  |  shape {W.shape}"):
                st.dataframe(pd.DataFrame(W).round(4),use_container_width=True)
