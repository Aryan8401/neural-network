
import streamlit as st
import numpy as np
import plotly.graph_objects as go

def sigmoid(x):   return 1/(1+np.exp(-np.clip(x,-500,500)))
def sigmoid_d(x): s=sigmoid(x); return s*(1-s)
def relu(x):      return np.maximum(0,x)
def relu_d(x):    return (x>0).astype(float)
ACTS={"ReLU":(relu,relu_d),"Sigmoid":(sigmoid,sigmoid_d)}

class AE:
    def __init__(self,input_dim,enc_dims,latent_dim,act="ReLU",lr=0.001):
        self.lr=lr; self.act,self.act_d=ACTS[act]; self.out_act,self.out_d=sigmoid,sigmoid_d
        enc=[input_dim]+list(enc_dims)+[latent_dim]; dec=[latent_dim]+list(reversed(enc_dims))+[input_dim]
        np.random.seed(42)
        self.enc_W=[np.random.randn(enc[i],enc[i+1])*np.sqrt(2/enc[i]) for i in range(len(enc)-1)]
        self.enc_b=[np.zeros((1,enc[i+1])) for i in range(len(enc)-1)]
        self.dec_W=[np.random.randn(dec[i],dec[i+1])*np.sqrt(2/dec[i]) for i in range(len(dec)-1)]
        self.dec_b=[np.zeros((1,dec[i+1])) for i in range(len(dec)-1)]
    def _fwd(self,X,Ws,bs,out=False):
        A=X; acts=[A]; zs=[]
        for i,(W,b) in enumerate(zip(Ws,bs)):
            z=A@W+b; zs.append(z)
            A=(self.out_act(z) if (i==len(Ws)-1 and out) else self.act(z)); acts.append(A)
        return acts,zs
    def encode(self,X): a,_=self._fwd(X,self.enc_W,self.enc_b); return a[-1]
    def forward(self,X):
        ea,ez=self._fwd(X,self.enc_W,self.enc_b); da,dz=self._fwd(ea[-1],self.dec_W,self.dec_b,out=True)
        return ea,ez,da,dz
    def _bwd(self,dA,zs,acts,Ws,out=False):
        dWs=[None]*len(Ws); dbs=[None]*len(Ws)
        for i in reversed(range(len(Ws))):
            dZ=dA*(self.out_d(zs[i]) if (i==len(Ws)-1 and out) else self.act_d(zs[i]))
            dWs[i]=acts[i].T@dZ/acts[i].shape[0]; dbs[i]=dZ.mean(0,keepdims=True); dA=dZ@Ws[i].T
        return dWs,dbs,dA
    def train_step(self,X):
        ea,ez,da,dz=self.forward(X); recon=da[-1]; loss=np.mean((recon-X)**2)
        dA=2*(recon-X)/X.shape[0]
        dWd,dbd,dAl=self._bwd(dA,dz,da,self.dec_W,out=True)
        dWe,dbe,_=self._bwd(dAl,ez,ea,self.enc_W)
        for W,dW,b,db in zip(self.enc_W,dWe,self.enc_b,dbe): W-=self.lr*dW; b-=self.lr*db
        for W,dW,b,db in zip(self.dec_W,dWd,self.dec_b,dbd): W-=self.lr*dW; b-=self.lr*db
        return loss

def run():
    from sklearn.datasets import make_blobs,make_moons
    c1,c2,c3=st.columns(3)
    with c1: dataset=st.selectbox("Dataset",["Blobs","Moons"]); n=st.slider("Samples",100,2000,500)
    with c2: latent=st.slider("Latent Dim",1,8,2); act=st.selectbox("Activation",["ReLU","Sigmoid"])
    with c3: lr=st.select_slider("LR",[0.1,0.01,0.001,0.0001],0.01); epochs=st.slider("Epochs",10,2000,500)
    np.random.seed(0)
    if dataset=="Blobs": X,y=make_blobs(n,centers=4,random_state=0)
    else: X,y=make_moons(n,noise=0.1,random_state=0)
    X=(X-X.min(0))/(X.ptp(0)+1e-8); X=X.astype(np.float32)
    enc_dims=[8]; arch=f"Input(2) → {enc_dims} → Latent({latent}) → {list(reversed(enc_dims))} → Output(2)"
    st.code(arch)
    if st.button("🚀 Train Autoencoder",use_container_width=True):
        ae=AE(2,enc_dims,latent,act,lr); progress=st.progress(0); status=st.empty(); losses=[]
        for ep in range(1,epochs+1):
            loss=ae.train_step(X); losses.append(loss)
            if ep%max(1,epochs//100)==0: progress.progress(ep/epochs); status.markdown(f"Epoch `{ep}` Loss: `{loss:.6f}`")
        progress.empty(); status.empty()
        _,_,da,_=ae.forward(X); X_r=da[-1]; Z=ae.encode(X)
        m1,m2=st.columns(2)
        with m1: st.markdown(f"<div class='metric-box'><div class='metric-val'>{losses[-1]:.5f}</div><div class='metric-label'>Final MSE</div></div>",unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-box'><div class='metric-val'>{latent}</div><div class='metric-label'>Latent Dims</div></div>",unsafe_allow_html=True)
        fig=go.Figure(go.Scatter(y=losses,mode="lines",line=dict(color="#fa709a",width=2)))
        fig.update_layout(plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="#e8e8f0",height=250,margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig,use_container_width=True)
        colors=["#6c63ff","#ff6584","#43e97b","#f7971e"]
        c1,c2=st.columns(2)
        with c1:
            fig2=go.Figure()
            for lbl in np.unique(y): m=y==lbl; fig2.add_trace(go.Scatter(x=X[m,0],y=X[m,1],mode="markers",marker=dict(size=5,color=colors[lbl%4]),name=f"C{lbl}"))
            fig2.update_layout(title="Original",plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="#e8e8f0",height=300,margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig2,use_container_width=True)
        with c2:
            fig3=go.Figure()
            for lbl in np.unique(y): m=y==lbl; fig3.add_trace(go.Scatter(x=X_r[m,0],y=X_r[m,1],mode="markers",marker=dict(size=5,color=colors[lbl%4],opacity=0.7),name=f"C{lbl}"))
            fig3.update_layout(title="Reconstructed",plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="#e8e8f0",height=300,margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig3,use_container_width=True)
        if latent==2:
            fig4=go.Figure()
            for lbl in np.unique(y): m=y==lbl; fig4.add_trace(go.Scatter(x=Z[m,0],y=Z[m,1],mode="markers",marker=dict(size=6,color=colors[lbl%4]),name=f"C{lbl}"))
            fig4.update_layout(title="Latent Space",plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="#e8e8f0",height=340,margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig4,use_container_width=True)
