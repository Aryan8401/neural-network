
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SOM:
    def __init__(self,gh,gw,dim,lr=0.5,sigma=None,epochs=1000):
        self.gh=gh; self.gw=gw; self.dim=dim; self.lr0=lr; self.sigma0=sigma or max(gh,gw)/2; self.epochs=epochs
        np.random.seed(42); self.weights=np.random.rand(gh,gw,dim)
        self.positions=np.array([[i,j] for i in range(gh) for j in range(gw)],dtype=float)
        self.loss_history=[]
    def _bmu(self,x): d=np.linalg.norm(self.weights-x,axis=2); f=d.argmin(); return np.unravel_index(f,(self.gh,self.gw))
    def _h(self,bmu,sigma):
        d=np.linalg.norm(self.positions-np.array(bmu),axis=1)
        return np.exp(-d**2/(2*sigma**2)).reshape(self.gh,self.gw,1)
    def train(self,X,cb=None):
        self.loss_history=[]; n=len(X)
        for ep in range(self.epochs):
            t=ep/self.epochs; lr=self.lr0*np.exp(-t*3); sigma=self.sigma0*np.exp(-t*3)
            el=0
            for i in np.random.permutation(n):
                x=X[i]; bmu=self._bmu(x); h=self._h(bmu,sigma)
                self.weights+=lr*h*(x-self.weights); el+=np.linalg.norm(x-self.weights[bmu])
            self.loss_history.append(el/n)
            if cb and ep%max(1,self.epochs//100)==0: cb(ep,el/n)
    def map_data(self,X): return [self._bmu(x) for x in X]
    def u_matrix(self):
        U=np.zeros((self.gh,self.gw))
        for i in range(self.gh):
            for j in range(self.gw):
                nb=[]; w=self.weights[i,j]
                if i>0: nb.append(self.weights[i-1,j])
                if i<self.gh-1: nb.append(self.weights[i+1,j])
                if j>0: nb.append(self.weights[i,j-1])
                if j<self.gw-1: nb.append(self.weights[i,j+1])
                if nb: U[i,j]=np.mean([np.linalg.norm(w-nb2) for nb2 in nb])
        return U

def run():
    c1,c2,c3=st.columns(3)
    with c1: gh=st.slider("Grid H",3,20,8); gw=st.slider("Grid W",3,20,8)
    with c2: lr=st.slider("LR",0.01,1.0,0.5,0.01); epochs=st.slider("Epochs",50,2000,300,step=50)
    with c3: dataset=st.selectbox("Dataset",["2D Blobs","Iris (4D)","Random"]); sigma=st.slider("Sigma",0.5,float(max(gh,gw)),float(max(gh,gw)//2),0.5)
    np.random.seed(0)
    if dataset=="2D Blobs":
        from sklearn.datasets import make_blobs; X,y=make_blobs(400,centers=5,random_state=0)
        X=(X-X.min(0))/(X.ptp(0)+1e-8)
    elif dataset=="Iris (4D)":
        from sklearn.datasets import load_iris; from sklearn.preprocessing import MinMaxScaler
        d=load_iris(); X=MinMaxScaler().fit_transform(d.data); y=d.target
    else: X=np.random.randn(300,2); X=(X-X.min(0))/(X.ptp(0)+1e-8); y=np.zeros(len(X),int)
    st.markdown(f"**Data:** `{X.shape}` | **Grid:** `{gh}×{gw}`")
    if st.button("🚀 Train SOM",use_container_width=True):
        som=SOM(gh,gw,X.shape[1],lr,sigma,epochs); progress=st.progress(0); status=st.empty()
        def cb(ep,loss): progress.progress(ep/epochs); status.markdown(f"Epoch `{ep}/{epochs}` Error: `{loss:.5f}`")
        som.train(X,cb=cb); progress.empty(); status.empty()
        m1,m2=st.columns(2)
        with m1: st.markdown(f"<div class='metric-box'><div class='metric-val'>{som.loss_history[-1]:.4f}</div><div class='metric-label'>Quant. Error</div></div>",unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-box'><div class='metric-val'>{gh*gw}</div><div class='metric-label'>Neurons</div></div>",unsafe_allow_html=True)
        fig=go.Figure(go.Scatter(y=som.loss_history,mode="lines",line=dict(color="#fd7900",width=2)))
        fig.update_layout(plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="#e8e8f0",height=240,margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig,use_container_width=True)
        U=som.u_matrix()
        fig2=go.Figure(go.Heatmap(z=U,colorscale="Hot",reversescale=True))
        fig2.update_layout(plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="#e8e8f0",height=360,margin=dict(l=20,r=20,t=30,b=20),title="U-Matrix")
        st.plotly_chart(fig2,use_container_width=True)
        n_dims=min(X.shape[1],4); cp_cols=st.columns(n_dims)
        for d in range(n_dims):
            fig_c=go.Figure(go.Heatmap(z=som.weights[:,:,d],colorscale="Viridis",showscale=False))
            fig_c.update_layout(plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="#e8e8f0",height=180,title=f"Dim {d+1}",margin=dict(l=5,r=5,t=40,b=5))
            with cp_cols[d]: st.plotly_chart(fig_c,use_container_width=True)
