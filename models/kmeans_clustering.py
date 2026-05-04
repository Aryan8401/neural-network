
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_blobs,make_moons,make_circles

COLORS=["#6c63ff","#ff6584","#43e97b","#f7971e","#38b2f5","#f093fb","#fa709a","#a18cd1","#ffecd2","#4facfe"]

class KMeans:
    def __init__(self,k,init="K-Means++",max_iter=100,tol=1e-4):
        self.k=k; self.init=init; self.max_iter=max_iter; self.tol=tol
        self.centroids=self.labels=None; self.inertia_history=[]; self.centroid_history=[]; self.label_history=[]
    def _init_c(self,X):
        np.random.seed(42)
        if self.init=="Random": return X[np.random.choice(len(X),self.k,replace=False)].copy()
        centers=[X[np.random.randint(len(X))]]
        for _ in range(self.k-1):
            d=np.array([min(np.linalg.norm(x-c)**2 for c in centers) for x in X])
            centers.append(X[np.random.choice(len(X),p=d/d.sum())])
        return np.array(centers)
    def fit(self,X):
        self.centroids=self._init_c(X); self.inertia_history=[]; self.centroid_history=[self.centroids.copy()]; self.label_history=[]
        for _ in range(self.max_iter):
            d=np.linalg.norm(X[:,None]-self.centroids[None],axis=2); labels=d.argmin(1); self.label_history.append(labels.copy())
            inertia=sum(np.linalg.norm(X[labels==k]-self.centroids[k])**2 for k in range(self.k) if (labels==k).any())
            self.inertia_history.append(inertia)
            nc=np.array([X[labels==k].mean(0) if (labels==k).any() else self.centroids[k] for k in range(self.k)])
            self.centroid_history.append(nc.copy()); shift=np.linalg.norm(nc-self.centroids); self.centroids=nc; self.labels=labels
            if shift<self.tol: break
        return self

def run():
    c1,c2,c3=st.columns(3)
    with c1: dataset=st.selectbox("Dataset",["Blobs","Uniform","Moons","Circles"]); n=st.slider("Samples",50,1000,300)
    with c2: k=st.slider("K (Clusters)",2,10,3); init=st.selectbox("Init",["K-Means++","Random"])
    with c3: max_iter=st.slider("Max Iterations",5,100,30); run_elbow=st.checkbox("Elbow Method",True)
    np.random.seed(42)
    if dataset=="Blobs": X,_=make_blobs(n,centers=k,random_state=42)
    elif dataset=="Moons": X,_=make_moons(n,noise=0.1,random_state=42)
    elif dataset=="Circles": X,_=make_circles(n,noise=0.05,factor=0.4,random_state=42)
    else: X=np.random.rand(n,2)*10
    if st.button("🚀 Run K-Means",use_container_width=True):
        km=KMeans(k,init,max_iter); km.fit(X)
        n_it=len(km.label_history)
        m1,m2,m3=st.columns(3)
        with m1: st.markdown(f"<div class='metric-box'><div class='metric-val'>{n_it}</div><div class='metric-label'>Iterations</div></div>",unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-box'><div class='metric-val'>{km.inertia_history[-1]:.1f}</div><div class='metric-label'>Final Inertia</div></div>",unsafe_allow_html=True)
        with m3: st.markdown(f"<div class='metric-box'><div class='metric-val'>{k}</div><div class='metric-label'>Clusters</div></div>",unsafe_allow_html=True)
        step=st.slider("Iteration",1,n_it,n_it)
        labels=km.label_history[step-1]; centroids=km.centroid_history[step]
        fig=go.Figure()
        for ki in range(k):
            mask=labels==ki; fig.add_trace(go.Scatter(x=X[mask,0],y=X[mask,1],mode="markers",marker=dict(size=5,color=COLORS[ki%len(COLORS)],opacity=0.7),name=f"C{ki+1}"))
        fig.add_trace(go.Scatter(x=centroids[:,0],y=centroids[:,1],mode="markers",marker=dict(size=14,color=[COLORS[i%len(COLORS)] for i in range(k)],symbol="x",line=dict(width=2,color="white")),name="Centroids"))
        fig.update_layout(plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="#e8e8f0",height=380,xaxis=dict(gridcolor="#1a1a26"),yaxis=dict(gridcolor="#1a1a26"),margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig,use_container_width=True)
        fig2=go.Figure(go.Scatter(x=list(range(1,len(km.inertia_history)+1)),y=km.inertia_history,mode="lines+markers",line=dict(color="#a18cd1",width=2),marker=dict(size=6,color="#f7971e")))
        fig2.update_layout(plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="#e8e8f0",xaxis_title="Iteration",yaxis_title="Inertia",xaxis=dict(gridcolor="#1a1a26"),yaxis=dict(gridcolor="#1a1a26"),margin=dict(l=20,r=20,t=30,b=20),height=260)
        st.plotly_chart(fig2,use_container_width=True)
        if run_elbow:
            k_range=range(1,min(11,n//10+2)); inertias=[]
            for ki in k_range:
                km_e=KMeans(ki,init,max_iter); km_e.fit(X); inertias.append(km_e.inertia_history[-1])
            fig3=go.Figure(go.Scatter(x=list(k_range),y=inertias,mode="lines+markers",line=dict(color="#43e97b",width=2),marker=dict(size=8,color="#ff6584")))
            fig3.update_layout(plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="#e8e8f0",xaxis_title="K",yaxis_title="Inertia",xaxis=dict(gridcolor="#1a1a26",dtick=1),yaxis=dict(gridcolor="#1a1a26"),margin=dict(l=20,r=20,t=30,b=20),height=280)
            st.plotly_chart(fig3,use_container_width=True)
        rows=[{"Cluster":ki+1,"Size":(km.labels==ki).sum(),"Cx":round(km.centroids[ki,0],4),"Cy":round(km.centroids[ki,1],4)} for ki in range(k)]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
