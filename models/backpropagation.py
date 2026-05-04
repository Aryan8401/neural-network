
import streamlit as st
import numpy as np
import pandas as pd

def sigmoid(x):   return 1/(1+np.exp(-np.clip(x,-500,500)))
def sigmoid_d(x): s=sigmoid(x); return s*(1-s)
def relu(x):      return np.maximum(0,x)
def relu_d(x):    return (x>0).astype(float)
def tanh_fn(x):   return np.tanh(x)
def tanh_d(x):    return 1-np.tanh(x)**2

ACTS={"Sigmoid":(sigmoid,sigmoid_d),"ReLU":(relu,relu_d),"Tanh":(tanh_fn,tanh_d)}

def run():
    st.markdown("### 🔧 Network Setup")
    c1,c2,c3=st.columns(3)
    with c1: n_inputs=st.number_input("Input neurons",1,6,2); hidden_act=st.selectbox("Hidden Activation",list(ACTS.keys()))
    with c2: n_hidden=st.number_input("Hidden neurons",1,8,3); out_act=st.selectbox("Output Activation",list(ACTS.keys()),index=0)
    with c3: lr=st.slider("Learning Rate",0.001,1.0,0.1,0.001,format="%.3f"); loss_type=st.selectbox("Loss",["MSE","Binary Cross-Entropy"])
    st.markdown("---"); st.markdown("#### Input Sample")
    x_cols=st.columns(int(n_inputs)); x_vals=[]
    for i,c in enumerate(x_cols):
        with c: x_vals.append(c.number_input(f"x{i+1}",value=round(0.5+i*0.1,2),format="%.3f",key=f"xv_{i}"))
    X=np.array(x_vals); target=st.number_input("Target (y)",0.0,1.0,0.8,0.01,format="%.3f")
    np.random.seed(42)
    W1=np.round(np.random.randn(int(n_inputs),int(n_hidden))*0.5,3)
    b1=np.zeros(int(n_hidden)); W2=np.round(np.random.randn(int(n_hidden),1)*0.5,3); b2=np.zeros(1)
    with st.expander("Edit W1"):
        W1_rows=[]
        for i in range(int(n_inputs)):
            row=[]; rcols=st.columns(int(n_hidden))
            for j,c in enumerate(rcols):
                with c: row.append(c.number_input(f"w1[{i},{j}]",value=float(W1[i,j]),format="%.3f",key=f"w1_{i}_{j}"))
            W1_rows.append(row)
        W1=np.array(W1_rows)
    if st.button("▶️ Run Backpropagation Step", use_container_width=True):
        h_fn,h_d=ACTS[hidden_act]; o_fn,o_d=ACTS[out_act]
        Z1=X@W1+b1; A1=h_fn(Z1); Z2=A1@W2+b2; A2=o_fn(Z2); y_hat=A2[0]
        if loss_type=="MSE": loss=0.5*(y_hat-target)**2; dL_dyhat=y_hat-target
        else:
            eps=1e-8; loss=-(target*np.log(y_hat+eps)+(1-target)*np.log(1-y_hat+eps))
            dL_dyhat=(y_hat-target)/((y_hat+eps)*(1-y_hat+eps))
        dL_dZ2=dL_dyhat*o_d(Z2); dL_dW2=A1.reshape(-1,1)*dL_dZ2
        dL_dA1=dL_dZ2@W2.T; dL_dZ1=dL_dA1*h_d(Z1); dL_dW1=np.outer(X,dL_dZ1)
        W2_new=W2-lr*dL_dW2; W1_new=W1-lr*dL_dW1
        st.markdown("## Forward Pass")
        st.code(f"Z1 = {np.round(Z1,4)}"); st.code(f"A1 = {np.round(A1,4)}")
        st.code(f"Z2 = {np.round(Z2,4)}"); st.code(f"y_hat = {round(y_hat,6)}")
        m1,m2=st.columns(2)
        with m1: st.markdown(f"<div class='metric-box'><div class='metric-val'>{y_hat:.5f}</div><div class='metric-label'>Prediction</div></div>",unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-box'><div class='metric-val'>{loss:.5f}</div><div class='metric-label'>{loss_type} Loss</div></div>",unsafe_allow_html=True)
        st.markdown("## Backward Pass")
        st.code(f"dL/dy_hat = {round(dL_dyhat,6)}")
        st.code(f"dL/dZ2    = {np.round(dL_dZ2,6)}")
        st.code(f"dL/dW2    = {np.round(dL_dW2.flatten(),6)}")
        st.code(f"dL/dZ1    = {np.round(dL_dZ1,6)}")
        st.code(f"dL/dW1 =\n{np.round(dL_dW1,6)}")
        st.markdown("## Weight Update")
        c1,c2=st.columns(2)
        with c1:
            df=pd.DataFrame({"Old W2":W2.flatten(),"Gradient":dL_dW2.flatten(),"New W2":W2_new.flatten()}).round(5)
            st.dataframe(df,use_container_width=True)
        with c2:
            st.markdown(f"W1 updated. Max delta: `{np.abs(dL_dW1*lr).max():.5f}`")
        st.success("One backprop step complete!")
