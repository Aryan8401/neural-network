
import streamlit as st
import numpy as np
import plotly.graph_objects as go

class CharRNN:
    def __init__(self,vocab_size,hidden_size=64,lr=0.01):
        self.V=vocab_size; self.H=hidden_size; self.lr=lr
        np.random.seed(42); s=0.01
        self.Wxh=np.random.randn(hidden_size,vocab_size)*s
        self.Whh=np.random.randn(hidden_size,hidden_size)*s
        self.Why=np.random.randn(vocab_size,hidden_size)*s
        self.bh=np.zeros((hidden_size,1)); self.by=np.zeros((vocab_size,1))
        self.mWxh=np.zeros_like(self.Wxh); self.mWhh=np.zeros_like(self.Whh)
        self.mWhy=np.zeros_like(self.Why); self.mbh=np.zeros_like(self.bh); self.mby=np.zeros_like(self.by)

    def forward(self,inputs,hprev):
        xs,hs,ys,ps={},{},{},{}; hs[-1]=hprev.copy(); loss=0
        for t,ix in enumerate(inputs[:-1]):
            xs[t]=np.zeros((self.V,1)); xs[t][ix]=1
            hs[t]=np.tanh(self.Wxh@xs[t]+self.Whh@hs[t-1]+self.bh)
            ys[t]=self.Why@hs[t]+self.by
            ey=np.exp(ys[t]-ys[t].max()); ps[t]=ey/ey.sum()
            loss+=-np.log(ps[t][inputs[t+1],0]+1e-8)
        return loss,xs,hs,ps

    def backward(self,inputs,xs,hs,ps):
        dWxh=np.zeros_like(self.Wxh); dWhh=np.zeros_like(self.Whh)
        dWhy=np.zeros_like(self.Why); dbh=np.zeros_like(self.bh); dby=np.zeros_like(self.by)
        dhnext=np.zeros_like(hs[0])
        for t in reversed(range(len(inputs)-1)):
            dy=ps[t].copy(); dy[inputs[t+1]]-=1
            dWhy+=dy@hs[t].T; dby+=dy
            dh=self.Why.T@dy+dhnext; dhraw=(1-hs[t]**2)*dh
            dbh+=dhraw; dWxh+=dhraw@xs[t].T; dWhh+=dhraw@hs[t-1].T; dhnext=self.Whh.T@dhraw
        for d in [dWxh,dWhh,dWhy,dbh,dby]: np.clip(d,-5,5,out=d)
        return dWxh,dWhh,dWhy,dbh,dby

    def adagrad(self,grads):
        for p,dp,m in [(self.Wxh,grads[0],self.mWxh),(self.Whh,grads[1],self.mWhh),
                       (self.Why,grads[2],self.mWhy),(self.bh,grads[3],self.mbh),(self.by,grads[4],self.mby)]:
            m+=dp*dp; p-=self.lr*dp/(np.sqrt(m)+1e-8)

    def sample(self,seed_ix,n,temperature=1.0):
        h=np.zeros((self.H,1)); x=np.zeros((self.V,1)); x[seed_ix]=1; result=[seed_ix]
        for _ in range(n):
            h=np.tanh(self.Wxh@x+self.Whh@h+self.bh); y=(self.Why@h+self.by)/temperature
            ey=np.exp(y-y.max()); p=(ey/ey.sum()).ravel(); ix=np.random.choice(len(p),p=p)
            x=np.zeros((self.V,1)); x[ix]=1; result.append(ix)
        return result

    def train_step(self,inputs,hprev):
        loss,xs,hs,ps=self.forward(inputs,hprev)
        grads=self.backward(inputs,xs,hs,ps); self.adagrad(grads)
        return loss,hs[len(inputs)-2]

SAMPLE_TEXTS={
    "Hello World":"hello world hello world hello world hello world hello world "*4,
    "Days":"monday tuesday wednesday thursday friday saturday sunday "*8,
    "Numbers":"one two three four five six seven eight nine ten "*6,
    "Custom":""
}

def run():
    preset=st.selectbox("Preset sequence",list(SAMPLE_TEXTS.keys()))
    if preset=="Custom": text=st.text_area("Enter custom text (min 30 chars)",height=120)
    else: text=SAMPLE_TEXTS[preset]; st.text_area("Preview",text,height=80,disabled=True)
    text=text.strip()
    if len(text)<20: st.warning("Need at least 20 characters."); return
    c1,c2,c3=st.columns(3)
    with c1: hidden_size=st.select_slider("Hidden Size",[16,32,64,128,256],64); seq_len=st.slider("Sequence Length",5,50,20)
    with c2: lr=st.select_slider("Learning Rate",[0.1,0.01,0.001],0.01); epochs=st.slider("Iterations",100,5000,1000,step=100)
    with c3: temperature=st.slider("Temperature",0.1,2.0,0.8,0.1); gen_len=st.slider("Generate Chars",50,500,200)
    chars=sorted(set(text)); vocab={c:i for i,c in enumerate(chars)}; ix2char={i:c for c,i in vocab.items()}
    data=[vocab[c] for c in text]; V=len(chars)
    st.markdown(f"**Vocab:** `{V}` chars | **Corpus:** `{len(data)}`")
    st.code("".join(chars),language=None)
    if st.button("🚀 Train RNN", use_container_width=True):
        rnn=CharRNN(V,hidden_size,lr); hprev=np.zeros((hidden_size,1))
        smooth=-np.log(1/V); losses=[]; progress=st.progress(0); status=st.empty()
        for i in range(epochs):
            start=(i*seq_len)%(len(data)-seq_len-1); inputs=data[start:start+seq_len+1]
            loss,hprev=rnn.train_step(inputs,hprev); smooth=0.999*smooth+0.001*loss; losses.append(smooth)
            if i%max(1,epochs//100)==0: progress.progress((i+1)/epochs); status.markdown(f"Step `{i+1}/{epochs}` Loss: `{smooth:.4f}`")
        progress.empty(); status.empty()
        gen=rnn.sample(vocab.get(text[0],0),gen_len,temperature)
        gen_text="".join(ix2char[ix] for ix in gen)
        m1,m2=st.columns(2)
        with m1: st.markdown(f"<div class='metric-box'><div class='metric-val'>{smooth:.4f}</div><div class='metric-label'>Final Loss</div></div>",unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-box'><div class='metric-val'>{np.exp(smooth):.2f}</div><div class='metric-label'>Perplexity</div></div>",unsafe_allow_html=True)
        fig=go.Figure(go.Scatter(y=losses,mode="lines",line=dict(color="#38b2f5",width=2)))
        fig.update_layout(plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="#e8e8f0",
                          xaxis_title="Iteration",yaxis_title="Loss",xaxis=dict(gridcolor="#1a1a26"),
                          yaxis=dict(gridcolor="#1a1a26"),margin=dict(l=20,r=20,t=30,b=20),height=280)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown(f"<div style='background:#111118;border:1px solid #2a2a40;border-radius:12px;padding:1.2rem;font-family:monospace;font-size:0.9rem;color:#43e97b;'>{gen_text}</div>",unsafe_allow_html=True)
        seed_input=st.text_input("Predict next char from seed:",text[:8])
        if seed_input and all(c in vocab for c in seed_input):
            h=np.zeros((hidden_size,1))
            for c in seed_input:
                x=np.zeros((V,1)); x[vocab[c]]=1; h=np.tanh(rnn.Wxh@x+rnn.Whh@h+rnn.bh)
            y=rnn.Why@h+rnn.by; ey=np.exp(y-y.max()); probs=(ey/ey.sum()).ravel()
            top5=probs.argsort()[-5:][::-1]
            for rank,idx in enumerate(top5,1):
                st.markdown(f"`{rank}.` `'{ix2char[idx]}'` — {probs[idx]*100:.2f}%")
