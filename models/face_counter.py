
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import cv2
    CV2_OK=True
except: CV2_OK=False
try:
    from PIL import Image
    PIL_OK=True
except: PIL_OK=False

def run():
    if not CV2_OK: st.error("pip install opencv-python"); return
    if not PIL_OK: st.error("pip install Pillow"); return
    c1,c2,c3=st.columns(3)
    with c1: scale=st.slider("Scale Factor",1.05,1.5,1.1,0.01); min_n=st.slider("Min Neighbors",1,15,5)
    with c2: min_size=st.slider("Min Face Size",10,150,30); box_style=st.selectbox("Box Style",["solid","brackets"])
    with c3: show_labels=st.checkbox("Show Labels",True); color_hex=st.color_picker("Box Color","#00ff64")
    def hex_bgr(h): h=h.lstrip("#"); return (int(h[4:6],16),int(h[2:4],16),int(h[0:2],16))
    color=hex_bgr(color_hex)
    uploaded=st.file_uploader("Upload image",type=["png","jpg","jpeg"])
    if uploaded:
        pil=Image.open(uploaded); cv_img=cv2.cvtColor(np.array(pil.convert("RGB")),cv2.COLOR_RGB2BGR)
        gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
        col1,col2=st.columns(2)
        with col1: st.markdown("**Original**"); st.image(pil,use_container_width=True)
        if st.button("Detect Faces",use_container_width=True):
            cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
            faces=cascade.detectMultiScale(gray,scale,min_n,minSize=(min_size,min_size))
            out=cv_img.copy()
            for i,(x,y,w,h) in enumerate(faces if len(faces) else []):
                if box_style=="solid": cv2.rectangle(out,(x,y),(x+w,y+h),color,2)
                else:
                    t=15
                    for pt in [(x,y,x+t,y),(x,y,x,y+t),(x+w,y,x+w-t,y),(x+w,y,x+w,y+t),
                               (x,y+h,x+t,y+h),(x,y+h,x,y+h-t),(x+w,y+h,x+w-t,y+h),(x+w,y+h,x+w,y+h-t)]:
                        cv2.line(out,(pt[0],pt[1]),(pt[2],pt[3]),color,3)
                if show_labels: cv2.putText(out,f"#{i+1}",(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2)
            with col2: st.markdown("**Detected**"); st.image(cv2.cvtColor(out,cv2.COLOR_BGR2RGB),use_container_width=True)
            n=len(faces) if len(faces) else 0
            st.markdown(f"<div class='metric-box'><div class='metric-val' style='font-size:3rem'>{n}</div><div class='metric-label'>Faces Detected</div></div>",unsafe_allow_html=True)
            if n>0:
                rows=[{"Face":f"#{i+1}","x":x,"y":y,"W":w,"H":h,"Area":w*h} for i,(x,y,w,h) in enumerate(faces)]
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                fig=go.Figure(go.Bar(x=[r["Face"] for r in rows],y=[r["Area"] for r in rows],marker_color="#6c63ff"))
                fig.update_layout(plot_bgcolor="#0a0a0f",paper_bgcolor="#0a0a0f",font_color="#e8e8f0",height=240,margin=dict(l=20,r=20,t=30,b=20))
                st.plotly_chart(fig,use_container_width=True)
