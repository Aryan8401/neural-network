
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

try:
    import cv2
    CV2_OK=True
except: CV2_OK=False
try:
    from PIL import Image
    PIL_OK=True
except: PIL_OK=False

def pil_to_cv(p): return cv2.cvtColor(np.array(p.convert("RGB")),cv2.COLOR_RGB2BGR)
def cv_to_rgb(c): return cv2.cvtColor(c,cv2.COLOR_BGR2RGB)

def run():
    if not CV2_OK: st.error("pip install opencv-contrib-python"); return
    if not PIL_OK: st.error("pip install Pillow"); return
    if "registered_faces" not in st.session_state: st.session_state.registered_faces={}
    if "attendance_log" not in st.session_state: st.session_state.attendance_log=[]
    tab1,tab2,tab3=st.tabs(["Register Faces","Mark Attendance","Attendance Log"])
    with tab1:
        name=st.text_input("Person Name")
        uploaded=st.file_uploader("Upload face photos",type=["png","jpg","jpeg"],accept_multiple_files=True)
        scale=st.slider("Scale Factor",1.05,1.5,1.1,0.05); min_n=st.slider("Min Neighbors",1,10,5)
        cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
        if st.button("Register") and name and uploaded:
            crops=[]
            for uf in uploaded:
                img=pil_to_cv(Image.open(uf)); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces=cascade.detectMultiScale(gray,scale,min_n,minSize=(30,30))
                if len(faces)>0:
                    x,y,w,h=faces[0]; crops.append(cv2.resize(gray[y:y+h,x:x+w],(100,100)))
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                st.image(cv_to_rgb(img),caption=uf.name,width=180)
            if crops:
                if name not in st.session_state.registered_faces: st.session_state.registered_faces[name]=[]
                st.session_state.registered_faces[name].extend(crops); st.success(f"Registered {name} with {len(crops)} face(s).")
            else: st.warning("No faces detected.")
        for n,faces in st.session_state.registered_faces.items(): st.markdown(f"- **{n}** — {len(faces)} sample(s)")
    with tab2:
        if not st.session_state.registered_faces: st.info("Register faces first.")
        else:
            test=st.file_uploader("Test image",type=["png","jpg","jpeg"],key="test")
            conf_thresh=st.slider("Confidence Threshold",0,150,80)
            if test and st.button("Detect & Mark"):
                img=pil_to_cv(Image.open(test)); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                recognizer=cv2.face.LBPHFaceRecognizer_create()
                labels,faces=[],[]
                label2name={}; lbl=0
                for nm,fcs in st.session_state.registered_faces.items():
                    label2name[lbl]=nm
                    for f in fcs: faces.append(f); labels.append(lbl)
                    lbl+=1
                try: recognizer.train(faces,np.array(labels))
                except Exception as e: st.error(str(e)); st.stop()
                cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
                detected=cascade.detectMultiScale(gray,1.1,5,minSize=(30,30))
                present=set(); results=[]
                for i,(x,y,w,h) in enumerate(detected if len(detected) else []):
                    roi=cv2.resize(gray[y:y+h,x:x+w],(100,100))
                    lbl_p,conf=recognizer.predict(roi); nm_p=label2name.get(lbl_p,"Unknown")
                    ok=conf<conf_thresh; color=(0,255,0) if ok else (0,0,255)
                    cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                    cv2.putText(img,f"{nm_p}({conf:.0f})",(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                    if ok: present.add(nm_p)
                    results.append({"Face":f"#{i+1}","Name":nm_p,"Confidence":round(conf,2),"Status":"✅" if ok else "❌"})
                st.image(cv_to_rgb(img),use_container_width=True)
                if results: st.dataframe(pd.DataFrame(results),use_container_width=True,hide_index=True)
                now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for nm in st.session_state.registered_faces:
                    st.session_state.attendance_log.append({"Timestamp":now,"Name":nm,"Status":"Present" if nm in present else "Absent"})
                st.success(f"Present: {', '.join(present) if present else 'None'}")
    with tab3:
        if not st.session_state.attendance_log: st.info("No records yet.")
        else:
            df=pd.DataFrame(st.session_state.attendance_log)
            st.dataframe(df,use_container_width=True,hide_index=True)
            if st.button("Clear Log"): st.session_state.attendance_log=[]; st.rerun()
