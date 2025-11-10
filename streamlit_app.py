# ============================================================
# Delivery Mood (NLP + Voice)
# Full A3 build — fixed for Streamlit Cloud
# ============================================================

import streamlit as st
st.set_page_config(page_title="Delivery Mood (NLP + Voice)", page_icon="🛵", layout="wide")

# ------------------------------------------------------------
# Imports (only after page_config)
# ------------------------------------------------------------
import time, io, re, json
from collections import deque
import numpy as np
import pandas as pd
import requests

# Optional imports
try:
    import langid
    HAS_LANGID = True
except Exception:
    HAS_LANGID = False

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

# ------------------------------------------------------------
# Secrets & Config
# ------------------------------------------------------------
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", None)
WHISPER_MODEL = st.secrets.get("WHISPER_MODEL", "distil-whisper/distil-large-v3")

# ------------------------------------------------------------
# Lexicons
# ------------------------------------------------------------
ROMAN_INDIC_HINTS = {
    "te": {"machaa","ra","rey","inka","entha","em","cheyyi","unnaa","bokka","pani"},
    "hi": {"bhai","jaldi","aana","bhejo","bahut","karna","nahi","kya","kyun"},
    "ta": {"dei","da","enna","poda","thambi","sapadu","seri"},
    "kn": {"maga","yenu","banni","beda","maadi"},
    "ml": {"eda","mone","entha","cheyya","poyi","vallam"},
}

BAD_WORDS = {"generic":{"idiot","stupid","useless","trash","damn","shit"},
             "chat":{"bokka","pichhi","pani","sala","chutiya"}}

POS_TOK = {"good","great","awesome","nice","tasty","fresh","fast","love","super","mast"}
NEG_TOK = {"bad","late","delay","cold","slow","rude","refund","worst","missing"}

LAUGH_TOK = {"lol","haha","hehe","lmao"}
JOY_EMOJI = {"😊","🙂","😄","😍","🤩","👍"}
NEG_EMOJI = {"😞","😕","☹️","🙁","😔"}
ANGER_EMOJI = {"😡","🤬"}

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def detect_language(text:str) -> str:
    t = text.strip().lower()
    if not t: return "und"
    # Roman heuristics
    scores = {lang:sum(1 for w in vocab if re.search(rf"\b{w}\b",t))
              for lang,vocab in ROMAN_INDIC_HINTS.items()}
    scores = {k:v for k,v in scores.items() if v>0}
    if scores: return max(scores,key=scores.get)
    if HAS_LANGID:
        lid,_ = langid.classify(t); return lid
    return "und"

def sentiment_score(text:str):
    if not text.strip(): return 0,"Neutral","😐",[]
    t = text.lower(); score,trig=0,[]

    for w in POS_TOK:
        if w in t: score+=8; trig.append(w)
    for w in NEG_TOK:
        if w in t: score-=10; trig.append(w)
    if any(e in text for e in JOY_EMOJI): score+=12
    if any(e in text for e in NEG_EMOJI): score-=8
    if any(e in text for e in ANGER_EMOJI): score-=18
    if any(x in t for x in LAUGH_TOK): score=max(score,2)

    score=int(max(-100,min(100,score)))
    if score<=-60: return score,"Angry","😡",trig
    if score<=-30: return score,"Frustrated","😠",trig
    if score<=-6:  return score,"Disappointed","😕",trig
    if score>=45:  return score,"Delighted","🤩",trig
    if score>=15:  return score,"Satisfied","🙂",trig
    return score,"Neutral","😐",trig

def toxicity_flag(text:str)->bool:
    t=text.lower()
    hit=any(re.search(rf"\b{w}\b",t) for w in BAD_WORDS["generic"]|BAD_WORDS["chat"])
    if hit and any(x in t for x in LAUGH_TOK): return False
    return hit

def update_cps(key:str,text:str):
    now=time.time()
    meta=st.session_state.get(key,{"len":0,"ts":now,"cps":0.0})
    dt=max(0.3,now-meta["ts"])
    cps=(len(text)-meta["len"])/dt
    st.session_state[key]={"len":len(text),"ts":now,"cps":max(0.0,cps)}
    return st.session_state[key]["cps"]

# ------------------------------------------------------------
# Whisper API
# ------------------------------------------------------------
def hf_whisper_transcribe(audio:bytes)->dict:
    if not HF_API_TOKEN: raise RuntimeError("Missing HF_API_TOKEN")
    url=f"https://api-inference.huggingface.co/models/{WHISPER_MODEL}"
    r=requests.post(url,headers={"Authorization":f"Bearer {HF_API_TOKEN}"},data=audio,timeout=60)
    r.raise_for_status(); js=r.json()
    txt=(js.get("text") or js[0].get("text","")) if isinstance(js,list) else js.get("text","")
    lang=js.get("language") or detect_language(txt)
    return {"text":txt.strip(),"language":lang}

# ------------------------------------------------------------
# Voice Buffer
# ------------------------------------------------------------
class VoiceBuffer:
    def __init__(self): self.frames=[]; self.meter=deque(maxlen=40)
    def add(self,arr:np.ndarray):
        mono=arr.mean(axis=1) if arr.ndim==2 else arr
        self.frames.append((mono*32767).astype(np.int16).tobytes())
        rms=float(np.sqrt(np.mean(mono**2)+1e-12))
        self.meter.append(rms)
    def clear(self): self.frames.clear(); self.meter.clear()
    def as_wav(self):
        import wave; b=io.BytesIO()
        with wave.open(b,"wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(48000)
            wf.writeframes(b"".join(self.frames))
        return b.getvalue()

if "voicebuf" not in st.session_state: st.session_state.voicebuf=VoiceBuffer()

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tabs=st.tabs(["📊 Dashboard","✍️ Text Review","🎙️ Voice Review","📒 Logs"])

# ---------------- DASHBOARD ----------------
with tabs[0]:
    st.subheader("Live Mood Dashboard")
    snap=st.session_state.get("last_feedback",
        {"source":"—","text":"—","lang":"und","score":0,"mood":"Neutral","emo":"😐","toxic":False,"cps":0.0})
    c1,c2,c3=st.columns(3)
    c1.metric("Latest Mood",f"{snap['emo']} {snap['mood']}")
    c1.metric("Typing speed",f"{snap['cps']:.1f} cps")
    c2.metric("Language",snap["lang"])
    c2.metric("Text Sentiment",f"{snap['score']} ({snap['mood']})")
    c3.metric("Toxicity","Yes" if snap["toxic"] else "No")
    c3.metric("Source",snap["source"])

# ---------------- TEXT REVIEW ----------------
with tabs[1]:
    st.subheader("Review (auto detects language while you type)")

    txt=st.text_area("Type your message",
        value=st.session_state.get("last_text","entha babu entha sepu wait cheyyali"),
        key="txt_in",height=100)

    # instant rerun for live mood
    if txt!=st.session_state.get("prev_txt",""):
        st.session_state["prev_txt"]=txt
        st.experimental_rerun()

    cps=update_cps("cps_meta",txt)
    lang=detect_language(txt)
    score,mood,emo,_=sentiment_score(txt)
    tox=toxicity_flag(txt)
    g=int((score+100)/2)

    st.progress(g,text=f"{mood} • {g}/100")

    c1,c2,c3=st.columns(3)
    c1.metric("Language",lang)
    c2.metric("Sentiment",f"{score} ({emo} {mood})")
    c3.metric("Toxicity","Yes" if tox else "No")

    st.caption(f"Typing speed: **{cps:.1f} chars/sec**")
    st.session_state.setdefault("cps_series",deque(maxlen=50))
    st.session_state.cps_series.append(cps)
    st.line_chart(pd.DataFrame({"cps":list(st.session_state.cps_series)}))

    if st.button("Save → Dashboard",type="primary"):
        st.session_state["last_text"]=txt
        st.session_state["last_feedback"]={"source":"text","text":txt,"lang":lang,
            "score":score,"mood":mood,"emo":emo,"toxic":tox,"cps":cps}
        st.session_state.setdefault("log",[]).append(
            {"ts":time.strftime("%H:%M:%S"),"source":"text","lang":lang,
             "score":score,"mood":mood,"toxic":tox,"text":txt[:100]})
        st.success("Saved to dashboard")

# ---------------- VOICE REVIEW ----------------
with tabs[2]:
    st.subheader("Live Voice (Whisper)")

    if not HAS_WEBRTC:
        st.warning("`streamlit-webrtc` missing; mic disabled.")
    else:
        meter=st.empty()
        def audio_callback(frame):
            arr=frame.to_ndarray().T.astype(np.float32)/32768.0
            st.session_state.voicebuf.add(arr)
            lvl=min(1.0,float(np.mean(st.session_state.voicebuf.meter[-10:])*20))
            meter.progress(int(lvl*100),text="Mic level")
            return frame

        ctx=webrtc_streamer(key="mic",mode=WebRtcMode.SENDONLY,
            audio_receiver_size=512,
            rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video":False,"audio":True},
            audio_frame_callback=audio_callback)

        c1,c2=st.columns(2)
        if c1.button("🧹 Clear Buffer"): st.session_state.voicebuf.clear()
        if c2.button("🧠 Transcribe & Analyze",type="primary"):
            if not HF_API_TOKEN:
                st.error("Add HF_API_TOKEN in Secrets.")
            else:
                with st.spinner("Transcribing..."):
                    try:
                        out=hf_whisper_transcribe(st.session_state.voicebuf.as_wav())
                        text,lang=out["text"],out["language"]
                    except Exception as e:
                        st.exception(e); text,lang="","und"
                st.write("**Transcript:**",text or "—")
                st.write("**Language:**",lang)
                s,m,e,_=sentiment_score(text)
                t=toxicity_flag(text)
                g=int((s+100)/2)
                st.progress(g,text=f"{m} • {g}/100")
                c1,c2=st.columns(2)
                c1.metric("Voice Sentiment",f"{s} ({e} {m})")
                c2.metric("Toxicity","Yes" if t else "No")

                if st.button("Save Voice → Dashboard"):
                    st.session_state["last_feedback"]={"source":"voice","text":text,
                        "lang":lang,"score":s,"mood":m,"emo":e,"toxic":t,
                        "cps":st.session_state.get('cps_meta',{}).get('cps',0.0)}
                    st.session_state.setdefault("log",[]).append(
                        {"ts":time.strftime("%H:%M:%S"),"source":"voice",
                         "lang":lang,"score":s,"mood":m,"toxic":t,"text":text[:100]})
                    st.success("Saved to dashboard")

    if not HF_API_TOKEN:
        st.info("⚠️ Add HF_API_TOKEN in Secrets to enable Whisper ASR.")

# ---------------- LOGS ----------------
with tabs[3]:
    st.subheader("Logs")
    log=st.session_state.get("log",[])
    if not log: st.info("No entries yet.")
    else:
        st.dataframe(pd.DataFrame(log),use_container_width=True,hide_index=True)
        if st.button("Clear Log"): st.session_state["log"]=[]; st.success("Cleared.")

st.caption("Pan-India chat + voice mood detector • auto language, slang & typing meter • HuggingFace Whisper ASR")
