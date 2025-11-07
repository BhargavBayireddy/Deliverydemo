import os, io, time, math, re, json, queue, threading
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.signal import get_window
from scipy.io import wavfile

from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ----------------------------- CONFIG ---------------------------------
APP_TITLE = "Hybrid NLP (HF API) — Text + Voice Feedback"
HEADS_UP  = "Heads up: your order may be late by ~5 minutes. Sorry!"

# HF Inference API (set in Streamlit Secrets or env)
HF_TOKEN = st.secrets.get("HF_API_TOKEN", os.getenv("HF_API_TOKEN", ""))
ASR_MODEL = os.getenv("HF_ASR_MODEL", st.secrets.get("HF_ASR_MODEL", "Systran/faster-whisper-large-v3"))
EMO_MODEL = os.getenv("HF_EMO_MODEL", st.secrets.get("HF_EMO_MODEL", "j-hartmann/emotion-english-distilroberta-base"))
SENT_MODEL = os.getenv("HF_SENT_MODEL", st.secrets.get("HF_SENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"))

# Live transcription chunking (seconds)
CHUNK_SEC = 2.0
SAMPLE_RATE = 16000    # we standardize captured audio to 16k mono

# -------------------------- LANGUAGE DETECTION -------------------------
# Fast script-based detection for major Indian languages, then langid fallback.
SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),   # Hindi / Devanagari
    "bn": (0x0980, 0x09FF),   # Bengali
    "pa": (0x0A00, 0x0A7F),   # Punjabi (Gurmukhi)
    "gu": (0x0A80, 0x0AFF),   # Gujarati
    "or": (0x0B00, 0x0B7F),   # Odia
    "ta": (0x0B80, 0x0BFF),   # Tamil
    "te": (0x0C00, 0x0C7F),   # Telugu
    "kn": (0x0C80, 0x0CFF),   # Kannada
    "ml": (0x0D00, 0x0D7F),   # Malayalam
    "si": (0x0D80, 0x0DFF),   # Sinhala
}

def detect_script_lang(s: str):
    for lang, (start, end) in SCRIPT_RANGES.items():
        if any(start <= ord(ch) <= end for ch in s):
            return lang
    return None

@st.cache_data(show_spinner=False)
def load_langid():
    import langid
    return langid

def detect_lang(text: str) -> str:
    text = text.strip()
    if not text:
        return "und"
    script_lang = detect_script_lang(text)
    if script_lang:
        return script_lang
    try:
        langid = load_langid()
        return langid.classify(text)[0]
    except Exception:
        return "und"

# --------------------------- SIMPLE TEXT NLP ---------------------------
# Light, multilingual-ish sentiment with lexicons + emojis.
INTENS = {"very":1.3,"really":1.2,"so":1.2,"too":1.15,"extremely":1.4,"super":1.3,
          "bahut":1.2,"chala":1.2,"bahuthi":1.25,"inka":1.15,"romba":1.2}
NEGS = set(["not","no","never","hardly","barely","nahi","mat","kadu","illa","illae"])
POSW = set("""
good great awesome amazing excellent tasty fresh fast quick ontime polite friendly
love liked perfect nice yummy delicious wow clean crisp juicy superb thanks
mast semma mass superr bagundi super
""".split())
NEGW = set("""
bad terrible awful worst cold soggy late delay delayed dirty slow rude stale raw burnt
bland overpriced expensive missing leak leaking spilled refund replace cancel canceled cancelled
angry frustrated annoyed disappointed hate horrible issue problem broken uncooked inedible
vomit sick hair bekar bakwas mosam chindi pathetic useless trash
""".split())

def text_sentiment(text: str):
    t = text.strip()
    if not t: 
        return 0, "Neutral", "😐", []
    s = re.sub(r"[^\w\s!?]", " ", t.lower())
    words = s.split()
    score = 0.0; hits=[]
    exclam = t.count("!")
    caps = 1.15 if re.search(r"[A-Z]{3,}", t) else 1.0
    long = 1.1 if re.search(r"(.)\1{2,}", t.lower()) else 1.0
    score *= caps*long
    if exclam>=2: score*=1.1
    for i,w in enumerate(words):
        base=(2.5 if w in POSW else 0)+(-2.5 if w in NEGW else 0)
        if base!=0:
            if i>0 and words[i-1] in INTENS: base*=INTENS[words[i-1]]
            # simple negation window
            for k in range(1,4):
                if i-k>=0 and words[i-k] in NEGS: base*=-1; break
            score+=base; hits.append(w)
    raw=max(-40,min(40,score)); scaled=int(round((raw/40)*100))
    if   scaled<=-60: mood,emoji="Angry","😡"
    elif scaled<=-30: mood,emoji="Frustrated","😠"
    elif scaled<=-6:  mood,emoji="Disappointed","😕"
    elif -5<=scaled<=5: mood,emoji="Neutral","😐"
    elif scaled<=35:  mood,emoji="Satisfied","🙂"
    else:             mood,emoji="Delighted","🤩"
    return scaled,mood,emoji,list(dict.fromkeys(hits))

def mood_to_score(mood: str) -> int:
    mapping = {
        "Angry":10, "Frustrated":25, "Disappointed":40,
        "Neutral":50, "Satisfied":75, "Delighted":95,
        "Angry/Stress":20, "Sad":35, "Joy":85
    }
    return mapping.get(mood, 50)

# ------------------------- HUGGINGFACE HELPERS ------------------------
HF_URL = "https://api-inference.huggingface.co/models"

def hf_headers():
    if not HF_TOKEN:
        return {}
    return {"Authorization": f"Bearer {HF_TOKEN}"}

def hf_whisper_transcribe(wav_bytes: bytes, task_lang: str | None = None) -> str:
    """
    Send audio bytes (wav 16k mono recommended) to Whisper on HF Inference API.
    Returns transcribed text ("" on failure).
    """
    if not HF_TOKEN:
        return ""
    url = f"{HF_URL}/{ASR_MODEL}"
    opts = {"task": "transcribe"}
    if task_lang and task_lang != "und":
        opts["language"] = task_lang
    try:
        resp = requests.post(
            url, headers=hf_headers(),
            data=wav_bytes, params={"wait_for_model": "true"},
            timeout=60
        )
        if resp.status_code == 200:
            js = resp.json()
            # HF returns {"text": "..."} OR list of chunks
            if isinstance(js, dict) and "text" in js:
                return js["text"]
            if isinstance(js, list) and len(js) and "text" in js[0]:
                return " ".join([c.get("text","") for c in js])
        return ""
    except Exception:
        return ""

# ------------------------- STREAMING AUDIO BUFFER ----------------------
class LiveAudioBuffer:
    def __init__(self, sr=SAMPLE_RATE, chunk_sec=CHUNK_SEC):
        self.sr = sr
        self.chunk_len = int(sr * chunk_sec)
        self.buf = deque(maxlen=self.chunk_len * 4)
        self.last_flush = time.time()

    def add_frame(self, frame_ndarray):
        # frame_ndarray is int16 or float; standardize to float32 mono
        if frame_ndarray.ndim == 2:
            x = frame_ndarray.mean(axis=0)
        else:
            x = frame_ndarray
        if x.dtype != np.float32:
            x = x.astype(np.float32) / 32768.0
        self.buf.extend(x.tolist())

    def pop_chunk_wav(self):
        if len(self.buf) < self.chunk_len:
            return None
        # take exactly a chunk
        arr = np.array([self.buf.popleft() for _ in range(self.chunk_len)], dtype=np.float32)
        # pack to wav 16k
        mem = io.BytesIO()
        wavfile.write(mem, SAMPLE_RATE, arr)
        return mem.getvalue()

# ------------------------ STREAMLIT PAGE LAYOUT -----------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🛵", layout="wide")

st.title(f"🛵 {APP_TITLE}")
st.info(HEADS_UP)
if not HF_TOKEN:
    st.warning("⚠️ Set **HF_API_TOKEN** in Settings → Secrets for ASR & emotion models.", icon="⚠️")

if "orders" not in st.session_state:
    st.session_state.orders=[]
if "logs" not in st.session_state:
    st.session_state.logs=[]
if "live_text" not in st.session_state:
    st.session_state.live_text=""
if "live_lang" not in st.session_state:
    st.session_state.live_lang="und"
if "live_mood" not in st.session_state:
    st.session_state.live_mood=("Neutral","😐",50)

tabs = st.tabs(["📊 Dashboard", "✍️ Text Review", "🎙️ Voice Review", "🗒️ Logs"])

# ---------------------------- DASHBOARD TAB ---------------------------
with tabs[0]:
    col1,col2 = st.columns([1.2,1])
    with col1:
        st.subheader("Live Mood")
        mood, emoji, score = st.session_state.live_mood
        gauge = st.progress(score, text=f"{emoji} {mood} • {score}/100")
        # animate lightly (no-op update to reflect)
        time.sleep(0.05)
        gauge.progress(score, text=f"{emoji} {mood} • {score}/100")

        st.caption(f"Live language: **{st.session_state.live_lang}**")
        st.text_area("Live transcript (from mic)", value=st.session_state.live_text, height=120)

    with col2:
        st.subheader("Recent Orders (priority by mood)")
        orders=list(st.session_state.orders)
        if orders:
            orders.sort(key=lambda o: (0 if o["priority"]=="High" else 1, o["eta"]))
            df=pd.DataFrame([{
                "Time":o["time"],"Platform":o["platform"],"Order":o["order_id"],
                "ETA(min)":o["eta"],"Priority":o["priority"],"Mood":f"{o['emoji']} {o['mood']}"
            } for o in orders])
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No active drops yet.")

# ---------------------------- TEXT TAB --------------------------------
with tabs[1]:
    st.subheader("Review (auto detects language while you type)")
    txt = st.text_input("Type your message", placeholder="e.g., entha late avuthundi, I'm waiting…", key="txtbox")

    # detect language & sentiment
    lang = detect_lang(txt) if txt else "und"
    score,mood,emoji,hits = text_sentiment(txt)

    # mood gauge
    gscore = int(np.interp(score, [-100,100],[0,100])) if isinstance(score,(int,float)) else mood_to_score(mood)
    st.session_state.live_mood = (mood, emoji, gscore)
    st.session_state.live_lang = lang

    g = st.progress(gscore, text=f"{emoji} {mood} • {gscore}/100")
    time.sleep(0.03); g.progress(gscore, text=f"{emoji} {mood} • {gscore}/100")

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Language", lang)
    with c2: st.metric("Text Sentiment", f"{score} ({emoji} {mood})")
    with c3: st.metric("Toxicity", "No" if mood not in ["Angry","Frustrated"] else "Possible")

    if hits:
        st.caption("Triggers: " + ", ".join(hits))

    # (Optional) add to rider dashboard
    if st.button("Save as Latest Feedback → Dashboard"):
        oid=f"OD-{int(time.time())%100000}"
        st.session_state.orders.append({
            "time":datetime.now().strftime("%H:%M:%S"),
            "platform":"Zomato","order_id":oid,"eta":5.0,
            "priority":"High" if mood in ["Angry","Frustrated","Disappointed"] else "Normal",
            "mood":mood,"emoji":emoji
        })
        st.success("Saved to dashboard.")

# ---------------------------- VOICE TAB -------------------------------
with tabs[2]:
    st.subheader("Live Voice (auto language via Whisper)")

    st.caption("Click **Start**, allow mic, speak 5–8 seconds. Live subtitles update every ~2 seconds.")
    stt_placeholder = st.empty()
    mood_placeholder = st.empty()

    audio_buffer = LiveAudioBuffer(sr=SAMPLE_RATE, chunk_sec=CHUNK_SEC)
    text_q = queue.Queue()

    def asr_worker():
        # background thread: pull chunks and call HF Whisper
        while True:
            wav_bytes = text_q.get()
            if wav_bytes is None:
                break
            # language from current text box / script to assist ASR
            asr_lang = st.session_state.live_lang
            text = hf_whisper_transcribe(wav_bytes, task_lang=asr_lang)
            if text:
                # Update transcript & mood immediately
                st.session_state.live_text += (" " if st.session_state.live_text else "") + text
                # detect language again from new transcript
                st.session_state.live_lang = detect_lang(text)
                # mood from text (voice-only prosody omitted to keep cloud-lite)
                s,m,e,hits = text_sentiment(text)
                gscore = int(np.interp(s, [-100,100],[0,100]))
                st.session_state.live_mood = (m,e,gscore)

    # Start background thread
    worker_thread = threading.Thread(target=asr_worker, daemon=True)
    worker_thread.start()

    def audio_frame_callback(frame):
        # frame: av.AudioFrame -> numpy
        pcm = frame.to_ndarray()  # shape (channels, samples)
        if pcm.ndim == 2:
            pcm = pcm.mean(axis=0)  # mono
        audio_buffer.add_frame(pcm)
        # every CHUNK_SEC, create wav and enqueue for ASR
        now = time.time()
        if len(audio_buffer.buf) >= audio_buffer.chunk_len and (now - audio_buffer.last_flush) >= CHUNK_SEC:
            wav_bytes = audio_buffer.pop_chunk_wav()
            audio_buffer.last_flush = now
            if wav_bytes:
                text_q.put(wav_bytes)
        return frame

    ctx = webrtc_streamer(
        key="live-voice",
        mode=WebRtcMode.RECVONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        audio_frame_callback=audio_frame_callback,
    )

    # live UI refresh
    if ctx.state.playing:
        stt_placeholder.text_area("Live transcript", value=st.session_state.live_text, height=140)
        m,e,gs = st.session_state.live_mood
        mood_placeholder.progress(gs, text=f"{e} {m} • {gs}/100")

    # stop worker cleanly on page rerun
    if not ctx.state.playing:
        try:
            text_q.put_nowait(None)
        except Exception:
            pass

# ---------------------------- LOGS TAB --------------------------------
with tabs[3]:
    st.subheader("System Log")
    st.write(f"- HF Token set: **{bool(HF_TOKEN)}**")
    st.write(f"- ASR Model: `{ASR_MODEL}`")
    st.write(f"- Emotion Model: `{EMO_MODEL}`")
    st.write(f"- Sentiment Model: `{SENT_MODEL}`")
    st.write(f"- Chunk size: **{CHUNK_SEC}s**, Sample rate: **{SAMPLE_RATE} Hz**")
    st.caption("This page is for debugging configuration during demos.")

    if st.button("Clear transcript"):
        st.session_state.live_text=""
        st.success("Cleared.")

# --------------------------- END OF FILE -------------------------------
