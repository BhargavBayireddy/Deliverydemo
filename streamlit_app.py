# -------------------- Imports --------------------
import os, io, re, time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go

from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
from scipy.io.wavfile import write as wav_write

# Lingua language detector (safe dynamic import)
from lingua import Language, LanguageDetectorBuilder

# -------------------- Page config (must be first Streamlit call) --------------------
st.set_page_config(page_title="Delivery Mood (NLP + Voice)", page_icon="📦", layout="wide")

# -------------------- Build language detector safely --------------------
_possible = [
    "ENGLISH","HINDI","BENGALI","MARATHI","TELUGU","TAMIL",
    "KANNADA","MALAYALAM","GUJARATI","PUNJABI","URDU"
]
LANGUAGES = []
for name in _possible:
    if hasattr(Language, name):
        LANGUAGES.append(getattr(Language, name))
DETECTOR = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()

# -------------------- Globals --------------------
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", os.getenv("HF_API_TOKEN", ""))
VADER = SentimentIntensityAnalyzer()

HINGLISH_POS = {
    "mast","bahut accha","bahut acha","badiya","solid","awesome","masttt","op",
    "sahi","thanku","thanks","shukriya","great","nice","fast","jaldi","on time",
    "super","superb"
}
HINGLISH_NEG = {
    "late","faltu","ghatiya","bakwas","bekar","worst","slow","thanda","cold",
    "over spicy","chipku","smell","stale","basi","bokka","entha late","avuthrhadhi",
    "vasta ledu","ledu"
}
BAD_WORDS = {
    "idiot","stupid","dumb","bloody","hell","shit","crap",
    "bhosdi","chutiya","madarchod","haraami","harami","kutta","gandu","saala",
    "puka","thendi","panni","naayi"
}
FUNNY_MARKERS = {"lol","lmao","rofl","😂","🤣","hehe","haha","xD","😅","😆","😜","bro","bhai","anna","ra"}

# -------------------- Helpers --------------------
@dataclass
class NLPResult:
    lang_code: str
    sentiment_score: float
    toxicity: bool
    funny_context: bool
    mood_score: float
    label: str

def normalize_roman(s: str) -> str:
    s = s.lower()
    s = s.replace("tym","time").replace("plz","please").replace("pls","please")
    s = s.replace("gd","good").replace("bt","but").replace("bcz","because").replace("bc"," ")
    s = s.replace("late ga","late ").replace("entha","so ")
    return s

def detect_language_code(s: str) -> str:
    if not s.strip():
        return "und"
    try:
        lang = DETECTOR.detect_language_of(s)
        mapping = {
            Language.ENGLISH:"en", Language.HINDI:"hi", Language.BENGALI:"bn",
            Language.MARATHI:"mr", Language.TELUGU:"te", Language.TAMIL:"ta",
            Language.KANNADA:"kn", Language.MALAYALAM:"ml", Language.GUJARATI:"gu",
            Language.PUNJABI:"pa", Language.URDU:"ur"
        }
        return mapping.get(lang,"und")
    except Exception:
        return "und"

def detect_toxicity_and_funny(s: str) -> Tuple[bool,bool]:
    low = s.lower()
    bad = any(re.search(rf"\b{re.escape(w)}\b", low) for w in BAD_WORDS)
    funny = any(tok in low for tok in FUNNY_MARKERS)
    return bad, funny

def sentiment_and_mood(text: str) -> Tuple[float,float,str]:
    base = VADER.polarity_scores(text)["compound"]
    bonus = 0.0
    low = text.lower()
    if any(p in low for p in HINGLISH_POS): bonus += 0.15
    if any(n in low for n in HINGLISH_NEG): bonus -= 0.20
    comp = np.clip(base + bonus, -1.0, 1.0)
    mood = float(np.interp(comp, [-1,0,1],[5,50,95]))
    label = "neutral"
    if comp <= -0.55: label = "angry"
    elif comp < -0.15: label = "negative"
    elif comp > 0.55: label = "delighted"
    elif comp > 0.15: label = "positive"
    return comp, mood, label

def analyze_text(text: str) -> NLPResult:
    lang = detect_language_code(text)
    norm = normalize_roman(text)
    comp, mood, label = sentiment_and_mood(norm)
    toxic, funny = detect_toxicity_and_funny(text)
    if funny and comp < 0.15:
        comp = np.clip(comp + 0.10, -1.0, 1.0)
        mood = float(np.interp(comp, [-1,0,1],[5,50,95]))
    return NLPResult(lang, comp, toxic, funny, mood, label)

# -------------------- Typing speed → mood nudge --------------------
def update_typing_speed_meter(key="live_text") -> Tuple[float,float]:
    now = time.time()
    if "ts_last" not in st.session_state:
        st.session_state.ts_last = now
        st.session_state.len_last = 0
    prev_t = st.session_state.ts_last
    prev_n = st.session_state.len_last
    cur = st.session_state.get(key,"")
    cur_n = len(cur)
    dt = max(now - prev_t, 1e-3)
    dch = max(cur_n - prev_n, 0)
    cps = dch / dt
    wpm = cps * 60 / 5
    st.session_state.ts_last, st.session_state.len_last = now, cur_n
    return cps, wpm

def typing_nudge_to_mood(wpm: float) -> float:
    wpm = float(np.clip(wpm, 0, 120))
    if wpm > 70: return -6.0
    if wpm < 15: return +4.0
    return 0.0

# -------------------- Whisper voice (optional) --------------------
WHISPER_ENDPOINT = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
def hf_whisper_transcribe(wav_bytes: bytes) -> Optional[str]:
    if not HF_API_TOKEN: return None
    try:
        r = requests.post(
            WHISPER_ENDPOINT,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            data=wav_bytes, timeout=60
        )
        r.raise_for_status()
        js = r.json()
        if isinstance(js, dict):
            return js.get("text") or js.get("generated_text")
        if isinstance(js, list) and js and "text" in js[0]:
            return js[0]["text"]
    except Exception:
        return None

class AudioProcessor(AudioProcessorBase):
    def __init__(self): self.frames: List[np.ndarray] = []
    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray()
        if pcm.ndim > 1: pcm = pcm.mean(axis=0)
        self.frames.append(pcm.astype(np.float32))
        return frame

def collect_audio_and_transcribe() -> Optional[str]:
    ctx = webrtc_streamer(
        key="voice",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )
    st.caption("🎙 Click red stop → then 'Transcribe & Analyze'")
    if ctx and ctx.audio_processor:
        if st.button("Transcribe & Analyze"):
            samples = np.concatenate(ctx.audio_processor.frames) if ctx.audio_processor.frames else np.array([],dtype=np.float32)
            if samples.size == 0:
                st.warning("No audio captured.")
                return None
            sr_in, sr_out = 48000, 16000
            ratio = sr_in // sr_out if sr_in % sr_out == 0 else 3
            samples_ds = samples[::ratio]
            buf = io.BytesIO()
            wav_write(buf, sr_out, (samples_ds*32767).astype(np.int16))
            if HF_API_TOKEN:
                with st.spinner("Transcribing..."):
                    return hf_whisper_transcribe(buf.getvalue())
            else:
                st.info("HF_API_TOKEN not set.")
    return None

# -------------------- UI helpers --------------------
def gauge(mood: float, title: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(np.clip(mood,0,100)),
        number={"suffix":" /100"},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"thickness":0.35},
            "steps":[
                {"range":[0,25],"color":"#ffd6d6"},
                {"range":[25,50],"color":"#ffecc7"},
                {"range":[50,75],"color":"#e0f3ff"},
                {"range":[75,100],"color":"#d6ffe3"},
            ],
            "threshold":{"line":{"color":"#333","width":3},"value":mood}
        },
        title={"text":title}
    ))
    fig.update_layout(height=250, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

def badge(txt): 
    st.markdown(f"<span style='background:#f6f9fe;border:1px solid #ddeaff;padding:4px 8px;border-radius:6px'>{txt}</span>",unsafe_allow_html=True)

# -------------------- App --------------------
st.title("📦 Delivery Mood — Minimal (Text + Voice, Gauge)")

if not HF_API_TOKEN:
    st.warning("Set `HF_API_TOKEN` in Streamlit Secrets to enable voice transcription.", icon="⚠️")

tab_text, tab_voice, tab_dash = st.tabs(["✍️ Text Review","🎤 Voice Review","📊 Dashboard"])

with tab_text:
    st.session_state.setdefault("live_text","")
    live = st.text_input("Type your message:", key="live_text", placeholder="e.g., entha babu entha sepu wait cheyyali")
    cps, wpm = update_typing_speed_meter("live_text")
    res = analyze_text(live)
    mood_adj = float(np.clip(res.mood_score + typing_nudge_to_mood(wpm),0,100))

    c1,c2,c3,c4 = st.columns(4)
    with c1: badge(f"Lang: **{res.lang_code}**")
    with c2: badge(f"Sent: **{res.sentiment_score:.2f}**")
    with c3: badge(f"Toxic: {'Yes' if res.toxicity else 'No'} {'(funny 😉)' if res.funny_context else ''}")
    with c4: badge(f"WPM: {wpm:.1f}")

    gauge(mood_adj, f"Live Mood • {res.label.capitalize()}")

    if res.toxicity and not res.funny_context:
        st.error("🚨 Escalate to support (angry customer).")
    elif res.label in {"angry","negative"}:
        st.warning("⚠️ Send apology + ETA confirmation.")
    else:
        st.success("✅ Normal mood; routine reply OK.")

with tab_voice:
    transcript = collect_audio_and_transcribe()
    if transcript:
        st.write("**Transcript:**", transcript)
        r2 = analyze_text(transcript)
        gauge(r2.mood_score, f"Voice Mood • {r2.label.capitalize()}")
        st.caption(f"Lang={r2.lang_code} | Sent={r2.sentiment_score:.2f} | Toxic={'Yes' if r2.toxicity else 'No'}")

with tab_dash:
    st.write("Sample messages → quick mood map")
    demo = [
        ["entha babu entha sepu wait cheyyali", *analyze_text("entha babu entha sepu wait cheyyali").__dict__.values()],
        ["food super on time thanku", *analyze_text("food super on time thanku").__dict__.values()],
        ["bhai order late ho gaya 😂", *analyze_text("bhai order late ho gaya 😂").__dict__.values()],
        ["time bokka ra", *analyze_text("time bokka ra").__dict__.values()]
    ]
    df = pd.DataFrame(demo, columns=["text","lang","sentiment","toxic","funny","mood","label"])
    st.dataframe(df, use_container_width=True, hide_index=True)
