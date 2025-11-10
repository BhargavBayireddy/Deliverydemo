# -------------------- Imports --------------------
import os
import io
import time
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import requests

import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go

# Voice (WebRTC mic)
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
from scipy.io.wavfile import write as wav_write

# Language ID (pure-Python, no native deps)
from lingua import Language, LanguageDetectorBuilder

# -------------------- Page config (MUST be first Streamlit call) --------------------
st.set_page_config(
    page_title="Delivery Mood (NLP + Voice)",
    page_icon="📦",
    layout="wide"
)

# -------------------- Globals --------------------
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", os.getenv("HF_API_TOKEN", ""))  # Whisper Inference API token (optional)

# Build Lingua detector once
LANGUAGES = [
    Language.ENGLISH, Language.HINDI, Language.BENGALI, Language.MARATHI,
    Language.TELUGU, Language.TAMIL, Language.KANNADA, Language.MALAYALAM,
    Language.GUJARATI, Language.PUNJABI, Language.URDU
]
DETECTOR = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()

# Sentiment & simple emotion
VADER = SentimentIntensityAnalyzer()

# Basic slang & abuse dictionaries (roman + English). Extend as needed.
HINGLISH_POS = {
    "mast","bahut accha","bahut acha","badiya","solid","awesome","masttt","op","sahi","thanku",
    "thanks","shukriya","great","nice","fast","jaldi","on time","time kiya","super","superb"
}
HINGLISH_NEG = {
    "late","faltu","ghatiya","bakwass","bakwas","bekar","bekaar","worst","slow","thanda",
    "cold","garam","over spicy","chipku","leak","messy","hair","hairfall","smell","stale","basi",
    "time bokka","bokka","entha late","avuthrhadhi","kavala","kavali","vasta ledu","ledu"
}
BAD_WORDS = {
    # English
    "idiot","stupid","dumb","bloody","hell","shit","crap","bastard",
    # Hinglish/common roman slurs (keep lightweight & safe)
    "bhosdi","bhosdike","bhosdi ke","chutiya","chu**ya","chut*", "madarchod","mc","bc",
    "haraami","harami","kutta","suvar","gandu","lund","gaand","sala","saala",
    # Telugu/Tamil/Kannada/Malayalam roman (mild/trimmed)
    "puka","pukri","kena","kena payal","thol","thendi","panni","naayi"
}
FUNNY_MARKERS = {"lol","lmao","rofl","😂","🤣","hehe","haha","xD","😅","😆","😜","bro","bhai","anna","ra"}

# Text normalizer for Hinglish/Indic roman
def normalize_roman(text: str) -> str:
    s = text.lower()
    # common shorthand → englishy cues
    s = re.sub(r"\bacha\b","good",s)
    s = re.sub(r"\bachaa\b","good",s)
    s = re.sub(r"\baccha\b","good",s)
    s = re.sub(r"\bbakwa(s|ss)\b","bad",s)
    s = s.replace("tym","time").replace("plz","please").replace("pls","please")
    s = s.replace("gd","good").replace("bt","but").replace("bcz","because").replace("bc"," ")
    s = s.replace("late ga","late ").replace("entha","so ")
    return s

@dataclass
class NLPResult:
    lang_code: str
    sentiment_score: float      # -1..1 (VADER compound)
    toxicity: bool
    funny_context: bool
    mood_score: float           # 0..100 (for gauge)
    label: str                  # "angry/negative/neutral/positive/delighted"

# -------------------- Core NLP --------------------
def detect_language_code(s: str) -> str:
    """Return ISO-ish 2 letters (best-effort) using Lingua; fallback to 'und'."""
    text = s.strip()
    if not text:
        return "und"
    try:
        lang = DETECTOR.detect_language_of(text)
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
    """Simple lexical toxicity; funny if laughter/emojis present (reduces harshness)."""
    low = s.lower()
    bad = any(re.search(rf"\b{re.escape(w)}\b", low) for w in BAD_WORDS)
    funny = any(tok in low for tok in FUNNY_MARKERS)
    return bad, funny

def sentiment_and_mood(text: str) -> Tuple[float,float,str]:
    """
    Return:
      compound (-1..1), mood (0..100), label
    Mood mixes VADER + slang cues.
    """
    base = VADER.polarity_scores(text)["compound"]  # -1..1
    bonus = 0.0
    low = text.lower()

    if any(p in low for p in HINGLISH_POS): bonus += 0.15
    if any(n in low for n in HINGLISH_NEG): bonus -= 0.20

    comp = np.clip(base + bonus, -1.0, 1.0)
    mood = float(np.interp(comp, [-1, 0, 1], [5, 50, 95]))  # keep range (5..95)
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
    # If funny markers present, soften toxicity a bit for UI messaging; keep flag unchanged
    if funny and comp < 0.15:
        comp = np.clip(comp + 0.10, -1.0, 1.0)
        mood = float(np.interp(comp, [-1,0,1],[5,50,95]))
    return NLPResult(lang, comp, toxic, funny, mood, label)

# -------------------- Typing-speed → live mood nudge --------------------
def update_typing_speed_meter(text_key: str = "live_text") -> Tuple[float, float]:
    """
    Returns (cps, wpm). Stores state across reruns to estimate typing speed.
    """
    now = time.time()
    if "ts_last" not in st.session_state:
        st.session_state.ts_last = now
        st.session_state.len_last = 0

    prev_t = st.session_state.ts_last
    prev_n = st.session_state.len_last

    cur = st.session_state.get(text_key, "")
    cur_n = len(cur)
    dt = max(now - prev_t, 1e-3)
    dch = max(cur_n - prev_n, 0)

    cps = dch / dt                       # chars per second
    wpm = cps * 60 / 5                   # rough words/min (5 chars/word)

    # persist
    st.session_state.ts_last = now
    st.session_state.len_last = cur_n

    return cps, wpm

def typing_nudge_to_mood(wpm: float) -> float:
    """
    Convert typing speed to a small mood delta:
      very fast typing often correlates with agitation; very slow with apathy.
    """
    # clamp wpm 0..120 → delta -6..+4
    wpm = float(np.clip(wpm, 0, 120))
    if wpm > 70:
        return -6.0
    if wpm < 15:
        return +4.0
    return 0.0

# -------------------- Voice via WebRTC + Whisper (optional) --------------------
WHISPER_ENDPOINT = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"

def hf_whisper_transcribe(wav_bytes: bytes) -> Optional[str]:
    if not HF_API_TOKEN:
        return None
    try:
        resp = requests.post(
            WHISPER_ENDPOINT,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            data=wav_bytes,
            timeout=60,
        )
        resp.raise_for_status()
        js = resp.json()
        # HF Inference returns list/dict variants; try both
        if isinstance(js, list) and js and "text" in js[0]:
            return js[0]["text"]
        if isinstance(js, dict) and "text" in js:
            return js["text"]
        # Some endpoints return generated_text
        return js.get("generated_text")
    except Exception:
        return None

class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.frames: List[np.ndarray] = []

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # collect audio frames into numpy
        pcm = frame.to_ndarray()
        # mixdown to mono if needed
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=0, dtype=np.float32)
        self.frames.append(pcm.astype(np.float32))
        return frame

def collect_audio_and_transcribe() -> Optional[str]:
    """Start/stop WebRTC capture and transcribe after user clicks."""
    ctx = webrtc_streamer(
        key="voice",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )
    st.caption("Click the red stop button to finish recording, then press **Transcribe & Analyze**.")
    transcribed = None
    if ctx and ctx.audio_processor:
        if st.button("🎤 Transcribe & Analyze"):
            # stitch frames, write WAV in-memory, 16kHz
            samples = np.concatenate(ctx.audio_processor.frames) if ctx.audio_processor.frames else np.array([], dtype=np.float32)
            if samples.size == 0:
                st.warning("No audio captured yet.")
            else:
                # resample to 16k (if necessary) – assume ~48k from browser; downsample simple pick-every-3
                sr_in = 48000
                sr_out = 16000
                ratio = sr_in // sr_out if sr_in % sr_out == 0 else 3
                samples_ds = samples[::ratio].astype(np.float32)
                buf = io.BytesIO()
                wav_write(buf, sr_out, (samples_ds * 32767).astype(np.int16))
                wav_bytes = buf.getvalue()
                if HF_API_TOKEN:
                    with st.spinner("Transcribing via Whisper…"):
                        transcribed = hf_whisper_transcribe(wav_bytes)
                else:
                    st.info("HF_API_TOKEN not set → skipping ASR (add it in Streamlit Secrets).")
    return transcribed

# -------------------- UI helpers --------------------
def gauge(mood: float, title: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(np.clip(mood, 0, 100)),
        number={"suffix": " /100"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.35},
            "steps": [
                {"range":[0,25], "color":"#ffd6d6"},
                {"range":[25,50], "color":"#ffecc7"},
                {"range":[50,75], "color":"#e0f3ff"},
                {"range":[75,100], "color":"#d6ffe3"},
            ],
            "threshold": {"line":{"color":"#333","width":3}, "thickness":0.8, "value":mood}
        },
        title={"text": title}
    ))
    fig.update_layout(height=260, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

def header_badge(text: str):
    st.markdown(f"<div style='padding:6px 10px;border-radius:8px;background:#f6f9fe;border:1px solid #e5eefc;display:inline-block'>{text}</div>", unsafe_allow_html=True)

# -------------------- App --------------------
st.title("📦 Delivery Mood — Minimal (Text + Voice, Gauge)")

# Tip for ASR
if not HF_API_TOKEN:
    st.warning("Set **HF_API_TOKEN** in your environment / Streamlit Secrets to enable Whisper ASR.", icon="⚠️")

tab_text, tab_voice, tab_dash = st.tabs(["✍️ Text Review", "🎤 Voice Review", "📊 Dashboard"])

# ---------- Text Review ----------
with tab_text:
    st.session_state.setdefault("live_text","")
    live = st.text_input("Type your message (Hinglish/English/Indian languages roman OK):",
                         key="live_text", placeholder="e.g., 'entha late avuthrhadhi, i am waiting'")

    # Live typing metrics
    cps, wpm = update_typing_speed_meter("live_text")

    # NLP
    res = analyze_text(live)
    mood_adj = float(np.clip(res.mood_score + typing_nudge_to_mood(wpm), 0, 100))

    c1,c2,c3,c4 = st.columns([1,1,1,1])
    with c1:
        header_badge(f"Language: **{res.lang_code}**")
    with c2:
        header_badge(f"Sentiment: **{res.sentiment_score:.2f}**")
    with c3:
        header_badge(f"Toxicity: **{'Yes' if res.toxicity else 'No'}** {'(funny 😉)' if res.funny_context else ''}")
    with c4:
        header_badge(f"Typing speed: **{wpm:.1f} wpm**")

    gauge(mood_adj, f"Live Mood • {res.label.capitalize()}")

    # Action hints (for a delivery dashboard)
    st.write("### Suggested actions")
    if res.toxicity and not res.funny_context:
        st.error("High priority: escalate to rider + send apology with ETA & coupon.")
    elif res.label in {"angry","negative"}:
        st.warning("Medium priority: proactive apology + ETA clarification.")
    else:
        st.success("Normal priority: acknowledge & close politely.")

# ---------- Voice Review ----------
with tab_voice:
    st.caption("Uses WebRTC mic. Record ~6–10 seconds, then click **Transcribe & Analyze**.")
    transcript = collect_audio_and_transcribe()
    if transcript:
        st.write("**Transcript:**", transcript)
        res_v = analyze_text(transcript)
        gauge(res_v.mood_score, f"Voice Mood • {res_v.label.capitalize()}")
        st.write(f"Language: **{res_v.lang_code}** | Sentiment: **{res_v.sentiment_score:.2f}** | Toxicity: **{'Yes' if res_v.toxicity else 'No'}** {'(funny 😉)' if res_v.funny_context else ''}")

# ---------- Dashboard ----------
with tab_dash:
    st.write("A compact dashboard to demo live mood logic.")
    demo_rows = [
        ["entha babu entha sepu wait cheyyali", *analyze_text("entha babu entha sepu wait cheyyali").__dict__.values()],
        ["time bokka ra", *analyze_text("time bokka ra").__dict__.values()],
        ["food super on time thanku", *analyze_text("food super on time thanku").__dict__.values()],
        ["bhai order late ho gaya 😂", *analyze_text("bhai order late ho gaya 😂").__dict__.values()],
    ]
    cols = ["text","lang","sentiment","toxic","funny","mood","label"]
    df = pd.DataFrame(demo_rows, columns=cols)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Note: Mood mixes VADER sentiment + Indian slang cues + funny-context relief, with a small typing-speed nudge.")
