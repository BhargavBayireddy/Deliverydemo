# -*- coding: utf-8 -*-
# Hybrid NLP Dashboard (Text + Voice) — Indian Languages + Roman chat slang
# Mode: Version B (full dashboard)
#
# Features
# - Auto language detect: native Indic scripts + Roman (Hinglish/Tenglish/Tanglish/Kanglish)
# - Sentiment + Emotion + Toxicity via HuggingFace Inference API
# - Slang / bad-words detector with context handling (joke/teasing vs abuse)
# - Typing-speed → stress gauge
# - Live voice capture (WebRTC) with waveform + Whisper transcription + prosody emotion
#
# Env: set HF_API_TOKEN in Streamlit secrets

import io, os, re, json, time, math, queue
import random
import requests
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Optional helpers
try:
    import fasttext  # for Roman language clue
    HAS_FASTTEXT = True
except Exception:
    HAS_FASTTEXT = False

from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

# --------------------------------------------
# CONFIG
# --------------------------------------------
st.set_page_config(page_title="Hybrid NLP (Text + Voice)", page_icon="🛵", layout="wide")

HF_API = "https://api-inference.huggingface.co/models"
HF_TOKEN = st.secrets.get("HF_API_TOKEN", None)

# HuggingFace models (stable + free-tier friendly)
MODELS = dict(
    sentiment="cardiffnlp/twitter-xlm-roberta-base-sentiment",    # neg/neu/pos (multi-lingual)
    emotion="j-hartmann/emotion-english-distilroberta-base",      # 7 emotions (works ok cross-lingual too)
    toxicity="unitary/toxic-bert",                                # toxicity
    whisper="openai/whisper-small"                                # multilingual ASR
)

# WebRTC TURN/STUN (public Google STUN is OK)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --------------------------------------------
# LOAD SLANG LEXICON
# --------------------------------------------
@st.cache_resource
def load_slang():
    path = os.path.join("data", "slang_words.json")
    if not os.path.exists(path):
        # Minimal fallback list
        seed = {
            "bokka": {"lang": ["te","te_rom"], "severity": 3, "notes": "Telugu slang"},
            "puka": {"lang": ["te","te_rom"], "severity": 4, "notes": "abusive"},
            "pani": {"lang": ["te","te_rom"], "severity": 1, "notes": "mild (depends)"},
            "bhosdi": {"lang": ["hi","hi_rom"], "severity": 5, "notes": "abusive"},
            "bsdk": {"lang": ["hi","hi_rom"], "severity": 4, "notes": "abusive short"},
            "fuck": {"lang": ["en"], "severity": 4, "notes": "abusive"},
            "bloody": {"lang": ["en"], "severity": 2, "notes": "mild"}
        }
        return seed
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

SLANG = load_slang()

LAUGHTER_TOKENS = {"lol","lmao","haha","hahaha","😂","😅","😆","hehe","rofl"}
NEGATE_TOKENS = {"no","not","never","mat","kadu","illa","kadhu","illa","venam","vendam","illa","beda"}
SECOND_PERSON = {"you","nee","nuv","ni","tera","tera","tum","tu","bro","bhai","babu"}

# --------------------------------------------
# UTILS: HuggingFace Inference
# --------------------------------------------
def hf_headers():
    if not HF_TOKEN:
        return {}
    return {"Authorization": f"Bearer {HF_TOKEN}"}

def hf_classify(model, inputs):
    """
    Generic HF classification (pipelines return list of dicts).
    """
    url = f"{HF_API}/{model}"
    r = requests.post(url, headers=hf_headers(), json={"inputs": inputs, "options":{"wait_for_model":True}})
    r.raise_for_status()
    return r.json()

def hf_asr_whisper(audio_bytes, sampling_rate=16000):
    """
    Whisper-small via HF API. Accepts WAV bytes.
    """
    url = f"{HF_API}/{MODELS['whisper']}"
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    r = requests.post(url, headers=hf_headers(), files=files, data={"task":"transcribe"})
    r.raise_for_status()
    js = r.json()
    # HF returns dict with 'text' or list
    if isinstance(js, dict) and "text" in js:
        return js["text"]
    if isinstance(js, list) and js and "text" in js[0]:
        return js[0]["text"]
    return ""

# --------------------------------------------
# LANGUAGE DETECTION
# --------------------------------------------
INDIC_BLOCKS = {
    "hi": (0x0900, 0x097F),  # Devanagari
    "te": (0x0C00, 0x0C7F),  # Telugu
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "ml": (0x0D00, 0x0D7F),  # Malayalam
    "bn": (0x0980, 0x09FF),  # Bengali
    "gu": (0x0A80, 0x0AFF),  # Gujarati
    "pa": (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    "or": (0x0B00, 0x0B7F),  # Odia
}

ROMAN_HINTS = dict(
    te_rom={"entha","sepu","ivvu","ra","naku","anta","bokka","enduku","cheyyali","ekkada","tiffin","nuvvu","telu","aavthundi","late"},
    hi_rom={"bhai","jaldi","intezar","khana","bhook","der","bekaar","faltu","kya","kab","ghar","bhaiya","bhukkad"},
    ta_rom={"enna","thambi","saar","seekiram","sapadu","sapadu","seri","podu","ippadi"},
    kn_rom={"yaako","beka","illi","hegide","thumba","kelsa"},
    ml_rom={"entha","evide","vannu","poyi","sheri","vada","pani","kochu"}
)

def detect_script_lang(text):
    # If any Indic script char → return its language
    for ch in text:
        cp = ord(ch)
        for lang, (lo,hi) in INDIC_BLOCKS.items():
            if lo <= cp <= hi:
                return lang
    return None

@st.cache_data(show_spinner=False)
def load_fasttext_model():
    if not HAS_FASTTEXT:
        return None
    # tiny language id model
    # Will be downloaded on first use (fastText wheel embeds model path sometimes)
    return None

def detect_language(text):
    """
    Returns ISO-ish code among: te, ta, hi, kn, ml, en, etc.
    Approach:
    1) Indic script check
    2) Roman hint lexicons (priority to te/hi/ta/kn/ml)
    3) FastText (if available)
    4) Fallback english/und
    """
    txt = text.strip()
    if not txt:
        return "und"

    # 1) Indic script
    lang = detect_script_lang(txt)
    if lang: return lang

    # 2) Roman hints (score-based)
    scores = defaultdict(int)
    tokens = re.findall(r"[a-zA-Z]+", txt.lower())
    for t in tokens:
        for lang_key, vocab in ROMAN_HINTS.items():
            if t in vocab:
                scores[lang_key] += 2
        # bonus: Telugu vowel patterns
        if re.search(r"(ivv|cheyy|sepu|ayya|amma)", t): scores["te_rom"] += 1
        if re.search(r"(jaldi|bhai|der|khana|nahi)", t): scores["hi_rom"] += 1
        if re.search(r"(seekiram|sapadu|seri)", t): scores["ta_rom"] += 1
        if re.search(r"(yaka|beka|illi|hegide)", t): scores["kn_rom"] += 1
        if re.search(r"(entha|evide|poyi|sheri)", t): scores["ml_rom"] += 1
    if scores:
        best = max(scores, key=scores.get)
        return best.split("_")[0]  # te_rom -> te

    # 3) fastText fallback (optional)
    # Skipped to avoid large model download on Streamlit Cloud free tier.

    # 4) Fallback
    if re.search(r"[a-zA-Z]", txt):
        return "en"
    return "und"

# --------------------------------------------
# SLANG / BAD WORDS with CONTEXT
# --------------------------------------------
def slang_scan(text, lang_hint="und"):
    low = text.lower()
    toks = re.findall(r"\w+", low)
    hits = []
    for i, t in enumerate(toks):
        if t in SLANG:
            # language check (if provided)
            entry = SLANG[t]
            ok = True
            if "lang" in entry and lang_hint not in ("und","en"):
                if lang_hint not in entry["lang"]:
                    ok = False
            if ok:
                ctx = toks[max(0,i-3):i+4]
                joking = any(x in LAUGHTER_TOKENS for x in ctx)
                neg = any(x in NEGATE_TOKENS for x in ctx)
                direct = any(x in SECOND_PERSON for x in ctx)
                sev = int(entry.get("severity", 3))
                # Context mod
                if joking: sev -= 1
                if neg: sev -= 1
                if direct: sev += 1
                sev = max(1, min(5, sev))
                hits.append({"word": t, "context": " ".join(ctx), "severity": sev, "joking": joking, "negated": neg, "directed": direct})
    # Aggregate decision
    toxicity_flag = any(h["severity"] >= 3 for h in hits)
    return hits, toxicity_flag

# --------------------------------------------
# TEXT → Sentiment / Emotion / Toxicity
# --------------------------------------------
def softmax(z):
    ez = np.exp(z - np.max(z))
    return ez / ez.sum()

def label_from_hf_scores(res):
    # HF returns list of list or list; unify
    arr = res
    if isinstance(res, list) and len(res)>0 and isinstance(res[0], dict) and "label" in res[0]:
        arr = [res]
    # pick top label
    top = sorted(arr[0], key=lambda x: -float(x.get("score",0)))[:1][0]
    return top["label"], float(top["score"])

def classify_text_all(text):
    if not text.strip():
        return dict(sentiment=("NEU", 0.5), emotion=("neutral", 0.5), toxicity=("no", 0.5))
    out = {}
    # Sentiment
    try:
        r = hf_classify(MODELS["sentiment"], text)
        lab, sc = label_from_hf_scores(r)
        # normalize to NEG/NEU/POS
        out["sentiment"] = (lab.upper(), sc)
    except Exception:
        out["sentiment"] = ("NEU", 0.33)
    # Emotion
    try:
        r = hf_classify(MODELS["emotion"], text)
        lab, sc = label_from_hf_scores(r)
        out["emotion"] = (lab.lower(), sc)
    except Exception:
        out["emotion"] = ("neutral", 0.33)
    # Toxicity (model-level)
    try:
        r = hf_classify(MODELS["toxicity"], text)
        lab, sc = label_from_hf_scores(r)
        tox_yes = lab.lower() in ("toxic","LABEL_1","abusive")
        out["toxicity"] = ("yes" if tox_yes else "no", sc)
    except Exception:
        out["toxicity"] = ("no", 0.33)
    return out

# --------------------------------------------
# TYPING SPEED → Mood gauge
# --------------------------------------------
def typing_speed_update():
    # keep last length/time in session
    now = time.time()
    txt = st.session_state.get("input_text","")
    last = st.session_state.get("ts_meta", {"len":0, "t":now})
    dt = max(0.25, now - last["t"])
    cps = max(0.0, (len(txt) - last["len"])/dt)
    st.session_state["ts_meta"] = {"len": len(txt), "t": now, "cps": cps}

def cps_to_mood(cps):
    # basic mapping
    if cps < 3: return "Calm", 0.3
    if cps < 6: return "Normal", 0.6
    return "Stressed", 0.9

def gauge(value, label="Meter"):
    # simple progress bar gauge
    st.progress(min(100,int(value*100)), text=f"{label}: {int(value*100)}/100")

# --------------------------------------------
# VOICE CAPTURE (live) + WAVEFORM
# --------------------------------------------
class AudioBuffer:
    def __init__(self, max_seconds=10, sr=16000):
        self.q = deque(maxlen=max_seconds*sr)
        self.sr = sr
    def add(self, pcm16):
        # pcm16 is numpy int16 mono
        self.q.extend(pcm16.tolist())
    def wav_bytes(self):
        import soundfile as sf
        arr = np.array(self.q, dtype=np.int16)
        bio = io.BytesIO()
        if len(arr)==0:
            return b""
        sf.write(bio, arr, self.sr, format="WAV", subtype="PCM_16")
        return bio.getvalue()

def webrtc_audio_receiver_factory(shared_buf: AudioBuffer):
    def recv(frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().mean(axis=0).astype(np.int16)  # mono
        shared_buf.add(pcm)
        return frame
    return recv

# --------------------------------------------
# UI
# --------------------------------------------
st.title("🛵 Hybrid NLP (HF API) — Text + Voice + Dashboard")

tabs = st.tabs(["📊 Dashboard", "✍️ Text Review", "🎙️ Voice Review", "📒 Logs"])

# Persistent store
if "logs" not in st.session_state:
    st.session_state.logs = []

# ------------- TEXT REVIEW -------------
with tabs[1]:
    st.caption("Type even **1 word** — we auto-detect language (Indic & Roman) + slang + sentiment + emotion + toxicity.")
    txt = st.text_input("Type your message", key="input_text", on_change=typing_speed_update)
    typing_speed_update()
    cps = st.session_state.get("ts_meta",{}).get("cps",0.0)
    mood, mood_v = cps_to_mood(cps)

    lang = detect_language(txt)
    slang_hits, tox_from_slang = slang_scan(txt, lang_hint=lang)
    clf = classify_text_all(txt)

    # Fuse toxicity with slang context
    tox_final = "yes" if (tox_from_slang or clf["toxicity"][0]=="yes") else "no"

    # Sentiment bar (simple)
    sent_label, sent_score = clf["sentiment"]
    em_label, em_score = clf["emotion"]

    st.write("#### Live Meters")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Language", lang)
    with c2: st.metric("Text Sentiment", f"{sent_label} ({int(sent_score*100)})")
    with c3: st.metric("Dominant Emotion", em_label)
    with c4: st.metric("Toxicity", "Yes" if tox_final=="yes" else "No")

    gauge(mood_v, f"Typing Speed → {mood} ({cps:.1f} cps)")
    st.progress( min(100, 50 if sent_label=="NEU" else (80 if sent_label=="POS" else 20)),
                 text=f"Overall Tone • {em_label} • {int(sent_score*100)}/100")

    if slang_hits:
        st.warning("Slang detected (context-aware):")
        st.dataframe(pd.DataFrame(slang_hits), use_container_width=True, hide_index=True)
    else:
        st.info("No slang detected.")

    if st.button("Save as Latest Feedback → Dashboard"):
        st.session_state.logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "modality": "text",
            "text": txt,
            "lang": lang,
            "sentiment": sent_label,
            "emotion": em_label,
            "toxicity": tox_final,
            "cps": round(cps,1),
            "slang_count": len(slang_hits)
        })
        st.success("Saved to Dashboard.")

# ------------- VOICE REVIEW -------------
with tabs[2]:
    st.caption("Click **Start**, speak for 5–8 seconds. Waveform animates live. Click **Transcribe & Analyze** to run Whisper + NLP.")
    buf = AudioBuffer(max_seconds=12, sr=16000)

    webrtc_ctx = webrtc_streamer(
        key="live-voice",
        mode=WebRtcMode.RECVONLY,
        audio_receiver_size=1024,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
        on_audio_frame=webrtc_audio_receiver_factory(buf),
    )

    # Waveform preview
    import plotly.graph_objects as go
    samples = np.array(buf.q, dtype=np.int16)
    if samples.size > 0:
        seg = samples[-16000*3:]  # last 3s
        x = np.arange(len(seg))/16000.0
        fig = go.Figure(go.Scatter(y=seg, x=x, mode="lines"))
        fig.update_layout(height=160, margin=dict(l=0,r=0,t=10,b=0), yaxis_title="Amp")
        st.plotly_chart(fig, use_container_width=True)

    colA, colB = st.columns([1,1])
    with colA:
        st.button("Start", help="Use the START in the grey mic panel (browser prompt).")
    with colB:
        if st.button("🧠 Transcribe & Analyze"):
            if not HF_TOKEN:
                st.error("Set HF_API_TOKEN in Streamlit Secrets to enable Whisper.")
            else:
                wav_bytes = buf.wav_bytes()
                if not wav_bytes:
                    st.error("No audio captured yet. Speak after pressing Start.")
                else:
                    with st.spinner("Transcribing..."):
                        text = hf_asr_whisper(wav_bytes, 16000)
                    st.text_area("Transcribed Text", value=text, height=100)
                    # Run text pipeline
                    lang = detect_language(text)
                    slang_hits, tox_from_slang = slang_scan(text, lang_hint=lang)
                    clf = classify_text_all(text)
                    tox_final = "yes" if (tox_from_slang or clf["toxicity"][0]=="yes") else "no"
                    sent_label, sent_score = clf["sentiment"]
                    em_label, em_score = clf["emotion"]

                    st.write("#### Voice Analysis")
                    m1,m2,m3,m4 = st.columns(4)
                    with m1: st.metric("Language", lang)
                    with m2: st.metric("Sentiment", f"{sent_label} ({int(sent_score*100)})")
                    with m3: st.metric("Emotion", em_label)
                    with m4: st.metric("Toxicity", "Yes" if tox_final=="yes" else "No")

                    if slang_hits:
                        st.warning("Slang detected (context-aware):")
                        st.dataframe(pd.DataFrame(slang_hits), use_container_width=True, hide_index=True)
                    st.success("Voice result ready. Save to Dashboard if needed.")
                    if st.button("Save Voice → Dashboard"):
                        st.session_state.logs.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "modality": "voice",
                            "text": text,
                            "lang": lang,
                            "sentiment": sent_label,
                            "emotion": em_label,
                            "toxicity": tox_final,
                            "cps": None,
                            "slang_count": len(slang_hits)
                        })
                        st.success("Saved to Dashboard.")

# ------------- DASHBOARD -------------
with tabs[0]:
    st.caption("Latest saved items from Text/Voice.")
    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs)
        st.dataframe(df, use_container_width=True, hide_index=True)
        # Small KPIs
        pos = (df["sentiment"]=="POS").sum()
        neg = (df["sentiment"]=="NEG").sum()
        neu = (df["sentiment"]=="NEU").sum()
        tox = (df["toxicity"]=="yes").sum()
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("POS", pos)
        with c2: st.metric("NEG", neg)
        with c3: st.metric("NEU", neu)
        with c4: st.metric("Toxic", tox)
    else:
        st.info("No items yet. Save from Text or Voice tabs.")

# ------------- LOGS -------------
with tabs[3]:
    st.json(st.session_state.logs)
