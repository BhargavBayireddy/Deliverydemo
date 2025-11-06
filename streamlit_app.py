# -*- coding: utf-8 -*-
# 🛵 Hybrid NLP (HuggingFace API) — Single Page, Auto Detect, WebRTC Mic
# - Text: language detect, multilingual sentiment, emotion, toxicity (rules), aspects
# - Voice: live mic (WebRTC) → HF Whisper ASR (auto language) + local prosody → voice mood
# - Fusion: text + voice → Priority + smart reply + coupon
# - UI: one page, auto-updates while typing / speaking
#
# Setup:
#   1) pip install -r requirements.txt
#   2) Set env var HF_API_TOKEN (Streamlit Cloud: Settings → Secrets)
#   3) streamlit run app.py

import os, io, time, re, math, collections, json, base64
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import get_window, lfilter
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile

from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av  # needs FFmpeg presence on the host

# ------------------------------- Config -------------------------------
st.set_page_config(page_title="Hybrid NLP (HF API) — Text + Voice", page_icon="🎙️", layout="wide")
st.title("🛵 Hybrid NLP for Delivery Feedback (Text + Voice)")

st.caption("Auto detects language, sentiment & emotion from text and live voice. Uses HuggingFace Inference API.")

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
if not HF_API_TOKEN:
    st.warning("⚠️ Set HF_API_TOKEN in environment / Streamlit Secrets for ASR, language & emotion models.")

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

# HuggingFace model choices (you can swap these to other hosted models)
HF_LANG_ID_MODEL   = "papluca/xlm-roberta-base-language-detection"         # text-classification → labels like 'en', 'hi', ...
HF_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"        # text-classification (multi-lingual, 3 classes)
HF_EMOTION_MODEL   = "cardiffnlp/twitter-xlm-roberta-base-emotion"          # text-classification (joy, anger, fear, etc.)
HF_WHISPER_MODEL   = "openai/whisper-small"                                 # automatic-speech-recognition

# Mic/ASR buffering
SAMPLE_RATE = 16000
BUF_SECONDS = 8
BUF_SAMPLES = SAMPLE_RATE * BUF_SECONDS

# ------------------------------- App State -------------------------------
if "voice_buf" not in st.session_state:
    st.session_state.voice_buf = collections.deque(maxlen=BUF_SAMPLES)
if "last_asr_text" not in st.session_state:
    st.session_state.last_asr_text = ""
if "last_asr_lang" not in st.session_state:
    st.session_state.last_asr_lang = "unknown"
if "asr_ts" not in st.session_state:
    st.session_state.asr_ts = 0.0
if "orders" not in st.session_state:
    st.session_state.orders = []  # Delivery Boy Dashboard

# ------------------------------- Lightweight NLP -------------------------------
INTENS = {"very":1.3,"really":1.2,"so":1.2,"too":1.15,"extremely":1.4,"super":1.3,
          "bahut":1.2,"chala":1.2,"baga":1.2,"romba":1.2}
NEGS = set(["not","no","never","hardly","barely","scarcely","isnt","wasnt","arent","dont","didnt","cant","wont",
            "nahi","mat","kadu","illa","kalla","leda"])
POSW = set("good great awesome amazing excellent tasty fresh fast quick ontime polite friendly love liked perfect nice yummy delicious wow clean crisp juicy recommend thanks superb mast semma mass superr bagundi super".split())
NEGW = set("bad terrible awful worst cold soggy late delay delayed dirty slow rude stale raw burnt bland overpriced expensive missing leak leaking spill spilled refund replace cancel canceled cancelled angry frustrated annoyed disappointed hate horrible issue problem broken uncooked inedible vomit sick hair bekar faltu bakwas worsttt waste dappa sappa mosam chindi pathetic useless trash".split())

TOX_PAT = re.compile(r"\b(chor|fraud|mc|bc|madarchod|bhenchod|saala|saale|kutte|kutti|bloody|idiot|stupid|poda|podi|paka|thuu|nonsense)\b", re.I)

ASPECTS = {
    "Food": set(""" food biryani pizza burger rice curry roti dosa idli sambar shawarma roll fries taste spicy cold hot stale raw burnt oily salty sweet chutney sauce portion quantity fresh """.split()),
    "Delivery": set("delivery rider driver boy courier late delay delayed ontime fast slow call behaviour attitude rude polite location route otp helmet bag vehicle bike app map wrong house address stairs").split(),
    "App": set("app ui ux payment upi card wallet refund replace cancel cancelled update crash bug otp login coupon promo offer support chat service email ticket").split(),
}

def tokenize(text: str):
    return re.findall(r"[a-zA-Z]+", text.lower())

def text_aspects(text:str):
    toks = set(tokenize(text))
    tags = []
    for asp, vocab in ASPECTS.items():
        if toks & set(vocab): tags.append(asp)
    return tags or ["General"]

def text_toxic(text:str):
    return bool(TOX_PAT.search(text))

def rule_sentiment(text:str):
    t = text.strip()
    if not t: return 0, "Neutral", "😐", []
    s = re.sub(r"[^\w\s!?]", " ", t.lower())
    words = s.split()
    score = 0.0; hits = []
    exclam = t.count("!")
    caps = 1.15 if re.search(r"[A-Z]{3,}", t) else 1.0
    elong = 1.1 if re.search(r"(.)\1{2,}", t.lower()) else 1.0
    if any(x in t for x in ["😡","🤬"]): score -= 12
    if any(x in t for x in ["😊","🙂","😄","😍","🤩","👍","👏"]): score += 6
    for i,w in enumerate(words):
        base = (2.5 if w in POSW else 0) + (-2.5 if w in NEGW else 0)
        if base!=0:
            if i>0 and words[i-1] in INTENS: base*=INTENS[words[i-1]]
            for k in range(1,4):
                if i-k>=0 and words[i-k] in NEGS: base*=-1; break
            score += base; hits.append(w)
    score *= caps*elong
    if exclam>=2: score *= 1.1
    raw = max(-40, min(40, score))
    scaled = int(round((raw/40)*100))
    if scaled <= -60: mood, emoji = "Angry", "😡"
    elif scaled <= -30: mood, emoji = "Frustrated", "😠"
    elif scaled <= -6:  mood, emoji = "Disappointed", "😕"
    elif -5 <= scaled <= 5: mood, emoji = "Neutral", "😐"
    elif scaled <= 35: mood, emoji = "Satisfied", "🙂"
    else: mood, emoji = "Delighted", "🤩"
    return scaled, mood, emoji, list(dict.fromkeys(hits))

# ------------------------------- HF Inference Helpers -------------------------------
def hf_text_classification(model: str, text: str):
    """Returns list of dicts [{'label': 'xx', 'score': 0.9}, ...]"""
    if not HF_API_TOKEN or not text.strip(): return []
    url = f"https://api-inference.huggingface.co/models/{model}"
    try:
        r = requests.post(url, headers=HEADERS, json={"inputs": text, "options": {"wait_for_model": True}}, timeout=25)
        r.raise_for_status()
        out = r.json()
        # Some models return nested list
        if isinstance(out, list) and out and isinstance(out[0], list):
            out = out[0]
        return out if isinstance(out, list) else []
    except Exception:
        return []

def hf_asr_bytes(model: str, wav_bytes: bytes):
    """Send wav/pcm bytes to HF ASR model. Returns {'text': '...'} or {}."""
    if not HF_API_TOKEN or not wav_bytes: return {}
    url = f"https://api-inference.huggingface.co/models/{model}"
    try:
        r = requests.post(url, headers=HEADERS, data=wav_bytes, timeout=60)
        r.raise_for_status()
        out = r.json()
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}

def detect_lang_hf(text: str) -> str:
    res = hf_text_classification(HF_LANG_ID_MODEL, text)
    if not res: return "unknown"
    # pick highest score label
    res = sorted(res, key=lambda x: x.get("score",0), reverse=True)
    return res[0].get("label","unknown")

def sentiment_hf(text: str):
    # 3 labels: negative/neutral/positive
    preds = hf_text_classification(HF_SENTIMENT_MODEL, text)
    if not preds: return None
    # Map to -100..100
    scores = {p["label"].lower(): p["score"] for p in preds}
    neg = scores.get("negative",0); neu=scores.get("neutral",0); pos=scores.get("positive",0)
    val = int(round((pos - neg) * 100))
    if val <= -60: mood, emoji = "Angry", "😡"
    elif val <= -30: mood, emoji = "Frustrated", "😠"
    elif val <= -6:  mood, emoji = "Disappointed", "😕"
    elif -5 <= val <= 5: mood, emoji = "Neutral", "😐"
    elif val <= 35: mood, emoji = "Satisfied", "🙂"
    else: mood, emoji = "Delighted", "🤩"
    return dict(score=val, mood=mood, emoji=emoji, raw=preds)

def emotion_hf(text: str):
    preds = hf_text_classification(HF_EMOTION_MODEL, text)
    if not preds: return ("neutral", {})
    preds = sorted(preds, key=lambda x: x.get("score",0), reverse=True)
    label = preds[0]["label"].lower()
    scores = {p["label"].lower(): p["score"] for p in preds}
    return (label, scores)

# ------------------------------- Prosody (voice mood) -------------------------------
def frame_signal(x, fs, frame_ms=40, hop_ms=10):
    N = int(fs*frame_ms/1000); H = int(fs*hop_ms/1000)
    if N < 256: N, H = 640, 160
    w = get_window("hann", N)
    frames = []
    for i in range(0, len(x)-N+1, H):
        frames.append(x[i:i+N] * w)
    return np.array(frames), N, H

def autocorr_pitch(frame, fs, fmin=75, fmax=400):
    f = frame - np.mean(frame)
    if np.allclose(f, 0): return 0.0, -np.inf
    r = np.correlate(f, f, mode="full"); r = r[len(r)//2:]
    r0 = r[0] + 1e-9
    r = r / r0
    lag_min = int(fs/fmax); lag_max = min(int(fs/fmin), len(r)-1)
    if lag_max <= lag_min: return 0.0, -np.inf
    idx = np.argmax(r[lag_min:lag_max]) + lag_min
    peak = np.clip(r[idx], 1e-6, 0.999999)
    f0 = fs/idx if idx>0 else 0.0
    hnr = 10*np.log10(peak/(1-peak))
    return f0, hnr

def spectral_centroid(fr, fs):
    X = np.abs(rfft(fr)) + 1e-12
    F = rfftfreq(len(fr), 1.0/fs)
    return float(np.sum(F*X)/np.sum(X))

def cpp_proxy(fr, fs):
    X = np.abs(rfft(fr)) + 1e-9
    c = np.real(np.fft.irfft(np.log(X)))
    imin = int(fs/400); imax = min(int(fs/75), len(c)-1)
    roi = c[imin:imax]
    return float(np.max(roi) - np.median(roi))

def extract_prosody(x, fs=SAMPLE_RATE):
    if len(x) < fs//2: return {}
    x = lfilter([1, -0.97], [1], x).astype(np.float32)
    frames, N, H = frame_signal(x, fs)
    f0s, hnrs, ens, cents, cpps, voiced = [], [], [], [], [], []
    for fr in frames:
        f0, hnr = autocorr_pitch(fr, fs)
        e = float(np.sqrt(np.mean(fr**2)+1e-12))
        cent = spectral_centroid(fr, fs)
        cpp = cpp_proxy(fr, fs)
        zcr = np.mean(np.sign(fr)[:-1]*np.sign(fr)[1:] < 0)
        v = (e > 0.01) and (zcr < 0.18)
        f0s.append(f0); hnrs.append(hnr); ens.append(e); cents.append(cent); cpps.append(cpp); voiced.append(v)
    f0s = np.array(f0s); hnrs=np.array(hnrs); ens=np.array(ens); cents=np.array(cents); cpps=np.array(cpps); voiced=np.array(voiced)
    vf = voiced & (f0s>0)
    f0_v = f0s[vf]; en_v = ens[voiced]
    mean_f0 = float(np.mean(f0_v)) if f0_v.size else 0.0
    std_f0  = float(np.std(f0_v)) if f0_v.size else 0.0
    range_f0= float(np.max(f0_v)-np.min(f0_v)) if f0_v.size else 0.0
    mean_hnr= float(np.mean(hnrs[vf])) if np.any(vf) else -np.inf
    mean_cpp= float(np.mean(cpps[vf])) if np.any(vf) else 0.0
    mean_int= float(np.mean(en_v)) if en_v.size else float(np.mean(ens))
    std_int = float(np.std(en_v)) if en_v.size else float(np.std(ens))
    mean_cent=float(np.mean(cents[voiced])) if np.any(voiced) else float(np.mean(cents))
    hop_sec = 0.01
    total = len(frames)*hop_sec
    pause = float(np.sum(~voiced)*hop_sec)
    onsets = np.sum((voiced[1:] & ~voiced[:-1]))
    rate = onsets/(total+1e-9)
    pause_segs = np.sum((~voiced[1:] & voiced[:-1]))
    avg_pause = (pause/pause_segs) if pause_segs>0 else 0.0
    return dict(mean_f0=mean_f0,std_f0=std_f0,range_f0=range_f0,mean_int=mean_int,std_int=std_int,
                mean_hnr=mean_hnr,mean_cpp=mean_cpp,centroid=mean_cent,speaking_rate=rate,
                total_dur=total,pause_dur=pause,pause_segments=int(pause_segs),avg_pause=avg_pause)

def voice_mood(F):
    if not F: return "Neutral","🙂",["too little audio"], 40.0
    f0=F["mean_f0"]; rng=F["range_f0"]; inten=F["mean_int"]; intsd=F["std_int"]
    hnr=F["mean_hnr"]; cpp=F["mean_cpp"]; cent=F["centroid"]; rate=F["speaking_rate"]
    pr = F["pause_dur"]/(F["total_dur"]+1e-9)
    wide=rng>80; high_f0=f0>190; low_f0=(0<f0<120); dyn=intsd>0.02
    high_e=inten>0.06; low_e=inten<0.02; fast=rate>2.2; slow=rate<1.0
    many_pauses=(pr>0.45 or F["pause_segments"]>=4); high_cent=cent>2500; low_cent=cent<1600; weak_cpp=cpp<0.06; strong_cpp=cpp>0.12
    # stress proxy
    stress = float(np.clip(60*(inten*6) + 20*(1 - (cpp*5/2.5)) + 20*(1-pr), 0, 100))
    if (high_f0 or wide) and (high_e or dyn) and (fast or not many_pauses) and (high_cent or hnr<8):
        return "Angry/Stress","😠",["high pitch/variation","high energy","fast/pressed"], stress
    if (low_f0) and (low_e or not dyn) and (slow or many_pauses) and (weak_cpp or low_cent):
        return "Sad","😔",["low/flat pitch","low energy","slow & many pauses"], stress
    if (not low_f0) and (dyn or high_e) and (strong_cpp or hnr>=10) and (rate>=1.2) and (not many_pauses):
        return "Joy","😄",["lively energy","good CPP/HNR"], stress
    return "Neutral","🙂",["balanced features"], stress

# ------------------------------- Fusion & Actions -------------------------------
def fuse_priority(text_score:int, voice_stress:float, tox:bool)->int:
    neg_sev = max(0, -text_score)  # 0..100
    p = 0.6*neg_sev + 0.4*voice_stress
    if tox: p += 15
    return int(max(0, min(100, round(p))))

def smart_reply(text_mood:str, aspects:list, priority:int, language:str)->str:
    base = "English"
    if language.startswith("hi"): base="Hindi"
    if language.startswith("ta"): base="Tamil"
    if language.startswith("te"): base="Telugu"
    a = " / ".join(aspects[:2]) if aspects else "General"
    if priority >= 75:
        return f"[{base}] Sorry for the experience. I’m expediting your order now and noted issue about {a}. I’ll keep you updated. 🙏"
    if text_mood in ["Frustrated","Disappointed"]:
        return f"[{base}] Apologies for the trouble with {a}. I’m checking a faster resolution and will update shortly."
    if text_mood in ["Satisfied","Delighted"]:
        return f"[{base}] Thank you for the feedback! I’ll make sure {a} stays consistent. 😊"
    return f"[{base}] I’m on it! I’ll update you with the latest status shortly."

def coupon_trigger(priority:int, tox:bool)->str|None:
    if priority >= 85: return "SORRY15"
    if priority >= 70 and not tox: return "SORRY10"
    return None

# ------------------------------- UI: Inputs (Text + Mic) -------------------------------
st.markdown("### ✍️ Review (auto detects language while you type)")
text = st.text_input(" ", value="Biryani late ayindi, next time faster please.")
st.caption("Tip: Type even 1 word (e.g., 'entha late') — language & mood update instantly.")

st.markdown("### 🎙️ Live Voice (auto language via Whisper)")
st.caption("Click **Start**, allow microphone, speak 5–8 seconds. Processing runs automatically.")

class AudioBufferProcessor:
    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray()
        if pcm.ndim == 2:
            pcm = pcm.mean(axis=0)
        pcm = pcm.astype(np.float32)
        if frame.sample_rate != SAMPLE_RATE:
            step = max(1, int(frame.sample_rate / SAMPLE_RATE))
            pcm = pcm[::step]
        mx = np.max(np.abs(pcm)) + 1e-6
        pcm = pcm / mx
        st.session_state.voice_buf.extend(pcm.tolist())
        return frame

webrtc_streamer(
    key="live-voice",
    mode=WebRtcMode.RECVONLY,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    ),
    audio_processor_factory=AudioBufferProcessor,
    async_processing=True,
)

# Small mic visual
with st.container():
    mic_busy = "🔴 Recording…" if len(st.session_state.voice_buf) > SAMPLE_RATE else "🎤 Mic ready"
    st.write(f"**{mic_busy}**")

# ------------------------------- Analysis: TEXT -------------------------------
lang_txt = detect_lang_hf(text) if text.strip() else "unknown"
tox_txt = text_toxic(text)
aspects_txt = text_aspects(text)

# Use HF sentiment; fallback to rule-based if HF not available
sent_res = sentiment_hf(text) if text.strip() else None
if sent_res:
    score_txt = sent_res["score"]; mood_txt = sent_res["mood"]; emoji_txt = sent_res["emoji"]
else:
    score_txt, mood_txt, emoji_txt, _hits = rule_sentiment(text)

emo_label, emo_scores = emotion_hf(text) if text.strip() else ("neutral", {})

c1,c2,c3,c4 = st.columns(4)
with c1: st.metric("Text Lang", lang_txt)
with c2: st.metric("Text Sentiment", f"{score_txt} ({emoji_txt} {mood_txt})")
with c3: st.metric("Toxicity", "⚠️ Yes" if tox_txt else "No")
with c4: st.metric("Text Emotion", emo_label)

st.caption("Aspects: " + ", ".join(aspects_txt))

# ------------------------------- Analysis: VOICE (auto every few seconds) -------------------------------
sig = np.array(st.session_state.voice_buf, dtype=np.float32)

auto_process = True  # always on; processes every ~3s if enough audio
voice_text = st.session_state.last_asr_text
voice_lang = st.session_state.last_asr_lang

def _encode_wav_16k(sig_np: np.ndarray) -> bytes:
    arr = (np.clip(sig_np, -1, 1) * 32767.0).astype(np.int16)
    buff = io.BytesIO()
    wavfile.write(buff, SAMPLE_RATE, arr)
    return buff.getvalue()

if auto_process and len(sig) > SAMPLE_RATE*3 and time.time() - st.session_state.asr_ts > 3.0:
    # ASR on last 6 seconds window
    chunk = sig[-SAMPLE_RATE*6:]
    wav_bytes = _encode_wav_16k(chunk)
    asr = hf_asr_bytes(HF_WHISPER_MODEL, wav_bytes)
    if asr and "text" in asr:
        st.session_state.last_asr_text = asr["text"].strip()
        st.session_state.last_asr_lang = detect_lang_hf(st.session_state.last_asr_text) if st.session_state.last_asr_text else "unknown"
        st.session_state.asr_ts = time.time()

voice_text = st.session_state.last_asr_text
voice_lang = st.session_state.last_asr_lang

st.text_area("Transcribed voice (auto)", value=voice_text or "", height=80, help="Auto from Whisper (HF). You can also edit this.")

# Prosody → Voice Mood (always from the latest buffer)
F = extract_prosody(sig[-SAMPLE_RATE*6:]) if len(sig) > SAMPLE_RATE*2 else {}
mood_v, emoji_v, reasons_v, stress = voice_mood(F)

v1,v2,v3,v4 = st.columns(4)
with v1: st.metric("Voice Mood", f"{emoji_v} {mood_v}")
with v2: st.metric("Stress", f"{int(stress)}/100")
with v3: st.metric("Voice Lang", voice_lang)
with v4: st.metric("Mean F0", f"{F.get('mean_f0',0):.0f} Hz")

# Tiny waveform
if len(sig) > SAMPLE_RATE:
    L = min(len(sig), SAMPLE_RATE*3)
    t = np.arange(L)/SAMPLE_RATE
    fig_w, ax = plt.subplots(figsize=(6,2.0))
    ax.plot(t, sig[-L:], linewidth=0.9)
    ax.set_xlabel("s"); ax.set_ylabel("amp"); ax.set_title("Waveform (last 3s)"); ax.grid(alpha=0.3)
    st.pyplot(fig_w, clear_figure=True)

# ------------------------------- Fusion -------------------------------
# Prefer typed text sentiment if present; else use ASR transcript for text-based signals
fusion_text = text if text.strip() else (voice_text or "")
fusion_lang = lang_txt if text.strip() else voice_lang

# Recompute sentiment for fusion if we switched source
if fusion_text == text:
    f_score, f_mood = score_txt, mood_txt
else:
    sr = sentiment_hf(fusion_text) if fusion_text else None
    if sr:
        f_score, f_mood = sr["score"], sr["mood"]
    else:
        f_score, f_mood, _e, _h = *rule_sentiment(fusion_text), None

priority = fuse_priority(f_score, stress, tox_txt)
reply = smart_reply(f_mood, text_aspects(fusion_text), priority, fusion_lang)
coupon = coupon_trigger(priority, tox_txt)

p1,p2,p3 = st.columns(3)
with p1: st.metric("Priority", f"{priority}/100")
with p2: st.metric("Final Text Mood", f_mood)
with p3: st.metric("Final Lang", fusion_lang)

st.markdown("**Smart Reply:** " + reply)
if coupon:
    st.success(f"🎟️ Offer coupon: **{coupon}**")

# ------------------------------- Review Box + Delivery Boy Dashboard -------------------------------
st.markdown("---")
st.subheader("🧾 Review box (saved) + 👷 Delivery Boy Dashboard")

# Save on click
if st.button("Save to dashboard"):
    oid = f"OD-{int(time.time())%100000}"
    st.session_state.orders.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Order": oid,
        "TextLang": lang_txt,
        "TextMood": f"{mood_txt}",
        "VoiceMood": f"{mood_v}",
        "FinalLang": fusion_lang,
        "Priority": priority,
        "Reply": reply[:120] + ("…" if len(reply)>120 else ""),
        "Coupon": coupon or "-",
    })
    st.success("Saved.")

# Show dashboard
orders = list(st.session_state.orders)
if orders:
    df = pd.DataFrame(orders)
    # sort: high priority first
    df = df.sort_values(by="Priority", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No saved interactions yet.")

st.caption("Tip: All analysis is local + HF API. Replace model IDs with your own deployed endpoints if you want faster responses.")
