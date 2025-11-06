import io, time, re, math, collections
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from langdetect import detect
from scipy.signal import get_window, lfilter
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile

# Live voice deps
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av  # needs FFmpeg on PATH

# ----------------------------- UI copy -----------------------------
st.set_page_config(page_title="Hybrid NLP: Text + Voice Feedback", page_icon="🎙️", layout="wide")
st.title("🛵 Hybrid NLP for Delivery Feedback (Text + Voice)")

st.info("Heads up: your order may be late by ~5 minutes. Sorry! (Prototype focuses on NLP + Emotion)")

# ----------------------------- Globals -----------------------------
SAMPLE_RATE = 16000
BUF_SECONDS = 6.0
BUF_SAMPLES = int(SAMPLE_RATE * BUF_SECONDS)

if "voice_buf" not in st.session_state:
    st.session_state.voice_buf = collections.deque(maxlen=BUF_SAMPLES)
if "log_rows" not in st.session_state:
    st.session_state.log_rows = []

# ----------------------------- TEXT NLP -----------------------------
# Lightweight lexicons for sentiment (multi-lingual-ish; code-mix friendly)
INTENS = {"very":1.3,"really":1.2,"so":1.2,"too":1.15,"extremely":1.4,"super":1.3,
          "bahut":1.2,"chala":1.2,"baga":1.2,"romba":1.2}
NEGS = set(["not","no","never","hardly","barely","scarcely","isnt","wasnt","arent","dont","didnt","cant","wont",
            "nahi","mat","kadu","illa","kaduu","kalla","leda"])
POSW = set("good great awesome amazing excellent tasty fresh fast quick ontime polite friendly love liked perfect nice yummy delicious wow clean crisp juicy recommend thanks superb mast semma mass superr bagundi super".split())
NEGW = set("bad terrible awful worst cold soggy late delay delayed dirty slow rude stale raw burnt bland overpriced expensive missing leak leaking spill spilled refund replace cancel canceled cancelled angry frustrated annoyed disappointed hate horrible issue problem broken uncooked inedible vomit sick hair bekar faltu bakwas worsttt waste dappa sappa mosam chindi pathetic useless trash".split())

# Toxicity patterns (Hinglish + multi-lingual slang, keep short)
TOX_PAT = re.compile(r"\b(chor|fraud|mc|bc|madarchod|bhenchod|saala|saale|kutte|kutti|bloody|idiot|stupid|poda|podi|tho|paka|thuu|nonsense)\b", re.I)

# Aspect dictionaries (tiny but effective)
ASPECTS = {
    "Food": set("food biryani pizza burger rice curry roti dosa idli sambar shawarma roll fries taste spicy cold hot stale raw burnt oily salty sweet chutney sauce portion quantity fresh".split()),
    "Delivery": set("delivery rider driver boy courier late delay delayed ontime fast slow call behaviour attitude rude polite location route otp helmet bag vehicle bike app map wrong house address stairs".split()),
    "App": set("app ui ux payment upi card wallet refund replace cancel cancelled update crash bug otp login coupon promo offer support chat service email ticket".split()),
}

# 8-emotion word lists (lite NRC-style)
EMO = {
    "anger": set("angry furious rage mad pissed annoyed frustrated irate".split()),
    "disgust": set("disgust dirty filthy yuck gross smelly".split()),
    "fear": set("scared fear afraid worry anxious unsafe".split()),
    "sadness": set("sad unhappy disappointed upset crying low dull".split()),
    "joy": set("happy joy glad awesome great satisfied delighted".split()),
    "trust": set("trust reliable consistent honest dependable".split()),
    "anticipation": set("hope expect waiting excited looking forward".split()),
    "surprise": set("surprised shocked astonished wow unexpected".split()),
}

def lang_detect(text:str)->str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

def tokenize(text:str):
    # simple alnum tokens; keep accents basic
    return re.findall(r"[a-zA-Z]+", text.lower())

def text_sentiment(text:str):
    """
    Returns (score -100..100, label, emoji, hits)
    """
    t = text.strip()
    if not t:
        return 0, "Neutral", "😐", []

    s = re.sub(r"[^\w\s!?]", " ", t.lower())
    words = s.split()
    score = 0.0; hits = []
    exclam = t.count("!")
    caps = 1.15 if re.search(r"[A-Z]{3,}", t) else 1.0
    elong = 1.1 if re.search(r"(.)\1{2,}", t.lower()) else 1.0

    # emojis
    if any(x in t for x in ["😡","🤬"]): score -= 12
    if any(x in t for x in ["😊","🙂","😄","😍","🤩","👍","👏"]): score += 6

    for i, w in enumerate(words):
        base = (2.5 if w in POSW else 0) + (-2.5 if w in NEGW else 0)
        if base!=0:
            if i>0 and words[i-1] in INTENS: base *= INTENS[words[i-1]]
            # scope negations
            for k in range(1,4):
                if i-k>=0 and words[i-k] in NEGS:
                    base *= -1; break
            score += base; hits.append(w)

    score *= caps * elong
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

def text_aspects(text:str):
    tokens = set(tokenize(text))
    tags = []
    for aspect, vocab in ASPECTS.items():
        if tokens & vocab:
            tags.append(aspect)
    if not tags: tags = ["General"]
    return tags

def text_toxic(text:str):
    return bool(TOX_PAT.search(text))

def text_emotion8(text:str):
    tokens = set(tokenize(text))
    scores = {k: len(tokens & v) for k,v in EMO.items()}
    label = max(scores, key=scores.get) if any(scores.values()) else "neutral"
    return label, scores

# ----------------------------- VOICE PROSODY -----------------------------
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
    # pre-emphasis
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
    if not F: return "Neutral","🙂",["too little audio"], 0.5, 0.5, 40.0
    f0=F["mean_f0"]; f0sd=F["std_f0"]; rng=F["range_f0"]
    inten=F["mean_int"]; intsd=F["std_int"]; hnr=F["mean_hnr"]; cpp=F["mean_cpp"]
    cent=F["centroid"]; rate=F["speaking_rate"]; pauses=F["pause_segments"]; pr=F["pause_dur"]/(F["total_dur"]+1e-9)
    wide = rng>80; flat=f0sd<15
    high_f0=f0>190; low_f0=(0<f0<120)
    high_e=inten>0.06; low_e=inten<0.02; dyn=intsd>0.02
    fast=rate>2.2; slow=rate<1.0
    many_pauses=(pauses>=4) or (pr>0.45)
    high_cent=cent>2500; low_cent=cent<1600
    weak_cpp=cpp<0.06; strong_cpp=cpp>0.12

    # arousal / valence proxies
    arousal = float(np.clip(0.5*((inten*20)+(rate/3.0)+((cent-1200)/2000)), 0, 1))
    valence = float(np.clip(0.5*((cpp*5)+(hnr/15.0)-pr), 0, 1))
    stress  = float(np.clip(60*arousal + 20*(1-valence) + 20*(1 if (weak_cpp or hnr<8) else 0), 0, 100))

    if (high_f0 or wide) and (high_e or dyn) and (fast or not many_pauses) and (high_cent or hnr<8):
        return "Angry/Stress","😠",["high pitch/variation","high energy","fast/pressed"], arousal, valence, stress
    if (low_f0 or flat) and (low_e or not dyn) and (slow or many_pauses) and (weak_cpp or low_cent):
        return "Sad","😔",["low/flat pitch","low energy","slow & many pauses"], arousal, valence, stress
    if (not low_f0) and (dyn or high_e) and (strong_cpp or hnr>=10) and (rate>=1.2) and (not many_pauses):
        return "Joy","😄",["lively energy","good CPP/HNR"], arousal, valence, stress
    return "Neutral","🙂",["balanced features"], arousal, valence, stress

# ----------------------------- FUSION & ACTIONS -----------------------------
def fuse_priority(text_score:int, voice_stress:float, tox:bool)->int:
    """
    Combine text sentiment (-100..100) and voice stress (0..100) + toxicity.
    Heavier weight to negative sentiment & high stress.
    """
    # Map text negativity to 0..100 severity
    neg_sev = max(0, -text_score)  # 0..100
    # Weighted blend
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
        return f"[{base}] Sorry for the delay and inconvenience. I’m expediting your order now. I understand your concern about {a}. I’ll keep you updated every few minutes. 🙏"
    if text_mood in ["Frustrated","Disappointed"]:
        return f"[{base}] Apologies for the trouble with {a}. I’m checking a faster route and will update the ETA shortly. Thank you for your patience."
    if text_mood in ["Satisfied","Delighted"]:
        return f"[{base}] Thank you for the feedback! I’ll make sure the {a} stays consistent. 😊"
    return f"[{base}] I’m on it! I’ll update you with the latest ETA shortly."

def coupon_trigger(priority:int, tox:bool)->str|None:
    if priority >= 85: return "SORRY15"
    if priority >= 70 and not tox: return "SORRY10"
    return None

# ----------------------------- LAYOUT -----------------------------
tab_text, tab_voice, tab_fuse, tab_log = st.tabs(["✍️ Text Feedback", "🎙️ Voice Emotion", "⚖️ Fusion & Actions", "📋 Log"])

# ---- TEXT PANEL ----
with tab_text:
    st.caption("Type a short review in any language (Hinglish/Telugu/Tamil/English).")
    text = st.text_area("Customer's message", height=140, value="Biryani was cold but the delivery boy was polite. Please hurry next time.")
    lng = lang_detect(text) if text.strip() else "unknown"
    score, mood, emoji, hits = text_sentiment(text)
    aspects = text_aspects(text)
    tox = text_toxic(text)
    emo8, emo_scores = text_emotion8(text)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Language", lng)
    with c2: st.metric("Text Sentiment", f"{score} ({emoji} {mood})")
    with c3: st.metric("Toxicity", "⚠️ Yes" if tox else "No")
    with c4: st.metric("Dominant Emotion", emo8)

    st.caption("Aspects: " + ", ".join(aspects))
    if hits:
        st.caption("Sentiment Triggers: " + ", ".join(hits))

# ---- VOICE PANEL ----
with tab_voice:
    st.caption("Click **Start** and allow mic. Speak 5–10s.")
    class AudioBufferProcessor:
        def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
            pcm = frame.to_ndarray()
            if pcm.ndim==2: pcm = pcm.mean(axis=0)
            pcm = pcm.astype(np.float32)
            # resample (cheap) if needed
            if frame.sample_rate != SAMPLE_RATE:
                step = max(1, int(frame.sample_rate / SAMPLE_RATE))
                pcm = pcm[::step]
            # normalize softly
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

    sig = np.array(st.session_state.voice_buf, dtype=np.float32)
    if sig.size > 0:
        F = extract_prosody(sig[-BUF_SAMPLES:], SAMPLE_RATE)
        mood_v, emoji_v, reasons_v, arousal, valence, stress = voice_mood(F)

        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Voice Mood", f"{emoji_v} {mood_v}")
        with c2: st.metric("Mean F0", f"{F.get('mean_f0',0):.0f} Hz")
        with c3: st.metric("Stress Index", f"{stress:.0f}/100")

        m1,m2,m3,m4 = st.columns(4)
        with m1: st.metric("Pitch Range", f"{F.get('range_f0',0):.0f} Hz")
        with m2: st.metric("Intensity", f"{F.get('mean_int',0):.3f}")
        with m3: st.metric("CPP", f"{F.get('mean_cpp',0):.3f}")
        with m4: st.metric("Centroid", f"{F.get('centroid',0):.0f} Hz")

        st.caption("Why (voice): " + ", ".join(reasons_v))

        # Waveform (last 3s)
        L = min(len(sig), SAMPLE_RATE*3)
        t = np.arange(L)/SAMPLE_RATE
        fig_w, ax = plt.subplots(figsize=(6,2.0))
        ax.plot(t, sig[-L:], linewidth=0.9)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.set_title("Waveform (last 3s)"); ax.grid(alpha=0.3)
        st.pyplot(fig_w, clear_figure=True)

        # Spectrum (0..4 kHz of last ~2s)
        seg = sig[-min(len(sig), 2*SAMPLE_RATE):]
        if len(seg)>256:
            win = get_window("hann", len(seg))
            X = np.abs(rfft(seg*win))
            freqs = rfftfreq(len(seg), 1.0/SAMPLE_RATE)
            fig_s, ax2 = plt.subplots(figsize=(6,2.0))
            ax2.plot(freqs, 20*np.log10(X+1e-9), linewidth=0.9)
            ax2.set_xlim(0, 4000)
            ax2.set_xlabel("Hz"); ax2.set_ylabel("dB"); ax2.set_title("Spectrum"); ax2.grid(alpha=0.3)
            st.pyplot(fig_s, clear_figure=True)

# ---- FUSION & ACTIONS ----
with tab_fuse:
    # Compute (recompute from state)
    sig = np.array(st.session_state.voice_buf, dtype=np.float32)
    if sig.size>0:
        F = extract_prosody(sig[-BUF_SAMPLES:], SAMPLE_RATE)
        mood_v, emoji_v, reasons_v, arousal, valence, stress = voice_mood(F)
    else:
        mood_v, emoji_v, reasons_v, stress = "Neutral","🙂",["no audio"], 40.0

    priority = fuse_priority(score, stress, tox)
    action_reply = smart_reply(mood, text_aspects(text), priority, lang_detect(text) if text else "en")
    coupon = coupon_trigger(priority, tox)

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Priority Score", f"{priority}/100")
    with c2: st.metric("Text Mood", f"{mood}")
    with c3: st.metric("Voice Mood", f"{mood_v}")

    st.markdown("**Smart Reply:** " + action_reply)
    if coupon:
        st.success(f"🎟️ Offer coupon: **{coupon}**")

    if st.button("➕ Log this interaction"):
        st.session_state.log_rows.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Lang": lang_detect(text) if text else "unknown",
            "TextScore": score,
            "TextMood": mood,
            "VoiceMood": mood_v,
            "Priority": priority,
            "Toxic": "Yes" if tox else "No",
            "Aspects": ", ".join(text_aspects(text)),
            "Coupon": coupon or "-",
        })
        st.success("Logged.")

# ---- LOG (Dashboard) ----
with tab_log:
    if st.session_state.log_rows:
        df = pd.DataFrame(st.session_state.log_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No interactions logged yet.")
