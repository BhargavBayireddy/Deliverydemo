# -*- coding: utf-8 -*-
# Hybrid Delivery Feedback — Live Voice + Live Text + Realtime Mood
# Features:
# - Auto language detection (pycld3 + langid + script heuristics) for code-mix
# - Live subtitles from microphone (streaming faster-whisper tiny)
# - Realtime prosody (pitch/energy/rate) + sentiment → heartbeat meter
# - Text typing sentiment + CPS (chars/sec) → mood in realtime
# - Logs to CSV (./logs/session_events.csv)
# - Tabs: Text | Voice | Dashboard

import os, io, time, math, queue, threading, re, pathlib, datetime as dt
from collections import deque, Counter

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ----------- lightweight language ID -----------
import langid
langid.set_languages(None)
try:
    import pycld3
    HAS_CLD3 = True
except Exception:
    HAS_CLD3 = False

# ----------- audio / webrtc / whisper -----------
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from scipy.signal import lfilter
from scipy.io import wavfile

# faster-whisper (tiny) for low-latency subtitles
from faster_whisper import WhisperModel

# -------------------------- CONFIG --------------------------
st.set_page_config("Hybrid NLP + Voice (Live)", "🛵", layout="wide")

# Model paths / device
WHISPER_MODEL_SIZE = os.getenv("WHISPER_SIZE", "tiny")  # tiny/base/small
WHISPER_DEVICE = "cuda" if os.getenv("WHISPER_DEVICE") == "cuda" else "cpu"
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type="int8")

# Ensure log folder
LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(exist_ok=True)
CSV_PATH = LOG_DIR / "session_events.csv"
if not CSV_PATH.exists():
    pd.DataFrame(columns=[
        "ts","mode","text","lang","emotion","score","cps","f0","energy","rate"
    ]).to_csv(CSV_PATH, index=False)

# -------------------------- UTILS --------------------------
SENT_POS = set("""
good great awesome amazing excellent tasty fresh fast quick ontime polite friendly love liked perfect nice yummy delicious wow clean crisp juicy recommend thanks superb mast semma mass superr super
chala bagundi bagunnadi manchi superga ardam ayyindi nalla azhagu romba nalla
""".split())

SENT_NEG = set("""
bad terrible awful worst cold soggy late delay delayed dirty slow rude stale raw burnt bland overpriced expensive missing leak leaking spill spilled refund replace cancel canceled cancelled angry frustrated annoyed disappointed hate horrible issue problem broken uncooked inedible vomit sick hair bekar faltu bakwas waste chindi pathetic useless trash
mosam chala slow ga alasyam ayindi avvadu chetta thappu panni poya
""".split())

NEGATORS = {"not","no","never","hardly","barely","scarcely","isnt","arent","dont","didnt","cant","wont",
            "nahi","kadu","illa","mat","kadhu","alla"}
INTENS = {"very":1.3,"really":1.2,"so":1.2,"too":1.15,"extremely":1.35,"super":1.25,"bahut":1.25,"chala":1.2,"romba":1.2}

EMOJI_POS = {"😊","🙂","😄","😍","🤩","👍","👏","✨","❤️","😁"}
EMOJI_NEG = {"😡","🤬","😤","😠","😞","😢","😭","👎","🙄","🤦"}

def script_hint(s: str):
    # quick unicode script flags for Indic hints
    flags = {
        "te": any(0C00 <= ord(ch) <= 0C7F for ch in s),  # Telugu
        "ta": any(0B80 <= ord(ch) <= 0BFF for ch in s),  # Tamil
        "hi": any(0900 <= ord(ch) <= 097F for ch in s),  # Devanagari
    }
    for k,v in flags.items():
        if v: return k
    return None

def detect_lang(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "und"
    # emoji-only lines → skip
    if all((not ch.isalnum()) for ch in text):
        return "und"
    # unicode script hint first
    hint = script_hint(text)
    candidates = Counter()
    if hint: candidates[hint] += 2.5

    # langid
    lang_lid, _ = langid.classify(text)
    candidates[lang_lid] += 1.5

    # cld3
    if HAS_CLD3:
        r = pycld3.get_language(text)
        if r and r.is_reliable:
            candidates[r.language] += 2.0

    # very short → rely on hint + langid
    if len(text) < 6 and hint:
        return hint

    best, _ = candidates.most_common(1)[0] if candidates else ("en",1)
    # normalize to ISO-ish two-letter we use
    mapping = {"te":"te","ta":"ta","hi":"hi","en":"en","mr":"hi","bn":"hi","ur":"hi"}
    return mapping.get(best, best[:2] if best else "en")

def sentiment_rule(text: str):
    """
    Fast multi-lingual-ish rule sentiment.
    Returns score [-100,100], mood label, emoji.
    """
    t = text.strip()
    if not t:
        return 0,"Neutral","😐"
    low = re.sub(r"[^\w\s!?]", " ", t.lower())
    words = low.split()
    score = 0.0

    # emoji impact
    if any(e in t for e in EMOJI_NEG): score -= 15
    if any(e in t for e in EMOJI_POS): score += 10

    # exclaim & caps
    exclam = t.count("!")
    if exclam >= 2: score *= 1.1
    if re.search(r"[A-Z]{3,}", t): score *= 1.1

    for i,w in enumerate(words):
        base = 0.0
        if w in SENT_POS: base += 2.5
        if w in SENT_NEG: base -= 2.8
        if base != 0:
            if i>0 and words[i-1] in INTENS: base *= INTENS[words[i-1]]
            # simple scope negation within 3 tokens back
            for k in range(1,4):
                if i-k>=0 and words[i-k] in NEGATORS:
                    base *= -1; break
            score += base

    raw = max(-40, min(40, score))
    scaled = int(round((raw/40.0)*100))
    if scaled <= -60: return scaled, "Angry", "😡"
    if scaled <= -30: return scaled, "Frustrated", "😠"
    if scaled <= -6:  return scaled, "Disappointed", "😕"
    if scaled <= 35:  return scaled, "Neutral", "😐"
    return scaled, "Delighted", "🤩"

# ------------------- HEARTBEAT (pulse meter) -------------------
def render_heartbeat(level: float, label: str):
    """
    level: 0..1 (anger/arousal)
    """
    pct = int(max(0,min(100, level*100)))
    color = "#22c55e"  # green
    if pct >= 75: color = "#ef4444"  # red
    elif pct >= 50: color = "#f59e0b"  # amber
    elif pct >= 30: color = "#eab308"  # yellow

    html = f"""
    <div style="width:100%;padding:8px 0">
      <div style="font-weight:700;margin-bottom:6px">Mood Pulse: {label} ({pct}%)</div>
      <div style="width:100%;height:16px;background:#eee;border-radius:10px;overflow:hidden;position:relative">
        <div style="width:{pct}%;height:100%;background:{color};animation:pulse 1.0s ease-in-out infinite;border-radius:10px"></div>
      </div>
    </div>
    <style>
    @keyframes pulse {{
      0% {{ filter: brightness(0.95); }}
      50% {{ filter: brightness(1.1); }}
      100% {{ filter: brightness(0.95); }}
    }}
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)

# ------------------- PROSODY (voice mood) -------------------
def frame_signal(x, fs, frame_ms=40, hop_ms=10):
    N = max(128, int(fs*frame_ms/1000))
    H = max(16, int(fs*hop_ms/1000))
    frames = []
    for i in range(0, len(x)-N+1, H):
        frames.append(x[i:i+N])
    return np.array(frames), N, H

def autocorr_pitch(frame, fs, fmin=75, fmax=400):
    f = frame - np.mean(frame)
    if np.allclose(f, 0): return 0.0
    r = np.correlate(f, f, mode="full")[len(f)-1:]
    r /= (r[0] + 1e-9)
    lag_min, lag_max = int(fs/fmax), int(fs/fmin)
    lag_max = min(lag_max, len(r)-1)
    if lag_max <= lag_min: return 0.0
    seg = r[lag_min:lag_max]
    idx = np.argmax(seg) + lag_min
    return fs/idx if idx>0 else 0.0

def prosody_features(x, fs):
    # pre-emphasis + simple VAD
    x = lfilter([1,-0.97],[1],x).astype(np.float32)
    frames, _, _ = frame_signal(x, fs)
    if frames.size == 0:
        return 0.0,0.0,0.0
    f0s, energies = [], []
    for fr in frames:
        e = float(np.sqrt(np.mean(fr**2))+1e-12)
        energies.append(e)
        f0s.append(autocorr_pitch(fr, fs))
    f0 = float(np.median([f for f in f0s if f>0])) if np.any(np.array(f0s)>0) else 0.0
    energy = float(np.mean(energies))
    # speaking rate proxy: zero-crossings of voiced mask
    voiced = np.array(energies)> (np.mean(energies)*0.6)
    onsets = np.sum((voiced[1:] & ~voiced[:-1]))
    dur = len(frames)*0.01
    rate = onsets/(dur+1e-9)
    return f0, energy, rate

def voice_mood_from_prosody(f0, energy, rate):
    arousal = 0.0
    # normalize rough thresholds
    if f0 >= 190: arousal += 0.4
    if energy >= 0.06: arousal += 0.3
    if rate >= 2.2: arousal += 0.3
    arousal = max(0.0, min(1.0, arousal))
    if arousal >= 0.75: return "Angry", "😠", arousal
    if arousal >= 0.5:  return "Excited", "😄", arousal
    if arousal <= 0.2:  return "Sad", "😔", arousal
    return "Neutral", "🙂", arousal

# ------------------- LOGGING -------------------
def append_log(row: dict):
    df = pd.DataFrame([row])
    df.to_csv(CSV_PATH, mode="a", header=False, index=False)

# ------------------- UI -------------------
st.title("🛵 Hybrid NLP + Voice (Live) — Auto Language + Realtime Mood")
st.caption("Type or speak. The app auto-detects language (incl. code-mix), shows live mood, and logs events to CSV.")

tabs = st.tabs(["📄 Text Feedback", "🎙️ Voice (Live)", "📊 Dashboard"])

# ------------------- TEXT TAB -------------------
with tabs[0]:
    st.subheader("Review (auto language + live mood)")
    tip = st.caption("Tip: start typing — language & mood update instantly. Heartbeat shows arousal/anger.")
    if "text_meta" not in st.session_state:
        st.session_state.text_meta = {"prev_len":0,"prev_ts":time.time(),"cps":0.0}
    txt = st.text_area("Type here", key="text_box", height=120, placeholder="entha late avuthrandi...")
    now = time.time()
    meta = st.session_state.text_meta
    dt_span = max(0.2, now - meta["prev_ts"])
    cps = max(0.0, (len(txt) - meta["prev_len"]) / dt_span)
    st.session_state.text_meta = {"prev_len": len(txt), "prev_ts": now, "cps": cps}

    lang = detect_lang(txt)
    score, mood, emoji = sentiment_rule(txt)

    # anger / arousal from cps + sentiment
    cps_norm = min(1.0, cps/8.0)  # >=8 cps → high
    sent_norm = max(0.0, (-score)/100.0)  # negative pushes arousal
    arousal = max(0.0, min(1.0, 0.6*sent_norm + 0.4*cps_norm))

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Language", lang.upper())
    with c2: st.metric("Mood", f"{emoji} {mood}")
    with c3: st.metric("Typing Speed", f"{cps:.1f} cps")

    render_heartbeat(arousal, f"Arousal from typing & sentiment")

    if st.button("Save entry"):
        append_log({
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "mode":"text",
            "text": txt[:200],
            "lang": lang,
            "emotion": mood,
            "score": score,
            "cps": round(cps,2),
            "f0": "", "energy":"", "rate":""
        })
        st.success("Saved to logs/session_events.csv")

# ------------------- VOICE TAB -------------------
with tabs[1]:
    st.subheader("Live Voice (Google-style subtitles)")

    rtc = webrtc_streamer(
        key="live-voice",
        mode=WebRtcMode.RECVONLY,
        audio_receiver_size=1024,
        rtc_configuration=RTCConfiguration(
            {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": False, "audio": True},
    )

    # rolling buffer for ~5 seconds audio (16kHz mono)
    if "audio_buf" not in st.session_state:
        st.session_state.audio_buf = deque(maxlen=16_000*6)  # 6 sec
        st.session_state.sub_text = ""
        st.session_state.lang_voice = "und"
        st.session_state.last_proc = 0.0

    sub_placeholder = st.empty()
    kpi1,kpi2,kpi3 = st.columns(3)
    kpi1.metric("Voice Language","—")
    kpi2.metric("Voice Mood","—")
    kpi3.metric("Speaking Rate","—")

    # read frames
    if rtc and rtc.state.playing and rtc.audio_receiver:
        while True:
            try:
                frame = rtc.audio_receiver.get_frame(timeout=0.02)
            except queue.Empty:
                break
            if frame:
                pcm = frame.to_ndarray().astype(np.int16)  # shape (samples, channels?) -> ndarray
                # collapse channels
                if pcm.ndim == 2:
                    pcm = pcm.mean(axis=1).astype(np.int16)
                st.session_state.audio_buf.extend(pcm.tolist())

        # process every ~1.2 sec
        if (time.time() - st.session_state.last_proc) > 1.2 and len(st.session_state.audio_buf) > 16_000:
            st.session_state.last_proc = time.time()
            # make small wav in-memory
            arr = np.array(st.session_state.audio_buf, dtype=np.int16)
            wav_bytes = io.BytesIO()
            wavfile.write(wav_bytes, 16000, arr)
            wav_bytes.seek(0)

            # fast transcription (tiny)
            segments, _ = whisper_model.transcribe(wav_bytes, language=None, beam_size=1, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300))
            subs = []
            for seg in segments:
                subs.append(seg.text.strip())
            text_live = " ".join(subs).strip()
            if text_live:
                st.session_state.sub_text = text_live
                # language (from transcript text)
                lang_v = detect_lang(text_live)
                st.session_state.lang_voice = lang_v

                # prosody (quick)
                arr_f = arr.astype(np.float32)/32768.0
                f0, energy, rate = prosody_features(arr_f, 16000)
                vmood, vemoji, ar = voice_mood_from_prosody(f0, energy, rate)

                sub_placeholder.info(f"**You said:** {text_live}")
                kpi1.metric("Voice Language", lang_v.upper())
                kpi2.metric("Voice Mood", f"{vemoji} {vmood}")
                kpi3.metric("Speaking Rate", f"{rate:.2f} onsets/s")

                # heartbeat from voice arousal
                render_heartbeat(ar, f"Arousal from voice (F0/energy/rate)")

                # auto-log rolling events (each process tick)
                append_log({
                    "ts": dt.datetime.now().isoformat(timespec="seconds"),
                    "mode":"voice",
                    "text": text_live[:200],
                    "lang": lang_v,
                    "emotion": vmood,
                    "score": "", "cps":"",
                    "f0": round(f0,1), "energy": round(float(energy),3), "rate": round(rate,2)
                })

    st.caption("Subtitles update about every second while you speak. (Model: faster-whisper tiny)")

# ------------------- DASHBOARD TAB -------------------
with tabs[2]:
    st.subheader("Recent Events")
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        df = df.tail(200).iloc[::-1].reset_index(drop=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        cm1, cm2 = st.columns(2)
        with cm1:
            st.caption("Top languages")
            if not df.empty:
                langs = df["lang"].value_counts().head(8)
                st.bar_chart(langs)
        with cm2:
            st.caption("Emotions distribution")
            if not df.empty:
                emos = df["emotion"].value_counts().head(8)
                st.bar_chart(emos)
    else:
        st.info("No logs yet. Speak or type to populate the dashboard.")

# ------------------- Footer -------------------
st.markdown("---")
st.caption("Built for real-time multilingual feedback: language auto-detect, live subtitles, prosody-based emotion, heartbeat arousal meter, and CSV logging.")
