# -*- coding: utf-8 -*-
# 🛵 Delivery Mood Demo — Pan-India NLP + Voice (HF Whisper) + Live Gauge
# Single-file Streamlit app that runs on Streamlit Cloud (Python 3.11).
# - Language auto-detect (native scripts + romanized Indic via langid + heuristics)
# - Sentiment/Emotion/Toxicity via local HF model (export_model/) or rule fallback
# - Live typing-speed -> Mood Gauge (energy 0–100)
# - WebRTC mic -> HF Whisper API transcription -> same NLP analysis
# - Keeps Review box + Delivery Boy Dashboard
#
# SECRETS:
#   In Streamlit Cloud -> Settings -> Secrets add:
#     HF_API_TOKEN = "hf_xxxxxxxxxxxxxxxxx"
#
# REQS (pin these in requirements.txt):
#   streamlit==1.39.0
#   streamlit-webrtc==0.47.7
#   transformers==4.44.2
#   datasets==3.0.1
#   torch==2.3.1
#   langid==1.1.6
#   numpy==1.26.4
#   pandas==2.2.3
#   soundfile==0.12.1
#   av==12.0.0
#
# If export_model/ exists (tokenizer+config+pytorch_model.bin), the app uses it.
# Otherwise it uses a robust rule-based classifier as fallback.

import os, io, re, time, json, math, queue, base64
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import requests

# ---- optional but safe ----
try:
    import langid  # pure-python, good for Romanized guesses too
    HAS_LANGID = True
except Exception:
    HAS_LANGID = False

# HF pipeline (optional)
HF_MODEL_READY = False
try:
    from transformers import pipeline
    if os.path.isdir("export_model"):
        clf = pipeline("text-classification", model="export_model", tokenizer="export_model", top_k=None)
        HF_MODEL_READY = True
    else:
        clf = None
except Exception:
    clf = None
    HF_MODEL_READY = False

# ---- WebRTC (mic) ----
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

st.set_page_config(page_title="Delivery Mood (NLP + Voice)", page_icon="🛵", layout="wide")

# ========== Hard-coded slang & cues (Pan-India, compact) ==========
SLANG_ABUSIVE = set("""
bokka maddi bekar bakwas ullu kameena kutta chut* l*nd gaandu chutiya ch**ya sala
poda dei bloody waste idiot bastard suvar* panni puka mosam chindi
""".split())
SLANG_FRIENDLY = set("bro anna maga macha machan yaar ra re daa di bruh dude bhaiya thala boss babu".split())
SLANG_POS = set("keka mass semma super mast thopu vera_level nalla nice awesome".split())
NEG_CUES = set("late delay dirty slow cold soggy overpriced refund cancel cancelled rude disappointed hate angry worst horrible pathetic".split())
POS_CUES = set("good great awesome amazing excellent tasty fresh quick yummy clean crisp recommend thanks superb love".split())

EMOJI_POS = "😀😃😄😁😆😊😍🤩👍👏✨🔥🥳😌🙂"
EMOJI_NEG = "😠😡🤬💢😞😟😢😭☹️😒"
EMOJI_MIX = "🙄😏🤨😐"

ROMAN_HINTS = {
    "te": ["inka", "chala", "bokka", "ra", "randi", "inka", "enthaa", "ayyoo", "inka fast", "pani", "em", "antava"],
    "ta": ["dei", "machan", "anna", "podu", "romba", "seri", "sappa", "semma", "mass", "vera level"],
    "hi": ["bhai", "bahut", "yaar", "nahi", "jaldi", "accha", "bkl", "mc", "bc", "mast", "bekar"],
    "ml": ["alle", "nalla", "cha", "pani", "eda", "enthe", "entha", "polichu"],
    "kn": ["maga", "bega", "sari", "swalpa", "eshtu", "madkond", "kannada"],
}

SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),  # Devanagari
    "te": (0x0C00, 0x0C7F),  # Telugu
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "ml": (0x0D00, 0x0D7F),  # Malayalam
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "bn": (0x0980, 0x09FF),  # Bengali
    "gu": (0x0A80, 0x0AFF),  # Gujarati
    "pa": (0x0A00, 0x0A7F),  # Gurmukhi
    "or": (0x0B00, 0x0B7F),  # Odia
}

def detect_script_lang(s: str) -> str | None:
    for ch in s:
        cp = ord(ch)
        for lang, (a, b) in SCRIPT_RANGES.items():
            if a <= cp <= b:
                return lang
    return None

def guess_roman_lang(s: str) -> str:
    s_low = s.lower()
    # heuristic by hints
    scores = {k: 0 for k in ROMAN_HINTS}
    for lang, hints in ROMAN_HINTS.items():
        for h in hints:
            if h in s_low:
                scores[lang] += 1
    best = max(scores, key=lambda k: scores[k])
    if scores[best] > 0:
        return best
    # langid fallback
    if HAS_LANGID:
        code, _ = langid.classify(s_low)
        # map en->und for roman-indic
        if code in {"hi", "te", "ta", "ml", "kn", "bn", "gu", "pa", "ur"}:
            return code
    return "en"

def auto_language(text: str) -> str:
    if not text.strip():
        return "und"
    native = detect_script_lang(text)
    if native:
        return native
    # roman
    return guess_roman_lang(text)

# ========== Rule Sentiment/Emotion/Toxicity (fallback) ==========
def rule_classifier(text: str):
    t = text.strip()
    if not t:
        return {"sentiment": "neutral", "emotion": "neutral", "toxicity": 0, "reasons": []}

    low = t.lower()
    reasons = []
    score = 0

    # emojis
    if any(e in t for e in EMOJI_POS): score += 3; reasons.append("positive emoji")
    if any(e in t for e in EMOJI_NEG): score -= 4; reasons.append("negative emoji")
    if any(e in t for e in EMOJI_MIX): reasons.append("mixed emoji")

    # cues
    pos_hits = [w for w in POS_CUES if w in low]
    neg_hits = [w for w in NEG_CUES if w in low]
    score += 2 * len(pos_hits) - 2.5 * len(neg_hits)
    if pos_hits: reasons.append(f"+{','.join(pos_hits)}")
    if neg_hits: reasons.append(f"-{','.join(neg_hits)}")

    # slang
    if any(w in low for w in SLANG_ABUSIVE):
        reasons.append("abusive slang")
        tox = 4
        score -= 3
    else:
        tox = 0

    # ALLCAPS boost anger
    if re.search(r"[A-Z]{4,}", t):
        score -= 1.5; reasons.append("ALLCAPS")

    # punctuation intensity
    exc = t.count("!")
    if exc >= 2:
        score -= 0.5; reasons.append("!!")

    # map to classes
    if score <= -3:
        senti, emo = "negative", "angry"
    elif score <= -1:
        senti, emo = "negative", "disappointed"
    elif score < 1:
        senti, emo = "neutral", "neutral"
    elif score < 3:
        senti, emo = "positive", "satisfied"
    else:
        senti, emo = "positive", "delighted"

    return {"sentiment": senti, "emotion": emo, "toxicity": tox, "reasons": reasons}

def hf_classifier(text: str):
    try:
        out = clf(text)[0]  # list of dicts if top_k=None
        # Expect format: list of {"label": "...", "score": ...} — adapt as needed
        # We'll derive sentiment from labels; emotion coarse mapping
        # If your model outputs "negative/neutral/positive"
        labels = {d["label"].lower(): d["score"] for d in out}
        sentiment = max(labels, key=labels.get) if labels else "neutral"
        # crude emotion mapping
        if sentiment == "negative":
            emo = "angry" if "anger" in labels else "disappointed"
        elif sentiment == "positive":
            emo = "delighted"
        else:
            emo = "neutral"
        return {"sentiment": sentiment, "emotion": emo, "toxicity": 0, "reasons": [f"model:{sentiment}"]}
    except Exception as e:
        return rule_classifier(text)

# ========== Mood Gauge from typing + text ==========
def update_typing_speed(txt: str):
    now = time.time()
    meta = st.session_state.get("typing_meta", {"t": now, "n": 0, "cps": 0.0})
    dt = max(0.25, now - meta["t"])
    dn = max(0, len(txt) - meta["n"])
    cps = dn / dt
    st.session_state["typing_meta"] = {"t": now, "n": len(txt), "cps": cps}
    arousal = int(min(100, (cps / 10.0) * 100))  # saturates at ~10 cps
    return cps, arousal

def mood_score(model_energy: float, arousal_typing: int) -> int:
    # model_energy: 0..1 prob mapped; here: positive/anger treated as high-energy; sad low
    # We treat delighted/angry as high energy; neutral mid; sad/disappointed low.
    return int(0.6 * (model_energy * 100) + 0.4 * arousal_typing)

def energy_from_emotion(emotion: str) -> float:
    emotion = (emotion or "").lower()
    if emotion in ("angry", "anger", "delighted", "joy", "excited"):
        return 0.9
    if emotion in ("satisfied", "happy"):
        return 0.7
    if emotion in ("neutral",):
        return 0.5
    if emotion in ("disappointed", "sad"):
        return 0.25
    return 0.5

# ========== HF Whisper transcription ==========
HF_API_TOKEN = os.environ.get("HF_API_TOKEN") or st.secrets.get("HF_API_TOKEN", None)
WHISPER_MODEL = "openai/whisper-small"  # can edit in sidebar

def hf_whisper_transcribe(wav_bytes: bytes, model_name: str = WHISPER_MODEL, lang_hint: str | None = None) -> str:
    if not HF_API_TOKEN:
        raise RuntimeError("Missing HF_API_TOKEN secret.")
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    # HF accepts: {"inputs": audio bytes} or raw bytes with headers; we'll send raw with params
    params = {}
    if lang_hint:
        params["language"] = lang_hint
    resp = requests.post(url, headers=headers, data=wav_bytes, params=params, timeout=90)
    resp.raise_for_status()
    js = resp.json()
    # HF whisper returns list or dict depending on model; normalize
    if isinstance(js, dict) and "text" in js:
        return js["text"]
    if isinstance(js, list) and len(js) and "text" in js[0]:
        return js[0]["text"]
    # fallback
    return str(js)

# ========== Mic capture helper ==========
def frames_to_wav(frames, sample_rate=16000):
    """Concatenate int16 mono pcm to WAV bytes."""
    import soundfile as sf
    pcm = b"".join(frames)
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    # write wav
    buf = io.BytesIO()
    sf.write(buf, arr, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()

# ========== UI ==========
st.title("🛵 Delivery Mood — Text + Voice (Pan-India, Auto-Detect)")

colA, colB = st.columns([1.2, 1])

# --- LEFT: Text Review with live language + mood ---
with colA:
    st.subheader("✍️ Text Review (auto language + mood)")
    txt = st.text_area("Type here…", height=120, key="review_text")
    # live language
    lang = auto_language(txt)
    st.caption(f"Detected language: **{lang.upper()}**" if lang != "und" else "Detected language: **UND**")

    # typing speed
    cps, ar_typ = update_typing_speed(txt)
    st.caption(f"Typing speed: **{cps:.1f} chars/sec**")

    # classify
    if HF_MODEL_READY:
        pred = hf_classifier(txt)
    else:
        pred = rule_classifier(txt)

    # energy & gauge
    energy = energy_from_emotion(pred["emotion"])
    gauge = mood_score(energy, ar_typ)

    gcol1, gcol2, gcol3 = st.columns(3)
    with gcol1: st.metric("Sentiment", pred["sentiment"].title())
    with gcol2: st.metric("Emotion", pred["emotion"].title())
    with gcol3: st.metric("Toxicity", pred["toxicity"])

    st.progress(gauge, text=f"Mood Energy {gauge}/100")
    if pred["reasons"]:
        st.caption("Why: " + ", ".join(pred["reasons"][:6]))

    # Save to dashboard
    if st.button("➕ Add to Dashboard"):
        oid = f"OD-{int(time.time())%100000}"
        rec = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "mode": "Text",
            "order_id": oid,
            "lang": lang,
            "typing_cps": round(cps,1),
            "sentiment": pred["sentiment"],
            "emotion": pred["emotion"],
            "toxicity": pred["toxicity"],
            "mood_energy": gauge,
            "snippet": (txt[:120] + "…") if len(txt) > 120 else txt
        }
        st.session_state.setdefault("orders", []).append(rec)
        st.success(f"Added to dashboard: {oid}")

# --- RIGHT: Voice (WebRTC mic -> HF Whisper) ---
with colB:
    st.subheader("🎙️ Voice Review (mic → Whisper → mood)")
    if not HAS_WEBRTC:
        st.info("Mic capture not available in this environment. Try file upload below or run locally.")
    else:
        st.caption("Click **Start** to record. Speak for 4–8 seconds. Click **Stop**, then **Transcribe**.")
        audio_frames = []
        status_placeholder = st.empty()

        def recv_audio(frame):
            # frame is av.AudioFrame
            pcm16 = frame.to_ndarray().astype(np.int16).tobytes()
            audio_frames.append(pcm16)
            return frame

        webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True, "video": False},
)


        if webrtc_ctx and webrtc_ctx.state.playing:
            status_placeholder.info("🎤 Recording… speak now")

        trans_txt = st.text_area("Transcript (editable):", height=100, key="voice_transcript")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📝 Transcribe (HF Whisper)"):
                try:
                    wav_bytes = frames_to_wav(audio_frames, sample_rate=16000)
                    hint = auto_language(trans_txt) if trans_txt.strip() else None
                    text = hf_whisper_transcribe(wav_bytes, WHISPER_MODEL, lang_hint=hint)
                    st.session_state["voice_transcript"] = text
                    st.success("Transcribed!")
                    st.session_state["voice_last"] = text
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
        with c2:
            if st.button("⟲ Use last transcript"):
                text = st.session_state.get("voice_last", "")
                if text:
                    st.session_state["voice_transcript"] = text
                else:
                    st.info("No previous transcript yet.")

        # show updated transcript text area
        if "voice_transcript" in st.session_state:
            st.session_state["voice_transcript"] = st.text_area(
                "Transcript (editable):", value=st.session_state["voice_transcript"], height=100)

        # analyze transcript
        text_for_nlp = st.session_state.get("voice_transcript", "").strip()
        if text_for_nlp:
            vlang = auto_language(text_for_nlp)
            if HF_MODEL_READY:
                vpred = hf_classifier(text_for_nlp)
            else:
                vpred = rule_classifier(text_for_nlp)
            v_energy = mood_score(energy_from_emotion(vpred["emotion"]), arousal_typing=60)  # medium arousal for voice
            vc1, vc2, vc3 = st.columns(3)
            with vc1: st.metric("Lang", vlang.upper())
            with vc2: st.metric("Emotion", vpred["emotion"].title())
            with vc3: st.metric("Toxicity", vpred["toxicity"])
            st.progress(v_energy, text=f"Mood Energy {v_energy}/100")
            if vpred["reasons"]:
                st.caption("Why: " + ", ".join(vpred["reasons"][:6]))

            if st.button("➕ Add to Dashboard (Voice)"):
                oid = f"OD-{int(time.time())%100000}"
                rec = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "mode": "Voice",
                    "order_id": oid,
                    "lang": vlang,
                    "typing_cps": None,
                    "sentiment": vpred["sentiment"],
                    "emotion": vpred["emotion"],
                    "toxicity": vpred["toxicity"],
                    "mood_energy": v_energy,
                    "snippet": (text_for_nlp[:120] + "…") if len(text_for_nlp) > 120 else text_for_nlp
                }
                st.session_state.setdefault("orders", []).append(rec)
                st.success(f"Added to dashboard: {oid}")
        else:
            st.caption("No transcript yet.")

st.markdown("---")
st.subheader("👷 Delivery Boy Dashboard")
orders = st.session_state.get("orders", [])
if orders:
    # High priority = Angry/Disappointed or high toxicity
    def priority(o):
        if o["toxicity"] >= 3 or o["emotion"] in ("angry", "disappointed"):
            return "High"
        return "Normal"

    rows = []
    for o in orders:
        rows.append({
            "⏰ Time": o["time"],
            "Mode": o["mode"],
            "OrderID": o["order_id"],
            "Lang": o["lang"].upper() if o["lang"] else "UND",
            "Typing (cps)": o["typing_cps"] if o["typing_cps"] is not None else "—",
            "Sentiment": o["sentiment"].title(),
            "Emotion": o["emotion"].title(),
            "Toxicity": o["toxicity"],
            "Priority": priority(o),
            "Mood": o["mood_energy"],
            "Snippet": o["snippet"],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Priority","Mood"], ascending=[True, False])
    st.dataframe(df, hide_index=True, use_container_width=True)
else:
    st.info("No entries yet. Add from Text or Voice panels above.")
