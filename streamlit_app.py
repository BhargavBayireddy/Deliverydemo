# -*- coding: utf-8 -*-
# Delivery Mood (NLP + Voice) — Streamlit
# - Text review: auto language guess (langid + Indic slang lexicons)
# - Typing-speed meter → affects live mood gauge
# - Profanity/slang severity → affects mood gauge
# - Voice review: WebRTC capture → Whisper (HF Inference API) → text, lang, mood
# - Dashboard + Logs

import io, time, math, re, json, queue
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
from better_profanity import profanity
from langid.langid import LanguageIdentifier, model
import plotly.graph_objects as go

# WebRTC
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ------------- Config -------------
st.set_page_config(page_title="Delivery Mood (NLP + Voice)", page_icon="🛵", layout="wide")
HF_TOKEN = st.secrets.get("HF_API_TOKEN", None)
WHISPER_MODEL = "openai/whisper-large-v3-turbo"  # HF Inference model id
ENABLE_ASR = HF_TOKEN is not None

# ------------- Globals -------------
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
profanity.load_censor_words()

# Add India chat-slang (roman scripts) to profanity detector (won't censor UI; we only score)
INDIA_SLANG = [
    "bokka","p**ka","bokka*", "boddi","puka","puka*", "bokkaaa", "sala","bhosdi","bhosdike","bhenchod",
    "mc","bc","madarchod","chutiya","chu***","choot","gaand","gand","haraami","haraamkhor","kutte","kutta",
    "bloody", "saala","kameena","pandu", "waste fellow", "worst fellow", "nuisance"
]
profanity.add_censor_words([w for w in INDIA_SLANG if "*" not in w])

# Quick romanized cue words to bias lang guess for Indic chat
LEX = {
    "te": {"inka","enni","entha","bokka","cheyyi","cheppara","vachava","naku","kavali","inka late","vegga"},
    "hi": {"kya","bhai","jaldi","kaam","mat","nahi","bahut","sab","bhaiya","bhook","aaya"},
    "ta": {"enna","seri","romba","sapadu","vaanga","seekiram","thambi","machan","pannunga"},
    "kn": {"yen","madri","beka","baro","idu","ninna"},
    "ml": {"entha","verum","poyi","cheyyu","vannu","kazhicho","ninte"},
}

LANG_NAME = {
    "en":"English","hi":"Hindi","te":"Telugu","ta":"Tamil","kn":"Kannada","ml":"Malayalam",
    "bn":"Bengali","mr":"Marathi","gu":"Gujarati","pa":"Punjabi","ur":"Urdu","und":"Unknown"
}

# Sentiment lists (very lightweight, multilingual-ish)
POS = set("good great awesome amazing tasty fresh quick ontime polite friendly love perfect nice superb wow yummy delicious thanks".split())
NEG = set("bad terrible awful worst cold soggy late delay dirty slow rude stale raw burnt bland overpriced missing leak spilled refund replace cancel angry frustrated annoyed disappointed horrible issue problem broken inedible vomit sick hair waste nuisance".split())

INTENS = {"very":1.3,"really":1.2,"so":1.2,"too":1.15,"extremely":1.4,"super":1.3,"bahut":1.2,"chala":1.2,"romba":1.2}

# ------------- Small helpers -------------

def detect_lang(text: str) -> str:
    """langid + Indic roman-cues; returns ISO-639-1 or 'und'"""
    t = (text or "").strip()
    if not t:
        return "und"
    # langid guess
    lid, prob = identifier.classify(t)
    # romanized bias
    low = t.lower()
    counts = {k: sum(1 for w in v if w in low) for k,v in LEX.items()}
    if counts and max(counts.values()) >= 1:
        top = max(counts, key=lambda k: counts[k])
        # if langid is uncertain or bias strong, switch
        if prob < 0.85 or counts[top] >= 2:
            return top
    return lid if lid in LANG_NAME else "und"

def profanity_score(text: str) -> Tuple[int, List[str]]:
    """Return severity (0..100) and matched terms"""
    low = text.lower()
    found = []
    sev = 0
    for w in INDIA_SLANG:
        if w.replace("*","") and w.replace("*","") in low:
            found.append(w.replace("*",""))
            sev += 12
    # common English profanities via better_profanity wordlist
    if profanity.contains_profanity(low):
        sev = max(sev, 25)
    return min(100, sev), list(dict.fromkeys(found))

def sentiment_score(text: str) -> Tuple[int,str,str,List[str]]:
    """Lightweight polarity → (-100..100, mood, emoji, hits)"""
    t = text.strip()
    if not t:
        return 0,"Neutral","😐",[]
    low = re.sub(r"[^\w\s!?]"," ",t.lower())
    words = low.split()
    score = 0.0; hits=[]
    exclam = t.count("!")
    caps = 1.2 if re.search(r"[A-Z]{3,}", t) else 1.0
    longrep = 1.15 if re.search(r"(.)\1{3,}", low) else 1.0
    mul = caps*longrep*(1.05 if exclam>=2 else 1.0)

    for i,w in enumerate(words):
        base = 0
        if w in POS: base = 2.3
        if w in NEG: base = -2.6
        if base != 0 and i>0 and words[i-1] in INTENS:
            base *= INTENS[words[i-1]]
        score += base
        if base != 0: hits.append(w)

    raw = max(-40, min(40, score*mul))
    scaled = int(round(raw/40*100))
    if scaled <= -60: mood,emoji="Angry","😡"
    elif scaled <= -30: mood,emoji="Frustrated","😠"
    elif scaled <= -6: mood,emoji="Disappointed","😕"
    elif -5 <= scaled <= 5: mood,emoji="Neutral","😐"
    elif scaled <= 35: mood,emoji="Satisfied","🙂"
    else: mood,emoji="Delighted","🤩"
    return scaled, mood, emoji, list(dict.fromkeys(hits))

def mood_value(base_sent: int, prof_sev: int, typing_cps: float) -> Tuple[int,str]:
    """
    Combine sentiment(-100..100), profanity severity(0..100), typing speed cps.
    Faster angry typing drags mood down a bit.
    """
    val = base_sent
    val -= int(prof_sev * 0.35)         # profanity penalty
    if base_sent <= 0 and typing_cps >= 6.0:
        val -= 10                        # fast-angry typing
    return max(-100, min(100, val)), ("fast typing" if typing_cps >= 6.0 else "")

def gauge(value: int, title="Mood"):
    v = (value + 100)/2  # map -100..100 → 0..100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={'suffix': " / 100"},
        gauge={'axis': {'range':[0,100]},
               'bar': {'thickness': 0.3},
               'steps': [
                   {'range': [0,20], 'color': '#ef4444'},
                   {'range': [20,40], 'color': '#f97316'},
                   {'range': [40,60], 'color': '#e5e7eb'},
                   {'range': [60,80], 'color': '#a7f3d0'},
                   {'range': [80,100], 'color': '#34d399'},
               ]},
        title={'text': title}
    ))
    fig.update_layout(height=240, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

def hf_whisper_transcribe(audio_bytes: bytes) -> Dict[str, Any]:
    if not ENABLE_ASR:
        return {"ok": False, "error": "HF_API_TOKEN missing. Set it in Streamlit Secrets."}
    url = f"https://api-inference.huggingface.co/models/{WHISPER_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    params = {"return_timestamps": False, "task": "transcribe"}  # auto language
    try:
        r = requests.post(url, headers=headers, data=audio_bytes, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        # HF may return {"text": "..."} or list of segments; normalize
        text = js.get("text") if isinstance(js, dict) else None
        lang = js.get("language", None) if isinstance(js, dict) else None
        return {"ok": True, "text": text or "", "lang": lang or "und", "raw": js}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ------------- State -------------
if "logs" not in st.session_state: st.session_state.logs=[]
if "latest_text" not in st.session_state: st.session_state.latest_text=""
if "latest_lang" not in st.session_state: st.session_state.latest_lang="und"
if "meta" not in st.session_state: st.session_state.meta={"len":0,"ts":time.time(),"cps":0.0}
if "voice_transcript" not in st.session_state: st.session_state.voice_transcript=""
if "voice_lang" not in st.session_state: st.session_state.voice_lang="und"
if "last_mood" not in st.session_state: st.session_state.last_mood=0

# ------------- UI -------------
tabs = st.tabs(["📊 Dashboard", "✍️ Text Review", "🎙️ Voice Review", "📄 Logs"])

# ===== Dashboard =====
with tabs[0]:
    st.subheader("Live Mood Gauge")
    gauge(st.session_state.last_mood, "Overall mood")

    if st.session_state.latest_text:
        st.write("**Latest text**:", st.session_state.latest_text)
        st.caption(f"Language: {LANG_NAME.get(st.session_state.latest_lang,'Unknown')}")

    if st.session_state.voice_transcript:
        st.write("**Last voice transcript**:", st.session_state.voice_transcript)
        st.caption(f"Voice language: {LANG_NAME.get(st.session_state.voice_lang,'Unknown')}")

# ===== Text Review =====
with tabs[1]:
    st.subheader("Review (auto detects language while you type)")

    def _on_review_change():
        now = time.time()
        txt = st.session_state._review_box
        meta = st.session_state.meta
        dt = max(0.25, now - meta["ts"])
        cps = max(0.0, (len(txt) - meta["len"]) / dt)
        st.session_state.meta = {"len": len(txt), "ts": now, "cps": cps}

    # Widget -> keep in its own key; then copy into state fields we control
    if "_review_box" not in st.session_state: st.session_state._review_box = ""
    st.session_state._review_box = st.text_input(
        "Type your message", value=st.session_state._review_box, on_change=_on_review_change
    )
    text = st.session_state._review_box

    lang = detect_lang(text)
    base_sent, mood, emoji, hits = sentiment_score(text)
    prof_sev, prof_terms = profanity_score(text)
    cps = st.session_state.meta["cps"]

    final_mood, speed_note = mood_value(base_sent, prof_sev, cps)
    st.session_state.last_mood = final_mood
    st.session_state.latest_text = text
    st.session_state.latest_lang = lang

    st.progress(int((final_mood+100)/2), text=f"{emoji} {mood} • Gauge {(final_mood+100)//2}/100")

    c1,c2,c3 = st.columns(3)
    c1.metric("Language", LANG_NAME.get(lang,"Unknown"))
    c2.metric("Text Sentiment", f"{base_sent} ({emoji} {mood})")
    c3.metric("Toxicity/Slang", "Yes" if prof_sev>0 else "No")

    c4,c5,c6 = st.columns(3)
    c4.metric("Typing speed", f"{cps:.1f} cps" + (" • fast" if speed_note else ""))
    c5.metric("Profanity score", prof_sev)
    c6.metric("Triggers", ", ".join(hits[:5]) if hits else "—")

    st.caption("Tip: Type even 1–2 words (e.g., 'entha late') — language & mood update instantly.")

    if st.button("Save as Latest Feedback → Dashboard"):
        st.session_state.logs.append({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "text",
            "text": text, "lang": lang,
            "sent": base_sent, "mood": final_mood, "cps": cps,
            "profanity": prof_terms
        })
        st.success("Saved!")

# ===== Voice Review =====
with tabs[2]:
    st.subheader("Live Voice (auto language via Whisper)")
    if not ENABLE_ASR:
        st.warning("Set **HF_API_TOKEN** in Streamlit Secrets to enable Whisper ASR.")
    st.caption("Click **Start**, allow microphone, speak ~6–8s. Then click **Transcribe & Analyze**.")

    # WebRTC receiver
    audio_frames: "queue.Queue[bytes]" = queue.Queue()

    def audio_callback(frame):
        # PCM16 bytes
        pcm = frame.to_ndarray().astype(np.int16).tobytes()
        audio_frames.put(pcm)

    webrtc_ctx = webrtc_streamer(
        key="voice-asr",
        mode=WebRtcMode.RECVONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        audio_frame_callback=audio_callback,
    )

    colA, colB = st.columns([1,1])
    captured = st.button("🎙️ Capture 6s")
    if captured:
        st.info("Recording 6 seconds...")
        time.sleep(6)

    do_trans = st.button("🧠 Transcribe & Analyze", type="primary")
    if do_trans:
        # Gather frames → WAV (16 kHz)
        raw = b"".join(list(audio_frames.queue))
        if not raw:
            st.error("No audio captured. Click Start, speak, then try again.")
        else:
            # Pack as WAV float32 using simple header via soundfile-less approach
            # We assume 48000 Hz from WebRTC; downsample by picking every 3rd sample to ~16k
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)/32768.0
            if arr.size > 0:
                arr = arr[::3]  # crude downsample
            # Write minimal WAV
            import wave
            buff = io.BytesIO()
            with wave.open(buff, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((arr*32767.0).astype(np.int16).tobytes())
            audio_bytes = buff.getvalue()

            with st.spinner("Calling Whisper…"):
                res = hf_whisper_transcribe(audio_bytes)

            if not res["ok"]:
                st.error(f"ASR error: {res['error']}")
            else:
                txt = res["text"] or ""
                lang_from_asr = res.get("lang","und") or "und"
                # Fallback refine with our detector if Whisper didn't fill it
                lang_final = lang_from_asr if lang_from_asr!="und" else detect_lang(txt)

                st.session_state.voice_transcript = txt
                st.session_state.voice_lang = lang_final

                base_sent, mood, emoji, hits = sentiment_score(txt)
                prof_sev, prof_terms = profanity_score(txt)
                final_mood, _ = mood_value(base_sent, prof_sev, typing_cps=0.0)
                st.session_state.last_mood = final_mood

                st.text_area("Transcript (editable)", value=st.session_state.voice_transcript, height=140)
                c1,c2,c3 = st.columns(3)
                c1.metric("Language", LANG_NAME.get(lang_final,"Unknown"))
                c2.metric("Text Sentiment", f"{base_sent} ({emoji} {mood})")
                c3.metric("Toxicity/Slang", "Yes" if prof_sev>0 else "No")
                gauge(final_mood, "Voice mood")

                st.session_state.logs.append({
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": "voice",
                    "text": txt, "lang": lang_final,
                    "sent": base_sent, "mood": final_mood, "cps": 0.0,
                    "profanity": prof_terms
                })
                st.success("Voice processed & saved to logs.")

# ===== Logs =====
with tabs[3]:
    st.subheader("Logs")
    if not st.session_state.logs:
        st.info("No entries yet.")
    else:
        df = pd.DataFrame(st.session_state.logs)
        st.dataframe(df, use_container_width=True)
# -*- coding: utf-8 -*-
# Delivery Mood (NLP + Voice) — Streamlit
# - Text review: auto language guess (langid + Indic slang lexicons)
# - Typing-speed meter → affects live mood gauge
# - Profanity/slang severity → affects mood gauge
# - Voice review: WebRTC capture → Whisper (HF Inference API) → text, lang, mood
# - Dashboard + Logs

import io, time, math, re, json, queue
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
from better_profanity import profanity
from langid.langid import LanguageIdentifier, model
import plotly.graph_objects as go

# WebRTC
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ------------- Config -------------
st.set_page_config(page_title="Delivery Mood (NLP + Voice)", page_icon="🛵", layout="wide")
HF_TOKEN = st.secrets.get("HF_API_TOKEN", None)
WHISPER_MODEL = "openai/whisper-large-v3-turbo"  # HF Inference model id
ENABLE_ASR = HF_TOKEN is not None

# ------------- Globals -------------
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
profanity.load_censor_words()

# Add India chat-slang (roman scripts) to profanity detector (won't censor UI; we only score)
INDIA_SLANG = [
    "bokka","p**ka","bokka*", "boddi","puka","puka*", "bokkaaa", "sala","bhosdi","bhosdike","bhenchod",
    "mc","bc","madarchod","chutiya","chu***","choot","gaand","gand","haraami","haraamkhor","kutte","kutta",
    "bloody", "saala","kameena","pandu", "waste fellow", "worst fellow", "nuisance"
]
profanity.add_censor_words([w for w in INDIA_SLANG if "*" not in w])

# Quick romanized cue words to bias lang guess for Indic chat
LEX = {
    "te": {"inka","enni","entha","bokka","cheyyi","cheppara","vachava","naku","kavali","inka late","vegga"},
    "hi": {"kya","bhai","jaldi","kaam","mat","nahi","bahut","sab","bhaiya","bhook","aaya"},
    "ta": {"enna","seri","romba","sapadu","vaanga","seekiram","thambi","machan","pannunga"},
    "kn": {"yen","madri","beka","baro","idu","ninna"},
    "ml": {"entha","verum","poyi","cheyyu","vannu","kazhicho","ninte"},
}

LANG_NAME = {
    "en":"English","hi":"Hindi","te":"Telugu","ta":"Tamil","kn":"Kannada","ml":"Malayalam",
    "bn":"Bengali","mr":"Marathi","gu":"Gujarati","pa":"Punjabi","ur":"Urdu","und":"Unknown"
}

# Sentiment lists (very lightweight, multilingual-ish)
POS = set("good great awesome amazing tasty fresh quick ontime polite friendly love perfect nice superb wow yummy delicious thanks".split())
NEG = set("bad terrible awful worst cold soggy late delay dirty slow rude stale raw burnt bland overpriced missing leak spilled refund replace cancel angry frustrated annoyed disappointed horrible issue problem broken inedible vomit sick hair waste nuisance".split())

INTENS = {"very":1.3,"really":1.2,"so":1.2,"too":1.15,"extremely":1.4,"super":1.3,"bahut":1.2,"chala":1.2,"romba":1.2}

# ------------- Small helpers -------------

def detect_lang(text: str) -> str:
    """langid + Indic roman-cues; returns ISO-639-1 or 'und'"""
    t = (text or "").strip()
    if not t:
        return "und"
    # langid guess
    lid, prob = identifier.classify(t)
    # romanized bias
    low = t.lower()
    counts = {k: sum(1 for w in v if w in low) for k,v in LEX.items()}
    if counts and max(counts.values()) >= 1:
        top = max(counts, key=lambda k: counts[k])
        # if langid is uncertain or bias strong, switch
        if prob < 0.85 or counts[top] >= 2:
            return top
    return lid if lid in LANG_NAME else "und"

def profanity_score(text: str) -> Tuple[int, List[str]]:
    """Return severity (0..100) and matched terms"""
    low = text.lower()
    found = []
    sev = 0
    for w in INDIA_SLANG:
        if w.replace("*","") and w.replace("*","") in low:
            found.append(w.replace("*",""))
            sev += 12
    # common English profanities via better_profanity wordlist
    if profanity.contains_profanity(low):
        sev = max(sev, 25)
    return min(100, sev), list(dict.fromkeys(found))

def sentiment_score(text: str) -> Tuple[int,str,str,List[str]]:
    """Lightweight polarity → (-100..100, mood, emoji, hits)"""
    t = text.strip()
    if not t:
        return 0,"Neutral","😐",[]
    low = re.sub(r"[^\w\s!?]"," ",t.lower())
    words = low.split()
    score = 0.0; hits=[]
    exclam = t.count("!")
    caps = 1.2 if re.search(r"[A-Z]{3,}", t) else 1.0
    longrep = 1.15 if re.search(r"(.)\1{3,}", low) else 1.0
    mul = caps*longrep*(1.05 if exclam>=2 else 1.0)

    for i,w in enumerate(words):
        base = 0
        if w in POS: base = 2.3
        if w in NEG: base = -2.6
        if base != 0 and i>0 and words[i-1] in INTENS:
            base *= INTENS[words[i-1]]
        score += base
        if base != 0: hits.append(w)

    raw = max(-40, min(40, score*mul))
    scaled = int(round(raw/40*100))
    if scaled <= -60: mood,emoji="Angry","😡"
    elif scaled <= -30: mood,emoji="Frustrated","😠"
    elif scaled <= -6: mood,emoji="Disappointed","😕"
    elif -5 <= scaled <= 5: mood,emoji="Neutral","😐"
    elif scaled <= 35: mood,emoji="Satisfied","🙂"
    else: mood,emoji="Delighted","🤩"
    return scaled, mood, emoji, list(dict.fromkeys(hits))

def mood_value(base_sent: int, prof_sev: int, typing_cps: float) -> Tuple[int,str]:
    """
    Combine sentiment(-100..100), profanity severity(0..100), typing speed cps.
    Faster angry typing drags mood down a bit.
    """
    val = base_sent
    val -= int(prof_sev * 0.35)         # profanity penalty
    if base_sent <= 0 and typing_cps >= 6.0:
        val -= 10                        # fast-angry typing
    return max(-100, min(100, val)), ("fast typing" if typing_cps >= 6.0 else "")

def gauge(value: int, title="Mood"):
    v = (value + 100)/2  # map -100..100 → 0..100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={'suffix': " / 100"},
        gauge={'axis': {'range':[0,100]},
               'bar': {'thickness': 0.3},
               'steps': [
                   {'range': [0,20], 'color': '#ef4444'},
                   {'range': [20,40], 'color': '#f97316'},
                   {'range': [40,60], 'color': '#e5e7eb'},
                   {'range': [60,80], 'color': '#a7f3d0'},
                   {'range': [80,100], 'color': '#34d399'},
               ]},
        title={'text': title}
    ))
    fig.update_layout(height=240, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

def hf_whisper_transcribe(audio_bytes: bytes) -> Dict[str, Any]:
    if not ENABLE_ASR:
        return {"ok": False, "error": "HF_API_TOKEN missing. Set it in Streamlit Secrets."}
    url = f"https://api-inference.huggingface.co/models/{WHISPER_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    params = {"return_timestamps": False, "task": "transcribe"}  # auto language
    try:
        r = requests.post(url, headers=headers, data=audio_bytes, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        # HF may return {"text": "..."} or list of segments; normalize
        text = js.get("text") if isinstance(js, dict) else None
        lang = js.get("language", None) if isinstance(js, dict) else None
        return {"ok": True, "text": text or "", "lang": lang or "und", "raw": js}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ------------- State -------------
if "logs" not in st.session_state: st.session_state.logs=[]
if "latest_text" not in st.session_state: st.session_state.latest_text=""
if "latest_lang" not in st.session_state: st.session_state.latest_lang="und"
if "meta" not in st.session_state: st.session_state.meta={"len":0,"ts":time.time(),"cps":0.0}
if "voice_transcript" not in st.session_state: st.session_state.voice_transcript=""
if "voice_lang" not in st.session_state: st.session_state.voice_lang="und"
if "last_mood" not in st.session_state: st.session_state.last_mood=0

# ------------- UI -------------
tabs = st.tabs(["📊 Dashboard", "✍️ Text Review", "🎙️ Voice Review", "📄 Logs"])

# ===== Dashboard =====
with tabs[0]:
    st.subheader("Live Mood Gauge")
    gauge(st.session_state.last_mood, "Overall mood")

    if st.session_state.latest_text:
        st.write("**Latest text**:", st.session_state.latest_text)
        st.caption(f"Language: {LANG_NAME.get(st.session_state.latest_lang,'Unknown')}")

    if st.session_state.voice_transcript:
        st.write("**Last voice transcript**:", st.session_state.voice_transcript)
        st.caption(f"Voice language: {LANG_NAME.get(st.session_state.voice_lang,'Unknown')}")

# ===== Text Review =====
with tabs[1]:
    st.subheader("Review (auto detects language while you type)")

    def _on_review_change():
        now = time.time()
        txt = st.session_state._review_box
        meta = st.session_state.meta
        dt = max(0.25, now - meta["ts"])
        cps = max(0.0, (len(txt) - meta["len"]) / dt)
        st.session_state.meta = {"len": len(txt), "ts": now, "cps": cps}

    # Widget -> keep in its own key; then copy into state fields we control
    if "_review_box" not in st.session_state: st.session_state._review_box = ""
    st.session_state._review_box = st.text_input(
        "Type your message", value=st.session_state._review_box, on_change=_on_review_change
    )
    text = st.session_state._review_box

    lang = detect_lang(text)
    base_sent, mood, emoji, hits = sentiment_score(text)
    prof_sev, prof_terms = profanity_score(text)
    cps = st.session_state.meta["cps"]

    final_mood, speed_note = mood_value(base_sent, prof_sev, cps)
    st.session_state.last_mood = final_mood
    st.session_state.latest_text = text
    st.session_state.latest_lang = lang

    st.progress(int((final_mood+100)/2), text=f"{emoji} {mood} • Gauge {(final_mood+100)//2}/100")

    c1,c2,c3 = st.columns(3)
    c1.metric("Language", LANG_NAME.get(lang,"Unknown"))
    c2.metric("Text Sentiment", f"{base_sent} ({emoji} {mood})")
    c3.metric("Toxicity/Slang", "Yes" if prof_sev>0 else "No")

    c4,c5,c6 = st.columns(3)
    c4.metric("Typing speed", f"{cps:.1f} cps" + (" • fast" if speed_note else ""))
    c5.metric("Profanity score", prof_sev)
    c6.metric("Triggers", ", ".join(hits[:5]) if hits else "—")

    st.caption("Tip: Type even 1–2 words (e.g., 'entha late') — language & mood update instantly.")

    if st.button("Save as Latest Feedback → Dashboard"):
        st.session_state.logs.append({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "text",
            "text": text, "lang": lang,
            "sent": base_sent, "mood": final_mood, "cps": cps,
            "profanity": prof_terms
        })
        st.success("Saved!")

# ===== Voice Review =====
with tabs[2]:
    st.subheader("Live Voice (auto language via Whisper)")
    if not ENABLE_ASR:
        st.warning("Set **HF_API_TOKEN** in Streamlit Secrets to enable Whisper ASR.")
    st.caption("Click **Start**, allow microphone, speak ~6–8s. Then click **Transcribe & Analyze**.")

    # WebRTC receiver
    audio_frames: "queue.Queue[bytes]" = queue.Queue()

    def audio_callback(frame):
        # PCM16 bytes
        pcm = frame.to_ndarray().astype(np.int16).tobytes()
        audio_frames.put(pcm)

    webrtc_ctx = webrtc_streamer(
        key="voice-asr",
        mode=WebRtcMode.RECVONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        audio_frame_callback=audio_callback,
    )

    colA, colB = st.columns([1,1])
    captured = st.button("🎙️ Capture 6s")
    if captured:
        st.info("Recording 6 seconds...")
        time.sleep(6)

    do_trans = st.button("🧠 Transcribe & Analyze", type="primary")
    if do_trans:
        # Gather frames → WAV (16 kHz)
        raw = b"".join(list(audio_frames.queue))
        if not raw:
            st.error("No audio captured. Click Start, speak, then try again.")
        else:
            # Pack as WAV float32 using simple header via soundfile-less approach
            # We assume 48000 Hz from WebRTC; downsample by picking every 3rd sample to ~16k
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)/32768.0
            if arr.size > 0:
                arr = arr[::3]  # crude downsample
            # Write minimal WAV
            import wave
            buff = io.BytesIO()
            with wave.open(buff, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((arr*32767.0).astype(np.int16).tobytes())
            audio_bytes = buff.getvalue()

            with st.spinner("Calling Whisper…"):
                res = hf_whisper_transcribe(audio_bytes)

            if not res["ok"]:
                st.error(f"ASR error: {res['error']}")
            else:
                txt = res["text"] or ""
                lang_from_asr = res.get("lang","und") or "und"
                # Fallback refine with our detector if Whisper didn't fill it
                lang_final = lang_from_asr if lang_from_asr!="und" else detect_lang(txt)

                st.session_state.voice_transcript = txt
                st.session_state.voice_lang = lang_final

                base_sent, mood, emoji, hits = sentiment_score(txt)
                prof_sev, prof_terms = profanity_score(txt)
                final_mood, _ = mood_value(base_sent, prof_sev, typing_cps=0.0)
                st.session_state.last_mood = final_mood

                st.text_area("Transcript (editable)", value=st.session_state.voice_transcript, height=140)
                c1,c2,c3 = st.columns(3)
                c1.metric("Language", LANG_NAME.get(lang_final,"Unknown"))
                c2.metric("Text Sentiment", f"{base_sent} ({emoji} {mood})")
                c3.metric("Toxicity/Slang", "Yes" if prof_sev>0 else "No")
                gauge(final_mood, "Voice mood")

                st.session_state.logs.append({
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": "voice",
                    "text": txt, "lang": lang_final,
                    "sent": base_sent, "mood": final_mood, "cps": 0.0,
                    "profanity": prof_terms
                })
                st.success("Voice processed & saved to logs.")

# ===== Logs =====
with tabs[3]:
    st.subheader("Logs")
    if not st.session_state.logs:
        st.info("No entries yet.")
    else:
        df = pd.DataFrame(st.session_state.logs)
        st.dataframe(df, use_container_width=True)
