# -*- coding: utf-8 -*-
# Delivery Mood (Text + Voice) — lightweight, Cloud-safe
# Works without HF token / big models. Upgrades gracefully if optional libs exist.

import time
import re
import math
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---- Optional imports (fail-safe) -------------------------------------------
_HAS_WEBRTC = False
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    _HAS_WEBRTC = True
except Exception:
    _HAS_WEBRTC = False

try:
    import langid
    langid.set_languages(['en','hi','te','ta','bn','ml','mr','pa','gu','kn','ur'])  # bias to India langs
    _HAS_LANGID = True
except Exception:
    _HAS_LANGID = False

# -----------------------------------------------------------------------------
# Streamlit page config MUST be first command
st.set_page_config(page_title="Delivery Mood (NLP + Voice)", page_icon="🛵", layout="wide")

# -----------------------------------------------------------------------------
# Small slang dictionaries (extend any time)
SLANG_HINTS: Dict[str, str] = {
    # lang -> slang tokens (romanized)
    "te": r"(entha|inka|bokka|baava|ra|ayya|emaina|cheyyali|ivvu|naaku|pani|inka|emi|matlad|ivvaru)",
    "hi": r"(bhai|yaar|jaldi|faltu|bakwaas|gussa|thoda|kab|kaise|kya|achha|bura|bhooka|kaam)",
    "ta": r"(da|dei|enna|podu|poda|sapadu|saar|machan|seri|romba)",
    "ml": r"(eda|chetta|ammachi|sheri|polichu|nalla|pani|ente|ivide)",
    "kn": r"(maga|sari|banni|yaake|neenu|madbeku|tumba|ondhu|illi)",
    "mr": r"(khup|zal|ka|bhau|kuthe|zara|kiti|mhantat|jhala)",
    "bn": r"(koto|khub|valo|bhalo|tumi|kothay|kinte|khawa|dada)",
    "pa": r"(paaji|veer|kidda|balle|bhen|chod|karna|hona|menu|kida)",
    "gu": r"(kem|cho|su|mane|bhai|mara|khub|saaru|tame)",
    "ur": r"(bhai|zara|kyun|acha|bura|kitna|khana|intezaar)",
    # English fallback slang
    "en": r"(bro|dude|wtf|ffs|lol|omg|pls|broo|yaar|late|hungry|cold|spicy|refund|delay|order)",
}

BAD_WORDS = [
    # mild+strong; the detector also checks for jokey context
    "bokka","boka","saale","kutte","bsdk","mc","bkc","bhenchod","madarchod","gandu",
    "fuck","shit","bloody","stupid","idiot","nonsense","bhosdi","harami","chutiya","ullu"
]

JOKE_MARKERS = ["lol","lmao","😂","🤣","😅","jk","just kidding","hehe","haha"]

# -----------------------------------------------------------------------------
# Utilities

def now_ms() -> float:
    return time.time()

def rule_boost_lang(text: str) -> Tuple[str, float]:
    """
    Heuristic boost for romanized Indian slang.
    Returns (lang, score in 0..1). 0 means no strong hint.
    """
    s = text.lower()
    best = ("", 0.0)
    for lang, pat in SLANG_HINTS.items():
        if re.search(rf"\b{pat}\b", s):
            # score: number of matches (cap at 1.0)
            n = len(re.findall(rf"\b{pat}\b", s))
            score = min(0.25 + 0.15 * n, 0.95)
            if score > best[1]:
                best = (lang, score)
    return best

def detect_language(text: str) -> str:
    text = text.strip()
    if not text:
        return "und"
    # First, rule boost on slang
    hint_lang, hint_score = rule_boost_lang(text)
    if hint_score >= 0.6:
        return hint_lang
    # Next, langid if available
    if _HAS_LANGID:
        lang, _ = langid.classify(text)
        # If rule suggested something and langid is uncertain, use hint
        return hint_lang if hint_lang and hint_score >= 0.4 else lang
    # Fallback basic guess
    if re.search(r"[అ-హ]", text):  # Telugu Unicode
        return "te"
    if re.search(r"[ािीुूेैोौँं]", text):  # Devanagari (Hindi etc.)
        return "hi"
    return hint_lang or "en"

def soft_sentiment(text: str) -> Tuple[int, str]:
    """
    Lightweight sentiment scorer: -1..+1 then mapped to (-100..+100, label).
    Looks at exclamation, negations, emoticons, some keywords.
    """
    s = text.lower()
    pos = ["great","good","tasty","awesome","nice","fast","love","super","best","thanks","thx","😍","😋","😊","👍"]
    neg = ["bad","cold","stale","late","delay","refund","poor","hate","worst","slow","raw","hair","fly","🤬","😠","😡","👎"]

    score = 0
    for w in pos: score += 1 if w in s else 0
    for w in neg: score -= 1 if w in s else 0
    score += s.count("!") * 0.2
    score = max(-4, min(4, score))
    scaled = int(score * 25)  # -100..100

    label = "Positive" if scaled > 20 else "Negative" if scaled < -20 else "Neutral"
    # Politeness / apology
    if re.search(r"\b(sorry|apolog(y|ise)|pls|please)\b", s):
        scaled = min(100, scaled + 10)
    return scaled, label

def detect_toxicity(text: str) -> Tuple[bool, str]:
    s = text.lower()
    bad = any(re.search(rf"\b{re.escape(w)}\b", s) for w in BAD_WORDS))
    joking = any(m in s for m in JOKE_MARKERS)
    if bad and joking:
        return (True, "Toxic but jokey context (low priority)")
    if bad:
        return (True, "Likely toxic / abusive")
    return (False, "No")

def mood_from_sentiment_and_speed(sent: int, cps: float) -> Tuple[str, float]:
    """
    Combine sentiment (-100..100) with characters-per-second typing speed
    to estimate arousal (0..100). Returns (dominant_emotion, arousal_0_100)
    """
    # Arousal from speed (cap at 12 cps)
    arousal = min(100.0, (cps / 12.0) * 100.0)
    # Valence from sentiment
    if sent > 25 and arousal < 40: dom = "calm 😊"
    elif sent > 25 and arousal >= 40: dom = "excited 😄"
    elif sent < -25 and arousal >= 40: dom = "angry 😡"
    elif sent < -25 and arousal < 40: dom = "disappointed 😞"
    else: dom = "neutral 😐"
    return dom, float(round(arousal, 1))

def plot_gauge(value: float, title: str) -> go.Figure:
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge={'axis': {'range': [0, 100]},
               'bar': {'thickness': 0.25},
               'steps': [
                   {'range': [0, 33], 'color': "#dfe7fd"},
                   {'range': [33, 66], 'color': "#c1d3fe"},
                   {'range': [66, 100], 'color': "#a5b4fc"},
               ]},
        title={'text': title},
        number={'suffix': " / 100"}
    )).update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))

def init_state():
    st.session_state.setdefault("latest", {})
    st.session_state.setdefault("log", [])
    st.session_state.setdefault("prev_len", 0)
    st.session_state.setdefault("prev_time", now_ms())
    st.session_state.setdefault("cps", 0.0)
    st.session_state.setdefault("text_box", "")
    st.session_state.setdefault("prev_txt", "")

init_state()

# Live textbox callback to update typing speed + rerun
def _on_text_change():
    txt = st.session_state.text_box
    t = now_ms()
    dt = max(0.05, (t - st.session_state.prev_time))
    dlen = max(0, len(txt) - st.session_state.prev_len)
    st.session_state.cps = round(dlen / dt, 2)
    st.session_state.prev_len = len(txt)
    st.session_state.prev_time = t
    st.session_state.prev_txt = txt
    st.rerun()  # modern API (replaces experimental_rerun)

# -----------------------------------------------------------------------------
# Layout
st.title("🛵 Delivery Mood — Hybrid NLP (Text + Voice)")

tab_dash, tab_text, tab_voice, tab_logs = st.tabs(["📊 Dashboard", "✍️ Text Review", "🎤 Voice Review", "🧾 Logs"])

# --------------------------- Text Review -------------------------------------
with tab_text:
    st.subheader("Review (auto-detect language while you type)")
    txt = st.text_area(
        "Type your message",
        key="text_box",
        height=80,
        placeholder="e.g., 'entra babu entha sepu wait cheyyali / bhai jaldi bhejo'",
        on_change=_on_text_change
    )

    # Analysis
    lang = detect_language(txt)
    sent_score, sent_label = soft_sentiment(txt)
    toxic_flag, toxic_label = detect_toxicity(txt)
    dom, arousal = mood_from_sentiment_and_speed(sent_score, st.session_state.cps)

    # Display strip
    st.progress(int(np.interp(sent_score, [-100, 0, 100], [0, 50, 100])), text=f"{sent_label} • {int(np.interp(sent_score, [-100, 0, 100], [0,50,100]))}/100")

    c1,c2,c3,c4 = st.columns([1,1,1,1])
    with c1: st.metric("Language", lang)
    with c2: st.metric("Text Sentiment", f"{sent_score} ({sent_label})")
    with c3: st.metric("Toxicity", "Yes" if toxic_flag else "No", help=toxic_label)
    with c4: st.metric("Typing Speed", f"{st.session_state.cps} cps")

    st.plotly_chart(plot_gauge(arousal, "Arousal (typing-speed meter)"), use_container_width=True)
    st.info(f"Dominant Emotion → **{dom}**")

    if st.button("Save as Latest Feedback → Dashboard"):
        item = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "text",
            "text": txt,
            "lang": lang,
            "sent": sent_score,
            "sent_label": sent_label,
            "toxic": toxic_flag,
            "emotion": dom,
            "cps": st.session_state.cps
        }
        st.session_state.latest = item
        st.session_state.log.insert(0, item)
        st.success("Saved to Dashboard ✓")

# --------------------------- Voice Review ------------------------------------
with tab_voice:
    st.subheader("Live Voice (upload is stable; mic is optional)")
    colu, colm = st.columns([1,1])

    with colu:
        audio_file = st.file_uploader("Upload voice (WAV/MP3/M4A)", type=["wav","mp3","m4a"])
        if audio_file:
            st.audio(audio_file)
            # Fake analysis: we don't do ASR here to stay dependency-light.
            # In your next iteration, call HF Whisper API and pass transcript to the same pipeline.
            fake_transcript = "auto: (demo) received voice; integrate Whisper for real transcript"
            lang_v = detect_language(fake_transcript)
            sent_v, lbl_v = soft_sentiment(fake_transcript)
            toxic_v, tox_lbl_v = detect_toxicity(fake_transcript)
            dom_v, arousal_v = mood_from_sentiment_and_speed(sent_v, 0.0)
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Voice Lang", lang_v)
            with c2: st.metric("Sentiment", f"{sent_v} ({lbl_v})")
            with c3: st.metric("Toxicity", "Yes" if toxic_v else "No", help=tox_lbl_v)
            st.plotly_chart(plot_gauge(arousal_v, "Arousal (voice)"), use_container_width=True)
            if st.button("Save Voice → Dashboard"):
                item = {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": "voice",
                    "text": fake_transcript,
                    "lang": lang_v,
                    "sent": sent_v,
                    "sent_label": lbl_v,
                    "toxic": toxic_v,
                    "emotion": dom_v,
                    "cps": 0.0
                }
                st.session_state.latest = item
                st.session_state.log.insert(0, item)
                st.success("Saved to Dashboard ✓")

    with colm:
        if _HAS_WEBRTC:
            st.caption("Mic (WebRTC) — experimental; works if browser/server allow mic")
            rtc_cfg = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            webrtc_streamer(
                key="voice",
                mode=WebRtcMode.SENDONLY,
                audio_receiver_size=256,
                media_stream_constraints={"audio": True, "video": False},
                rtc_configuration=rtc_cfg
            )
        else:
            st.warning("`streamlit-webrtc` not available on this server. Upload voice on the left (recommended).")

# --------------------------- Dashboard ---------------------------------------
with tab_dash:
    st.subheader("Delivery Dashboard (latest item)")
    latest = st.session_state.latest
    if latest:
        c1,c2,c3,c4 = st.columns([2,1,1,1])
        with c1:
            st.write(f"**When:** {latest['ts']}  |  **Mode:** {latest['mode']}  |  **Lang:** `{latest['lang']}`")
            st.write(f"**Message:** {latest['text'] or '(voice)'}")
        with c2: st.metric("Sentiment", f"{latest['sent']} ({latest['sent_label']})")
        with c3: st.metric("Toxic?", "Yes" if latest['toxic'] else "No")
        with c4: st.metric("Typing cps", f"{latest.get('cps',0)}")
        st.plotly_chart(plot_gauge(
            mood_from_sentiment_and_speed(latest['sent'], latest.get('cps',0))[1],
            "Arousal"), use_container_width=True)
        st.success(f"Dominant Emotion → **{latest['emotion']}**")
    else:
        st.info("No items yet. Add from **Text Review** or **Voice Review**.")

# --------------------------- Logs --------------------------------------------
with tab_logs:
    st.subheader("Recent items")
    if st.session_state.log:
        df = pd.DataFrame(st.session_state.log)
        st.dataframe(df, use_container_width=True, height=300)
    else:
        st.caption("No logs yet.")

# Footer tip
st.caption("Tip: the model uses rule-boosted language hints for romanized slang + a lightweight sentiment/toxicity scorer. Plug in Whisper/HF APIs later for ASR/emotions.")
