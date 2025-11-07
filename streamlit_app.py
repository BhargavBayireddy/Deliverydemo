# -*- coding: utf-8 -*-
# Hybrid NLP (HF API) — Text + Voice
# - Language detection: fastText lid.176 (works for romanized/“chat” text)
# - Voice: WebRTC -> Whisper (HF Inference API) -> transcript + language
# - Mood: lightweight lexicon/rules -> score (-100..+100) + gauge
# - UI: Tabs (Dashboard, Text Review, Voice Review, Logs)

import os, io, time, re, json, tempfile, requests, pathlib
import numpy as np
import streamlit as st

# Optional: mic capture (web)
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

# ---------- Settings ----------
HF_ASR_MODEL = "openai/whisper-small"     # HuggingFace Inference API
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODEL_DIR = pathlib.Path("./models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
FT_BIN = MODEL_DIR / "lid.176.bin"

st.set_page_config(page_title="Hybrid NLP (HF API) — Text + Voice", page_icon="🛵", layout="wide")

# ---------- Small utilities ----------
def human_lang_name(code: str) -> str:
    # Minimal, high-coverage mapping; unknown -> code
    m = {
        "en":"English","hi":"Hindi","te":"Telugu","ta":"Tamil","kn":"Kannada","ml":"Malayalam",
        "mr":"Marathi","bn":"Bengali","gu":"Gujarati","pa":"Punjabi","or":"Odia","ur":"Urdu",
        "ne":"Nepali","si":"Sinhala","as":"Assamese","sd":"Sindhi",
        "fr":"French","es":"Spanish","de":"German","it":"Italian","pt":"Portuguese","ru":"Russian",
        "ar":"Arabic","tr":"Turkish","id":"Indonesian","ms":"Malay","vi":"Vietnamese",
        "zh":"Chinese","ja":"Japanese","ko":"Korean","fa":"Persian","ps":"Pashto"
    }
    return m.get(code, code or "unknown")

@st.cache_resource(show_spinner=False)
def load_fasttext():
    # Try import fasttext; if build fails (some hosts), fall back to fasttext-wheel package name
    try:
        import fasttext
    except Exception:
        import importlib
        fasttext = importlib.import_module("fasttext")
    # Ensure model file
    if not FT_BIN.exists():
        try:
            r = requests.get(FASTTEXT_URL, timeout=60)
            r.raise_for_status()
            FT_BIN.write_bytes(r.content)
        except Exception as e:
            st.error("Failed to download fastText lid.176 model.")
            raise e
    model = fasttext.load_model(str(FT_BIN))
    return model

FT = load_fasttext()

# Romanized/“chat” keyword boosts (very small lists; extend as needed)
ROMA_HINTS = {
    "te": {"entha","sepu","cheyyali","inka","ayyindi","ivvandi","babu","anna","ra","pls","fast ga"},
    "hi": {"kya","kaise","kyu","bahut","mast","jaldi","yar","yaar","nahi","krdo","jaldi se"},
    "ta": {"enna","epdi","venum","seri","romba","sapadu","sapten","da","machan","machaa"},
    "kn": {"yenu","swalpa","tumba","ondhu","hegidde","banni"},
    "ml": {"entha","cheyyu","vannu","kitti","alle"},
}

def detect_lang_fasttext(text: str):
    """Return (lang_code, prob, name). Works for romanized text; adds keyword boosts."""
    clean = (text or "").strip()
    if not clean:
        return "unknown", 0.0, "unknown"
    labels, probs = FT.predict(clean.replace("\n"," "), k=1)
    code = labels[0].replace("__label__", "")
    prob = float(probs[0])

    # Boost for romanized Indic when fastText is unsure (<0.75)
    if prob < 0.75:
        low = clean.lower()
        for iso, hints in ROMA_HINTS.items():
            if any(h in low for h in hints):
                # soft override if the winner isn't already the same language
                code = iso
                prob = max(prob, 0.85)
                break

    return code, prob, human_lang_name(code)

# --------- Mood (rule-based) ----------
NEG = {
    "late","delay","waiting","slow","bad","worst","rude","cold","soggy","stale","refund","cancel",
    "bekar","bakwas","chindi","waste","problem","issue","hate","angry","frustrated","disappointed",
    # romanized
    "entha","sepu","cheyyali","inka","ayyindi","bahut bura","bahut late","romba late","venum asap"
}
POS = {"good","great","awesome","amazing","excellent","tasty","fresh","fast","quick","polite","love",
       "perfect","nice","yummy","delicious","super","mast","semma","satisfying","hot","crispy","clean"}

def sentiment_score(text: str) -> tuple[int,str,str]:
    t = (text or "").strip()
    if not t:
        return 0,"Neutral","😐"
    low = t.lower()
    ex = t.count("!")
    caps = 1.15 if re.search(r"[A-Z]{3,}", t) else 1.0

    score = 0
    # simple token-wise scoring
    for w in re.findall(r"[a-zA-Z]+", low):
        if w in POS: score += 6
        if w in NEG: score -= 8
    # emojis
    if any(x in t for x in ["😡","🤬"]): score -= 20
    if any(x in t for x in ["😊","🙂","😄","😍","🤩","👍","👏"]): score += 16

    score = int(np.clip((score* caps + ex*2), -100, 100))
    if score <= -60: mood, emoji = "Angry", "😡"
    elif score <= -30: mood, emoji = "Frustrated", "😠"
    elif score <= -6: mood, emoji = "Disappointed", "😕"
    elif score <= 5: mood, emoji = "Neutral", "😐"
    elif score <= 40: mood, emoji = "Satisfied", "🙂"
    else: mood, emoji = "Delighted", "🤩"
    return score, mood, emoji

def gauge_bar(score: int, label: str):
    # score -100..100 -> 0..100 bar
    val = int(np.interp(score, [-100,100],[0,100]))
    st.progress(val, text=f"{label} • {val}/100")

# -------- Whisper (HF Inference API) ----------
def hf_headers():
    tok = os.getenv("HF_API_TOKEN", "")
    headers = {"Accept": "application/json"}
    if tok: headers["Authorization"] = f"Bearer {tok}"
    return headers

def whisper_transcribe(wav_bytes: bytes):
    """
    Send audio to HF Whisper. Returns dict {text, language?, raw} or raises.
    """
    url = f"https://api-inference.huggingface.co/models/{HF_ASR_MODEL}"
    r = requests.post(url, headers=hf_headers(), data=wav_bytes, timeout=60)
    r.raise_for_status()
    js = r.json()
    # HF whisper returns single dict or a list (space for different repos); normalize
    if isinstance(js, list) and js and isinstance(js[0], dict):
        js = js[0]
    text = js.get("text") or ""
    # Some servers include "language" key (ISO)
    lang = js.get("language")
    return {"text": text, "language": lang, "raw": js}

def pcm16_to_wav(pcm16: bytes, sample_rate: int = 16000) -> bytes:
    import wave, struct
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return buf.getvalue()

# ----------- UI / State -----------
if "logs" not in st.session_state:
    st.session_state.logs = []

st.title("🧪 Hybrid NLP (HF API) — Text + Voice")
st.caption("Auto detects language, sentiment & emotion from text and live voice. Uses fastText (176-langs) and Whisper (HF API).")

if not os.getenv("HF_API_TOKEN"):
    st.warning("⚠️ Set **HF_API_TOKEN** in your environment / Streamlit Secrets to enable Whisper ASR.", icon="⚠️")

tabs = st.tabs(["📊 Dashboard","✍️ Text Review","🎙️ Voice Review","🧾 Logs"])

# ------------- Dashboard -------------
with tabs[0]:
    st.subheader("Recent Items")
    if not st.session_state.logs:
        st.info("No events yet. Use *Text Review* or *Voice Review* to add entries.")
    else:
        for item in reversed(st.session_state.logs[-12:]):
            with st.container(border=True):
                st.markdown(f"**Time:** {item['time']}  •  **Source:** {item['src']}")
                st.markdown(f"**Language:** {human_lang_name(item['lang'])}  ({item['lang_conf']:.2f})")
                st.markdown(f"**Text:** {item['text']}")
                gauge_bar(item["score"], f"{item['emoji']} {item['mood']}")
                st.caption(json.dumps(item.get("extra", {}), ensure_ascii=False))

# ------------- Text Review -------------
with tabs[1]:
    st.subheader("Review (auto detects language while you type)")
    txt = st.text_input("Type your message", value="", placeholder="e.g., 'entra babu entha sepu wait cheyyali'")
    lang, conf, lname = "unknown", 0.0, "unknown"
    if txt:
        code, prob, lname = detect_lang_fasttext(txt)
        lang, conf = code, prob
        score, mood, emoji = sentiment_score(txt)
        gauge_bar(score, f"{emoji} {mood}")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Language", f"{lname}")
        with col2: st.metric("Text Sentiment", f"{score} ({emoji} {mood})")
        with col3:
            tox = "Yes" if any(w in txt.lower() for w in ["abuse","idiot","stupid","bhosd","madarch"]) else "No"
            st.metric("Toxicity", tox)

        if st.button("Save as Latest Feedback → Dashboard", type="primary"):
            st.session_state.logs.append({
                "time": time.strftime("%H:%M:%S"),
                "src": "text",
                "text": txt,
                "lang": lang,
                "lang_conf": conf,
                "score": score,
                "mood": mood,
                "emoji": emoji,
                "extra": {"detected_language": lname}
            })
            st.success("Saved to dashboard.")

# ------------- Voice Review -------------
with tabs[2]:
    st.subheader("Live Voice (auto language via Whisper)")
    st.caption("Click **Start**, allow microphone, speak for 5–8 seconds. Processing runs automatically.")
    if not HAS_WEBRTC:
        st.error("Mic capture not available in this environment. Try locally or on HTTPS.")
    else:
        # Simple RECVONLY mic; we just read a few frames, then process when user clicks 'Transcribe'
        ctx = webrtc_streamer(
            key="voice-demo",
            mode=WebRtcMode.RECVONLY,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True},
        )
        if "voice_pcm" not in st.session_state:
            st.session_state.voice_pcm = b""

        def pull_audio_frames(max_ms=6000):
            """Read ~N ms from receiver; return concatenated PCM16 bytes."""
            if not ctx or not ctx.state.playing or not ctx.audio_receiver:
                return b""
            pcm = []
            ms_total = 0
            start = time.time()
            while ms_total < max_ms and (time.time()-start) < 7.5:
                try:
                    frame = ctx.audio_receiver.get_frame(timeout=1.0)
                except Exception:
                    frame = None
                if frame is None:
                    continue
                if isinstance(frame, av.AudioFrame):
                    # Convert to mono 16k PCM16
                    f = frame.to_ndarray(format="s16", layout="mono")
                    pcm.append(f.tobytes())
                    ms_total += int(1000 * frame.samples / frame.sample_rate)
            return b"".join(pcm)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🎤 Capture 6s"):
                st.session_state.voice_pcm = pull_audio_frames(max_ms=6000)
                if st.session_state.voice_pcm:
                    st.success("Captured ~6s voice. Click **Transcribe & Analyze**.")
                else:
                    st.warning("No audio captured. Make sure mic permission is allowed.")
        with c2:
            if st.button("🧠 Transcribe & Analyze", type="primary"):
                if not st.session_state.voice_pcm:
                    st.error("No captured audio. Click 'Capture 6s' first.")
                else:
                    wav = pcm16_to_wav(st.session_state.voice_pcm, 16000)
                    try:
                        out = whisper_transcribe(wav)
                        transcript = out.get("text", "").strip()
                        wh_lang = out.get("language")
                        # Language: use Whisper's language if present else fastText on transcript
                        if wh_lang:
                            lang_code = wh_lang
                            conf = 0.90
                        else:
                            lang_code, conf, _ = detect_lang_fasttext(transcript)
                        # Mood on transcript
                        score, mood, emoji = sentiment_score(transcript)
                        gauge_bar(score, f"{emoji} {mood}")

                        st.markdown(f"**Transcript:** {transcript or '(empty)'}")
                        st.markdown(f"**Language:** {human_lang_name(lang_code)}  (conf ~ {conf:.2f})")

                        # Save to dashboard
                        st.session_state.logs.append({
                            "time": time.strftime("%H:%M:%S"),
                            "src": "voice",
                            "text": transcript,
                            "lang": lang_code,
                            "lang_conf": conf,
                            "score": score,
                            "mood": mood,
                            "emoji": emoji,
                            "extra": {"whisper": out.get("raw", {})}
                        })
                        st.success("Saved to dashboard.")
                    except Exception as e:
                        st.exception(e)

# ------------- Logs -------------
with tabs[3]:
    st.subheader("Raw Log (latest first)")
    if not st.session_state.logs:
        st.info("No logs yet.")
    else:
        for item in reversed(st.session_state.logs[-20:]):
            st.code(json.dumps(item, ensure_ascii=False, indent=2))
