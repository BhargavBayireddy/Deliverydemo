# -*- coding: utf-8 -*-
# 🛵 Delivery ETA + Mood (text + reliable live-voice prosody)
# - Keeps: review box, ETA map, Delivery Boy Dashboard
# - Fix: live mic via streamlit-webrtc AudioProcessor (stable)
# - Adds: Arousal–Valence meter + Stress Index, spectrum + waveform live
# - Upload fallback (WAV; MP3/M4A if PyAV+FFmpeg present)

import math, time, random, re, requests, io, collections
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import pydeck as pdk
import matplotlib.pyplot as plt

from scipy.signal import get_window, lfilter
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile

# -------- Optional codecs for uploads --------
try:
    import av
    HAS_AV = True
except Exception:
    av = None
    HAS_AV = False

# -------- WebRTC (live mic) --------
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

# ---------------- UI text ----------------
LANGS = {
  "en":{"title":"Delivery ETA + Mood (Demo)","headsup":"Heads up: your order may be late by ~5 minutes. Sorry!",
        "gps_btn":"📍 Use my GPS","late_by":lambda m:f"Your order may be ~{m} min late.",
        "review":"Write a review","submit":"Submit Review & Notify Rider",
        "dash":"Delivery Boy Dashboard","recalc":"Recalculate ETA",
        "angry_tip":"Customer is angry — take the shortest route, call & apologize.",
        "coupon":"If faster route isn’t possible, offer coupon: SORRY10",
        "typing_fast":"Typing fast (angry) 🚨",
        "manual":"Or enter location manually if GPS fails",
        "voice":"Voice (Prosody)",
        "record_btn":"🎙️ Start / Allow Mic",
        "upload_lbl":"…or upload a voice note (" + ("WAV/MP3/M4A" if HAS_AV else "WAV only") + ")",
        "analyze":"Analyze Uploaded Audio",
        "howto":"Speak 5–10s. We analyze pitch, energy, jitter, shimmer, HNR, CPP, centroid, pauses & rate.",
        "calib":"Calibrate (first 5s as your baseline)",
        "calib_done":"✅ Baseline saved and thresholds adapted to your voice."
        },
  "hi":{"title":"डिलीवरी ETA + मूड (डेमो)","headsup":"ध्यान दें: आपका ऑर्डर ~5 मिनट लेट हो सकता है। क्षमा करें!",
        "gps_btn":"📍 मेरा GPS लो","late_by":lambda m:f"आपका ऑर्डर ~{m} मिनट लेट हो सकता है।",
        "review":"रिव्यू लिखें","submit":"रिव्यू भेजें व राइडर को बताएँ",
        "dash":"डिलीवरी बॉय डैशबोर्ड","recalc":"ETA दोबारा निकालें",
        "angry_tip":"कस्टमर गुस्से में है — शॉर्टकट लें, कॉल करें और माफ़ी माँगें।",
        "coupon":"तेज़ रूट संभव नहीं तो कूपन दें: SORRY10",
        "typing_fast":"गुस्से में तेज़ टाइपिंग 🚨",
        "manual":"GPS काम न करे तो नीचे मैनुअल भरें",
        "voice":"वॉइस (प्रोसोडी)",
        "record_btn":"🎙️ माइक चालू करें",
        "upload_lbl":"…या वॉइस नोट अपलोड करें (" + ("WAV/MP3/M4A" if HAS_AV else "केवल WAV") + ")",
        "analyze":"वॉइस विश्लेषण",
        "howto":"5–10 सेकंड बोलें — हम पिच, ऊर्जा, जिटर, शिमर, HNR, CPP, सेंट्रॉइड, पॉज़ व रेट देखते हैं।",
        "calib":"कैलिब्रेट (पहले 5 सेकंड बेसलाइन)",
        "calib_done":"✅ बेसलाइन सेव — थ्रेशोल्ड आपकी आवाज़ के अनुसार।"
        },
  "te":{"title":"డెలివరీ ETA + మూడ్ (డెమో)","headsup":"మీ ఆర్డర్ ~5 నిమిషాలు ఆలస్యం కావచ్చు. సారీ!",
        "gps_btn":"📍 నా GPS తీసుకో","late_by":lambda m:f"మీ ఆర్డర్ సుమారు {m} ని ఆలస్యమవుతుంది.",
        "review":"రివ్యూ రాయండి","submit":"రివ్యూ పంపి రైడర్‌కు తెలపండి",
        "dash":"డెలివరీ బాయ్ డాష్‌బోర్డ్","recalc":"ETA మళ్లీ లెక్కించు",
        "angry_tip":"కస్టమర్ కోపంగా ఉన్నాడు — షార్ట్‌కట్ తీసుకుని ఫోన్ చేసి క్షమాపణ చెప్పండి.",
        "coupon":"త్వరగా వీలు లేకపోతే కూపన్ ఇవ్వండి: SORRY10",
        "typing_fast":"ఫాస్ట్ టైపింగ్ (కోపం) 🚨",
        "manual":"GPS రాకపోతే క్రింద మాన్యువల్ ఎంటర్ చేయండి",
        "voice":"వాయిస్ (ప్రోసొడీ)",
        "record_btn":"🎙️ మైక్ స్టార్ట్",
        "upload_lbl":"…లేదా వాయిస్ నోట్ అప్‌లోడ్ (" + ("WAV/MP3/M4A" if HAS_AV else "WAV మాత్రమే") + ")",
        "analyze":"వాయిస్ విశ్లేషణ",
        "howto":"5–10 సెకన్‌లు మాట్లాడండి — పిచ్, ఎనర్జీ, జిట్టర్/షిమ్మర్, HNR/CPP, సెంట్రాయిడ్, పాజ్/రేట్ చూస్తాం.",
        "calib":"క్యాలిబ్రేట్ (మొదటి 5సె బేస్‌లైన్)",
        "calib_done":"✅ బేస్‌లైన్ సేవ్ — మీ వాయిస్‌కి తగిన థ్రెషోల్డ్‌లు."
        },
  "ta":{"title":"டெலிவரி ETA + மனநிலை (டெமோ)","headsup":"உங்கள் ஆர்டர் ~5 நிமிடம் தாமதமாகலாம். மன்னிக்கவும்!",
        "gps_btn":"📍 என் GPS எடு","late_by":lambda m:f"உங்கள் ஆர்டர் சுமார் {m} நிமிடம் தாமதம்.",
        "review":"ரிவியூ எழுதவும்","submit":"ரிவியூ அனுப்பு & ரைடருக்கு தெரிவி",
        "dash":"டெலிவரி பாய் டாஷ்‌போர்டு","recalc":"ETA மீண்டும் கணக்கிடு",
        "angry_tip":"வாடிக்கையாளர் கோபம் — ஷார்ட்கட் எடுத்து மன்னிப்பு கேளுங்கள்.",
        "coupon":"வேகமான பாதை இல்லை என்றால் கூப்பன்: SORRY10",
        "typing_fast":"கோப டைப்பிங் 🚨",
        "manual":"GPS வேலை செய்யாவிட்டால் கீழே மானுவல் உள்ளிடவும்",
        "voice":"வாய்ஸ் (ப்ரொசோடி)",
        "record_btn":"🎙️ மைக் ஸ்டார்ட்",
        "upload_lbl":"…அல்லது வோய்ஸ் நோட் (" + ("WAV/MP3/M4A" if HAS_AV else "WAV மட்டும்") + ")",
        "analyze":"வாய்ஸ் பகுப்பாய்வு",
        "howto":"5–10 விநாடி பேசுங்கள் — Pitch/Energy/Jitter/Shimmer/HNR/CPP/Pauses/Rate.",
        "calib":"அடிப்படை ஒலி (5s) சேமிக்க",
        "calib_done":"✅ உங்கள் குரலுக்கு ஏற்ப அளவுகள் அமைக்கப்பட்டன."
        },
}

# ---------------- Helper: geodesy & ETA ----------------
def haversine_km(a_lat,a_lng,b_lat,b_lng):
    R=6371.0
    dLat=math.radians(b_lat-a_lat); dLon=math.radians(b_lng-a_lng)
    la=math.radians(a_lat); lb=math.radians(b_lat)
    h=math.sin(dLat/2)**2+math.cos(la)*math.cos(lb)*math.sin(dLon/2)**2
    return 2*R*math.asin(math.sqrt(h))

def random_nearby(lat,lng,km=3.0):
    d=km/111.0
    bear=random.random()*2*math.pi
    return lat + d*math.cos(bear), lng + d*math.sin(bear)/math.cos(math.radians(lat))

def openmeteo_now(lat,lng):
    try:
        u=f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current=precipitation,weather_code,wind_speed_10m"
        r=requests.get(u,timeout=8); r.raise_for_status()
        cur=r.json().get("current",{})
        return float(cur.get("precipitation",0)), int(cur.get("weather_code",0))
    except Exception:
        return 0.0, 0

def overpass_count(query):
    try:
        r=requests.post("https://overpass-api.de/api/interpreter", data={"data":query}, timeout=12)
        r.raise_for_status(); js=r.json()
        return len(js.get("elements",[]))
    except Exception:
        return 0

def bbox_around_line(a_lat,a_lng,b_lat,b_lng,km_buffer=0.35):
    lat_min=min(a_lat,b_lat)-km_buffer/111.0
    lat_max=max(a_lat,b_lat)+km_buffer/111.0
    avg=(a_lat+b_lat)/2.0
    lng_buf=km_buffer/(111.0*math.cos(math.radians(avg)))
    lng_min=min(a_lng,b_lng)-lng_buf
    lng_max=max(a_lng,b_lng)+lng_buf
    return lat_min,lng_min,lat_max,lng_max

def osm_counts_along_route(a_lat,a_lng,b_lat,b_lng):
    s,w,n,e=bbox_around_line(a_lat,a_lng,b_lat,b_lng,km_buffer=0.4)
    q_sig=f'[out:json][timeout:20];node["highway"="traffic_signals"]({s},{w},{n},{e});out;'
    q_work=f'[out:json][timeout:20];(node["highway"="construction"]({s},{w},{n},{e});way["highway"="construction"]({s},{w},{n},{e}););out;'
    return overpass_count(q_sig), overpass_count(q_work)

def peak_multiplier():
    hr=datetime.now().hour
    if 8<=hr<=11 or 17<=hr<=21: return 1.2
    if 23<=hr or hr<=5: return 0.95
    return 1.05

def estimate_eta(o_lat,o_lng,d_lat,d_lng):
    km=haversine_km(o_lat,o_lng,d_lat,d_lng)
    base_speed=26.0
    base=(km/base_speed)*60.0
    precip,wcode=openmeteo_now(d_lat,d_lng)
    try:
        sigs,works=osm_counts_along_route(o_lat,o_lng,d_lat,d_lng)
    except Exception:
        sigs,works=0,0
    peak=peak_multiplier()
    rain_add=5.0 if (precip>=2.0 or wcode in (51,53,55,61,63,65,80,81,82)) else 0.0
    signals_add=min(sigs, int(km*3+3))*0.67
    road_add=min(works,3)*2.5
    eta=base*peak + signals_add + rain_add + road_add
    reasons=[("Base (distance & rider speed)", round(base,1)),
             ("Peak-hour traffic", round(base*(peak-1),1))]
    if signals_add>0: reasons.append(("Traffic signals", round(signals_add,1)))
    if road_add>0: reasons.append(("Roadworks", round(road_add,1)))
    if rain_add>0: reasons.append(("Weather impact", round(rain_add,1)))
    return max(1.0, round(eta,1)), km, sigs, works, precip, reasons

# ---------------- Text sentiment (kept from your code) ----------------
INTENS={"very":1.3,"really":1.2,"so":1.2,"too":1.15,"extremely":1.4,"super":1.3,"bahut":1.2,"chala":1.2}
NEGS=set(["not","no","never","hardly","barely","scarcely","isnt","wasnt","arent","dont","didnt","cant","wont","nahi","mat","kadu","illa"])
POSw=set("good great awesome amazing excellent tasty fresh fast quick ontime polite friendly love liked perfect nice yummy delicious wow clean crisp juicy recommend thanks superb mast semma mass superr".split())
NEGw=set("bad terrible awful worst cold soggy late delay delayed dirty slow rude stale raw burnt bland overpriced expensive missing leak leaking spill spilled refund replace cancel canceled cancelled angry frustrated annoyed disappointed hate horrible issue problem broken uncooked inedible vomit sick hair bekar faltu bakwas worsttt waste dappa sappa mosam chindi pathetic useless trash".split())

def sentiment_score(text):
    t=text.strip()
    if not t: return 0,"Neutral","😐",[]
    s=re.sub(r"[^\w\s!?]", " ", t.lower())
    words=s.split()
    score=0.0; hits=[]
    exclam=t.count("!"); caps=1.2 if re.search(r"[A-Z]{3,}", t) else 1.0
    long=1.2 if re.search(r"(.)\1{2,}", t.lower()) else 1.0
    if any(x in t for x in ["😡","🤬"]): score -= 12
    if any(x in t for x in ["😊","🙂","😄","😍","🤩","👍","👏"]): score += 6
    score *= caps*long
    if exclam>=2: score*=1.15
    for i,w in enumerate(words):
        base=(2.5 if w in POSw else 0)+(-2.5 if w in NEGw else 0)
        if base!=0:
            if i>0 and words[i-1] in INTENS: base*=INTENS[words[i-1]]
            for k in range(1,4):
                if i-k>=0 and words[i-k] in NEGS: base*=-1; break
            score+=base; hits.append(w)
    raw=max(-40,min(40,score)); scaled=int(round((raw/40)*100))
    if scaled<=-60: mood,emoji="Angry","😡"
    elif scaled<=-30: mood,emoji="Frustrated","😠"
    elif scaled<=-6: mood,emoji="Disappointed","😕"
    elif -5<=scaled<=5: mood,emoji="Neutral","😐"
    elif scaled<=35: mood,emoji="Satisfied","🙂"
    else: mood,emoji="Delighted","🤩"
    low=t.lower()
    if any(k in low for k in ["refund","chargeback","return"]): mood,emoji="Frustrated","😠"
    if any(k in low for k in ["cancel","canceled","cancelled"]): mood,emoji="Angry","😡"
    return scaled,mood,emoji,list(dict.fromkeys(hits))

# ---------------- Prosody (voice) ----------------
SAMPLE_RATE = 16000
BUF_SECONDS = 6.0
BUF_SAMPLES = int(SAMPLE_RATE*BUF_SECONDS)

def to_mono_float(audio_bytes):
    # Try WAV first
    try:
        fs, data = wavfile.read(io.BytesIO(audio_bytes))
        arr = data.astype(np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        if np.max(np.abs(arr))>0:
            arr = arr/np.max(np.abs(arr))
        return fs, arr
    except Exception:
        pass
    # PyAV fallback for mp3/m4a if available
    if not HAS_AV:
        raise RuntimeError("Only WAV supported here (PyAV/FFmpeg not installed).")
    with av.open(io.BytesIO(audio_bytes)) as container:
        stream = next(s for s in container.streams if s.type=="audio")
        fs = stream.rate
        frames=[]
        for frame in container.decode(stream):
            ch = frame.to_ndarray().astype(np.float32)
            if ch.ndim==2: ch = ch.mean(axis=0)
            frames.append(ch/32768.0)
        arr = np.concatenate(frames) if frames else np.zeros(1,dtype=np.float32)
        if fs!=SAMPLE_RATE:
            step = max(1, int(fs/SAMPLE_RATE))
            arr = arr[::step]
            fs = int(fs/step)
        if np.max(np.abs(arr))>0:
            arr=arr/np.max(np.abs(arr))
        return fs, arr

def frame_signal(x, fs, frame_ms=40, hop_ms=10):
    N = int(fs*frame_ms/1000); H = int(fs*hop_ms/1000)
    if N<256: N,H = 640,160
    w = get_window("hann", N)
    frames=[]
    for i in range(0, len(x)-N+1, H):
        frames.append(x[i:i+N]*w)
    return np.array(frames), N, H

def autocorr_pitch(fr, fs, fmin=75, fmax=400):
    f = fr - np.mean(fr)
    if np.allclose(f,0): return 0.0, -np.inf
    r = np.correlate(f,f,mode="full"); r = r[len(r)//2:]
    r = r/(r[0]+1e-9)
    lag_min=int(fs/fmax); lag_max=min(int(fs/fmin), len(r)-1)
    if lag_max<=lag_min: return 0.0, -np.inf
    idx = np.argmax(r[lag_min:lag_max]) + lag_min
    peak = np.clip(r[idx], 1e-6, 0.999999)
    f0 = fs/idx if idx>0 else 0.0
    hnr = 10*np.log10(peak/(1-peak))
    return f0, hnr

def cepstral_peak_prominence(fr, fs):
    X = np.abs(rfft(fr)) + 1e-9
    c = np.real(np.fft.irfft(np.log(X)))
    q = np.arange(len(c))/fs
    imin=int(fs/400); imax=min(int(fs/75), len(c)-1)
    roi=c[imin:imax]
    return float(np.max(roi)-np.median(roi))

def spectral_centroid(fr, fs):
    X = np.abs(rfft(fr))+1e-12
    F = rfftfreq(len(fr), 1.0/fs)
    return float(np.sum(F*X)/np.sum(X))

def extract_prosody(x, fs=SAMPLE_RATE):
    if len(x)<fs//2: return {}
    # pre-emphasis
    x = lfilter([1,-0.97],[1], x).astype(np.float32)
    frames, N, H = frame_signal(x, fs)
    f0s, hnrs, ens, cents, cpps, voiced = [], [], [], [], [], []
    for fr in frames:
        f0, hnr = autocorr_pitch(fr, fs)
        e = float(np.sqrt(np.mean(fr**2)+1e-12))
        cent = spectral_centroid(fr, fs)
        cpp = cepstral_peak_prominence(fr, fs)
        zcr = np.mean(np.sign(fr)[:-1]*np.sign(fr)[1:]<0)
        v = (e>0.01) and (zcr<0.18)
        f0s.append(f0); hnrs.append(hnr); ens.append(e); cents.append(cent); cpps.append(cpp); voiced.append(v)
    f0s=np.array(f0s); hnrs=np.array(hnrs); ens=np.array(ens); cents=np.array(cents)
    cpps=np.array(cpps); voiced=np.array(voiced)
    vf = voiced & (f0s>0)
    f0_v = f0s[vf]; en_v = ens[voiced]
    mean_f0=float(np.mean(f0_v)) if f0_v.size else 0.0
    std_f0=float(np.std(f0_v)) if f0_v.size else 0.0
    range_f0=float(np.max(f0_v)-np.min(f0_v)) if f0_v.size else 0.0
    mean_hnr=float(np.mean(hnrs[vf])) if np.any(vf) else -np.inf
    mean_cpp=float(np.mean(cpps[vf])) if np.any(vf) else 0.0
    mean_int=float(np.mean(en_v)) if en_v.size else float(np.mean(ens))
    std_int=float(np.std(en_v)) if en_v.size else float(np.std(ens))
    mean_cent=float(np.mean(cents[voiced])) if np.any(voiced) else float(np.mean(cents))
    hop_sec=0.01; total=len(frames)*hop_sec
    pause=float(np.sum(~voiced)*hop_sec)
    onsets=np.sum((voiced[1:] & ~voiced[:-1]))
    rate=onsets/(total+1e-9)
    pause_segs=np.sum((~voiced[1:] & voiced[:-1]))
    avg_pause=(pause/pause_segs) if pause_segs>0 else 0.0
    return dict(mean_f0=mean_f0,std_f0=std_f0,range_f0=range_f0,mean_int=mean_int,std_int=std_int,
                mean_hnr=mean_hnr,mean_cpp=mean_cpp,centroid=mean_cent,speaking_rate=rate,
                total_dur=total,pause_dur=pause,pause_segments=int(pause_segs),avg_pause=avg_pause)

def decide_voice_mood(F, baseline=None):
    if not F: return "Neutral","🙂",["too little audio"], 0.0, (0.5,0.5)
    # adapt thresholds with baseline if available
    adj = lambda k, a,b: (F[k] - (baseline.get(k,0.0) if baseline else 0.0))
    f0 = F["mean_f0"]; f0sd=F["std_f0"]; rng=F["range_f0"]
    inten=F["mean_int"]; intsd=F["std_int"]; hnr=F["mean_hnr"]; cpp=F["mean_cpp"]
    cent=F["centroid"]; rate=F["speaking_rate"]; pauses=F["pause_segments"]; pr=F["pause_dur"]/(F["total_dur"]+1e-9)

    # demo thresholds (mildly adaptive on baseline mean_f0/intensity)
    base_f0 = baseline.get("mean_f0",0) if baseline else 0
    base_int = baseline.get("mean_int",0) if baseline else 0
    high_f0 = f0 > (190 if base_f0==0 else max(150, base_f0+40))
    low_f0  = 0<f0< (120 if base_f0==0 else max(80, base_f0-30))
    wide = rng>80; flat=f0sd<15
    high_e = inten> (0.06 if base_int==0 else base_int*1.6)
    low_e  = inten< (0.02 if base_int==0 else max(0.01, base_int*0.6))
    dyn_e  = intsd>0.02
    fast = rate>2.2; slow=rate<1.0
    many_pauses = (pauses>=4) or (pr>0.45)
    strong_cpp = cpp>0.12; weak_cpp=cpp<0.06
    high_cent = cent>2500; low_cent=cent<1600

    # Arousal & Valence (continuous 0..1)
    arousal = np.clip(0.5*( (inten*20) + (rate/3.0) + ((cent-1200)/2000) ), 0, 1)
    valence = np.clip(0.5*( (cpp*5) + (hnr/15.0) - pr ), 0, 1)

    # Stress Index 0..100 (more pauses lowers it a bit)
    stress = np.clip(60*(arousal) + 20*(1-valence) + 20*(1 if (weak_cpp or hnr<8) else 0), 0, 100)

    if (high_f0 or wide) and (high_e or dyn_e) and (fast or not many_pauses) and (high_cent or hnr<8):
        return "Angry/Stress","😠",["high pitch/variation","high energy","fast/pressed"], stress, (arousal,valence)
    if (low_f0 or flat) and (low_e or not dyn_e) and (slow or many_pauses) and (weak_cpp or low_cent):
        return "Sad","😔",["low/flat pitch","low energy","slow & many pauses"], stress, (arousal,valence)
    if (not low_f0) and (dyn_e or high_e) and (strong_cpp or hnr>=10) and (rate>=1.2) and (not many_pauses):
        return "Joy","😄",["lively energy","healthy pitch","good CPP/HNR"], stress, (arousal,valence)
    return "Neutral","🙂",["balanced features"], stress, (arousal,valence)

# ---------------- Streamlit state ----------------
st.set_page_config(page_title="Delivery ETA Demo (Text + Voice)", page_icon="🛵", layout="wide")
lang_key = st.selectbox("Language / भाषा / భాష / மொழி", list(LANGS.keys()), index=0)
T = LANGS[lang_key]
st.title(f"🛵 {T['title']}")
st.info(T["headsup"])

if "orders" not in st.session_state: st.session_state.orders=[]
if "meta" not in st.session_state: st.session_state.meta={"len":0,"ts":time.time(),"cps":0.0}
if "voice_buf" not in st.session_state: st.session_state.voice_buf = collections.deque(maxlen=BUF_SAMPLES)
if "baseline" not in st.session_state: st.session_state.baseline = None

# GPS via query params
q=st.experimental_get_query_params()
gps_lat=float(q.get("lat",[0])[0]) if "lat" in q else None
gps_lng=float(q.get("lon",[0])[0]) if "lon" in q else None

def gps_button(label):
    components.html(f"""
    <button style="padding:10px 14px;border:none;border-radius:8px;background:#0ea5e9;color:#fff;font-weight:600"
     onclick="(function(){{
       if(!navigator.geolocation){{alert('No geolocation');return;}}
       navigator.geolocation.getCurrentPosition(function(p){{
         const u=new URL(window.location.href);
         u.searchParams.set('lat', p.coords.latitude.toFixed(6));
         u.searchParams.set('lon', p.coords.longitude.toFixed(6));
         window.location.replace(u.toString());
       }}, function(){{alert('Location blocked');}}, {{enableHighAccuracy:true,timeout:10000}});
     }})()"> {label} </button>
    """, height=60)

PLATS=["Zomato","Swiggy","Rapido","Uber","Ola"]
ITEMS=["biryani","pizza","paneer roll","idli-dosa","veg pulao","burger","shawarma","chai & samosa"]
intro_platform=random.choice(PLATS); intro_item=random.choice(ITEMS)
st.caption(f"You ordered **{intro_item}** on **{intro_platform}**.")

left,right=st.columns([1.25,1])

# ---------------- Left: Auto ETA + Map ----------------
with left:
    st.subheader("Auto ETA")
    gps_button(T["gps_btn"])
    st.caption(T["manual"])
    c1,c2=st.columns(2)
    with c1:
        user_lat=st.number_input("Your Lat", value=float(gps_lat) if gps_lat else 17.3850, format="%.6f")
    with c2:
        user_lng=st.number_input("Your Lng", value=float(gps_lng) if gps_lng else 78.4867, format="%.6f")

    rider_lat,rider_lng = random_nearby(user_lat,user_lng, km=3.0)

    if st.button(T["recalc"], type="primary"):
        eta, km, sigs, works, precip, reasons = estimate_eta(rider_lat,rider_lng,user_lat,user_lng)
        st.success(T["late_by"](eta))
        cols=st.columns(max(1,len(reasons)))
        reason_text=[]
        for c,(label,mins) in zip(cols,reasons):
            with c: st.metric(label, f"{'+' if mins>=0 else ''}{mins} m")
            reason_text.append(f"{label} ({'+' if mins>=0 else ''}{mins}m)")
        oid=f"{intro_platform[:2].upper()}-{int(time.time())%10000}"
        st.session_state.orders.append({
            "time":datetime.now().strftime("%H:%M:%S"),
            "platform":intro_platform,"order_id":oid,
            "distance":round(km,1),"eta":eta,
            "signals":sigs,"works":works,"mood":"Neutral","emoji":"😐",
            "priority":"Normal","review":"","reasons":", ".join(reason_text)
        })
        mid_lat=(user_lat+rider_lat)/2; mid_lng=(user_lng+rider_lng)/2
        layer_points = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([
                {"name":"You","lat":user_lat,"lon":user_lng,"size":120},
                {"name":"Rider","lat":rider_lat,"lon":rider_lng,"size":120},
            ]),
            get_position='[lon, lat]', get_radius='size', get_fill_color=[0,128,0,160],
            pickable=True,
        )
        layer_line = pdk.Layer(
            "LineLayer",
            data=pd.DataFrame([{"from_lon":rider_lng,"from_lat":rider_lat,"to_lon":user_lng,"to_lat":user_lat}]),
            get_source_position='[from_lon, from_lat]',
            get_target_position='[to_lon, to_lat]',
            get_width=4, get_color=[0, 102, 204, 180],
        )
        st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9",
                                 initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lng, zoom=13),
                                 layers=[layer_points, layer_line]))

# ---------------- Right: Review + Voice ----------------
with right:
    tabs = st.tabs([T["review"], T["voice"]])

    # ---- Tab 1: Typed review (kept) ----
    with tabs[0]:
        seed = {"hi":"bhai delay ho gaya, jaldi bhejo!","te":"inka delay ayindi, koncham fast ga raa!",
                "ta":"konjam late aachu, seekiram vaanga!","en":"The delay is annoying, please hurry."}[lang_key]
        if "review" not in st.session_state: st.session_state.review = seed

        def _on_review_change():
            now=time.time()
            txt=st.session_state.review
            meta=st.session_state.meta
            dt=max(0.2, now-meta["ts"])
            cps=max(0.0, (len(txt)-meta["len"])/dt)
            st.session_state.meta={"len":len(txt),"ts":now,"cps":cps}

        st.text_area(" ", key="review", height=140, on_change=_on_review_change)
        cps=st.session_state.meta["cps"]
        score,mood,emoji,hits=sentiment_score(st.session_state.review)
        fast_flag = (cps>=6.0 and mood in ["Angry","Frustrated","Disappointed"])
        priority = "High" if (mood in ["Angry","Frustrated","Disappointed"] or fast_flag) else "Normal"

        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Mood", f"{emoji} {mood}")
        with c2: st.metric("Score", score)
        with c3: st.metric("Typing speed", f"{cps:.1f} cps" + (" • "+LANGS[lang_key]["typing_fast"] if fast_flag else ""))

        st.caption("Triggers: " + (", ".join(hits) if hits else "—"))

        if st.button(T["submit"]):
            if st.session_state.orders:
                st.session_state.orders[-1]["mood"]=mood
                st.session_state.orders[-1]["emoji"]=emoji
                st.session_state.orders[-1]["priority"]=priority
                st.session_state.orders[-1]["review"]=st.session_state.review[:200]
            if priority=="High":
                st.warning(LANGS[lang_key]["angry_tip"])
                st.info(LANGS[lang_key]["coupon"])
            st.toast(f"Rider notified with priority: {priority}", icon="✅")

    # ---- Tab 2: Voice (Prosody, live + upload) ----
    with tabs[1]:
        st.caption(T["howto"])
        # calibration toggle
        do_calib = st.checkbox(T["calib"], value=False)
        # live mic
        class AudioBufferProcessor:
            def recv_audio(self, frame: "av.AudioFrame"):
                pcm = frame.to_ndarray()
                if pcm.ndim==2: pcm = pcm.mean(axis=0)
                pcm = pcm.astype(np.float32)
                if frame.sample_rate!=SAMPLE_RATE:
                    step = max(1, int(frame.sample_rate/SAMPLE_RATE))
                    pcm = pcm[::step]
                # normalize softly
                if np.max(np.abs(pcm))>0:
                    pcm = pcm/(np.max(np.abs(pcm))+1e-6)
                st.session_state.voice_buf.extend(pcm.tolist())
                return frame

        webrtc_streamer(
            key="voice-live",
            mode=WebRtcMode.RECVONLY,
            client_settings=ClientSettings(
                rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": False, "audio": True},
            ),
            audio_processor_factory=AudioBufferProcessor,
            async_processing=True,
        )

        # analyze current buffer
        sig = np.array(st.session_state.voice_buf, dtype=np.float32)
        if sig.size>0:
            # calibration: use first 5s as baseline
            if do_calib and st.session_state.baseline is None and len(sig) > SAMPLE_RATE*5:
                st.session_state.baseline = extract_prosody(sig[:SAMPLE_RATE*5], SAMPLE_RATE)
                st.info(T["calib_done"])

            F = extract_prosody(sig[-BUF_SAMPLES:], SAMPLE_RATE)
            mood_v, emoji_v, reasons, stress, (arousal, valence) = decide_voice_mood(F, st.session_state.baseline or {})
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Voice Mood", f"{emoji_v} {mood_v}")
            with c2: st.metric("Mean Pitch (F0)", f"{F.get('mean_f0',0):.0f} Hz")
            with c3: st.metric("Speaking Rate", f"{F.get('speaking_rate',0):.2f}/s")

            m1,m2,m3,m4 = st.columns(4)
            with m1: st.metric("Pitch Range", f"{F.get('range_f0',0):.0f} Hz")
            with m2: st.metric("Intensity (RMS)", f"{F.get('mean_int',0):.3f}")
            with m3: st.metric("CPP", f"{F.get('mean_cpp',0):.3f}")
            with m4: st.metric("Centroid", f"{F.get('centroid',0):.0f} Hz")

            n1,n2 = st.columns(2)
            with n1: st.metric("Stress Index", f"{stress:.0f}/100")
            with n2:
                # Arousal–Valence meter
                fig, ax = plt.subplots(figsize=(3.6,3.0))
                ax.axhline(0.5,color='0.8'); ax.axvline(0.5,color='0.8')
                ax.scatter([arousal],[valence], s=80)
                ax.set_xlim(0,1); ax.set_ylim(0,1)
                ax.set_xlabel("Arousal"); ax.set_ylabel("Valence"); ax.set_title("A–V Meter")
                st.pyplot(fig, clear_figure=True)

            st.caption("Why: " + ", ".join(reasons))

            # Waveform (last 3s)
            L = min(len(sig), SAMPLE_RATE*3)
            t = np.arange(L)/SAMPLE_RATE
            fig_w, ax = plt.subplots(figsize=(6,2.2))
            ax.plot(t, sig[-L:], linewidth=0.9)
            ax.set_title("Waveform (last 3s)")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.grid(alpha=0.3)
            st.pyplot(fig_w, clear_figure=True)

            # Spectrum (first 2s of buffer)
            seg = sig[-min(len(sig), 2*SAMPLE_RATE):]
            win = get_window("hann", len(seg))
            X = np.abs(rfft(seg*win))
            freqs = rfftfreq(len(seg), 1.0/SAMPLE_RATE)
            fig_s, ax2 = plt.subplots(figsize=(6,2.2))
            ax2.plot(freqs, 20*np.log10(X+1e-9), linewidth=0.9)
            ax2.set_xlim(0, 4000)
            ax2.set_title("Spectrum (dB)")
            ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Level (dB)")
            ax2.grid(alpha=0.3)
            st.pyplot(fig_s, clear_figure=True)

            # Attach to latest order if exists
            if st.session_state.orders:
                st.session_state.orders[-1]["mood"] = mood_v
                st.session_state.orders[-1]["emoji"] = emoji_v
                st.session_state.orders[-1]["priority"] = "High" if mood_v in ["Angry/Stress","Sad"] else "Normal"

        # ----- Upload fallback -----
        types = ["wav","mp3","m4a","aac","ogg"] if HAS_AV else ["wav"]
        audio_upload = st.file_uploader(T["upload_lbl"], type=types)
        if st.button(T["analyze"]):
            if not audio_upload:
                st.error("No audio provided.")
            else:
                try:
                    fs, x = to_mono_float(audio_upload.read())
                    F = extract_prosody(x, fs)
                    mood_v, emoji_v, reasons, stress, _ = decide_voice_mood(F, st.session_state.baseline or {})
                    st.metric("Voice Mood", f"{emoji_v} {mood_v}")
                    st.caption("Why: " + ", ".join(reasons))
                except Exception as e:
                    st.exception(e)

# ---------------- Delivery Boy Dashboard (kept) ----------------
st.markdown("---")
st.subheader(f"👷 {LANGS[lang_key]['dash']}")
orders=list(st.session_state.orders)
if orders:
    orders.sort(key=lambda o: (0 if o["priority"]=="High" else 1, o["eta"]))
    df=pd.DataFrame([{
        "⏰ Time":o["time"],"Platform":o["platform"],"Order":o["order_id"],
        "Dist(km)":o["distance"],"ETA(min)":o["eta"],"Signals":o.get("signals",""),
        "Roadworks":o.get("works",""),"Priority":o["priority"],"Mood":f"{o['emoji']} {o['mood']}",
        "Reasons":o["reasons"],"Review":o["review"]
    } for o in orders])
    st.dataframe(df, hide_index=True, use_container_width=True)
else:
    st.info("No active drops yet.")
