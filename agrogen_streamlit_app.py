# agrogen_streamlit_app.py
"""
AgroGen - Single-file Streamlit app (no custom CSS/theme)
Features:
- Soil Analysis (region-aware)
- Groq Chatbot (preferred) with simple fallback
- Groq Vision for images + Hugging Face local fallback
- Weather (OpenWeather)
- Past Analyses via Django API (optional)
- Newsletter (RSS)
- Fertilizer estimation & Profitability
"""

# -------------- imports & page config (must be first Streamlit command) --------------
import os
import json
import base64
from io import BytesIO
from datetime import datetime

import streamlit as st
st.set_page_config(page_title="AgroGen — Research & Modern Farmers", layout="wide")

import pandas as pd
import numpy as np
from PIL import Image
import requests
from dotenv import load_dotenv

# Optional libs wrapped in try/except to avoid hard failures
try:
    import feedparser
except Exception:
    feedparser = None

try:
    from groq import Groq
except Exception:
    Groq = None

# HF local vision fallback
try:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except Exception:
    torch = None
    AutoImageProcessor = None
    AutoModelForImageClassification = None

# Text-to-speech (optional)
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# ---------------- load env and defaults ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DJANGO_API_URL = os.getenv("DJANGO_API_URL", "http://127.0.0.1:8000/api/soil/")
FERT_API_URL = os.getenv("FERT_API_URL")  # optional fertilizer price API
HF_VISION_MODEL = os.getenv("HF_VISION_MODEL", "nateraw/vit-base-beans")

# Use the working defaults you provided
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "llama-3.2-11b-vision")

# ---------------- Groq client init ----------------
groq_client = None
GROQ_READY = False
if GROQ_API_KEY and Groq is not None:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        GROQ_READY = True
    except Exception:
        GROQ_READY = False

# ---------------- Groq wrappers ----------------
def safe_groq_chat(messages, model=CHAT_MODEL, max_tokens=300, temperature=0.6):
    """Send messages to Groq chat. messages = [{'role':'user','content':'text'}, ...]"""
    if not GROQ_READY:
        raise RuntimeError("Groq not configured or client not available.")
    resp = groq_client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)

def safe_groq_vision(prompt_text, image_b64, model=VISION_MODEL, max_tokens=400, temperature=0.6):
    """Send text+image to Groq vision-capable model (account must support)."""
    if not GROQ_READY:
        raise RuntimeError("Groq not configured or client not available.")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }
    ]
    resp = groq_client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)

# ---------------- Weather helpers ----------------
def geocode_city(city: str):
    """Return (lat, lon, error_message)"""
    if not OPENWEATHER_API_KEY:
        return None, None, "OPENWEATHER_API_KEY not set in .env"
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city, "limit": 1, "appid": OPENWEATHER_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return None, None, f"Geocoding failed: status {r.status_code}"
        data = r.json()
        if isinstance(data, list) and data:
            return data[0].get("lat"), data[0].get("lon"), None
        return None, None, "Could not geocode the city."
    except Exception as e:
        return None, None, f"Geocode error: {e}"

def get_weather(lat: float, lon: float):
    """Return (weather_dict, error_message)"""
    if not OPENWEATHER_API_KEY:
        return None, "OPENWEATHER_API_KEY not set in .env"
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return None, f"Weather fetch failed: status {r.status_code}"
        data = r.json()
        if "main" not in data:
            return None, "Unexpected weather response"
        return {
            "temp": data["main"].get("temp"),
            "feels_like": data["main"].get("feels_like"),
            "humidity": data["main"].get("humidity"),
            "pressure": data["main"].get("pressure"),
            "wind_speed": data.get("wind", {}).get("speed"),
            "condition": data.get("weather", [{}])[0].get("description", "").title()
        }, None
    except Exception as e:
        return None, f"Weather error: {e}"

# ---------------- Soil scoring & recommendations ----------------
def soil_health_score(ph, n, p, k, moisture, oc, region=""):
    """Region-aware scoring (returns 0-100)."""
    region = (region or "").lower()
    score = 0
    # Simple region profiles (examples)
    profiles = {
        "rajasthan": {"ideal_ph": (7.0,8.2), "min_moist":8, "max_moist":25},
        "gujarat": {"ideal_ph": (6.5,8.0), "min_moist":10, "max_moist":30},
        "punjab": {"ideal_ph": (6.0,7.5), "min_moist":15, "max_moist":35},
        "kerala": {"ideal_ph": (5.0,6.5), "min_moist":25, "max_moist":60},
        "assam": {"ideal_ph": (4.5,5.8), "min_moist":30, "max_moist":70},
    }
    prof = None
    for k_ in profiles:
        if k_ in region:
            prof = profiles[k_]; break
    if not prof:
        prof = {"ideal_ph":(6.0,7.5), "min_moist":15, "max_moist":30}
    i_low, i_high = prof["ideal_ph"]
    m_low, m_high = prof["min_moist"], prof["max_moist"]
    # pH scoring
    if i_low <= ph <= i_high: score += 25
    elif abs(ph - i_low) <= 0.5 or abs(ph - i_high) <= 0.5: score += 15
    else: score += 5
    # NPK
    score += 20 if n >= 250 else 10 if n >= 150 else 5
    score += 15 if p >= 30 else 10 if p >= 15 else 5
    score += 15 if k >= 150 else 10 if k >= 80 else 5
    # organic carbon
    score += 10 if oc >= 1.5 else 5 if oc >= 0.8 else 2
    # moisture (region aware)
    if m_low <= moisture <= m_high: score += 10
    elif abs(moisture - m_low) <= 5 or abs(moisture - m_high) <= 5: score += 5
    else: score += 1
    return max(0, min(100, int(score)))

def recommend_crops(score, ph, region=""):
    r = (region or "").lower()
    if score >= 70:
        base = ["Wheat", "Maize", "Sunflower"]
    elif score >= 50:
        base = ["Millets", "Pulses", "Sorghum"]
    else:
        base = ["Legumes (green manure)", "Cover crops"]
    if any(x in r for x in ["rajasthan", "gujarat", "barmer"]):
        return ["Bajra (Pearl Millet)", "Cumin", "Groundnut", "Moong"]
    if any(x in r for x in ["punjab", "haryana", "uttar pradesh", "bihar"]):
        return ["Wheat", "Rice", "Maize", "Sugarcane"]
    if any(x in r for x in ["kerala", "karnataka", "tamil", "andhra"]):
        return ["Paddy", "Coconut", "Banana", "Spices"]
    return base

def fertilizer_recommendation(n, p, k, ph):
    rec = []
    if n < 150: rec.append("Nitrogen low — apply urea or compost")
    if p < 20: rec.append("Phosphorus low — apply rock phosphate/DAP")
    if k < 100: rec.append("Potassium low — apply MOP or ash")
    if ph < 5.5: rec.append("Acidic soil — apply lime")
    if ph > 8.0: rec.append("Alkaline soil — add organic matter / gypsum")
    if not rec: rec.append("Balanced — maintain current sustainable practices")
    return rec

# ---------------- Fertilizer pricing & estimate ----------------
INDIA_PRICES_FALLBACK = {"urea_per_kg":9.0, "dap_per_kg":40.0, "mop_per_kg":24.0}

def get_fertilizer_prices(region=None):
    if FERT_API_URL:
        try:
            r = requests.get(FERT_API_URL, params={"region": region} if region else {}, timeout=6)
            if r.status_code == 200:
                return r.json(), None
            return INDIA_PRICES_FALLBACK, f"FERT API returned {r.status_code}"
        except Exception as e:
            return INDIA_PRICES_FALLBACK, f"FERT API error: {e}"
    return INDIA_PRICES_FALLBACK, "Using fallback prices"

def estimate_fertilizer_cost_from_ppm(n_ppm, p_ppm, k_ppm, area_ha, region=None):
    # simple heuristic conversion from ppm deficit to kg fertilizer
    target_n, target_p, target_k = 250.0, 30.0, 150.0
    n_def = max(0.0, target_n - n_ppm)
    p_def = max(0.0, target_p - p_ppm)
    k_def = max(0.0, target_k - k_ppm)
    # heuristics
    kg_N_per_ha = n_def * 0.5
    kg_P_per_ha = p_def * 0.5
    kg_K_per_ha = k_def * 0.5
    kg_N_total = kg_N_per_ha * area_ha
    kg_P_total = kg_P_per_ha * area_ha
    kg_K_total = kg_K_per_ha * area_ha
    prices, note = get_fertilizer_prices(region)
    urea_price = prices.get("urea_per_kg", INDIA_PRICES_FALLBACK["urea_per_kg"])
    dap_price = prices.get("dap_per_kg", INDIA_PRICES_FALLBACK["dap_per_kg"])
    mop_price = prices.get("mop_per_kg", INDIA_PRICES_FALLBACK["mop_per_kg"])
    # nutrient content conversions
    urea_kg = kg_N_total / 0.46 if kg_N_total > 0 else 0.0
    dap_kg = kg_P_total / 0.18 if kg_P_total > 0 else 0.0
    mop_kg = kg_K_total / 0.50 if kg_K_total > 0 else 0.0
    cost = (urea_kg * urea_price) + (dap_kg * dap_price) + (mop_kg * mop_price)
    return round(cost, 2), {"urea_kg": round(urea_kg, 2), "dap_kg": round(dap_kg, 2), "mop_kg": round(mop_kg, 2)}, note

# ---------------- HF vision loader (optional) ----------------
@st.cache_resource
def load_hf_model(name=HF_VISION_MODEL):
    if AutoImageProcessor is None or AutoModelForImageClassification is None:
        return None, None, "transformers/torch not installed"
    try:
        proc = AutoImageProcessor.from_pretrained(name)
        model = AutoModelForImageClassification.from_pretrained(name)
        return proc, model, None
    except Exception as e:
        return None, None, str(e)

hf_processor, hf_model, hf_err = load_hf_model()

def analyze_leaf_image_hf(image: Image.Image):
    if hf_processor is None or hf_model is None:
        return None, None, hf_err or "HF model not available"
    try:
        img = image.convert("RGB")
        inputs = hf_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_model(**inputs)
        logits = outputs.logits
        predicted = int(logits.argmax(-1).item())
        label = hf_model.config.id2label.get(predicted, str(predicted)) if hasattr(hf_model.config, "id2label") else str(predicted)
        score = float(torch.softmax(logits, dim=-1)[0][predicted].item())
        return label, round(score * 100, 2), None
    except Exception as e:
        return None, None, str(e)

# ---------------- Newsletter (RSS) ----------------
def fetch_news_rss(query="agriculture India", max_items=6):
    if feedparser is None:
        return None, "feedparser not installed"
    q = requests.utils.quote(query)
    rss = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(rss)
        items = []
        for e in feed.entries[:max_items]:
            items.append({"title": e.get("title"), "link": e.get("link"), "published": e.get("published", "")})
        return items, None
    except Exception as e:
        return None, str(e)

# ---------------- Sidebar navigation ----------------
st.sidebar.title("AgroGen")
st.sidebar.caption("Research & modern Indian farmers — English only")

pages = ["Soil Analysis", "Chatbot", "Weather", "Image Disease Detector", "Past Analyses", "Newsletter", "Profitability"]
page = st.sidebar.radio("Navigate", pages)

# show basic status
st.sidebar.markdown("### Status")
st.sidebar.write({
    "Groq configured": GROQ_READY,
    "OpenWeather": bool(OPENWEATHER_API_KEY),
    "HF vision local": hf_processor is not None and hf_model is not None
})

st.title("🌾 AgroGen — Research & Modern Farmers")

# ---------------- Soil Analysis page ----------------
if page == "Soil Analysis":
    st.header("Soil Analysis")
    st.markdown("Enter soil parameters (region optional). Outputs region-aware score, crop suggestions and fertilizer advice.")
    with st.form("soil_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
            nitrogen = st.number_input("Nitrogen (N) ppm", min_value=0.0, value=200.0, step=1.0)
        with c2:
            phosphorus = st.number_input("Phosphorus (P) ppm", min_value=0.0, value=30.0, step=1.0)
            potassium = st.number_input("Potassium (K) ppm", min_value=0.0, value=150.0, step=1.0)
        with c3:
            moisture = st.number_input("Soil moisture (%)", min_value=0.0, value=20.0, step=0.5)
            organic_carbon = st.number_input("Organic Carbon (%)", min_value=0.0, value=0.8, step=0.1)
        region_input = st.text_input("Region / District (optional)")
        analyze = st.form_submit_button("Analyze Soil")
    if analyze:
        score = soil_health_score(ph, nitrogen, phosphorus, potassium, moisture, organic_carbon, region_input)
        crops = recommend_crops(score, ph, region_input)
        recs = fertilizer_recommendation(nitrogen, phosphorus, potassium, ph)
        st.metric("Soil Health Score", f"{score}/100")
        st.markdown("**Recommended crops (region-aware):**")
        st.write(", ".join(crops))
        st.markdown("**Nutrient suggestions:**")
        for r in recs:
            st.write("- " + r)
        # Fertilizer cost estimate for 1 ha
        cost_est, breakdown, note = estimate_fertilizer_cost_from_ppm(nitrogen, phosphorus, potassium, 1.0, region_input)
        st.markdown(f"**Estimated fertilizer cost for 1 ha (INR):** {cost_est} _(breakdown: {breakdown})_")
        # AI insight via Groq if available
        ai_prompt = f"As an agronomist give 2 short farmer-friendly recommendations from: pH={ph}, N={nitrogen}, P={phosphorus}, K={potassium}, Moisture={moisture}, OC={organic_carbon}, region={region_input}"
        if GROQ_READY:
            try:
                out = safe_groq_chat([{"role": "user", "content": ai_prompt}], model=CHAT_MODEL, max_tokens=120, temperature=0.4)
                st.info(out)
            except Exception as e:
                st.warning(f"Groq error: {e}")
        else:
            st.info("AI unavailable — use the suggestions above.")

# ---------------- Chatbot page ----------------
elif page == "Chatbot":
    st.header("AgroGen Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "🌾 Hello — I'm AgroGen. Ask about soil, crops, pests, weather."}]
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_input = st.chat_input("Ask a farming question...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        if GROQ_READY:
            try:
                messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[-20:]]
                reply = safe_groq_chat(messages, model=CHAT_MODEL, max_tokens=220, temperature=0.6)
            except Exception as e:
                reply = f"Groq error: {e}"
        else:
            # simple fallback
            txt = user_input.lower()
            if "ph" in txt or "soil" in txt:
                reply = "Check pH and NPK. For neutral pH (6–7.5) many crops grow well. Add compost to improve organic carbon."
            elif "weather" in txt:
                reply = "Check local weather before fertilizer; irrigate early morning/evening to reduce evaporation."
            else:
                reply = "Ask about soil, crop choice, fertilizer, or pest symptoms. Example: 'Best crop for pH 6.5?'"
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

# ---------------- Weather page ----------------
elif page == "Weather":
    st.header("Weather & Advice")
    city = st.text_input("City / District (e.g., Jaipur)", value="")
    if st.button("Get Weather"):
        if not city:
            st.warning("Enter a city or district name.")
        else:
            lat, lon, gerr = geocode_city(city)
            if gerr:
                st.error(gerr)
            else:
                weather, werr = get_weather(lat, lon)
                if werr:
                    st.error(werr)
                else:
                    st.markdown(f"**Weather in {city.title()}:**")
                    st.write(weather)
                    advice_prompt = f"As agronomist give 3 short actionable steps for farmers based on this weather: {weather}"
                    if GROQ_READY:
                        try:
                            advice = safe_groq_chat([{"role": "user", "content": advice_prompt}], model=CHAT_MODEL, max_tokens=120)
                        except Exception as e:
                            advice = f"Groq error: {e}"
                    else:
                        advice = "Avoid fertilizer before heavy rain; irrigate appropriately; protect newly sown fields from storms."
                    st.info(advice)

# ---------------- Image Disease Detector page ----------------
elif page == "Image Disease Detector":
    st.header("Image Disease Detector — Groq Vision + HF fallback")
    st.markdown("Upload leaf or soil image (jpg/png). If local HF model available it will be used; otherwise Groq Vision will be used if configured.")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    symp = st.text_input("Describe visible symptoms (optional)")
    if uploaded:
        try:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)
            # Try HF local first
            if hf_processor is not None and hf_model is not None:
                with st.spinner("Analyzing with local HF model..."):
                    label, conf, err = analyze_leaf_image_hf(img)
                if err:
                    st.error(f"Local HF analysis failed: {err}")
                else:
                    st.subheader("Diagnosis (Local HF)")
                    st.write(f"**Prediction:** {label}")
                    st.write(f"**Confidence:** {conf}%")
                    if "healthy" in (label or "").lower():
                        st.success("Leaf looks healthy.")
                    else:
                        st.warning("Possible issue detected — consult extension services.")
            elif GROQ_READY:
                with st.spinner("Analyzing with Groq Vision..."):
                    try:
                        buf = BytesIO()
                        img.convert("RGB").save(buf, format="JPEG")
                        img_b64 = base64.b64encode(buf.getvalue()).decode()
                        prompt = ("You are a plant pathologist. Analyze the image and return top 3 likely causes (disease/pest/deficiency), severity level, and 3 practical actions. "
                                  f"Farmer note: {symp or 'None'}")
                        diagnosis = safe_groq_vision(prompt, img_b64, model=VISION_MODEL, max_tokens=400)
                        st.subheader("Diagnosis / Suggestions (Groq Vision)")
                        st.write(diagnosis)
                    except Exception as e:
                        st.error(f"Groq vision error: {e}")
            else:
                st.warning("No vision model available. Install transformers+torch for local analysis, or set GROQ_API_KEY for Groq vision.")
        except Exception as e:
            st.error(f"Image handling failed: {e}")

# ---------------- Past Analyses page ----------------
elif page == "Past Analyses":
    st.header("Past Analyses (Django backend - optional)")
    st.markdown("Load soil records from configured Django API and download CSV.")
    if st.button("Load Past Records"):
        try:
            r = requests.get(DJANGO_API_URL, timeout=8)
            if r.status_code == 200:
                records = r.json()
                if not records:
                    st.info("No records found.")
                else:
                    df = pd.DataFrame(records)
                    if "created_at" in df.columns:
                        df["created_at"] = pd.to_datetime(df["created_at"])
                    st.dataframe(df, use_container_width=True)
                    if "created_at" in df.columns and "score" in df.columns:
                        st.markdown("### Score trend")
                        st.line_chart(df.set_index("created_at")["score"])
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Download CSV", data=csv, file_name="agrogen_records.csv", mime="text/csv")
            else:
                st.warning(f"Failed to fetch: {r.status_code}")
        except Exception as e:
            st.error(f"Error fetching records: {e}")

# ---------------- Newsletter page ----------------
elif page == "Newsletter":
    st.header("📬 Farming Newsletter — Latest Headlines")
    topic = st.text_input("Search topic", value="agriculture India")
    if st.button("Fetch latest headlines"):
        if feedparser is None:
            st.error("feedparser not installed. Install with: pip install feedparser")
        else:
            items, ferr = fetch_news_rss(topic, max_items=8)
            if ferr:
                st.error(ferr)
            else:
                st.markdown(f"### Top news for: **{topic}**")
                for it in items:
                    st.markdown(f"- [{it['title']}]({it['link']}) — {it.get('published','')}")
                st.info("Tip: connect an SMTP provider to email this newsletter to subscribers.")

# ---------------- Profitability page ----------------
elif page == "Profitability":
    st.header("🌾 Crop Profitability Prediction")
    st.markdown("Estimate profitability using expected yield, market price and production costs. Optionally include fertilizer cost estimation.")
    default_crop_stats = {
        "Wheat": {"yield": 3.5, "price": 22000, "cost": 20000},
        "Rice": {"yield": 3.8, "price": 24000, "cost": 30000},
        "Maize": {"yield": 4.0, "price": 18000, "cost": 18000},
        "Bajra": {"yield": 2.0, "price": 15000, "cost": 12000},
        "Pulses": {"yield": 1.0, "price": 45000, "cost": 15000},
    }
    crop_choice = st.selectbox("Crop", list(default_crop_stats.keys()) + ["Custom"])
    area_ha = st.number_input("Area (ha)", min_value=0.01, value=1.0, step=0.1, format="%.2f")
    region_choice = st.text_input("Region (optional)")
    if crop_choice != "Custom":
        d = default_crop_stats[crop_choice]
        default_yield = d["yield"]; default_price = d["price"]; default_cost = d["cost"]
    else:
        default_yield = 2.0; default_price = 20000; default_cost = 20000

    c1, c2, c3 = st.columns(3)
    with c1:
        expected_yield = st.number_input("Expected yield (t/ha)", value=float(default_yield), min_value=0.0, step=0.1)
    with c2:
        # make sure value, min_value, step are same types (ints)
        market_price = st.number_input("Market price (INR/tonne)",
                                       value=int(default_price),
                                       min_value=0,
                                       step=100)
    with c3:
        prod_cost_per_ha = st.number_input("Production cost (INR/ha)",
                                           value=int(default_cost),
                                           min_value=0,
                                           step=100)

    use_fert = st.checkbox("Estimate fertilizer cost from soil values", value=False)
    fert_cost = 0.0
    if use_fert:
        n_ppm = st.number_input("Nitrogen (ppm)", value=200.0, step=1.0)
        p_ppm = st.number_input("Phosphorus (ppm)", value=30.0, step=1.0)
        k_ppm = st.number_input("Potassium (ppm)", value=150.0, step=1.0)
        fert_cost, breakdown, note = estimate_fertilizer_cost_from_ppm(n_ppm, p_ppm, k_ppm, area_ha, region_choice)
        st.markdown(f"Estimated fertilizer cost: INR {fert_cost} _(breakdown: {breakdown})_")

    if st.button("Calculate Profitability"):
        total_prod_t = expected_yield * area_ha
        revenue = total_prod_t * market_price
        tot_cost = prod_cost_per_ha * area_ha + fert_cost
        profit = revenue - tot_cost
        profit_per_ha = profit / area_ha if area_ha > 0 else 0.0
        roi = (profit / tot_cost * 100) if tot_cost > 0 else float("inf")
        breakeven = (tot_cost / total_prod_t) if total_prod_t > 0 else float("inf")
        st.metric("Estimated production (t)", f"{total_prod_t:.2f}")
        st.metric("Estimated revenue (INR)", f"{revenue:,.2f}")
        st.metric("Estimated total cost (INR)", f"{tot_cost:,.2f}")
        st.metric("Estimated profit (INR)", f"{profit:,.2f}")
        st.write(f"Profit per ha: INR {profit_per_ha:,.2f}")
        if roi != float("inf"):
            st.write(f"ROI: {roi:.1f}%")
        else:
            st.write("ROI: N/A")
        if breakeven != float("inf"):
            st.info(f"Breakeven price per tonne: INR {breakeven:.2f}")

# Footer
st.markdown("---")
st.caption("AgroGen — for researchers & modern Indian farmers. Provide GROQ_API_KEY and OPENWEATHER_API_KEY in .env to enable AI & weather features.")
