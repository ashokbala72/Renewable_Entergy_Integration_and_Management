import pandas as pd
import streamlit as st
st.set_page_config(layout="wide")


import time
def countdown_and_timestamp(seconds=60):
    now = pd.Timestamp.now()
    remaining = seconds - int(time.time()) % seconds
    mins, secs = divmod(remaining, 60)
    st.caption(f"🕒 Last refreshed at {now.strftime('%H:%M:%S')}, ⏳ Auto-refreshing in {mins:02d}:{secs:02d}")
countdown_and_timestamp()


import time

def countdown(seconds=60):
    remaining = seconds - int(time.time()) % seconds
    mins, secs = divmod(remaining, 60)
    st.caption(f"⏳ Auto-refreshing in {mins:02d}:{secs:02d}")

st.markdown("""
<style>
/* Updated styling using stable data-testid selectors */
[data-testid="stTabs"] > div > div {
    background-color: #eaf6ff;
    border-radius: 10px;
    padding: 0.5rem;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #b3e5fc !important;
    font-weight: bold;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Tab styling */
    background-color: #eaf6ff;
    color: black;
    border-radius: 8px;
    margin-right: 5px;
    padding: 8px;
    font-weight: bold;
}
    background-color: #b3e5fc;
    color: black;
}
</style>
""", unsafe_allow_html=True)

from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=60 * 1000, key='data_refresh')

import numpy as np
import requests
from datetime import datetime, timedelta
from openai import OpenAI, OpenAIError
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("🔋 Renewable Forecast & Strategy Assistant")

# --- 1. Open-Meteo Weather Forecast (No API key needed) ---
def fetch_weather_forecast(lat=28.61, lon=77.23):  # Default: New Delhi
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly=temperature_2m,windspeed_10m,"
        f"shortwave_radiation&forecast_days=1&timezone=auto"
    )
    response = requests.get(url)
    data = response.json()
    hourly = data['hourly']
    df = pd.DataFrame({
        'time': pd.to_datetime(hourly['time']),
        'temperature_2m': hourly['temperature_2m'],
        'windspeed_10m': hourly['windspeed_10m'],
        'shortwave_radiation': hourly['shortwave_radiation']
    })
    return df.head(10)  # next 10 hours

# --- 2. Simulate Historical Production Data ---
def simulate_historical_production():
    base_time = pd.Timestamp.now() - pd.Timedelta(hours=24)
    timestamps = [base_time + pd.Timedelta(minutes=30*i) for i in range(48)]
    return pd.DataFrame({
        'timestamp': timestamps,
        'actual_output_mw': np.random.uniform(5, 20, size=48)
    })

# --- 3. Simulate Live Sensor Feed (Streaming) ---
def simulate_live_sensors_stream():
    base_time = pd.Timestamp.now() - pd.Timedelta(minutes=9)
    timestamps = [base_time + pd.Timedelta(minutes=i) for i in range(10)]
    return pd.DataFrame({
        'timestamp': timestamps,
        'voltage': np.random.uniform(410, 430, size=10),
        'current': np.random.uniform(15, 25, size=10),
        'inverter_status': np.random.choice(['ON', 'STANDBY', 'FAULT'], size=10)
    })

# --- 4. Predict Output ---
def predict_output(weather_df, sensor_df):
    forecast_df = weather_df.copy()
    forecast_df['voltage'] = sensor_df['voltage'].iloc[-1]
    forecast_df['current'] = sensor_df['current'].iloc[-1]
    forecast_df['inverter_status'] = sensor_df['inverter_status'].iloc[-1]
    forecast_df['predicted_output_mw'] = (
        0.4 * forecast_df['temperature_2m'] +
        0.5 * forecast_df['windspeed_10m'] +
        0.1 * forecast_df['shortwave_radiation'] / 10 +
        np.random.uniform(-1, 1, len(forecast_df))  # simulate variation
    )
    return forecast_df

# --- 5. GenAI Advisory ---
def get_genai_advice(row):
    prompt = f"""
You are a renewable energy grid strategist. Based on the following conditions:
- Temperature: {row['temperature_2m']} °C
- Wind Speed: {row['windspeed_10m']} m/s
- Irradiance: {row['shortwave_radiation']} W/m²
- Predicted Output: {row['predicted_output_mw']:.2f} MW
- Voltage: {row['voltage']} V
- Current: {row['current']} A
- Inverter Status: {row['inverter_status']}

Suggest:
1. Grid management strategy
2. Battery or storage recommendation
3. Resource allocation tip
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4,
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        return f"GenAI error: {e}"

# --- UI Section ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌤️ Weather Forecast",
    "📈 Historical Production",
    "🛰️ Live Sensor Feed",
    "🔋 Predicted Output",
    "🧠 GenAI Strategy"
])

with tab1:
    st.subheader("🌤️ Weather Forecast (Open-Meteo)")
    weather_df = fetch_weather_forecast()
    st.dataframe(weather_df)
    st.line_chart(weather_df.set_index("time")[["temperature_2m", "windspeed_10m"]])

with tab2:
    st.subheader("📈 Historical Production (Simulated)")
    hist_df = simulate_historical_production()
    st.line_chart(hist_df.set_index("timestamp")["actual_output_mw"])

with tab3:
    st.subheader("🛰️ Live Sensor Feed (Last 10 mins)")
    st.info("⏱️ This feed auto-refreshes every 60 seconds.")
    sensor_df = simulate_live_sensors_stream()
    st.dataframe(sensor_df)
    st.line_chart(sensor_df.set_index("timestamp")[["voltage", "current"]])

with tab4:
    st.subheader("🔋 Predicted Renewable Output")
    prediction_df = predict_output(weather_df, sensor_df)
    st.dataframe(prediction_df[['time', 'predicted_output_mw']])
    st.line_chart(prediction_df.set_index("time")["predicted_output_mw"])

with tab5:
    st.subheader("🧠 GenAI Strategy Recommendations")
    for _, row in prediction_df.iterrows():
        st.markdown(f"""
**🕒 Time:** {row['time']}
- 🌡️ Temp: {row['temperature_2m']} °C
- 💨 Wind: {row['windspeed_10m']} m/s
- ☀️ Irradiance: {row['shortwave_radiation']} W/m²
- ⚡ Voltage: {row['voltage']} V
- 🔌 Current: {row['current']} A
- 🛠️ Inverter: {row['inverter_status']}
- 🔋 Predicted Output: {row['predicted_output_mw']:.2f} MW

🧠 **Strategy Advice:**
{get_genai_advice(row)}

---
""")
