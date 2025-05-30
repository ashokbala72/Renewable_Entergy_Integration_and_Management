
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Refresh every 60s
st.set_page_config(layout="wide")
st_autorefresh(interval=60 * 1000, key='refresh_key')

now = pd.Timestamp.now()
remaining = 60 - int(time.time()) % 60
mins, secs = divmod(remaining, 60)
st.caption(f"🕓 Last refreshed at {now.strftime('%H:%M:%S')}, ⏳ Auto-refreshing in {mins:02d}:{secs:02d}")

def fetch_weather_forecast(lat=28.61, lon=77.23):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,windspeed_10m,shortwave_radiation&forecast_days=1&timezone=auto"
    response = requests.get(url)
    hourly = response.json()['hourly']
    return pd.DataFrame({
        'time': pd.to_datetime(hourly['time']),
        'temperature_2m': hourly['temperature_2m'],
        'windspeed_10m': hourly['windspeed_10m'],
        'shortwave_radiation': hourly['shortwave_radiation']
    }).head(10)

def simulate_historical_production():
    base_time = pd.Timestamp.now() - pd.Timedelta(hours=168)
    timestamps = [base_time + pd.Timedelta(hours=1)*i for i in range(168)]
    return pd.DataFrame({
        'timestamp': timestamps,
        'actual_output_mw': np.random.uniform(5, 20, size=168)
    })

def simulate_live_sensors_stream():
    base_time = pd.Timestamp.now() - pd.Timedelta(minutes=9)
    timestamps = [base_time + pd.Timedelta(minutes=i) for i in range(10)]
    return pd.DataFrame({
        'timestamp': timestamps,
        'voltage': np.random.uniform(410, 430, size=10),
        'current': np.random.uniform(15, 25, size=10),
        'inverter_status': np.random.choice(['ON', 'STANDBY', 'FAULT'], size=10)
    })

def predict_output(weather_df, sensor_df):
    forecast_df = weather_df.copy()
    forecast_df['voltage'] = sensor_df['voltage'].iloc[-1]
    forecast_df['current'] = sensor_df['current'].iloc[-1]
    forecast_df['inverter_status'] = sensor_df['inverter_status'].iloc[-1]
    forecast_df['predicted_output_mw'] = (
        0.4 * forecast_df['temperature_2m'] +
        0.5 * forecast_df['windspeed_10m'] +
        0.1 * forecast_df['shortwave_radiation'] / 10 +
        np.random.uniform(-1, 1, len(forecast_df))
    )
    return forecast_df

def fetch_uk_market_demand():
    try:
        # Placeholder for actual UK API. Simulating with random data.
        time_series = pd.date_range(start=pd.Timestamp.now(), periods=10, freq='H')
        demand = np.random.uniform(10, 25, size=10)
        return pd.DataFrame({'time': time_series, 'market_demand_mw': demand})
    except:
        return pd.DataFrame()

# Load actual data
weather_df = fetch_weather_forecast()
sensor_df = simulate_live_sensors_stream()
prediction_df = predict_output(weather_df, sensor_df)
hist_df = simulate_historical_production()
uk_demand_df = fetch_uk_market_demand()
uk_demand_df['predicted_output_mw'] = prediction_df['predicted_output_mw'].values

tabs = st.tabs([
    "🌤️ Weather Forecast",
    "📈 Historical Production",
    "🛰️ Live Sensor Feed",
    "📉 Demand vs Output",
    "🧠 GenAI Actions",
    "💬 Ask My Assistant",
    "📊 Trading Assistant",
])

with tabs[0]:
    st.subheader("🌤️ Weather Forecast (Open-Meteo)")
    st.dataframe(weather_df)
    st.line_chart(weather_df.set_index("time")[["temperature_2m", "windspeed_10m"]])

with tabs[1]:
    st.subheader("📈 Historical Production (Simulated)")
    st.line_chart(hist_df.set_index("timestamp")["actual_output_mw"])

with tabs[2]:
    st.subheader("🛰️ Live Sensor Feed (Last 10 mins)")
    st.info("⏱️ This feed auto-refreshes every 60 seconds.")
    st.dataframe(sensor_df)
    st.line_chart(sensor_df.set_index("timestamp")[["voltage", "current"]])




with tabs[3]:
    st.subheader("📉 Market Demand vs Predicted Output (UK Grid)")
    st.markdown("✅ **Data Source:** Live feed simulated from [National Grid ESO (UK)](https://www.nationalgrideso.com/energy-data-dashboard)")
    st.markdown("### 📋 Demand vs Output Table")
    st.dataframe(uk_demand_df[['time', 'market_demand_mw', 'predicted_output_mw']])

    st.markdown("### 📈 Demand vs Output Chart")
    st.line_chart(uk_demand_df.set_index("time")[["market_demand_mw", "predicted_output_mw"]])

    try:
        sample_df = uk_demand_df[['time', 'market_demand_mw', 'predicted_output_mw']].head()
        sample_df['time'] = sample_df['time'].dt.strftime('%Y-%m-%d %H:%M')
        sample_text = sample_df.to_string(index=False)

        insight_prompt = f"""
You are a grid optimization analyst. Based on the table below of market demand vs predicted output, write a one-line summary describing the current trend or gap.

{sample_text}
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": insight_prompt}],
            max_tokens=100
        )
        punchline = response.choices[0].message.content.strip()
        st.success(f"🔍 GenAI Insight: {punchline}")
    except Exception as e:
        st.warning("Unable to fetch GenAI punchline.")
        st.error(f"Debug info: {str(e)}")

with tabs[4]:
    st.subheader("🧠 GenAI Actions")
    try:
        latest_row = prediction_df.iloc[-1]
        latest_sensor = sensor_df.iloc[-1]
        context = f"""
Predicted Output: {latest_row['predicted_output_mw']:.2f} MW  
Temperature: {latest_row['temperature_2m']:.1f} °C  
Wind Speed: {latest_row['windspeed_10m']:.1f} m/s  
Solar Irradiance: {latest_row['shortwave_radiation']:.0f} W/m²  
Voltage: {latest_sensor['voltage']:.1f} V  
Current: {latest_sensor['current']:.1f} A  
Inverter Status: {latest_sensor['inverter_status']}
"""

        prompt = f"""
You are a renewable energy system advisor. Based on the following data, provide 3 specific and actionable recommendations to improve energy performance or grid reliability. For each recommendation, cite the data points that justify it.

Input Data:
{context}
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        st.markdown("### 📊 Input Parameters")
        st.markdown(context)
        st.markdown("### ✅ GenAI Action Recommendations")
        st.markdown(response.choices[0].message.content)
    except Exception as e:
        st.error(f"⚠️ Unable to generate GenAI Actions: {e}")


with tabs[5]:
    st.subheader("💬 Ask My Assistant")
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        question = st.text_input("💡 Ask a question about today's renewable performance:", key="ask_input_box")
        if question:
            try:
                latest_row = prediction_df.iloc[-1]
                latest_sensor = sensor_df.iloc[-1]
                context = f"""
Forecast Output: {latest_row['predicted_output_mw']:.2f} MW  
Temperature: {latest_row['temperature_2m']:.1f} °C  
Wind Speed: {latest_row['windspeed_10m']:.1f} m/s  
Irradiance: {latest_row['shortwave_radiation']:.0f} W/m²  
Voltage: {latest_sensor['voltage']:.1f} V  
Current: {latest_sensor['current']:.1f} A  
Inverter Status: {latest_sensor['inverter_status']}
"""
                prompt = f"""
You are a renewable energy assistant. Use the following input data and answer the user's question clearly and concisely.

Input Data:
{context}

User Question:
{question}
"""
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=250
                )
                st.markdown("### 🧠 Assistant Response")
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(f"❌ Error: {e}")



with tabs[6]:
    st.subheader("📊 Trading Assistant")
    try:
        latest_row = prediction_df.iloc[-1]
        market_row = uk_demand_df.iloc[-1]
        forecast_output = latest_row['predicted_output_mw']
        market_demand = market_row['market_demand_mw']
        price_estimate = round(np.random.uniform(80, 140), 2)  # GBP per MWh simulated
        timestamp = latest_row['time']
        context = f"""
Timestamp: {timestamp}
Forecast Output: {forecast_output:.2f} MW
Market Demand: {market_demand:.2f} MW
Estimated Trading Price: £{price_estimate} per MWh
"""

        prompt = f"""
You are a renewable energy trading advisor. Based on the following live forecast and market data, provide specific trading recommendations.

1. Should the operator sell now?
2. How many MWh should be sold?
3. What price range (in GBP) should be targeted?
4. Mention if there’s any surplus or shortage in supply.
5. For each point, clearly state what input influenced your decision.

Input Data:
{context}
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350
        )

        st.markdown("### 📈 Live Trading Data")
        st.markdown(context)
        st.markdown("### 💼 GenAI Trading Strategy with Input Rationale")
        st.markdown(response.choices[0].message.content)

    except Exception as e:
        st.error(f"❌ Trading Assistant failed: {e}")
