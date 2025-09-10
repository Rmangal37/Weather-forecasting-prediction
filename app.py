import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Load model and scaler ---
model = load_model("lstm_model.h5", compile=False)
scaler = joblib.load("scaler.save")

SEQ_LEN = 30  # must match training

# --- Page Config ---
st.set_page_config(page_title="Weather Forecasting", page_icon="🌦️", layout="wide")

# --- Custom Blue Theme CSS ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
        }
        h1, h2, h3, h4 {
            color: #e0f7fa;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #1565c0;
            color: white;
            border-radius: 10px 10px 0 0;
            padding: 10px;
            margin-right: 5px;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #1e88e5;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0d47a1 !important;
            color: #fff !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("📂 Upload a CSV file", type=["csv"])
n_forecast = st.sidebar.slider("📆 Forecast horizon (days)", 1, 14, 7)

# --- Main Title ---
st.title("🌦️ Weather Forecasting Dashboard")
st.markdown("### Powered by LSTM — providing **blue-sky insights** ⛅")

# --- File Handling ---
if uploaded_file is not None:
    try:
        # Parse date column properly
        df = pd.read_csv(uploaded_file, parse_dates=["Formatted Date"])
        df = df.set_index("Formatted Date").sort_index()
    except Exception:
        st.error("❌ Could not parse dates. Make sure your CSV has a 'Formatted Date' column.")
        st.stop()

    cols = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
            'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']

    if not all(col in df.columns for col in cols):
        st.error(f"❌ CSV must contain these columns: {cols}")
    else:
        # Scale features
        scaled = scaler.transform(df[cols])

        # Start with last SEQ_LEN values
        seq_input = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, len(cols))

        forecasts = []
        seq = seq_input.copy()

        # Use last known exogenous features (except temp)
        last_known_features = df[cols].iloc[-1].values[1:]

        for _ in range(n_forecast):
            pred = model.predict(seq, verbose=0)

            # Combine predicted temp with last known exogenous values
            temp_pred = np.hstack([pred, last_known_features.reshape(1, -1)])
            forecast = scaler.inverse_transform(temp_pred)[:, 0][0]
            forecasts.append(forecast)

            # Update sequence (autoregressive loop)
            next_step = np.hstack([pred, last_known_features.reshape(1, -1)])
            seq = np.append(seq[:, 1:, :], scaler.transform(next_step).reshape(1, 1, -1), axis=1)

        # Forecast DataFrame with real dates
        forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                       periods=n_forecast, freq="D")
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecasted Temp (°C)": forecasts
        })

        # --- Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "🔮 Forecast Table", "📈 Forecast Plot", "✅ Metrics"])

        with tab1:
            st.subheader("Uploaded Data")
            st.dataframe(df.head(20))

        with tab2:
            st.subheader("Forecasted Temperatures")
            st.dataframe(forecast_df)

        with tab3:
            st.subheader("Forecast Plot")
            fig = go.Figure()
            # Actual last 30 days
            fig.add_trace(go.Scatter(
                x=df.index[-30:], y=df['Temperature (C)'].values[-30:],
                mode="lines+markers", name="Actual Temp", line=dict(color="#42a5f5")
            ))
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecasts,
                mode="lines+markers", name="Forecast", line=dict(color="#0d47a1")
            ))
            fig.update_layout(
                title=f"{n_forecast}-Day Temperature Forecast",
                xaxis_title="Date", yaxis_title="Temp (°C)",
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.8)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("Forecast Accuracy (if actuals available)")
            try:
                actual_future = df['Temperature (C)'].values[-n_forecast:]
                mae = mean_absolute_error(actual_future, forecasts)
                rmse = np.sqrt(mean_squared_error(actual_future, forecasts))
                mape = np.mean(np.abs((actual_future - forecasts) / actual_future)) * 100

                st.metric("MAE", f"{mae:.2f}")
                st.metric("RMSE", f"{rmse:.2f}")
                st.metric("MAPE", f"{mape:.2f}%")
            except Exception:
                st.info("Upload future actuals in CSV to see accuracy metrics.")
else:
    st.info("⬅️ Upload a CSV file in the sidebar to get started.")
