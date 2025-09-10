Weather Forecasting with LSTM & SARIMAX

An end-to-end weather forecasting system that compares classical statistical models (SARIMAX) with deep learning (LSTM) for temperature prediction. The project also includes a Streamlit web application for interactive visualization, forecasting, and evaluation.


📌 Features

📊 Exploratory Data Analysis (EDA): Seasonal patterns, trends, and feature correlations

🤖 Modeling Approaches:

- SARIMAX (Seasonal AutoRegressive Integrated Moving Average with Exogenous variables): Captures linear trends, seasonal patterns, and relationships with external variables (humidity, pressure, etc.)

- LSTM (Long Short-Term Memory): A recurrent neural network designed to learn long-term dependencies in sequential weather data

🔄 Preprocessing Pipeline: Chronological split, MinMax scaling, sequence preparation

📈 Evaluation Metrics: MAE, RMSE, MAPE for comparing model performance

🌐 Streamlit Dashboard: Upload CSVs, generate forecasts, view accuracy metrics, and visualize results



🧠 Models

🔹 SARIMAX

Why SARIMAX?

- Handles seasonality (daily, yearly weather cycles)

- Incorporates exogenous features (humidity, wind speed, pressure, visibility)

- Works well for short-term linear patterns
  

Limitations:

- Struggles with highly nonlinear relationships

- Training can be slow for long time series with multiple seasonalities
  
  

🔹 LSTM

Why LSTM?

- Captures nonlinear patterns in weather data

- Learns long-term dependencies (e.g., yearly cycles, gradual changes)

- Performs well on multivariate forecasting (using multiple weather features together)
  

Limitations:

- Requires more data and computational resources

- Sensitive to hyperparameter tuning and scaling





📊 Results (Example)
Model	MAE (°C)	RMSE (°C)	MAPE (%)
SARIMAX	2.10	2.95	8.2%
LSTM	1.65	2.40	6.3%

(Values are illustrative — actual results depend on dataset and tuning.)



🛠️ Tech Stack

Languages & Libraries: Python, Pandas, NumPy, Scikit-learn

Statistical Modeling: Statsmodels (SARIMAX)

Deep Learning: TensorFlow / Keras (LSTM)

Visualization: Matplotlib, Seaborn, Plotly

Deployment: Streamlit




🚀 How to Run

1️⃣ Clone the repository
git clone https://github.com/yourusername/Weather-forecasting-prediction.git
cd Weather-forecasting-prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app.py



📊 Usage

- Upload a CSV file with at least 30 days of data

- Required columns:
['Formatted Date', 'Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']

- Select forecast horizon (1–14 days)



Outputs:

- Forecast table

- Interactive forecast plot

- Accuracy metrics (if actuals available)

  

🔮 Future Improvements

- Ensemble approach: combine SARIMAX + LSTM + Prophet

- Hyperparameter tuning with Optuna

- Deployment via Docker/Hugging Face Spaces

- Add AQI (Air Quality Index) prediction




📜 License

This project is licensed under the MIT License.
