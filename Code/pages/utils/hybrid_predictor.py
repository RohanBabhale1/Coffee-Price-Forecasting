# pages/utils/hybrid_predictor.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import streamlit as st 

MODEL_DIR = 'trained_models'
HISTORY_FILE = os.path.join(MODEL_DIR, 'y_aligned_history.csv')
TIME_STEP = 60 
PERIODS = [143, 687, 3200] 

@st.cache_resource
def load_keras_model(path):
    try:
        return load_model(path, compile=False)
    except Exception as e:
        st.error(f"Error loading Keras model from {path}: {e}")
        return None

@st.cache_resource
def load_joblib_asset(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading Joblib asset from {path}: {e}")
        return None

@st.cache_data 
def load_numpy_array(path):
    try:
        return np.load(path)
    except Exception as e:
        st.error(f"Error loading NumPy array from {path}: {e}")
        return None

@st.cache_data 
def load_csv_data(path):
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
    except Exception as e:
         st.error(f"Error loading CSV data from {path}: {e}")
         return None


def forecast_lstm_recursive(component_name, future_index):
    st.write(f"  * Forecasting LSTM: {component_name}")

    model_path = os.path.join(MODEL_DIR, f'lstm_{component_name}.keras')
    scaler_path = os.path.join(MODEL_DIR, f'scaler_{component_name}.joblib')
    sequence_path = os.path.join(MODEL_DIR, f'last_sequence_{component_name}.npy')

    model = load_keras_model(model_path)
    scaler = load_joblib_asset(scaler_path)
    last_real_sequence = load_numpy_array(sequence_path)

    if model is None or scaler is None or last_real_sequence is None:
        st.warning(f"Skipping LSTM forecast for {component_name} due to loading errors.")
        return pd.Series(np.nan, index=future_index)

    forecast_horizon = len(future_index)
    future_preds_scaled = []
    current_batch = last_real_sequence.reshape((1, TIME_STEP, 1))

    tf.get_logger().setLevel('ERROR')
    original_autograph_level = tf.autograph.set_verbosity(0) 

    for i in range(forecast_horizon):
        next_pred_scaled = model.predict(current_batch, verbose=0)
        future_preds_scaled.append(next_pred_scaled[0, 0])
        
        new_sequence = np.roll(current_batch, -1, axis=1) 
        new_sequence[0, -1, 0] = next_pred_scaled[0, 0] 
        current_batch = new_sequence

    tf.autograph.set_verbosity(original_autograph_level)
    tf.get_logger().setLevel('INFO') 

    predictions_scaled = np.array(future_preds_scaled).reshape(-1, 1)
    try:
        final_predictions = scaler.inverse_transform(predictions_scaled)
    except Exception as e:
         st.error(f"Error inverse transforming predictions for {component_name}: {e}")
         return pd.Series(np.nan, index=future_index)

    return pd.Series(final_predictions.flatten(), index=future_index)


def forecast_seasonal_naive_future(period, future_index):
    st.write(f"  * Forecasting Naive: Period {period}")

    naive_data_path = os.path.join(MODEL_DIR, f'naive_data_p{period}.csv')
    last_cycle = load_csv_data(naive_data_path)

    if last_cycle is None or last_cycle.empty:
        st.warning(f"Skipping Naive forecast for period {period} due to loading errors.")
        return pd.Series(np.nan, index=future_index)

    forecast_horizon = len(future_index)
    
    n_repeats = (forecast_horizon // period) + 1
    naive_forecast_vals = np.tile(last_cycle.values, n_repeats)[:forecast_horizon]

    return pd.Series(naive_forecast_vals, index=future_index)

@st.cache_data 
def get_hybrid_forecast(forecast_horizon_days: int):
    st.info("Loading models and generating forecast...")

    y_aligned_history = load_csv_data(HISTORY_FILE)
    if y_aligned_history is None or y_aligned_history.empty:
        st.error("Failed to load historical data. Cannot proceed.")
        return None, None 

    full_data_len = len(y_aligned_history)
    last_real_date = y_aligned_history.index[-1]

    future_index = pd.date_range(
        start=last_real_date + pd.Timedelta(days=1),
        periods=forecast_horizon_days
    )

    all_forecasts = {}

    st.write("* Forecasting Trend...")
    trend_model_path = os.path.join(MODEL_DIR, 'trend_model.joblib')
    poly_reg = load_joblib_asset(trend_model_path)
    if poly_reg is None:
         st.warning("Skipping Trend forecast due to loading error.")
         all_forecasts['Trend'] = pd.Series(np.nan, index=future_index)
    else:
        X_future_lr = np.arange(full_data_len, full_data_len + forecast_horizon_days).reshape(-1, 1)
        try:
            trend_forecast = poly_reg.predict(X_future_lr)
            all_forecasts['Trend'] = pd.Series(trend_forecast, index=future_index)
        except Exception as e:
            st.error(f"Error predicting trend: {e}")
            all_forecasts['Trend'] = pd.Series(np.nan, index=future_index)

    st.write("* Forecasting Seasonal_143...")
    lstm_s143_forecast = forecast_lstm_recursive('Seasonal_143', future_index)
    naive_s143_forecast = forecast_seasonal_naive_future(PERIODS[0], future_index) 
    all_forecasts['Seasonal_143'] = 0.5 * lstm_s143_forecast.fillna(0) + 0.5 * naive_s143_forecast.fillna(0)

    st.write("* Forecasting Seasonal_687...")
    lstm_s687_forecast = forecast_lstm_recursive('Seasonal_687', future_index)
    naive_s687_forecast = forecast_seasonal_naive_future(PERIODS[1], future_index) 
    all_forecasts['Seasonal_687'] = 0.5 * lstm_s687_forecast.fillna(0) + 0.5 * naive_s687_forecast.fillna(0)

    st.write("* Forecasting Seasonal_3200...")
    all_forecasts['Seasonal_3200'] = forecast_lstm_recursive('Seasonal_3200', future_index)

    st.write("* Forecasting Residual...")
    all_forecasts['Residual'] = forecast_lstm_recursive('Residual', future_index)

    st.write("* Recombining components...")
    forecast_df = pd.DataFrame(all_forecasts)

    if forecast_df.isnull().all().all():
        st.error("All forecast components failed or resulted in NaNs.")
        return None, y_aligned_history

    final_prediction_pd = forecast_df.fillna(0).sum(axis=1)
    final_prediction_pd.name = 'Close'

    st.success("Forecast generation complete.")
    return final_prediction_pd, y_aligned_history