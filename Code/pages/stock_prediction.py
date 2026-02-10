# pages/stock_prediction.py
import streamlit as st
from pages.utils.hybrid_predictor import get_hybrid_forecast, HISTORY_FILE, load_csv_data
from pages.utils.plotly_figure import plotly_table, Moving_average_forecast 
from pages.utils.model_trainer import train_new_models 
import pandas as pd
import numpy as np
import os
import datetime

st.set_page_config(
    page_title="Stock Prediction",
    page_icon="📈", 
    layout="wide",
)

st.title("📈 Stock Prediction (Hybrid LSTM Model)")

def get_file_timestamp(filepath):
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
    except FileNotFoundError:
        return None

trained_date = get_file_timestamp(HISTORY_FILE)
live_data_date = get_file_timestamp("full_stock_data.csv")

st.sidebar.subheader("Model Status")
if trained_date:
    st.sidebar.info(f"Models last trained: \n{trained_date.strftime('%Y-%m-%d %H:%M')}")
    if live_data_date and live_data_date > trained_date:
        st.sidebar.warning("New live data is available. Re-training is recommended for accurate predictions.")
        if st.sidebar.button("Train on New Data"):
            with st.status("Training new models... This may take several minutes.", expanded=True) as status:
                try:
                    train_new_models(status) 
                    status.update(label="Training complete! Clearing cache.", state="complete", expanded=False)
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Models retrained! Click 'Generate Forecast' to use the new models.")
                    st.rerun() 
                except Exception as e:
                    status.update(label="Training Failed!", state="error", expanded=True)
                    st.error(f"An error occurred: {e}")
    else:
        st.sidebar.success("Models are up-to-date with the latest data.")
else:
    st.sidebar.error("Model files not found!")
    st.sidebar.warning("Please run the 'Stock Analysis' page once to download data, then click here to train.")
    if st.sidebar.button("Train Initial Models"):
        if not live_data_date:
            st.sidebar.error("`full_stock_data.csv` not found. Please run the 'Stock Analysis' page first.")
        else:
            with st.status("Training models... This may take several minutes.", expanded=True) as status:
                try:
                    train_new_models(status)
                    status.update(label="Training complete! Clearing cache.", state="complete", expanded=False)
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Models trained! You can now generate forecasts.")
                    st.rerun()
                except Exception as e:
                    status.update(label="Training Failed!", state="error", expanded=True)
                    st.error(f"An error occurred: {e}")


col1, col2 = st.columns([1, 3])
with col1:
    st.info("Model trained for Coffee Futures (KC=F)") 
    forecast_days = st.number_input(
        "Days to Forecast",
        min_value=1,
        max_value=365, 
        value=30, 
        step=1,
        key="forecast_days_input" 
    )

st.sidebar.subheader("Model Performance")
st.sidebar.markdown(f"""
Based on a test split during development (90 - 10):
- **RMSE:** 10.5378
- **MAE:** 7.9016
- **MAPE:** 2.89%

""")

if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None
if "history_data" not in st.session_state:
    st.session_state.history_data = None
if "forecast_days_processed" not in st.session_state:
    st.session_state.forecast_days_processed = 0

if st.button("Generate Forecast", key="predict_button") or \
   (st.session_state.forecast_days_input != st.session_state.forecast_days_processed):

    st.session_state.forecast_result = None
    st.session_state.history_data = None

    if not os.path.exists(HISTORY_FILE):
        st.error(f"Error: Model files not found at '{HISTORY_FILE}'.")
        st.error("Please use the 'Train Models' button in the sidebar.")
    else:
        with st.spinner(f"Generating {forecast_days}-day forecast... Please wait."):
            forecast_res, history_res = get_hybrid_forecast(forecast_days)
            st.session_state.forecast_result = forecast_res
            st.session_state.history_data = history_res
            st.session_state.forecast_days_processed = forecast_days

if st.session_state.forecast_result is not None and not st.session_state.forecast_result.empty:
    st.subheader(f"Forecasted Close Price for the Next {st.session_state.forecast_days_processed} Days")

    st.write('##### Forecast Data')
    forecast_df_display = st.session_state.forecast_result.to_frame(name='Forecasted_Close')
    fig_table = plotly_table(forecast_df_display.round(3))
    fig_table.update_layout(height=300) 
    st.plotly_chart(fig_table, use_container_width=True)

    st.write('##### Forecast Plot')
    if st.session_state.history_data is not None and not st.session_state.history_data.empty:
         history_data_series = st.session_state.history_data
         if isinstance(history_data_series, pd.DataFrame):
             history_data_series = history_data_series.squeeze("columns")
         if history_data_series.name != 'Close': 
             history_data_series = history_data_series.rename('Close')

         combined_data = pd.concat([history_data_series, st.session_state.forecast_result])
         if combined_data.name != 'Close': 
            combined_data = combined_data.rename('Close')

         history_days_to_plot = 180 
         plot_start_index = max(0, len(history_data_series) - history_days_to_plot)

         combined_df_plot = combined_data.iloc[plot_start_index:].to_frame() 

         fig_forecast_plot = Moving_average_forecast(combined_df_plot, forecast_len=st.session_state.forecast_days_processed)
         st.plotly_chart(fig_forecast_plot, use_container_width=True)
    else:
        st.warning("Historical data not loaded, cannot display combined plot.")

elif st.session_state.get("predict_button") and st.session_state.forecast_result is None: 
    st.error("Forecast generation failed. Please check the console output or ensure model files exist and paths are correct.")

else:
    st.write("Select the number of days and click 'Generate Forecast' to see the prediction.")
