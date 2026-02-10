import streamlit as st
import pandas as pd
import numpy as np
import os
from pages.utils.hybrid_predictor import get_hybrid_forecast, HISTORY_FILE

st.set_page_config(
    page_title="Stakeholder Insights",
    page_icon="💡",
    layout="wide",
)

st.title("💡 Stakeholder Insights & Recommendations")
st.markdown("""
This page analyzes a 90-day forecast to provide tailored insights for different time horizons.
""")

def analyze_forecast(forecast_series, historical_series):
    """
    Performs an advanced analysis on a given forecast series.
    Returns a dictionary of metrics.
    """
    analysis = {}
    
    if forecast_series.empty or len(forecast_series) < 2:
        return {
            'trend': 'N/A', 'momentum': 0, 'volatility': 'N/A', 
            'start': 0, 'end': 0, 'peak': 0, 'trough': 0, 'pct_change': 0
        }

    analysis['start'] = forecast_series.iloc[0]
    analysis['end'] = forecast_series.iloc[-1]
    analysis['peak'] = forecast_series.max()
    analysis['trough'] = forecast_series.min()
    analysis['pct_change'] = ((analysis['end'] - analysis['start']) / analysis['start']) * 100

    x = np.arange(len(forecast_series))
    y = forecast_series.values
    analysis['momentum'] = np.polyfit(x, y, 1)[0]
    
    if analysis['momentum'] > 0.1: 
        analysis['trend'] = "Upward 🔼"
    elif analysis['momentum'] < -0.1: 
        analysis['trend'] = "Downward 🔽"
    else:
        analysis['trend'] = "Stable / Sideways ➖"

    try:
        forecast_vol = np.std(forecast_series)
        # Use last 90 days of history as a baseline
        history_vol = np.std(historical_series.iloc[-90:])
        
        if forecast_vol > (history_vol * 1.5):
            analysis['volatility'] = "High ⚠️"
        elif forecast_vol < (history_vol * 0.75):
            analysis['volatility'] = "Low ✅"
        else:
            analysis['volatility'] = "Normal ⚖️"
    except Exception:
        analysis['volatility'] = "N/A"

    return analysis

if not os.path.exists(HISTORY_FILE):
    st.error("Model files not found! Please go to the 'Stock Prediction' page and train the models first.")
    st.stop()

DEFAULT_HORIZON = 90 
if "forecast_result" not in st.session_state or \
   st.session_state.forecast_result is None or \
   len(st.session_state.forecast_result) < DEFAULT_HORIZON:
    
    with st.spinner(f"Forecast not found or is too short. Generating a new {DEFAULT_HORIZON}-day forecast..."):
        try:
            forecast_res, history_res = get_hybrid_forecast(DEFAULT_HORIZON)
            if forecast_res is None:
                st.error("Model files are missing. Please go to the 'Stock Prediction' page and click 'Train Initial Models'.")
                st.stop()
            st.session_state.forecast_result = forecast_res
            st.session_state.history_data = history_res
            st.success(f"Generated {DEFAULT_HORIZON}-day forecast.")
        except Exception as e:
            st.error(f"Could not generate forecast: {e}")
            st.stop()

try:
    forecast_all = st.session_state.forecast_result
    history_all = st.session_state.history_data
    
    forecast_short = forecast_all.iloc[:min(7, len(forecast_all))]
    forecast_medium = forecast_all.iloc[:min(30, len(forecast_all))]
    forecast_long = forecast_all 
    analysis_short = analyze_forecast(forecast_short, history_all)
    analysis_medium = analyze_forecast(forecast_medium, history_all)
    analysis_long = analyze_forecast(forecast_long, history_all)

except Exception as e:
    st.error(f"Could not analyze forecast data. Error: {e}")
    st.stop()

st.markdown("---")
st.subheader("Recommendations Based on Forecast Time Horizon")

tab_investor, tab_retailer, tab_producer = st.tabs(["📈 Investor (Short-Term)", "☕ Retailer (Medium-Term)", "🧑‍🌾 Producer (Long-Term)"])

with tab_investor:
    st.header(f"Investor / Trader Insights (Next {len(forecast_short)} Days)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trend", analysis_short['trend'])
    col2.metric("Volatility", analysis_short['volatility'])
    col3.metric("Forecasted Peak", f"${analysis_short['peak']:,.2f}")
    col4.metric("Forecasted Trough", f"${analysis_short['trough']:,.2f}")

    if analysis_short['trend'] == "Upward 🔼":
        st.success("**Outlook: BULLISH**")
        st.write(f"""
        **Potential Strategies:**
        * **Long Positions:** This suggests an opportunity for short-term long positions.
        * **Key Levels:** Prices are expected to trade between \t ${analysis_short['trough']:,.2f} \tand\t ${analysis_short['peak']:,.2f}.
        * **Volatility:** The market is predicted to be **{analysis_short['volatility']}**. If "High", consider using wider stops or reducing position size. If "Low", a breakout trade might be forming.
        """)
    elif analysis_short['trend'] == "Downward 🔽":
        st.error("**Outlook: BEARISH**")
        st.write(f"""
        **Potential Strategies:**
        * **Short Positions:** This suggests an opportunity for short-term short positions or buying put options.
        * **Key Levels:** Prices are expected to trade between \t ${analysis_short['trough']:,.2f} \t and \t ${analysis_short['peak']:,.2f}.
        * **Volatility:** The market is predicted to be **{analysis_short['volatility']}**. High volatility in a downtrend can be very risky; manage positions accordingly.
        """)
    else: # Stable
        st.info("**Outlook: NEUTRAL / SIDEWAYS**")
        st.write(f"""
        **Potential Strategies:**
        * **Range Trading:** The market is expected to be range-bound between ${analysis_short['trough']:,.2f} and ${analysis_short['peak']:,.2f}. This may favor options-selling strategies (like iron condors) that profit from low volatility.
        * **Volatility:** The predicted volatility is **{analysis_short['volatility']}**. If "Low", be cautious of a potential breakout.
        """)

with tab_retailer:
    st.header(f"Retailer / Coffee Shop Insights (Next {len(forecast_medium)} Days)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trend", analysis_medium['trend'])
    col2.metric("Volatility", analysis_medium['volatility'])
    col3.metric("Forecasted High", f"${analysis_medium['peak']:,.2f}")
    col4.metric("Forecasted Low", f"${analysis_medium['trough']:,.2f}")

    if analysis_medium['trend'] == "Upward 🔼":
        st.error("**Action: Secure Inventory**")
        st.write(f"""
        The 30-day forecast indicates that **prices for beans are likely to rise**.
        
        **Potential Actions:**
        * **Buy Now:** Consider stocking up on inventory now to avoid higher costs later.
        * **Lock in Contracts:** Negotiate fixed-price contracts with your suppliers for the coming month.
        * **Opportunity:** The best time to buy may be near the forecasted low of ${analysis_medium['trough']:,.2f}.
        """)
    elif analysis_medium['trend'] == "Downward 🔽":
        st.success("**Action: Delay Purchases**")
        st.write(f"""
        The 30-day forecast indicates that **prices for beans are likely to fall**.
        
        **Potential Actions:**
        * **Buy Later:** Hold off on large inventory purchases. Operate on a just-in-time (JIT) basis to capitalize on falling prices.
        * **Negotiate:** Use this forecast as leverage to negotiate better pricing from your suppliers.
        * **Target Price:** Aim to purchase inventory around the forecasted low of ${analysis_medium['trough']:,.2f}.
        """)
    else: 
        st.info("**Action: Maintain Normal Operations**")
        st.write(f"""
        The 30-day forecast predicts **stable prices** with low volatility.
        
        **Potential Actions:**
        * **Standard Purchasing:** No urgent action is needed. Continue with your normal inventory purchasing cycle.
        * **Stable Margins:** Your cost of goods should remain predictable, making financial planning easier.
        """)

with tab_producer:
    st.header(f"Producer / Farmer Insights (Full {len(forecast_long)}-Day Forecast)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trend", analysis_long['trend'])
    col2.metric("Volatility", analysis_long['volatility'])
    col3.metric("Forecasted Peak", f"${analysis_long['peak']:,.2f}")
    col4.metric("Forecasted Trough", f"${analysis_long['trough']:,.2f}")

    if analysis_long['trend'] == "Upward 🔼":
        st.success("**Action: Hold Inventory / Sell Later**")
        st.write(f"""
        The {len(forecast_long)}-day forecast is **bullish**, suggesting higher prices are coming.
        
        **Potential Actions:**
        * **Delay Sales:** If possible, hold onto your inventory. The best time to sell may be near the forecasted peak of ${analysis_long['peak']:,.2f}.
        * **Avoid Contracts:** Be cautious about locking in sales at today's prices.
        * **Spot Market:** Plan to sell on the spot market to capture the highest possible price.
        """)
    elif analysis_long['trend'] == "Downward 🔽":
        st.error("**Action: Sell Inventory / Lock in Contracts**")
        st.write(f"""
        The {len(forecast_long)}-day forecast is **bearish**, suggesting prices are likely to fall.
        
        **Potential Actions:**
        * **Sell Now:** This is a strong signal to sell your current inventory to avoid lower prices.
        * **Forward Contracts:** Aggressively seek out buyers and lock in forward sales contracts for your future harvest at today's prices.
        * **Hedging:** This is a critical time to consider hedging (e.g., selling futures) to protect against declining revenue.
        """)
    else:
        st.info("**Action: Maintain Normal Sales Cycle**")
        st.write(f"""
        The {len(forecast_long)}-day forecast predicts a **stable market**.
        
        **Potential Actions:**
        * **Secure Partners:** With price being less of a factor, focus on building long-term relationships and contracts with reliable buyers.
        * **Focus on Quality:** Differentiate your product based on quality to earn premiums above the stable market price.
        """)

st.markdown("---")
