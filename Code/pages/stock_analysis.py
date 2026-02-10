import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from pages.utils.plotly_figure import * 
import os
import pandas_ta as ta 

st.set_page_config(
    page_title="Stock Analysis",
    page_icon="bar_chart",
    layout="wide",
)

st.title("Stock Analysis")

today = datetime.date.today()

ticker = st.text_input("Stock Ticker", "KC=F")
st.subheader(ticker)

csv_file = "full_stock_data.csv"
try:
    with st.spinner(f"Downloading latest data for {ticker}..."):
        full_data = yf.download(ticker, period='max')
        if full_data.empty:
            st.error(f"No data returned for ticker {ticker}. Please check the ticker symbol.")
            st.stop()
        
        full_data.to_csv(csv_file)

    df = pd.read_csv(csv_file)

    if 'Price' in df.columns:
        df = pd.read_csv(csv_file, skiprows=[1, 2])
        df = df.rename(columns={'Price': 'Date'})
        df.to_csv(csv_file, index=False) # Re-save without index
        
        data = pd.read_csv(csv_file)
    else:
        data = pd.read_csv(csv_file)

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    else:
        st.error("Could not find 'Date' column after processing. Please check CSV.")
        st.stop()

except Exception as e:
    st.error(f"Error downloading or processing data: {e}")
    st.stop()

col1, col2 = st.columns(2)
last_10_df = data.tail(10).round(3)
fig_df = plotly_table(last_10_df) 
st.write('Historical Data (Last 10 days)')
st.plotly_chart(fig_df, use_container_width=True)

col1, col2, col3 = st.columns(3)
if len(data['Close']) >= 2:
    daily_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
    col1.metric("Daily Change", str(round(data['Close'].iloc[-1], 2)), str(round(daily_change, 2)))
else:
    col1.metric("Daily Change", f"{data['Close'].iloc[-1]:.2f}", "N/A")


col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,1,1,1,1,1])
num_period = ''

with col1:
    if st.button('5D'):
        num_period = '5d'
with col2:
    if st.button('1M'):
        num_period = '1mo'
with col3:
    if st.button('6M'):
        num_period = '6mo'
with col4:
    if st.button('YTD'):
        num_period = 'ytd'
with col5:
    if st.button('1Y'):
        num_period = '1y'
with col6:
    if st.button('5Y'):
        num_period = '5y'
with col7:
    if st.button('MAX'):
        num_period = 'max'

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    chart_type = st.selectbox('', ('Candle', 'Line'))
with col2:
    if chart_type == 'Candle':
        indicators = st.selectbox('', ('RSI', 'MACD'))
    else:
        indicators = st.selectbox('', ('RSI', 'Moving Average', 'MACD'))

def extract_period_data(df, period):
    df_indexed = df.copy() 
    if period == '5d':
        return df_indexed.tail(5)
    elif period == '1mo':
        return df_indexed.tail(30)
    elif period == '6mo':
         return df_indexed.tail(180)
    elif period == 'ytd':
        return df_indexed[df_indexed.index >= f"{today.year}-01-01"]
    elif period == '1y':
        return df_indexed.tail(365)
    elif period == '5y':
        return df_indexed.tail(1825)
    elif period == 'max':
        return df_indexed  
    else:
        return df 


filtered_data = extract_period_data(data, num_period)


if chart_type == 'Candle' and indicators == 'RSI':
    st.plotly_chart(candlestick(filtered_data, num_period), use_container_width=True)
    st.plotly_chart(RSI(filtered_data, num_period), use_container_width=True)

if chart_type == 'Candle' and indicators == "MACD":
    st.plotly_chart(candlestick(filtered_data, num_period), use_container_width=True)
    st.plotly_chart(MACD(filtered_data, num_period), use_container_width=True)

if chart_type == 'Line' and indicators == 'RSI':
    st.plotly_chart(close_chart(filtered_data, num_period), use_container_width=True)
    st.plotly_chart(RSI(filtered_data, num_period), use_container_width=True)

if chart_type == 'Line' and indicators == 'Moving Average':
    st.plotly_chart(Moving_average(filtered_data, num_period), use_container_width=True)

if chart_type == 'Line' and indicators == 'MACD':
     st.plotly_chart(close_chart(filtered_data, num_period), use_container_width=True)
     st.plotly_chart(MACD(filtered_data, num_period), use_container_width=True)