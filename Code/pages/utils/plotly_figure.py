# pages/utils/plotly_figure.py
import plotly.graph_objects as go
import dateutil.relativedelta
import pandas_ta as pta 
import datetime
import pandas as pd
import numpy as np 
import streamlit as st 

def plotly_table(df):
    """ Renders a DataFrame as a Plotly Figure table. """
    fig = go.Figure(data=[go.Table(
        header=dict(values=["<b>Date</b>"] + ["<b>" + str(i) + "</b>" for i in df.columns], 
                    line_color='#0078ff', fill_color='#0d1f3b',
                    align='center', font=dict(color='white', size=15), height=35),
        cells=dict(values=[df.index.strftime('%Y-%m-%d') if isinstance(df.index, pd.DatetimeIndex) else df.index] + [df[col] for col in df.columns], 
                   fill_color=[['#e1efff', '#f8fafd'] * (len(df) // 2 + 1)],
                   align='left', line_color='white',
                   font=dict(color='black', size=14))
    )])
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), title="Data Table")
    return fig

def filter_data(dataframe, num_period):
    """ Filters the dataframe based on the selected period. """
    dataframe = dataframe.copy()
    latest = dataframe.index[-1]

    if num_period == '1mo':
        date = latest - dateutil.relativedelta.relativedelta(months=1)
    elif num_period == '5mo':
        date = latest - dateutil.relativedelta.relativedelta(months=5)
    elif num_period == '6mo':
        date = latest - dateutil.relativedelta.relativedelta(months=6)
    elif num_period == '1y':
        date = latest - dateutil.relativedelta.relativedelta(years=1)
    elif num_period == '5y':
        date = latest - dateutil.relativedelta.relativedelta(years=5)
    elif num_period == 'ytd':
        date = datetime.datetime(latest.year, 1, 1).date() 
    else: 
        return dataframe 

    filtered_df = dataframe.loc[date:]
    if filtered_df.empty:
        return dataframe.tail(30)
    return filtered_df

def close_chart(dataframe, num_period=False):
    if num_period:
        dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe['Open'], mode='lines', name='Open', line=dict(width=2, color='#5ab7ff')))
    fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe['Close'], mode='lines', name='Close', line=dict(width=2, color='black')))
    fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe['High'], mode='lines', name='High', line=dict(width=2, color='#0078ff')))
    fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe['Low'], mode='lines', name='Low', line=dict(width=2, color='red')))
    fig.update_xaxes(title_text="Date", rangeslider_visible=True)
    fig.update_yaxes(title_text="Price")
    fig.update_layout(
        height=500, margin=dict(l=50, r=20, t=40, b=50),
        plot_bgcolor='white', paper_bgcolor='#e1efff',
        legend=dict(orientation='h', yanchor='top', y=1.02, xanchor='right', x=1),
        title="Stock Prices Over Time"
    )
    return fig

def candlestick(dataframe, num_period):
    dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dataframe.index, open=dataframe['Open'], high=dataframe['High'],
        low=dataframe['Low'], close=dataframe['Close'], name="Candlestick"
    ))
    fig.update_xaxes(title_text="Date", rangeslider_visible=True)
    fig.update_yaxes(title_text="Price")
    fig.update_layout(
        showlegend=True, height=500, margin=dict(l=50, r=20, t=40, b=50),
        plot_bgcolor='white', paper_bgcolor='#e1efff', title="Candlestick Chart"
    )
    return fig

def RSI(dataframe, num_period):
    df_copy = dataframe.copy() 
    if num_period:
        df_copy = filter_data(df_copy, num_period)
    
    if df_copy.empty or len(df_copy) < 14:
        st.warning(f"Not enough data to calculate RSI for {num_period}.")
        return go.Figure()
        
    df_copy['RSI'] = pta.rsi(df_copy['Close'], length=14) 

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_copy.index, y=df_copy['RSI'], mode='lines', name='RSI', line=dict(width=2, color='orange')))
    fig.add_trace(go.Scatter(x=df_copy.index, y=[70]*len(df_copy), mode='lines', name='Overbought', line=dict(width=2, color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df_copy.index, y=[30]*len(df_copy), fill='tonexty', mode='lines', name='Oversold', line=dict(width=2, color='#79da84', dash='dash')))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="RSI Value", range=[0, 100])
    fig.update_layout(
        height=250, margin=dict(l=50, r=20, t=40, b=50),
        plot_bgcolor='white', paper_bgcolor='#e1efff',
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),
        title="Relative Strength Index (RSI)"
    )
    return fig

def Moving_average(dataframe, num_period):
    df_copy = dataframe.copy() 
    if num_period:
        df_copy = filter_data(df_copy, num_period)
        
    if df_copy.empty:
        st.warning(f"Not enough data to calculate Moving Averages for {num_period}.")
        return go.Figure()
        
    df_copy['SMA_50'] = pta.sma(df_copy['Close'], 50) 
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_copy.index, y=df_copy['Open'], mode='lines', name='Open', line=dict(width=2, color='#5ab7ff')))
    fig.add_trace(go.Scatter(x=df_copy.index, y=df_copy['Close'], mode='lines', name='Close', line=dict(width=2, color='black')))
    fig.add_trace(go.Scatter(x=df_copy.index, y=df_copy['High'], mode='lines', name='High', line=dict(width=2, color='#0078ff')))
    fig.add_trace(go.Scatter(x=df_copy.index, y=df_copy['Low'], mode='lines', name='Low', line=dict(width=2, color='red')))
    fig.add_trace(go.Scatter(x=df_copy.index, y=df_copy['SMA_50'], mode='lines', name='SMA 50', line=dict(width=2, color='purple')))
    fig.update_xaxes(title_text="Date", rangeslider_visible=True)
    fig.update_yaxes(title_text="Price")
    fig.update_layout(
        height=500, margin=dict(l=50, r=20, t=44, b=50), 
        plot_bgcolor='white', paper_bgcolor='#e1efff',
        legend=dict(orientation='h', yanchor='top', y=1.02, xanchor="right", x=1),
        title="Moving Average (SMA 50)"
    )
    return fig

def MACD(dataframe, num_period=False):
    df_copy = dataframe.copy() 
    if num_period:
        df_copy = filter_data(df_copy, num_period)

    if df_copy.empty or len(df_copy) < 26: 
        st.warning(f"Not enough data to calculate MACD for {num_period}.")
        return go.Figure()
        
    macd = pta.macd(df_copy['Close'], fast=12, slow=26, signal=9) 
    if macd is None or macd.empty:
        st.warning(f"Could not calculate MACD for {num_period}.")
        return go.Figure()
        
    df_copy['MACD'] = macd['MACD_12_26_9']
    df_copy['MACD_Signal'] = macd['MACDs_12_26_9']
    df_copy['MACD_Hist'] = macd['MACDh_12_26_9']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_copy.index, y=df_copy['MACD'], mode='lines', name='MACD', line=dict(width=2, color='orange')))
    fig.add_trace(go.Scatter(x=df_copy.index, y=df_copy['MACD_Signal'], mode='lines', name='Signal', line=dict(width=2, color='red', dash='dash')))
    colors = ['green' if val >= 0 else 'red' for val in df_copy['MACD_Hist']]
    fig.add_trace(go.Bar(x=df_copy.index, y=df_copy['MACD_Hist'], name='Histogram', marker_color=colors))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="MACD Value")
    fig.update_layout(
        height=350, margin=dict(l=50, r=20, t=40, b=50),
        plot_bgcolor='white', paper_bgcolor='#e1efff',
        legend=dict(orientation='h', yanchor='top', y=1.02, xanchor='right', x=1),
        title="MACD Indicator"
    )
    return fig

def Moving_average_forecast(forecast, forecast_len=30):
    """
    Plot historical data and forecast for the prediction page.
    """
    fig = go.Figure()
    
    num_total_points = len(forecast)
    num_history_points = num_total_points - forecast_len
    
    if num_history_points <= 0:
        fig.add_trace(go.Scatter(
            x=forecast.index, y=forecast['Close'], mode='lines',
            name='Forecasted Close Price', line=dict(width=2, color='red', dash='dash')
        ))
    else:
        fig.add_trace(go.Scatter(
            x=forecast.index[:num_history_points], y=forecast['Close'].iloc[:num_history_points],
            mode='lines', name='Historical Close Price', line=dict(width=2, color='black')
        ))
        fig.add_trace(go.Scatter(
            x=forecast.index[num_history_points-1:], y=forecast['Close'].iloc[num_history_points-1:], 
            mode='lines', name='Forecasted Close Price', line=dict(width=2, color='red', dash='dash')
        ))

    fig.update_xaxes(title_text="Date", rangeslider_visible=True)
    fig.update_yaxes(title_text="Price")
    fig.update_layout(
        height=500, margin=dict(l=50, r=20, t=40, b=50),
        plot_bgcolor='white', paper_bgcolor='#e1efff',
        legend=dict(orientation='h', xanchor="right", yanchor="top", y=1.02, x=1),
        title=f"Stock Price Forecast (Historical + {forecast_len} Days Prediction)"
    )
    return fig