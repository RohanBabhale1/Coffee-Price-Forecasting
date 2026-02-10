import streamlit as st

st.set_page_config(
    page_title="Trading App",
    page_icon= "heavy_dollar_sign",
    layout="wide"
)

st.title("Coffee Futures Price Analysis and Forecasting")
st.header("Course Project : Statistics For Computer Science CS309")

pg = st.navigation([
    st.Page("pages/stock_analysis.py", title="Stock Analysis", icon="📊"),
    st.Page("pages/stock_prediction.py", title="Stock Prediction", icon="📈"),
    st.Page("pages/stakeholder_insights.py", title="Stakeholder Insights", icon="💡") 
])

pg.run()