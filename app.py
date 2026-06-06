import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page Setup
st.set_page_config(
    page_title="Alpha-VCP Sector Quant",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Alpha-VCP Sector-Rotation Quant Engine")
st.markdown("### **Top-Down Matrix: RRG Sector Tailwinds + Adaptive Base Breakouts**")
st.write("---")

# ==========================================
# SIDEBAR - LOOSER OPTIMIZED PARAMETERS
# ==========================================
with st.sidebar:
    st.header("🎯 ALGO OPTIMIZATION")
    
    st.subheader("🌋 1. Catalyst (Loose Defaults)")
    min_ep_gain = st.slider("Min Breakout Day Gain (%)", 3.5, 12.0, 4.5, 0.5)
    vol_shock_factor = st.slider("Volume Shock Multiple (x SMA50)", 1.5, 6.0, 2.2, 0.1)
    
    st.subheader("📦 2. Adaptive Box Width (VCP)")
    max_allowed_box_width = st.slider("Max Post-Breakout Box Width (%)", 5.0, 25.0, 12.5, 0.5)
    lookback_window = st.slider("Scan Lookback Window (Bars)", 3, 35, 20, 1)
    
    st.subheader("🔥 3. RRG / Sector Rotation Rules")
    min_sector_cohesion = st.slider("Min Co-Sector Stocks Triggered", 1, 5, 2, 1)
    
    st.subheader("🛡️ 4. Risk & Execution Management")
    exit_logic = st.selectbox("Exit Strategy", ["Close Below 10 EMA", "Breach of Box Floor"])
    target_profit = st.slider("Fixed Take-Profit (%) [0 = Trailing Only]", 0, 100, 35, 5)

# ==========================================
# DATA POOL MATRIX LOADER
# ==========================================
@st.cache_data(ttl=86400)
def load_broad_indian_universe():
    pool = []
    try:
        url_500 = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        df_500 = pd.read_csv(url_500)
        df_500.columns = [c.upper().strip() for c in df_500.columns]
        for _, row in df_500.iterrows():
            pool.append({
                'Symbol_YF': str(row['SYMBOL']).strip() + ".NS",
                'Name': row['COMPANY NAME'],
                'Sector': row['INDUSTRY'] if 'INDUSTRY' in df_500.columns else 'Broad Market'
            })
    except Exception:
        pass
    return pd.DataFrame(pool)

master_universe = load_broad_indian_universe()
TICKERS = master_universe['Symbol_YF'].tolist()
NAME_MAP = dict(zip(master_universe['Symbol_YF'], master_universe['Name']))
SECTOR_MAP = dict(zip(master_universe['Symbol_YF'], master_universe['Sector']))

@st.cache_data(ttl=1200)
def fetch_bulk_historical_data(ticker_list):
    end_dt = datetime.today().strftime('%Y-%m-%d')
    start_dt = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
    return yf.download(ticker_list, start=start_dt, end=end_dt, interval="1d", group_by='ticker', progress=False)

tab1, tab2 = st.tabs(["🔍 Sector-Filter Scanner", "📊 Quant Alpha Backtester"])

# ==========================================
# TAB 1: LIVE SCANNER WITH SECTOR MOMENTUM
# ==========================================
with tab1:
    st.subheader("📡 Live Institutional Engine (Hot Sector Filtering)")
    if st.button("🔥 RUN LIVE SCREENING WITH SECTOR FILTER"):
        with st.spinner("Analyzing market data for simultaneous sector collection..."):
            
            chunk_size = 80
            raw_detections = []
            sector_counts = {}
            
            # Phase 1: Scan and look for volume expansion across sectors
            for i in range(0, len(TICKERS), chunk_size):
                chunk = TICKERS[i:i+chunk_size]
                market_data = fetch_bulk_historical_data(chunk)
                
                for ticker in chunk:
                    try:
                        if ticker not in market_data.columns.levels
