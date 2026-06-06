import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# PAGE CONFIGURATION & QUANT THEME
# ==========================================
st.set_page_config(
    page_title="Alpha-VCP Dynamic Quant",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Alpha-VCP Matrix with Financial Metrics")
st.markdown("### **Top-Down Matrix: RRG Sector Tailwinds + Historical Quarterly Sales Track**")
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
    
    st.subheader("🛡️ 4. Risk & Advanced Profit Booking")
    st.info("💡 Exit Rules Locked:\n1. 20 EMA Breach + High Volume\n2. Candle Close below 30 EMA")

# ==========================================
# DATA POOL MATRIX LOADER (2500+ NSE Universe)
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
        
    try:
        url_all = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df_all = pd.read_csv(url_all)
        df_all.columns = [c.upper().strip() for c in df_all.columns]
        if 'SERIES' in df_all.columns:
            df_all = df_all[df_all['SERIES'].astype(str).str.upper().str.strip() == 'EQ']
        existing = {x['Symbol_YF'] for x in pool}
        for _, row in df_all.iterrows():
            sym = str(row['SYMBOL']).strip()
            if sym and sym.lower() != 'symbol' and (sym + ".NS") not in existing:
                pool.append({
                    'Symbol_YF': sym + ".NS",
                    'Name': row['NAME OF COMPANY'] if 'NAME OF COMPANY' in df_all.columns else sym,
                    'Sector': 'Broad Market'
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

# Fundamental Extraction for Display Only
def calculate_historical_sales_growth(ticker_symbol):
    """
    Fetches raw financials and calculates the percentage growth 
    between the two most recent quarters.
    """
    try:
        t_obj = yf.Ticker(ticker_symbol)
        quarterly_funda = t_obj.quarterly_financials
        
        revenue_labels = ['Total Revenue', 'Gross Sales', 'Operating Revenue']
        target_label = None
        for label in revenue_labels:
            if label in quarterly_funda.index:
                target_label = label
                break
                
        if target_label is not None:
            rev_series = quarterly_funda.loc[target_label].dropna()
            if len(rev_series) >= 2:
                # Index 0 = Latest Quarter, Index 1 = Previous Quarter
                latest_q_sales = float(rev_series.iloc[0])
                prev_q_sales = float(rev_series.iloc[1])
                
                if prev_q_sales > 0:
                    growth = ((latest_q_sales - prev_q_sales) / prev_q_sales) * 100
                    return f"{round(growth, 2)}%"
    except Exception:
        pass
    return "Data N/A"

tab1, tab2 = st.tabs(["🔍 Sector-Filter Scanner", "📊 Quant Alpha Backtester"])

# ==========================================
# TAB 1: LIVE SCANNER WITH INFO-ONLY SALES COLUMN
# ==========================================
with tab1:
    st.subheader("📡 Live Institutional Engine (Hot Sector Filtering)")
    if st.button("🔥 RUN LIVE SCREENING WITH SECTOR FILTER"):
        with st.spinner("Scanning market and fetching fundamental metrics..."):
            
            chunk_size = 80
            raw_detections = []
            sector_counts = {}
            
            # Phase 1: Technical & Structural Check
            for i in range(0, len(TICKERS), chunk_size):
                chunk = TICKERS[i:i+chunk_size]
                market_data = fetch_bulk_historical_data(chunk)
                
                for ticker in chunk:
                    try:
                        if ticker not in market_data.columns.levels[0]: 
                            continue
                            
                        df = market_data[ticker].dropna()
                        if len(df) < 100: 
                            continue
                        
                        df = df.copy()
                        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
                        df['SMA_50'] = df['Close'].rolling(window=50).mean()
                        df['Vol_SMA50'] = df['Volume'].rolling(window=50).mean()
                        df['Pct_Change'] = df['Close'].pct_change() * 100
                        
                        idx = len(df) - 1
                        current_close = float(df.iloc[idx]['Close'])
                        
                        if current_close < df.iloc[idx]['SMA_50']: 
                            continue
                        
                        # Find Catalyst Day
                        found_impulse = False
                        impulse_idx = -1
                        for lb in range(1, lookback_window + 1):
                            check_idx = idx - lb
                            if check_idx < 0: break
                            if df.iloc[check_idx]['Pct_Change'] >= min_ep_gain and df.iloc[check_idx]['Volume'] >= (vol_shock_factor * df.iloc[check_idx]['Vol_SMA50']):
                                found_impulse = True
                                impulse_idx = check_idx
                                break
                                
                        if found_impulse:
                            post_slice = df.iloc[impulse_idx + 1: idx + 1]
                            if len(post_slice) == 0: continue
                            
                            box_w = ((post_slice['High'].max() - post_slice['Low'].min()) / post_slice['Low'].min()) * 100
                            
                            if box_w <= max_allowed_box_width:
                                current_sector = SECTOR_MAP.get(ticker, "Broad Market")
                                sector_counts[current_sector] = sector_counts.get(current_sector, 0) + 1
                                
                                raw_detections.append({
                                    "Ticker_Raw": ticker,
                                    "Ticker": ticker.replace('.NS', ''),
                                    "Company Name": NAME_MAP.get(ticker, "Unknown Asset"),
                                    "Sector": current_sector,
                                    "Box Width (%)": round(box_w, 2),
                                    "Breakout Jump": f"+{round(df.iloc[impulse_idx]['Pct_Change'], 2)}%",
                                    "Volume Shock": f"{round(df.iloc[impulse_idx]['Volume'] / df.iloc[impulse_idx]['Vol_SMA50'], 1)}x",
                                    "Price": current_close
                                })
                    except Exception: 
                        continue
            
            # Phase 2: Live Sector Allocation & Dynamic Column Ingestion
            final_filtered_gems = []
            
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            for index, item in enumerate(raw_detections):
                sec = item["Sector"]
                status_text.text(f"Processing data column for: {item['Ticker']}...")
                
                # Check Sector Cohesion Rule
                if sector_counts.get(sec, 0) >= min_sector_cohesion:
                    # Fetch and fill column WITHOUT dropping the asset if sales are low
                    sales_pct = calculate_historical_sales_growth(item["Ticker_Raw"])
                    item["Sales Growth (Prev 2 Qtrs)"] = sales_pct
                    
                    item["Sector Strength Indicators"] = f"🔥 HOT SECTOR ({sector_counts[sec]} Assets Active)"
                    final_filtered_gems.append(item)
                    
                progress_bar.progress((index + 1) / len(raw_detections))
            
            status_text.empty()
            progress_bar.empty()
                    
            if final_filtered_gems:
                st.success(f"💎 Found {len(final_filtered_gems)} Technical Alpha Assets!")
                # Convert to dataframe and clean tracking columns
                final_df = pd.DataFrame(final_filtered_gems).drop(columns=['Ticker_Raw'])
                
                # Reordering columns to bring fundamental analytics to sight
                cols = list(final_df.columns)
                if "Sales Growth (Prev 2 Qtrs)" in cols:
                    cols.insert(4, cols.pop(cols.index("Sales Growth (Prev 2 Qtrs)")))
                    final_df = final_df[cols]
                
                st.dataframe(final_df, use_container_width=True)
            else:
                st.warning("No assets matched the combined technical structure and sector filters.")

# ==========================================
# TAB 2: PORTFOLIO BACKTESTER ENGINE WITH ADVANCED EXITS
# ==========================================
with tab2:
    st.subheader("📊 Dynamic Matrix Institutional Backtester")
    if st.button("🚀 EXECUTE SECTOR-VALIDATED BACKTEST"):
        with st.spinner("Running historical simulations using customized trailing exit boundaries..."):
            
            test_subset = TICKERS[:150] 
            bulk_history = fetch_bulk_historical_data(test_subset)
            
            all_trades = []
            
            for ticker in test_subset:
                try:
                    if ticker not in bulk_history.columns.levels[0]: 
                        continue
                        
                    df = bulk_history[ticker].dropna().copy()
                    if len(df) < 150: 
                        continue
                    
                    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
                    df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()
                    df['SMA_50'] = df['Close'].rolling(window=50).mean()
                    df['Vol_SMA50'] = df['Volume'].rolling(window=50).mean()
                    df['Pct_Change'] = df['Close'].pct_change() * 100
                    
                    in_position = False
                    entry_price = 0
                    entry_date = None
                    
                    for t in range(50, len(df) - 1):
                        current_row = df.iloc[t]
                        
                        if not in_position:
                            if current_row['Pct_Change'] >= min_ep_gain and current_row['Volume'] >= (vol_shock_factor * current_row['Vol_SMA50']):
                                lookback_slice = df.iloc[max(0, t-lookback_window):t+1]
                                box_w = ((lookback_slice['High'].max() - lookback_slice['Low'].min()) / lookback_slice['Low'].min()) * 100
                                
                                if box_w <= max_allowed_box_width and current_row['Close'] > current_row['SMA_50']:
                                    in_position = True
                                    entry_price = float(df.iloc[t+1]['Open']) 
                                    entry_date = df.index[t+1]
                        else:
                            current_close = float(df.iloc[t]['Close'])
                            current_volume = float(df.iloc[t]['Volume'])
                            avg_volume = float(df.iloc[t]['Vol_SMA50'])
                            ema20 = float(df.iloc[t]['EMA_20'])
                            ema30 = float(df.iloc[t]['EMA_30'])
                            
                            days_in_trade = (df.index[t] - entry_date).days
                            
                            exit_rule_1 = (current_close < ema20) and (current_volume > avg_volume)
                            exit_rule_2 = (current_close < ema30)
                            
                            if exit_rule_1 or exit_rule_2 or (t == len(df) - 2):
                                exit_price = current_close
                                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                                
                                exit_reason = "⚠️ 20 EMA High Vol Breach" if exit_rule_1 else "🛑 30 EMA Candle Close"
                                if t == len(df) - 2: exit_reason = "⏳ Horizon End"
                                
                                all_trades.append({
                                    "Asset": ticker.replace('.NS', ''),
                                    "Sector": SECTOR_MAP.get(ticker, "Broad Market"),
                                    "Entry Date": entry_date.strftime('%Y-%m-%d'),
                                    "Exit Date": df.index[t].strftime('%Y-%m-%d'),
                                    "Holding Period": f"{days_in_trade} Days",
                                    "Trade Return (%)": round(pnl_pct, 2),
                                    "Exit Reason": exit_reason,
                                    "Status": "🎯 Profit" if pnl_pct > 0 else "🛑 Loss"
                                })
                                in_position = False
                except Exception: 
                    continue
                
            if all_trades:
                trades_df = pd.DataFrame(all_trades)
                
                # Metrics Display
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🎯 Total Executed Trades", f"{len(trades_df)}")
                win_rate = (len(trades_df[trades_df['Trade Return (%)'] > 0]) / len(trades_df)) * 100
                m2.metric("📊 Net Win Rate Ratio", f"{round(win_rate, 2)}%")
                m3.metric("📈 Expected Return / Trade", f"{round(trades_df['Trade Return (%)'].mean(), 2)}%")
                
                returns_array = trades_df['Trade Return (%)'] / 100
                sharpe = (returns_array.mean() / returns_array.std() * np.sqrt(252)) if returns_array.std() != 0 else 0
                m4.metric("🏆 Strategy Sharpe Ratio", f"{round(sharpe, 2)}")
                
                # Graph Performance
                st.subheader("📈 Cumulative Strategy Returns Profile")
                trades_df['Cum_Returns'] = (1 + trades_df['Trade Return (%)']/100).cumprod() - 1
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=trades_df['Entry Date'], y=trades_df['Cum_Returns'] * 100, mode='lines+markers', name='Equity Line', line=dict(color='#06D6A0', width=2.5)))
                fig.update_layout(template="plotly_dark", title="Portfolio Performance Pathway (%)", xaxis_title="Timeline Execution", yaxis_title="Cumulative Net Gains (%)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.error("No trades executed. Adjust sidebar metrics to load simulation bounds.")
