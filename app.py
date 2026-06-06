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
    page_title="Alpha-VCP Quant Suite",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Alpha-VCP Institutional Quant Trading Engine")
st.markdown("### **Multi-Asset Structural Scanner & Vectorized Portfolio Backtester**")
st.write("---")

# ==========================================
# SIDEBAR - HYPERPARAMETER OPTIMIZATION (Best Quant Defaults)
# ==========================================
with st.sidebar:
    st.header("🎯 ALGO CONFIGURATION")
    
    st.subheader("🌋 1. Catalyst & Impulse Day")
    min_ep_gain = st.slider("Min Breakout Day Gain (%)", 4.0, 15.0, 5.5, 0.5)
    vol_shock_factor = st.slider("Volume Shock Multiple (x SMA50)", 2.0, 8.0, 3.0, 0.5)
    
    st.subheader("📦 2. Box Compression (VCP)")
    max_allowed_box_width = st.slider("Max Post-Breakout Box Width (%)", 4.0, 20.0, 9.5, 0.5)
    lookback_window = st.slider("Scan Lookback Window (Bars)", 3, 30, 15, 1)
    
    st.subheader("🛡️ 3. Execution & Risk Management")
    exit_logic = st.selectbox("Exit Strategy (Stop-Loss/Trailing)", ["Close Below 10 EMA", "Breach of Box Floor"])
    target_profit = st.slider("Fixed Take-Profit (%) [0 = Trailing Only]", 0, 50, 25, 5)
    
    st.write("---")
    require_strict_ma = st.checkbox("Strict Trend Filter (10>20>50>200 SMA)", value=False)

# ==========================================
# DATA ENGINE - CORES LOADING
# ==========================================
@st.cache_data(ttl=86400)
def load_broad_indian_universe():
    pool = []
    # Fetch Nifty 500
    try:
        url_500 = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        df_500 = pd.read_csv(url_500)
        df_500.columns = [c.upper().strip() for c in df_500.columns]
        for _, row in df_500.iterrows():
            pool.append({
                'Symbol_YF': str(row['SYMBOL']).strip() + ".NS",
                'Name': row['COMPANY NAME'],
                'Sector': row['INDUSTRY'] if 'INDUSTRY' in df_500.columns else 'Core Sector'
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

# ==========================================
# TABS DIVISION (AS USED IN INSTITUTIONAL UI)
# ==========================================
tab1, tab2 = st.tabs(["🔍 Live Market Setup Scanner", "📈 Quantitative Backtest Analytics"])

# Global state to share data between processing blocks
processed_chunks = None

with tab1:
    st.subheader("⚡ Real-Time Edge Detector (Adaptive Base Breakouts)")
    if st.button("🔥 RUN LIVE SCREENING PATTERNS"):
        with st.spinner("Decoding raw multi-asset order flows..."):
            
            chunk_size = 80
            detected_gems = []
            
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
                        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
                        df['SMA_50'] = df['Close'].rolling(window=50).mean()
                        df['Vol_SMA50'] = df['Volume'].rolling(window=50).mean()
                        df['Pct_Change'] = df['Close'].pct_change() * 100
                        
                        idx = len(df) - 1
                        current_close = float(df.iloc[idx]['Close'])
                        
                        # Trend Engine Filter
                        if require_strict_ma:
                            df['SMA_200'] = df['Close'].rolling(window=200).mean()
                            if not (df.iloc[idx]['EMA_10'] > df.iloc[idx]['EMA_20'] > df.iloc[idx]['SMA_50'] > df.iloc[idx]['SMA_200']):
                                continue
                        else:
                            if not (current_close > df.iloc[idx]['SMA_50']):
                                continue
                        
                        # Catalyst Check (Lookback for Expansion Pivot)
                        found_impulse = False
                        impulse_idx = -1
                        for lb in range(1, lookback_window + 1):
                            check_idx = idx - lb
                            if check_idx < 0: break
                            row_data = df.iloc[check_idx]
                            
                            if row_data['Pct_Change'] >= min_ep_gain and row_data['Volume'] >= (vol_shock_factor * row_data['Vol_SMA50']):
                                found_impulse = True
                                impulse_idx = check_idx
                                break
                                
                        if found_impulse:
                            post_impulse_slice = df.iloc[impulse_idx + 1: idx + 1]
                            if len(post_impulse_slice) == 0: continue
                            
                            highest_high = post_impulse_slice['High'].max()
                            lowest_low = post_impulse_slice['Low'].min()
                            box_width = ((highest_high - lowest_low) / lowest_low) * 100
                            
                            near_10ema = abs(current_close - df.iloc[idx]['EMA_10']) / df.iloc[idx]['EMA_10'] <= 0.025
                            near_support = abs(current_close - lowest_low) / lowest_low <= 0.035
                            
                            if box_width <= max_allowed_box_width:
                                label = "📈 Minervini VCP Compression"
                                if near_10ema: label = "⚡ Qullamaggie 10EMA Pullback"
                                elif near_support: label = "📦 Darvas Box Floor Support"
                                
                                detected_gems.append({
                                    "Ticker": ticker.replace('.NS', ''),
                                    "Company Name": NAME_MAP.get(ticker, "Market Asset"),
                                    "Sector": SECTOR_MAP.get(ticker, "Core Segment"),
                                    "Setup Architecture": label,
                                    "Box Consolidation Width": f"{round(box_width, 2)}%",
                                    "Catalyst Day Gain": f"+{round(df.iloc[impulse_idx]['Pct_Change'], 2)}%",
                                    "Volume Surge Mult": f"{round(df.iloc[impulse_idx]['Volume'] / df.iloc[impulse_idx]['Vol_SMA50'], 1)}x",
                                    "Current Price": f"₹{round(current_close, 2)}"
                                })
                    except Exception:
                        continue
                        
            if detected_gems:
                st.success(f"💎 Found {len(detected_gems)} Institutional Structures Match Criteria!")
                st.dataframe(pd.DataFrame(detected_gems), use_container_width=True)
            else:
                st.warning("No patterns found. Optimize configuration bounds via the sidebar parameters.")

with tab2:
    st.subheader("📊 Backtester Engine: Performance Attribution Analytics")
    st.markdown("This simulation scans **Nifty 500 components** dynamically over the past **24 Months** applying your custom parameters to track hypothetical equity curves.")
    
    if st.button("🚀 EXECUTE VECTORIZED PORTFOLIO BACKTEST"):
        with st.spinner("Backtesting all setups over historical timelines..."):
            
            # Select a stable subset of liquid assets for lightning fast matrix processing
            test_subset = TICKERS[:120] 
            bulk_history = fetch_bulk_historical_data(test_subset)
            
            all_trades = []
            portfolio_daily_returns = []
            
            for ticker in test_subset:
                try:
                    if ticker not in bulk_history.columns.levels[0]: continue
                    df = bulk_history[ticker].dropna().copy()
                    if len(df) < 150: continue
                    
                    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
                    df['SMA_50'] = df['Close'].rolling(window=50).mean()
                    df['Vol_SMA50'] = df['Volume'].rolling(window=50).mean()
                    df['Pct_Change'] = df['Close'].pct_change() * 100
                    
                    in_position = False
                    entry_price = 0
                    entry_date = None
                    stop_loss_level = 0
                    
                    # Core Loop Tracking Strategy Matrices
                    for t in range(50, len(df) - 1):
                        current_row = df.iloc[t]
                        
                        if not in_position:
                            # Search for Catalyst Event
                            if current_row['Pct_Change'] >= min_ep_gain and current_row['Volume'] >= (vol_shock_factor * current_row['Vol_SMA50']):
                                # Confirm consolidation pattern structure
                                lookback_slice = df.iloc[max(0, t-lookback_window):t+1]
                                box_w = ((lookback_slice['High'].max() - lookback_slice['Low'].min()) / lookback_slice['Low'].min()) * 100
                                
                                if box_w <= max_allowed_box_width and current_row['Close'] > current_row['SMA_50']:
                                    in_position = True
                                    entry_price = float(df.iloc[t+1]['Open']) # Next Day Open Execution
                                    entry_date = df.index[t+1]
                                    stop_loss_level = float(lookback_slice['Low'].min()) if exit_logic == "Breach of Box Floor" else float(current_row['EMA_10'])
                        else:
                            # Monitor Position Exits
                            current_close = float(df.iloc[t]['Close'])
                            days_in_trade = (df.index[t] - entry_date).days
                            
                            # Exit Condition Evaluation
                            hit_sl = current_close < stop_loss_level if exit_logic == "Breach of Box Floor" else current_close < df.iloc[t]['EMA_10']
                            hit_tp = (current_close >= entry_price * (1 + target_profit/100)) if target_profit > 0 else False
                            
                            if hit_sl or hit_tp or (t == len(df) - 2):
                                exit_price = float(df.iloc[t]['Close'])
                                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                                
                                all_trades.append({
                                    "Asset": ticker.replace('.NS', ''),
                                    "Entry Date": entry_date.strftime('%Y-%m-%d'),
                                    "Exit Date": df.index[t].strftime('%Y-%m-%d'),
                                    "Holding Period": f"{days_in_trade} Days",
                                    "Trade Return (%)": round(pnl_pct, 2),
                                    "Status": "🎯 Profit" if pnl_pct > 0 else "🛑 Stop-Loss"
                                })
                                in_position = False
                except Exception:
                    continue
            
            if all_trades:
                trades_df = pd.DataFrame(all_trades)
                
                # Math Metrics Extraction
                total_trades = len(trades_df)
                win_trades = len(trades_df[trades_df['Trade Return (%)'] > 0])
                win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
                avg_return = trades_df['Trade Return (%)'].mean()
                
                # Metrics Dashboard UI
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🎯 Total Executed Trades", f"{total_trades}")
                m2.metric("📊 Win Rate Ratio", f"{round(win_rate, 2)}%")
                m3.metric("📈 Expected Return / Trade", f"{round(avg_return, 2)}%")
                
                # Sharpe Ratio Estimation Formula
                returns_array = trades_df['Trade Return (%)'] / 100
                sharpe = (returns_array.mean() / returns_array.std() * np.sqrt(252)) if returns_array.std() != 0 else 0
                m4.metric("🏆 Strategy Sharpe Ratio", f"{round(sharpe, 2)}")
                
                # Equity Curve Compilation Logic
                st.subheader("📈 Simulated Cumulative Strategy Growth")
                trades_df['Cum_Returns'] = (1 + trades_df['Trade Return (%)']/100).cumprod() - 1
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=trades_df['Entry Date'], y=trades_df['Cum_Returns'] * 100, mode='lines+markers', name='Strategy Growth Curve', line=dict(color='#00FFCC', width=2.5)))
                fig.update_layout(title="Equity Growth Path (%)", template="plotly_dark", xaxis_title="Timeline Execution", yaxis_title="Cumulative Returns (%)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📋 Execution Trade Log")
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.error("No simulated trades occurred inside the database matrix with current constraints.")
