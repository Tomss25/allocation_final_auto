import streamlit as st
import yfinance as yf
try:
    import mstarpy
    MSTARPY_AVAILABLE = True
except ImportError:
    MSTARPY_AVAILABLE = False
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import io
import itertools
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components
from sklearn.covariance import LedoitWolf
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE GLOBALE E SESSION STATE ---
st.set_page_config(page_title="Portfolio Suite", layout="wide")

if 'shared_df' not in st.session_state:
    st.session_state.shared_df = None
if 'shared_assets' not in st.session_state:
    st.session_state.shared_assets = []
if 'shared_freq' not in st.session_state:
    st.session_state.shared_freq = 252

# ==============================================================================
# BLOCCO 1: FUNZIONI ESATTE DI "ALLOCAZIONE AUTO"
# ==============================================================================
BG_PRIMARY = "#F8F9FA"
BG_SECONDARY = "#FFFFFF"
BG_CARD = "#FFFFFF"
TEXT_PRIMARY = "#1A202C"
TEXT_SECONDARY = "#4A5568"
TEXT_MUTED = "#718096"
COLOR_HIGHLIGHT = "#1A365D" 
COLOR_ACCENT = "#2C5282"
BORDER_COLOR = "#E2E8F0"
CHART_COLORS = ["#1A365D", "#2C5282", "#3182CE", "#4299E1", "#63B3ED", "#90CDF4", "#BEE3F8", "#E6F2FF"]
COLOR_GREEN = "#38A169"
COLOR_RED = "#E53E3E"
COLOR_GOLD = "#D69E2E"
BORDER_RADIUS = "12px"

PLOTLY_LAYOUT = dict(template="plotly_white", paper_bgcolor=BG_SECONDARY, plot_bgcolor=BG_SECONDARY, colorway=CHART_COLORS)

def kpi_tile(label: str, value: str, sub: str = "", positive=None):
    color_class = "positive" if positive is True else "negative" if positive is False else ""
    sub_html = f'<div style="font-size: 0.72rem; color: {TEXT_MUTED}; margin-top: 0.4rem;">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div style="background: {BG_CARD}; border: 2px solid {BORDER_COLOR}; border-radius: {BORDER_RADIUS}; padding: 1.4rem; text-align: center; margin-bottom: 1rem;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {TEXT_MUTED}; text-transform: uppercase;">{label}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: {'green' if color_class=='positive' else 'red' if color_class=='negative' else TEXT_PRIMARY};">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def kpi_row(metrics: list):
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col: kpi_tile(m.get("label", ""), m.get("value", "—"), m.get("sub", ""), m.get("positive", None))

def allocation_table(assets: list, weights):
    df = pd.DataFrame({"Asset": assets, "Peso": weights}).sort_values("Peso", ascending=False).reset_index(drop=True)
    df["Allocazione"] = (df["Peso"] * 100).map(lambda x: f"{x:.2f} %")
    st.table(df[["Asset", "Allocazione"]])

ALIAS_MAP = {"SP500": "^GSPC", "NASDAQ": "^NDX", "DAX": "^GDAXI", "VIX": "^VIX", "GOLD": "GC=F", "OIL": "CL=F", "BTC": "BTC-USD"}

def fetch_historical_data(tickers_input, years, timeframe="Giornaliero"):
    start_date = datetime.now() - timedelta(days=years*365)
    end_date = datetime.now()
    all_series = {}
    interval_map = {"Giornaliero": "1d", "Settimanale": "1wk", "Mensile": "1mo"}
    yf_interval = interval_map.get(timeframe, "1d")
    for t in tickers_input:
        series = None
        try:
            df = yf.download(t, start=start_date, interval=yf_interval, progress=False)
            if not df.empty:
                col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                series = df[col].squeeze()
                if isinstance(series, pd.Series): series = series.ffill()
        except: pass
        if series is None and MSTARPY_AVAILABLE:
            try:
                fund = mstarpy.Funds(term=t, country="it")
                history = fund.nav(start_date=start_date, end_date=end_date, frequency="daily")
                if history:
                    df = pd.DataFrame(history)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    series = df['nav'].tz_localize(None)
                    if timeframe == "Settimanale": series = series.resample('W').last().ffill()
                    elif timeframe == "Mensile": series = series.resample('ME').last().ffill()
            except: pass
        if series is not None:
            series.name = t 
            all_series[t] = series
    if all_series: return pd.DataFrame(all_series).ffill().dropna()
    return None

def pie_chart(labels, values, title="Asset Allocation") -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.52, textinfo="label+percent"))
    fig.update_layout(title=dict(text=title, x=0.5))
    return fig

def get_optimal_weights_auto(mu, sigma, min_weight, max_weight, rf):
    num_assets = len(mu)
    actual_max_weight = max(max_weight, (1.0 / num_assets) + 0.01)
    def neg_sharpe(w):
        vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
        if vol <= 0: return 1e6
        return -(np.sum(mu * w) - rf) / vol
    res = minimize(neg_sharpe, [1./num_assets] * num_assets, bounds=[(min_weight, actual_max_weight)] * num_assets, constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}), method='SLSQP')
    return res.x if res.success else None

def prep_data_auto(df, assets, lookback, freq):
    valid_assets = [c for c in assets if c in df.columns]
    if not valid_assets: return None, None, "Nessun asset valido."
    df = df[valid_assets]
    df_res = df.resample(freq).last().dropna()
    returns = df_res.pct_change().dropna()
    ann_factor = 252 if freq == 'D' else 52 if freq == 'W' else 12
    return returns.mean() * ann_factor, returns.cov() * ann_factor, (df_res, returns, ann_factor)

def portfolio_metrics_auto(weights, mu, sigma, rf) -> dict:
    p_ret = float(np.sum(mu * weights))
    p_vol = float(np.sqrt(np.dot(weights.T, np.dot(sigma, weights))))
    return {"return": p_ret, "volatility": p_vol, "sharpe": (p_ret - rf) / p_vol if p_vol > 0 else 0.0}

# ==============================================================================
# BLOCCO 2: FUNZIONI ESATTE DI "ALLOCAZIONE A 3"
# ==============================================================================
def clean_asset_name_3(name):
    return re.sub(r'\s*\(.*\)', '', str(name)).strip()

def get_advanced_stats_3(weights, returns, annual_factor):
    weights = np.array(weights)
    port_series = returns.dot(weights)
    mean_ret = port_series.mean() * annual_factor
    volatility = port_series.std() * np.sqrt(annual_factor)
    sharpe = mean_ret / volatility if volatility != 0 else 0
    negative_returns = port_series[port_series < 0]
    downside_std = negative_returns.std() * np.sqrt(annual_factor)
    sortino = mean_ret / downside_std if downside_std != 0 else 0
    cumulative = (1 + port_series).cumprod()
    peak = cumulative.cummax()
    mdd = ((cumulative - peak) / peak).min()
    return mean_ret, volatility, sharpe, sortino, mdd

def get_avg_correlation_3(data, assets):
    if len(assets) < 2: return 1.0
    corr_matrix = data[list(assets)].corr()
    return corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()

def optimize_portfolio_3(returns, annual_factor, min_weight=0.0):
    n_assets = len(returns.columns)
    if n_assets * min_weight > 1.0: return None 
    def objective(w):
        ret = np.sum(returns.mean() * w) * annual_factor
        vol = np.sqrt(np.dot(w.T, np.dot(returns.cov() * annual_factor, w)))
        return -(ret / vol if vol > 0 else 0)
    res = minimize(objective, [1./n_assets]*n_assets, method='SLSQP', bounds=[(min_weight, 1)]*n_assets, constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}))
    return res.x if res.success else None

def find_best_optimized_combination_3(data, k, annual_factor, max_corr_threshold=1.0, min_w=0.0):
    assets = data.columns.tolist()
    if len(assets) < k or k * min_w > 1.0: return None, None, (0,0,0,0,0)
    best_sharpe, best_combo, best_weights, best_stats = -np.inf, None, None, None
    for combo in itertools.combinations(assets, k):
        if get_avg_correlation_3(data, combo) <= max_corr_threshold:
            subset = data[list(combo)].pct_change().dropna()
            w = optimize_portfolio_3(subset, annual_factor, min_weight=min_w)
            if w is not None:
                stats = get_advanced_stats_3(w, subset, annual_factor)
                if stats[2] > best_sharpe:
                    best_sharpe, best_combo, best_weights, best_stats = stats[2], combo, w, stats
    return best_combo, best_weights, best_stats

# ==============================================================================
# ROUTING APP
# ==============================================================================
page = st.sidebar.radio("Navigazione Moduli", ["Allocazione Auto", "Allocazione a 3"])
st.sidebar.divider()

if page == "Allocazione Auto":
    st.title("Portfolio Optimizer & Data Engine")
    
    with st.sidebar:
        st.header("1. Acquisizione Dati")
        data_source_tab = st.radio("Sorgente Dati", ["API", "Upload File"])
        tickers_input = []
        uploaded_file = None
        if data_source_tab == "API":
            raw_input = st.text_area("Tickers", "^GSPC\nSWDA.MI")
            years = st.selectbox("Anni Storico", [1, 3, 5, 10], index=1)
            tickers_input = [ALIAS_MAP.get(t, t) for t in re.findall(r"[\w\.\-\^\=]+", raw_input.upper())]
        else:
            uploaded_file = st.file_uploader("Carica File", type=["csv", "xlsx"])
            
        data_freq = st.selectbox("Frequenza", ["Giornaliero", "Settimanale", "Mensile"])
        
        st.header("2. Parametri Ottimizzazione")
        lookback = st.slider("Orizzonte Rolling (Anni)", 1, 10, 3)
        min_weight = st.slider("Peso Minimo", 0.0, 0.2, 0.0)
        max_weight = st.slider("Peso Massimo", 0.1, 1.0, 0.4)
        rf = st.number_input("Risk Free (%)", 0.0, 10.0, 3.0)/100
        
        if st.button("🚀 GENERA SERIE STORICHE"):
            df_data = None
            if data_source_tab == "API" and tickers_input:
                df_data = fetch_historical_data(tickers_input, years, data_freq)
            elif uploaded_file:
                try:
                    df_data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
                    df_data = df_data.select_dtypes(include=[np.number]).dropna()
                except: pass
                
            if df_data is not None and not df_data.empty:
                st.session_state.shared_df = df_data
                st.session_state.shared_assets = df_data.columns.tolist()
                st.session_state.shared_freq = 252 if data_freq == "Giornaliero" else 52 if data_freq == "Settimanale" else 12
                st.success("Dati caricati in memoria!")
            else: st.error("Errore caricamento dati.")

    if st.session_state.shared_df is not None:
        df = st.session_state.shared_df
        assets = st.session_state.shared_assets
        freq = 'D' if st.session_state.shared_freq == 252 else 'W' if st.session_state.shared_freq == 52 else 'M'
        
        st.subheader("Serie Storiche Generate")
        st.line_chart((df / df.iloc[0]) * 100)
        
        mu_strat, sigma_strat, meta = prep_data_auto(df, assets, lookback, freq)
        if mu_strat is not None:
            st.subheader("Ottimizzazione Markowitz")
            w_mk = get_optimal_weights_auto(mu_strat, sigma_strat, min_weight, max_weight, rf)
            if w_mk is not None:
                m_mk = portfolio_metrics_auto(w_mk, mu_strat, sigma_strat, rf)
                kpi_row([
                    {"label": "Rendimento", "value": f"{m_mk['return']*100:.2f}%"},
                    {"label": "Volatilità", "value": f"{m_mk['volatility']*100:.2f}%"},
                    {"label": "Sharpe", "value": f"{m_mk['sharpe']:.3f}"}
                ])
                c1, c2 = st.columns(2)
                with c1: allocation_table(assets, w_mk)
                with c2: st.plotly_chart(pie_chart(assets, w_mk))

elif page == "Allocazione a 3":
    st.title("🛡️ Quant Allocation: 3-Tier Model")
    
    if st.session_state.shared_df is None:
        st.warning("Nessun dato in memoria. Torna alla scheda 'Allocazione Auto' e genera le serie storiche prima di usare questo modulo.")
    else:
        df = st.session_state.shared_df
        assets = st.session_state.shared_assets
        annual_factor = st.session_state.shared_freq
        
        with st.sidebar:
            st.header("Filtri Strategici")
            max_corr_input = st.slider("Max Correlazione Ammessa", 0.0, 1.0, 1.0, 0.05)
            min_weight_pct = st.slider("Peso Minimo per Asset (%)", 0, 33, 10, 1)
            min_weight_val = min_weight_pct / 100.0

        with st.spinner('Calcolo Ottimizzazione Combinatoria (potrebbe richiedere tempo)...'):
            # L1
            temp_sharpes = {a: get_advanced_stats_3([1], df[[a]].pct_change().dropna(), annual_factor)[2] for a in assets}
            best_single = max(temp_sharpes, key=temp_sharpes.get)
            manual_asset = st.selectbox("Seleziona Linea 1", assets, index=assets.index(best_single) if best_single in assets else 0)
            l1_stats = get_advanced_stats_3([1], df[[manual_asset]].pct_change().dropna(), annual_factor)
            
            # L2 & L3
            forced_min_w = max(min_weight_val, 0.01)
            pair_assets, pair_weights, pair_stats = find_best_optimized_combination_3(df, 2, annual_factor, max_corr_input, forced_min_w)
            triplet_assets, triplet_weights, triplet_stats = find_best_optimized_combination_3(df, 3, annual_factor, max_corr_input, forced_min_w)

        st.subheader("Risultati Allocazione Combinatoria")
        table_data = []
        def make_row(label, asset_list, weights, stats):
            if stats is None: return None
            r, v, s, sort, mdd = stats
            comp_str = f"{clean_asset_name_3(asset_list)} (100%)" if isinstance(asset_list, str) else " + ".join([f"{clean_asset_name_3(a)} ({w*100:.0f}%)" for a, w in zip(asset_list, weights) if w > 0.001])
            return {"Strategia": label, "Allocazione": comp_str, "Sharpe": f"{s:.2f}", "Rendimento": f"{r*100:.1f}%", "Max DD": f"{mdd*100:.1f}%"}
            
        r1 = make_row("Linea 1", manual_asset, [1], l1_stats)
        if r1: table_data.append(r1)
        r2 = make_row("Linea 2 (Best Pair)", pair_assets, pair_weights, pair_stats)
        if r2: table_data.append(r2)
        r3 = make_row("Linea 3 (Best Triplet)", triplet_assets, triplet_weights, triplet_stats)
        if r3: table_data.append(r3)
        
        st.table(pd.DataFrame(table_data))
