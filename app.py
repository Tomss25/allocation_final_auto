"""
Unified Quantitative Allocation Platform
Architettura integrata: Motore Dati Condiviso + Multi-Model Routing
(Include Patch di Sanitizzazione Dati CSV e Forzatura Datetime)
"""

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

# ==========================================
# CONFIGURAZIONE PAGINA & CSS (Master Theme)
# ==========================================
st.set_page_config(page_title="Portfolio Optimizer & Data Engine", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

BG_PRIMARY      = "#F8F9FA"
BG_SECONDARY    = "#FFFFFF"
BG_CARD         = "#FFFFFF"
TEXT_PRIMARY    = "#1A202C"
TEXT_SECONDARY  = "#4A5568"
TEXT_MUTED      = "#718096"
COLOR_HIGHLIGHT = "#1A365D" 
COLOR_ACCENT    = "#2C5282"
BORDER_COLOR    = "#E2E8F0"
CHART_COLORS    = ["#1A365D", "#2C5282", "#3182CE", "#4299E1", "#63B3ED", "#90CDF4", "#BEE3F8", "#E6F2FF"]
COLOR_GREEN     = "#38A169"
COLOR_RED       = "#E53E3E"
COLOR_GOLD      = "#D69E2E"
BORDER_RADIUS   = "12px"

PLOTLY_LAYOUT = dict(
    template="plotly_white", paper_bgcolor=BG_SECONDARY, plot_bgcolor=BG_SECONDARY,
    font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif", size=12),
    colorway=CHART_COLORS, hovermode="x unified"
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
    .stApp {{ background-color: {BG_PRIMARY} !important; color: {TEXT_PRIMARY}; font-family: 'Inter', sans-serif; }}
    
    /* MODIFIED SIDEBAR CSS - NAVY BLUE WITH HIGH CONTRAST */
    [data-testid="stSidebar"] {{ background-color: #1A365D !important; border-right: none; }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] > p {{ color: #FFFFFF !important; }}
    [data-testid="stSidebar"] .section-header {{ color: #90CDF4 !important; border-bottom: 2px solid #2C5282; }}
    [data-testid="stSidebar"] hr {{ border-color: #2C5282 !important; margin: 1rem 0; }}
    [data-testid="stSidebar"] div[data-baseweb="select"] > div, [data-testid="stSidebar"] textarea, [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{ background-color: #0F2537 !important; color: #FFFFFF !important; border: 1px solid #2C5282 !important; }}
    
    h1, h2, h3, h4 {{ color: {COLOR_HIGHLIGHT} !important; font-weight: 700; }}
    .kpi-tile {{ background: linear-gradient(135deg, {BG_CARD} 0%, #F7FAFC 100%); border: 2px solid {BORDER_COLOR}; border-radius: {BORDER_RADIUS}; padding: 1.4rem 1.6rem; text-align: center; position: relative; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04); }}
    .kpi-tile::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, {COLOR_HIGHLIGHT}, {COLOR_ACCENT}); }}
    .kpi-label {{ font-size: 0.72rem; font-weight: 700; color: {TEXT_MUTED}; text-transform: uppercase; margin-bottom: 0.6rem; }}
    .kpi-value {{ font-size: 1.8rem; font-weight: 700; color: {TEXT_PRIMARY}; font-family: 'JetBrains Mono', monospace; }}
    .kpi-value.positive {{ color: {COLOR_GREEN}; }}
    .kpi-value.negative {{ color: {COLOR_RED}; }}
    .kpi-sub {{ font-size: 0.72rem; color: {TEXT_MUTED}; margin-top: 0.4rem; }}
    .section-header {{ font-size: 0.75rem; font-weight: 700; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; padding-bottom: 0.6rem; border-bottom: 2px solid {BORDER_COLOR}; }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# STATE MANAGEMENT CONDIVISO
# ==========================================
if 'shared_df' not in st.session_state: st.session_state.shared_df = None
if 'shared_assets' not in st.session_state: st.session_state.shared_assets = []
if 'shared_freq' not in st.session_state: st.session_state.shared_freq = "Giornaliero"
if 'data_source' not in st.session_state: st.session_state.data_source = None

# ==========================================
# ETL & DATA SANITIZATION (Patch Temporanea)
# ==========================================
def clean_and_interpolate_dataframe(df):
    """
    Individua valori anomali ('undefined'), standardizza il formato numerico
    e applica interpolazione lineare. (Inquina la volatilità reale).
    """
    df = df.replace('undefined', np.nan)
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'string':
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            
    n = len(df)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]): continue
        for i in range(n):
            if pd.isna(df.iloc[i, df.columns.get_loc(col)]):
                if i == 0:
                    if n > 2: df.iloc[i, df.columns.get_loc(col)] = (df.iloc[i+1, df.columns.get_loc(col)] + df.iloc[i+2, df.columns.get_loc(col)]) / 2
                elif i == n - 1:
                    if n > 2: df.iloc[i, df.columns.get_loc(col)] = (df.iloc[i-1, df.columns.get_loc(col)] + df.iloc[i-2, df.columns.get_loc(col)]) / 2
                else:
                    df.iloc[i, df.columns.get_loc(col)] = (df.iloc[i-1, df.columns.get_loc(col)] + df.iloc[i+1, df.columns.get_loc(col)]) / 2
    return df

# ==========================================
# UI HELPERS E GRAFICI
# ==========================================
def kpi_tile(label: str, value: str, sub: str = "", positive=None):
    color_class = "positive" if positive is True else "negative" if positive is False else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="kpi-tile"><div class="kpi-label">{label}</div><div class="kpi-value {color_class}">{value}</div>{sub_html}</div>', unsafe_allow_html=True)

def kpi_row(metrics: list):
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col: kpi_tile(m.get("label", ""), m.get("value", "—"), m.get("sub", ""), m.get("positive", None))

def allocation_table(assets: list, weights):
    df = pd.DataFrame({"Asset": assets, "Peso": weights}).sort_values("Peso", ascending=False).reset_index(drop=True)
    df["Allocazione"] = (df["Peso"] * 100).map(lambda x: f"{x:.2f} %")
    st.table(df[["Asset", "Allocazione"]])

def _base_fig(**kwargs) -> go.Figure: return go.Figure(layout=go.Layout(**{**PLOTLY_LAYOUT, **kwargs}))

def pie_chart(labels, values, title="Asset Allocation") -> go.Figure:
    total = sum(values) if sum(values) > 0 else 1
    legend_labels = [f"{l} ({(v/total)*100:.1f}%)" for l, v in zip(labels, values)]
    fig = go.Figure(go.Pie(labels=legend_labels, values=values, hole=0.52, textinfo="none"))
    fig.update_layout(**{**PLOTLY_LAYOUT, "title": dict(text=title, x=0.5)})
    return fig

def equity_line_chart(nav_df: pd.DataFrame, title="Equity Line Comparativa (Base 100)") -> go.Figure:
    fig = _base_fig(title=dict(text=title, x=0))
    for i, col in enumerate(nav_df.columns):
        fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df[col], name=col, mode="lines"))
    fig.update_layout(yaxis_title="NAV (Base 100)")
    return fig

def drawdown_chart(nav_df: pd.DataFrame, title="Drawdown Analysis") -> go.Figure:
    fig = _base_fig(title=dict(text=title, x=0))
    for i, col in enumerate(nav_df.columns):
        dd = (nav_df[col] - nav_df[col].cummax()) / nav_df[col].cummax() * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=col, mode="lines", fill="tozeroy"))
    fig.update_layout(yaxis_title="Drawdown (%)")
    return fig

def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ==========================================
# DATA ENGINE CONDIVISO
# ==========================================
ALIAS_MAP = {"SP500": "^GSPC", "NASDAQ": "^NDX", "DAX": "^GDAXI", "VIX": "^VIX", "GOLD": "GC=F", "OIL": "CL=F", "BTC": "BTC-USD"}

@st.cache_data(show_spinner=False)
def fetch_historical_data(tickers_input, years, timeframe):
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

# ==========================================
# CORE MATH: ALLOCAZIONE AUTO (Completo)
# ==========================================
def prep_data(df, assets, lookback, freq):
    valid_assets = [c for c in assets if c in df.columns]
    if not valid_assets: return None, None, "Nessun asset valido."
    df = df[valid_assets]
    if df.empty or len(df) < 10: return None, None, "Dati insufficienti."
    df_res = df.resample(freq).last().dropna()
    returns = df_res.pct_change().dropna()
    ann_factor = 252 if freq == 'D' else 52 if freq == 'W' else 12
    mu = returns.mean() * ann_factor
    sigma = returns.cov() * ann_factor
    return mu, sigma, (df_res, returns, ann_factor)

def portfolio_metrics(weights, mu, sigma, rf) -> dict:
    p_ret = float(np.sum(mu * weights))
    p_vol = float(np.sqrt(np.dot(weights.T, np.dot(sigma, weights))))
    p_sharpe = (p_ret - rf) / p_vol if p_vol > 0 else 0.0
    return {"return": p_ret, "volatility": p_vol, "sharpe": p_sharpe}

def compute_nav(returns_df: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    return (1 + returns_df).cumprod() * base

def max_drawdown(nav_series: pd.Series) -> float:
    rolling_max = nav_series.cummax()
    return float(((nav_series - rolling_max) / rolling_max).min())

def get_optimal_weights(mu, sigma, min_w, max_w, rf):
    num_assets = len(mu)
    actual_max = max(max_w, (1.0/num_assets) + 0.01)
    def neg_sharpe(w):
        vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
        if vol <= 0: return 1e6
        return -(np.sum(mu * w) - rf) / vol
    res = minimize(neg_sharpe, [1./num_assets]*num_assets, bounds=[(min_w, actual_max)]*num_assets, constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}), method='SLSQP')
    return res.x if res.success else None

def get_montecarlo_weights(mu, sigma, min_w, max_w, rf, num_sims=5000):
    num_assets = len(mu)
    weights = np.random.random((num_sims, num_assets))
    weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
    mask = ((np.max(weights, axis=1) <= (max_w + 0.01)) & (np.min(weights, axis=1) >= (min_w - 0.01)))
    valid_weights = weights[mask]
    if len(valid_weights) == 0: return None
    port_ret = np.dot(valid_weights, mu)
    port_vol = np.sqrt(np.sum(np.dot(valid_weights, sigma) * valid_weights, axis=1))
    return valid_weights[np.argmax((port_ret - rf) / port_vol)]

def get_gmv_weights(sigma, min_w, max_w):
    num_assets = len(sigma)
    actual_max = max(max_w, (1.0/num_assets) + 0.01)
    res = minimize(lambda w: np.dot(w.T, np.dot(sigma, w)), [1./num_assets]*num_assets, bounds=[(min_w, actual_max)]*num_assets, constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}), method='SLSQP')
    return res.x if res.success else None

def get_cvar_weights(returns_matrix, min_w, max_w, alpha=0.05):
    n_assets, n_scenarios = returns_matrix.shape[1], returns_matrix.shape[0]
    actual_max = max(max_w, (1.0/n_assets) + 0.01)
    def cvar_obj(params):
        w, gamma = params[:-1], params[-1]
        shortfall = np.maximum(-np.dot(returns_matrix, w) - gamma, 0)
        return gamma + np.sum(shortfall) / (alpha * n_scenarios)
    init_w = [1./n_assets]*n_assets
    init_gamma = -np.percentile(np.dot(returns_matrix, init_w), alpha * 100)
    res = minimize(cvar_obj, np.append(init_w, init_gamma), method='SLSQP', bounds=[(min_w, actual_max)]*n_assets + [(None, None)], constraints=({'type': 'eq', 'fun': lambda x: np.sum(x[:-1]) - 1}))
    return res.x[:-1] if res.success else None

# ==========================================
# CORE MATH: ALLOCAZIONE A 3 (Tier Model)
# ==========================================
def clean_asset_name_3(name): return re.sub(r'\s*\(.*\)', '', str(name)).strip()

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

# ==========================================
# ROUTING & SIDEBAR (Architettura Unificata)
# ==========================================
page = st.sidebar.radio("Navigazione Moduli", ["Allocazione Auto", "Allocazione a 3", "Nota Metodologica"])
st.sidebar.divider()

with st.sidebar:
    st.markdown('<div class="section-header">📊 Acquisizione Dati Condivisa</div>', unsafe_allow_html=True)
    input_type = st.radio("Sorgente Dati", ["API (Ticker/ISIN)", "Upload File (CSV/Excel)"], label_visibility="collapsed")
    tickers_input = []
    uploaded_file = None
    
    if input_type == "API (Ticker/ISIN)":
        raw_text = st.text_area("Inserisci Ticker/ISIN", "^GSPC\nSWDA.MI\nEIMI.MI\nGC=F", height=120)
        tickers_input = [ALIAS_MAP.get(t, t) for t in re.findall(r"[\w\.\-\^\=]+", raw_text.upper())]
        years = st.selectbox("Anni Storico", [1, 3, 5, 10, 20], index=2)
    else:
        uploaded_file = st.file_uploader("Carica File", type=["csv", "xlsx"])
        
    timeframe = st.selectbox("Frequenza Dati", ["Giornaliero", "Settimanale", "Mensile"])
    
    st.markdown('<div class="section-header">⚙️ Parametri Ottimizzazione Base</div>', unsafe_allow_html=True)
    lookback = st.slider("Orizzonte Rolling (Anni)", 1, 10, 3)
    min_weight = st.slider("Peso Minimo Asset", 0.0, 0.2, 0.0, step=0.01)
    max_weight = st.slider("Peso Massimo Asset", 0.1, 1.0, 0.40, step=0.05)
    rf = st.number_input("Tasso Risk Free (%)", 0.0, 10.0, 3.0, step=0.1) / 100
    
    if st.button("🚀 GENERA SERIE STORICHE", type="primary", use_container_width=True):
        with st.spinner("Acquisizione Dati in corso..."):
            df_temp = None
            if input_type == "API (Ticker/ISIN)" and tickers_input:
                df_temp = fetch_historical_data(tickers_input, years, timeframe)
                source_name = "Yahoo/Morningstar"
            elif input_type == "Upload File (CSV/Excel)" and uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        # Fallback di lettura per CSV
                        try:
                            df_temp = pd.read_csv(uploaded_file, sep=';', index_col=0)
                        except:
                            uploaded_file.seek(0)
                            df_temp = pd.read_csv(uploaded_file, sep=',', index_col=0)
                    else:
                        df_temp = pd.read_excel(uploaded_file, index_col=0)
                    
                    # FORZATURA ASSOLUTA DATETIME PER L'INDICE
                    df_temp.index = pd.to_datetime(df_temp.index, dayfirst=True, errors='coerce')
                    df_temp = df_temp[df_temp.index.notnull()]
                    
                    df_temp = clean_and_interpolate_dataframe(df_temp)
                    df_temp = df_temp.select_dtypes(include=[np.number]).dropna()
                    source_name = "Upload CSV/Excel"
                except Exception as e:
                    st.error(f"Errore lettura o sanitizzazione file: {e}")
            
            if df_temp is not None and not df_temp.empty:
                st.session_state.shared_df = df_temp
                st.session_state.shared_assets = df_temp.columns.tolist()
                st.session_state.shared_freq = timeframe
                st.session_state.data_source = source_name
                st.success("✅ Dati
