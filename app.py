"""
Unified Quantitative Allocation Platform
Architettura integrata: Motore Dati Condiviso + Multi-Model Routing
(Include Patch di Sanitizzazione Dati CSV)
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
                    # Leggiamo il file senza forzare il decimale per evitare errori con le stringhe
                    if uploaded_file.name.endswith('.csv'):
                        df_temp = pd.read_csv(uploaded_file, sep=';', index_col=0, parse_dates=True)
                    else:
                        df_temp = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
                    
                    # Applichiamo la patch di pulizia che scambia le virgole e interpola
                    df_temp = clean_and_interpolate_dataframe(df_temp)
                    
                    # Forziamo a numerico e droppiamo eventuali rimanenze sporche
                    df_temp = df_temp.select_dtypes(include=[np.number]).dropna()
                    source_name = "Upload CSV/Excel"
                except Exception as e:
                    st.error(f"Errore lettura o sanitizzazione file: {e}")
            
            if df_temp is not None and not df_temp.empty:
                st.session_state.shared_df = df_temp
                st.session_state.shared_assets = df_temp.columns.tolist()
                st.session_state.shared_freq = timeframe
                st.session_state.data_source = source_name
                st.success("✅ Dati Acquisiti e Condivisi in Memoria!")
            else:
                st.error("❌ Fallimento estrazione dati.")

# ==========================================
# RENDER VIEWS
# ==========================================
if page == "Nota Metodologica":
    st.title("Nota Metodologica & Assunzioni del Modello")
    st.markdown("""
    ### 1. Generazione e Trattamento Dati
    Il Data Engine acquisisce serie storiche in formato **Total Return (Adjusted Close)** tramite API (Yahoo Finance) o scraping NAV (Morningstar). L'uso di dati Total Return è critico per internalizzare l'effetto capitalizzato di dividendi e cedole. I dati vengono allineati temporalmente e i missing values riempiti tramite metodo *forward-fill* per evitare bias di look-ahead. 

    ### 2. Ottimizzazione del Portafoglio
    L'architettura unifica due filosofie quantitative:
    * **Motore Auto (Markowitz & CVaR):** Implementa l'ottimizzazione Media-Varianza classica (SLSQP). Per mitigare la fragilità statistica di Markowitz, il modulo integra l'ottimizzazione **Min-CVaR (Expected Shortfall a coda 5%)**, spostando il focus dal rischio simmetrico al Tail Risk puro. Include stress test via simulazione di Montecarlo e backtest Walk-Forward.
    * **Motore a 3-Tier:** Esplora a forza bruta il subspazio di tutte le combinazioni possibili di $k \in \{1, 2, 3\}$ asset, applicando l'ottimizzatore sui subset filtrati tramite matrici di correlazione massima.

    ### 3. Assunzioni Statistiche
    * **Stazionarietà:** I modelli classici assumono implicitamente che i rendimenti passati siano stimatori non distorti del futuro.
    * **Distribuzione:** La varianza assume normalità dei rendimenti, assioma fallace. Il modulo CVaR e le proiezioni stocastiche rilassano questa assunzione.

    ### 4. Limiti del Modello e Rischi (Overfitting)
    Il framework è profondamente esposto al rischio di **Curve-Fitting**. In particolare, il Motore a 3-Tier agisce come un selettore ex-post: estraendo la combinazione storica ottimale, tende a sovrastimare massicciamente le performance out-of-sample.
    """)

elif st.session_state.shared_df is None:
    st.markdown("""
    <div style="text-align:center; padding:6rem 2rem;">
        <div style="font-size:4rem; margin-bottom:1.5rem;">📊</div>
        <div style="font-size:1.3rem; font-weight:700; color:#1A365D; margin-bottom:0.8rem;">Benvenuto nel Portfolio Optimizer Unificato</div>
        <div style="font-size:0.95rem; color:#4A5568;">Configura i parametri nella <b>sidebar</b> e premi <b>GENERA SERIE STORICHE</b> per avviare entrambi i motori analitici.</div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Allocazione Auto":
    st.title("Portfolio Optimizer & Data Engine")
    df = st.session_state.shared_df
    all_assets = st.session_state.shared_assets
    freq_str = st.session_state.shared_freq
    freq_code = 'D' if freq_str == "Giornaliero" else 'W' if freq_str == "Settimanale" else 'M'
    
    st.markdown(f"**Sorgente:** {st.session_state.data_source} | **Periodo:** {df.index[0].strftime('%d/%m/%Y')} → {df.index[-1].strftime('%d/%m/%Y')} | **Asset:** {len(all_assets)}")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Serie Storiche", "📈 Markowitz", "🎲 Montecarlo", "🛡️ Antifragile", "📉 Backtest", "🔮 Proiezione", "💹 Mercato Live"])

    with st.spinner("Preparazione Dati Core..."):
        mu_strat, sigma_strat, meta = prep_data(df, all_assets, lookback, freq_code)
    
    if mu_strat is None:
        st.error(meta)
    else:
        df_res, returns_wf, ann_factor_opt = meta
        
        # TAB 1: SERIE STORICHE
        with tab1:
            st.markdown('<div class="section-header">Analisi Serie Storiche</div>', unsafe_allow_html=True)
            metrics = []
            for col in df.columns:
                s = df[col]
                ret = s.pct_change().dropna()
                metrics.append({"Ticker": col, "Prezzo": round(s.iloc[-1], 2), "Rend %": round(((s.iloc[-1]/s.iloc[0])-1)*100, 2), "Volat %": round(ret.std()*np.sqrt(ann_factor_opt)*100, 2), "Max DD %": round(((s-s.cummax())/s.cummax()).min()*100, 2)})
            
            c1, c2 = st.columns([2, 1])
            with c1: st.plotly_chart(equity_line_chart((df/df.iloc[0])*100, "Performance Storica"), use_container_width=True)
            with c2: st.dataframe(pd.DataFrame(metrics).set_index("Ticker"), height=400)
            
            st.markdown("#### Dati Storici Grezzi")
            st.dataframe(df.sort_index(ascending=False).round(2), use_container_width=True, height=300)
            df_export = df.copy()
            df_export.index.name = "Data"
            csv = df_export.to_csv(sep=";", decimal=",", encoding="utf-8-sig")
            st.download_button(label="📥 SCARICA CSV DATI", data=csv, file_name="serie_storiche.csv", mime="text/csv")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(df.pct_change().corr(), annot=True, cmap="RdYlGn", fmt=".2f", ax=ax)
            st.pyplot(fig)

        # TAB 2: MARKOWITZ
        with tab2:
            st.markdown('<div class="section-header">Mean-Variance Optimization</div>', unsafe_allow_html=True)
            w_mk = get_optimal_weights(mu_strat, sigma_strat, min_weight, max_weight, rf)
            if w_mk is None: w_mk = np.array([1.0/len(all_assets)]*len(all_assets))
            m_mk = portfolio_metrics(w_mk, mu_strat, sigma_strat, rf)
            kpi_row([
                {"label": "Rendimento Atteso", "value": f"{m_mk['return']*100:.2f}%", "positive": m_mk['return']>0},
                {"label": "Volatilità Attesa", "value": f"{m_mk['volatility']*100:.2f}%"},
                {"label": "Sharpe Ratio", "value": f"{m_mk['sharpe']:.3f}", "positive": m_mk['sharpe']>1}
            ])
            c1, c2 = st.columns(2)
            with c1: allocation_table(all_assets, w_mk)
            with c2: st.plotly_chart(pie_chart(all_assets, w_mk, "Markowitz Allocation"))

        # TAB 3: MONTECARLO
        with tab3:
            st.markdown('<div class="section-header">Simulazione Montecarlo (10k Scenari)</div>', unsafe_allow_html=True)
            with st.spinner("Generazione..."):
                w_mc = get_montecarlo_weights(mu_strat, sigma_strat, min_weight, max_weight, rf, 10000)
            if w_mc is None: w_mc = np.array([1.0/len(all_assets)]*len(all_assets))
            m_mc = portfolio_metrics(w_mc, mu_strat, sigma_strat, rf)
            kpi_row([
                {"label": "Rendimento Atteso", "value": f"{m_mc['return']*100:.2f}%", "positive": m_mc['return']>0},
                {"label": "Volatilità Attesa", "value": f"{m_mc['volatility']*100:.2f}%"},
                {"label": "Sharpe Ratio", "value": f"{m_mc['sharpe']:.3f}", "positive": m_mc['sharpe']>1}
            ])
            c1, c2 = st.columns(2)
            with c1: allocation_table(all_assets, w_mc)
            with c2: st.plotly_chart(pie_chart(all_assets, w_mc, "Montecarlo Best Sharpe"))

        # TAB 4: ANTIFRAGILE
        with tab4:
            st.markdown('<div class="section-header">Min CVaR & Ledoit-Wolf Shrinkage</div>', unsafe_allow_html=True)
            with st.spinner("Calcolo Matrici Robuste..."):
                lw = LedoitWolf()
                sigma_shrunk = lw.fit(returns_wf).covariance_ * ann_factor_opt
                w_gmv = get_gmv_weights(sigma_shrunk, min_weight, max_weight)
                w_cvar = get_cvar_weights(returns_wf.values, min_weight, max_weight)
                
            if w_gmv is None: w_gmv = np.array([1.0/len(all_assets)]*len(all_assets))
            if w_cvar is None: w_cvar = np.array([1.0/len(all_assets)]*len(all_assets))
            m_cvar = portfolio_metrics(w_cvar, mu_strat, sigma_strat, rf)
            
            st.markdown("#### Minimizzazione Rischio di Rovina (CVaR - 95%)")
            kpi_row([
                {"label": "Rendimento Atteso", "value": f"{m_cvar['return']*100:.2f}%"},
                {"label": "Volatilità Attesa", "value": f"{m_cvar['volatility']*100:.2f}%"},
                {"label": "Sharpe Ratio", "value": f"{m_cvar['sharpe']:.3f}"}
            ])
            c1, c2 = st.columns(2)
            with c1: st.markdown("##### Min-CVaR"); allocation_table(all_assets, w_cvar)
            with c2: st.markdown("##### GMV Shrinkage"); allocation_table(all_assets, w_gmv)

        # TAB 5: BACKTEST
        with tab5:
            st.markdown('<div class="section-header">Walk-Forward Backtest</div>', unsafe_allow_html=True)
            window_size = int(lookback * ann_factor_opt)
            if len(returns_wf) <= window_size:
                st.error("Dati insufficienti per il backtest. Abbassa l'orizzonte Rolling.")
            else:
                wf_ret_mk, wf_ret_mc, wf_ret_gmv, wf_ret_cvar, wf_dates = [], [], [], [], []
                w_opt_mk = w_opt_mc = w_opt_gmv = w_opt_cvar = np.array([1.0/len(all_assets)]*len(all_assets))
                pb = st.progress(0, text="Stress test storico...")
                steps = len(returns_wf) - window_size
                
                for i in range(window_size, len(returns_wf)):
                    if i % 10 == 0: pb.progress(min((i-window_size)/steps, 1.0))
                    win = returns_wf.iloc[i-window_size:i]
                    if win.std().sum() > 0:
                        mu_win, sig_win = win.mean()*ann_factor_opt, win.cov()*ann_factor_opt
                        res_mk = get_optimal_weights(mu_win, sig_win, min_weight, max_weight, rf)
                        if res_mk is not None: w_opt_mk = res_mk
                        res_mc = get_montecarlo_weights(mu_win, sig_win, min_weight, max_weight, rf, 1000)
                        if res_mc is not None: w_opt_mc = res_mc
                        try:
                            res_gmv = get_gmv_weights(LedoitWolf().fit(win).covariance_*ann_factor_opt, min_weight, max_weight)
                            if res_gmv is not None: w_opt_gmv = res_gmv
                        except: pass
                        res_cvar = get_cvar_weights(win.values, min_weight, max_weight)
                        if res_cvar is not None: w_opt_cvar = res_cvar
                        
                    next_ret = returns_wf.iloc[i].values
                    wf_ret_mk.append(np.sum(w_opt_mk * next_ret))
                    wf_ret_mc.append(np.sum(w_opt_mc * next_ret))
                    wf_ret_gmv.append(np.sum(w_opt_gmv * next_ret))
                    wf_ret_cvar.append(np.sum(w_opt_cvar * next_ret))
                    wf_dates.append(returns_wf.index[i])
                    
                pb.empty()
                df_wf = pd.DataFrame({"Markowitz": wf_ret_mk, "Montecarlo": wf_ret_mc, "GMV Shrinkage": wf_ret_gmv, "Min CVaR": wf_ret_cvar}, index=wf_dates)
                nav = compute_nav(df_wf)
                c1, c2 = st.columns(2)
                with c1: st.plotly_chart(equity_line_chart(nav, "Backtest"), use_container_width=True)
                with c2: st.plotly_chart(drawdown_chart(nav, "Drawdown"), use_container_width=True)
                
                st.markdown("#### Analisi Strategica del Backtest")
                
                mdd_assets = ((df_res - df_res.cummax()) / df_res.cummax()).min() * 100
                mdd_ports = ((nav - nav.cummax()) / nav.cummax()).min() * 100
                
                mdd_df = pd.DataFrame({
                    "Nome": list(mdd_assets.index) + list(mdd_ports.index),
                    "Tipologia": ["Asset Singolo"] * len(mdd_assets) + ["Portafoglio Simulato"] * len(mdd_ports),
                    "Max Drawdown (%)": list(mdd_assets.values) + list(mdd_ports.values)
                }).sort_values("Max Drawdown (%)")
                
                st.table(mdd_df.style.format({"Max Drawdown (%)": "{:.2f}%"}))
                
                st.markdown("Questo backtest mostra la simulazione storica, ma ti stai illudendo se pensi che questi siano i rendimenti che otterrai. Il modello esegue ribilanciamenti continui assumendo liquidità infinita, zero slippage e zero costi di transazione. Se il Drawdown dei portafogli rompe la tua soglia psicologica o non giustifica il rischio rispetto alla caduta libera degli asset singoli (vedi tabella sopra), il tuo modello teorico ha fallito. Smetti di guardare il rendimento assoluto e fissa questi numeri negativi: sono il prezzo che pagherai nei periodi di panico.")

        # TAB 6: PROIEZIONE
        with tab6:
            st.markdown('<div class="section-header">Cono di Probabilità (GBM)</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            anni_futuri = c1.slider("Anni Proiezione", 1, 10, 5)
            n_sim = c2.selectbox("Scenari Paralleli", [1000, 5000])
            
            with st.spinner("Calcolo traiettorie quantistiche..."):
                w_proj = w_cvar if 'w_cvar' in locals() else np.array([1.0/len(all_assets)]*len(all_assets))
                cov_proj = sigma_shrunk if 'sigma_shrunk' in locals() else sigma_strat
                p_mu = float(np.sum(mu_strat * w_proj))
                p_vol = float(np.sqrt(np.dot(w_proj.T, np.dot(cov_proj, w_proj))))
                
                giorni = 252 * anni_futuri
                dt = 1/252
                sim = np.zeros((giorni+1, n_sim))
                sim[0] = 100.0
                Z = np.random.standard_normal((giorni, n_sim))
                sim[1:] = np.exp((p_mu - 0.5*p_vol**2)*dt + p_vol*np.sqrt(dt)*Z)
                sim = np.cumprod(sim, axis=0)
                perc = np.percentile(sim, [5, 25, 50, 75, 95], axis=1)
                
                fig = _base_fig(title=dict(text="Cono Incertezza Pesi CVaR", x=0))
                dates = pd.date_range(start=pd.Timestamp.today(), periods=giorni+1, freq='B')
                fig.add_trace(go.Scatter(x=dates, y=perc[4], mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=dates, y=perc[0], mode='lines', fill='tonexty', fillcolor=_hex_to_rgba(COLOR_HIGHLIGHT, 0.1), name='Banda 5-95%'))
                fig.add_trace(go.Scatter(x=dates, y=perc[2], mode='lines', line=dict(color=COLOR_HIGHLIGHT, width=3), name='Mediana (P50)'))
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Analisi Strategica della Proiezione")
                st.markdown(f"**Prospettive a {anni_futuri} anni (Capitale Iniziale: 100):**")
                st.markdown(f"- **Scenario Pessimistico (5% probabilità):** Il capitale crolla a **{perc[0][-1]:.2f}** (CAGR: **{((perc[0][-1]/100)**(1/anni_futuri)-1)*100:.2f}%**).")
                st.markdown(f"- **Scenario Mediano (50% probabilità):** Il capitale arriva a **{perc[2][-1]:.2f}** (CAGR: **{((perc[2][-1]/100)**(1/anni_futuri)-1)*100:.2f}%**).")
                st.markdown(f"- **Scenario Ottimistico (95% probabilità):** Il capitale arriva a **{perc[4][-1]:.2f}** (CAGR: **{((perc[4][-1]/100)**(1/anni_futuri)-1)*100:.2f}%**).")
                
                st.markdown("Stai guardando un cono generato da un Moto Browniano Geometrico, un modello che assume ingenuamente che la volatilità futura sarà identica a quella passata. La linea mediana e le stime ottimistiche sono puro rumore statistico. Il tuo vero focus deve essere la **banda inferiore (5%)**. Se quella linea scende al di sotto del tuo capitale di sopravvivenza, la tua allocazione attuale ha un rischio di rovina matematica inaccettabile. Non usare questa proiezione per sognare profitti, usala per quantificare i tuoi rischi peggiori.")

        # TAB 7: LIVE
        with tab7:
            st.markdown('<div class="section-header">Live Market Widgets</div>', unsafe_allow_html=True)
            c1, c2 = st.columns([2,1])
            with c1: components.html('<iframe src="https://sslecal2.investing.com?ecoDayBackground=%23FFFFFF&defaultFont=%231A202C&borderColor=%23E2E8F0&columns=exc_flags,exc_currency,exc_importance,exc_actual,exc_forecast,exc_previous&features=datepicker,timezone&countries=25,32,6,37,72,22,17,39,14,10,35,43,56,36,110,11,26,12,4,5&calType=week&timeZone=8&lang=1" width="650" height="467" frameborder="0"></iframe>', height=500)
            with c2: components.html('<iframe src="https://ssltsw.investing.com?lang=1&forex=1,2,3,5,7,9,10&commodities=8830,8836,8831,8849,8833,8862,8832&indices=175,166,172,27,179,170,174&stocks=345,346,347,348,349,350,352&tabs=1,2,3,4" width="317" height="467"></iframe>', height=500)

elif page == "Allocazione a 3":
    st.title("🛡️ Quant Allocation: 3-Tier Model")
    df = st.session_state.shared_df
    assets = st.session_state.shared_assets
    freq_str = st.session_state.shared_freq
    ann_factor = 252 if freq_str == "Giornaliero" else 52 if freq_str == "Settimanale" else 12

    c1, c2 = st.columns(2)
    max_corr = c1.slider("Max Correlazione Ammessa", 0.0, 1.0, 1.0)
    min_w = c2.slider("Peso Minimo Combinatorio (%)", 0, 33, 10)/100.0

    temp_sharpes = {a: get_advanced_stats_3([1], df[[a]].pct_change().dropna(), ann_factor)[2] for a in assets}
    best_single = max(temp_sharpes, key=temp_sharpes.get)
    try: default_idx = assets.index(best_single)
    except: default_idx = 0
    
    manual_asset = st.selectbox("Linea 1 (Asset Manuale)", assets, index=default_idx)

    with st.spinner('Calcolo Ottimizzazione Combinatoria (Forza Bruta)...'):
        l1_ret_frame = df[[manual_asset]].pct_change().dropna()
        l1_stats = get_advanced_stats_3([1], l1_ret_frame, ann_factor)
        
        forced_min_w = max(min_w, 0.01)
        p_assets, p_w, p_stats = find_best_optimized_combination_3(df, 2, ann_factor, max_corr, forced_min_w)
        t_assets, t_w, t_stats = find_best_optimized_combination_3(df, 3, ann_factor, max_corr, forced_min_w)

    st.subheader("Performance Tier")
    table_data = []
    def make_row(label, a_list, w_list, stats):
        if stats is None: return None
        r, v, s, sort, mdd = stats
        comp = f"{clean_asset_name_3(a_list)} (100%)" if isinstance(a_list, str) else " + ".join([f"{clean_asset_name_3(a)} ({w*100:.0f}%)" for a, w in zip(a_list, w_list) if w > 0.001])
        return {"Strategia": label, "Allocazione": comp, "Sharpe": f"{s:.2f}", "Rend": f"{r*100:.1f}%", "Max DD": f"{mdd*100:.1f}%"}
        
    r1 = make_row("L1 (Manuale)", manual_asset, [1], l1_stats)
    if r1: table_data.append(r1)
    r2 = make_row("L2 (Best Pair)", p_assets, p_w, p_stats)
    if r2: table_data.append(r2)
    r3 = make_row("L3 (Best Triplet)", t_assets, t_w, t_stats)
    if r3: table_data.append(r3)
    
    if table_data: 
        st.table(pd.DataFrame(table_data))
        
        st.markdown("#### Visualizzazione Allocazioni")
        c_pie1, c_pie2, c_pie3 = st.columns(3)
        with c_pie1:
            if r1: st.plotly_chart(pie_chart([manual_asset], [1], "Linea 1"), use_container_width=True)
        with c_pie2:
            if r2 and p_assets: st.plotly_chart(pie_chart(list(p_assets), p_w, "Linea 2"), use_container_width=True)
        with c_pie3:
            if r3 and t_assets: st.plotly_chart(pie_chart(list(t_assets), t_w, "Linea 3"), use_container_width=True)
            
        st.markdown("#### Simulazione Storica Comparativa")
        common_idx = l1_ret_frame.index
        l2_series, l3_series = None, None
        if p_assets: 
            l2_series = df[list(p_assets)].pct_change().dropna().dot(p_w)
            common_idx = common_idx.intersection(l2_series.index)
        if t_assets: 
            l3_series = df[list(t_assets)].pct_change().dropna().dot(t_w)
            common_idx = common_idx.intersection(l3_series.index)
            
        chart_df = pd.DataFrame(index=common_idx)
        chart_df[f"L1: {clean_asset_name_3(manual_asset)}"] = (1 + l1_ret_frame.loc[common_idx][manual_asset]).cumprod() * 100
        if p_assets and l2_series is not None: chart_df["L2: Best Pair"] = (1 + l2_series.loc[common_idx]).cumprod() * 100
        if t_assets and l3_series is not None: chart_df["L3: Best Triplet"] = (1 + l3_series.loc[common_idx]).cumprod() * 100
        
        fig_comp = px.line(chart_df, x=chart_df.index, y=chart_df.columns, template='plotly_white')
        fig_comp.update_layout(yaxis_title="Valore (Base 100)", legend=dict(orientation="h", y=1.1, title=None))
        st.plotly_chart(fig_comp, use_container_width=True)
    else: 
        st.warning("Nessuna combinazione soddisfa i criteri.")
