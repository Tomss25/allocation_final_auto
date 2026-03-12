"""
Unified Quantitative Allocation Platform
Architettura integrata: Motore Dati Condiviso + Multi-Model Routing
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
st.set_page_config(page_title="Quant Platform Pro", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# Palette Navy / Light Blue
BG_PRIMARY      = "#F4F7FA"
BG_SECONDARY    = "#FFFFFF"
TEXT_PRIMARY    = "#0A192F" # Deep Navy
TEXT_SECONDARY  = "#495670"
COLOR_HIGHLIGHT = "#112240" # Navy Header
COLOR_ACCENT    = "#0070F3" # Bright Blue Accent
BORDER_COLOR    = "#E2E8F0"

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG_PRIMARY}; color: {TEXT_PRIMARY}; font-family: 'Inter', sans-serif; }}
    [data-testid="stSidebar"] {{ background-color: {BG_SECONDARY}; border-right: 1px solid {BORDER_COLOR}; }}
    h1, h2, h3 {{ color: {COLOR_HIGHLIGHT} !important; font-weight: 700; letter-spacing: -0.5px; }}
    .stButton > button {{ background-color: {COLOR_ACCENT} !important; color: white !important; border-radius: 8px !important; font-weight: 600; border: none; transition: 0.3s; }}
    .stButton > button:hover {{ background-color: {COLOR_HIGHLIGHT} !important; box-shadow: 0 4px 12px rgba(0,112,243,0.3); }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; background-color: transparent; border-bottom: 2px solid {BORDER_COLOR}; }}
    .stTabs [data-baseweb="tab"] {{ background-color: {BG_SECONDARY}; border-radius: 6px 6px 0 0; border: 1px solid {BORDER_COLOR}; border-bottom: none; color: {TEXT_SECONDARY}; }}
    .stTabs [aria-selected="true"] {{ background-color: {COLOR_HIGHLIGHT} !important; color: white !important; }}
    div[data-testid="stMetric"] {{ background: {BG_SECONDARY}; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# STATE MANAGEMENT
# ==========================================
if 'df_historical' not in st.session_state:
    st.session_state.df_historical = None
if 'data_freq' not in st.session_state:
    st.session_state.data_freq = "Giornaliero"
if 'ann_factor' not in st.session_state:
    st.session_state.ann_factor = 252
if 'current_view' not in st.session_state:
    st.session_state.current_view = "auto"

# ==========================================
# DATA ENGINE (Condiviso)
# ==========================================
ALIAS_MAP = {
    "SP500": "^GSPC", "NASDAQ": "^NDX", "DAX": "^GDAXI", "VIX": "^VIX", 
    "GOLD": "GC=F", "OIL": "CL=F", "BTC": "BTC-USD"
}

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
# CORE MATH: ALLOCAZIONE AUTO (Legacy)
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

def get_optimal_weights(mu, sigma, min_weight, max_weight, rf):
    n = len(mu)
    actual_max = max(max_weight, (1.0/n) + 0.01)
    def neg_sharpe(w):
        vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
        if vol <= 0: return 1e6
        return -(np.sum(mu * w) - rf) / vol
    res = minimize(neg_sharpe, [1./n]*n, bounds=[(min_weight, actual_max)]*n,
                   constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}), method='SLSQP')
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
    res = minimize(cvar_obj, np.append(init_w, init_gamma), method='SLSQP',
                   bounds=[(min_w, actual_max)]*n_assets + [(None, None)],
                   constraints=({'type': 'eq', 'fun': lambda x: np.sum(x[:-1]) - 1}))
    return res.x[:-1] if res.success else None

# ==========================================
# CORE MATH: ALLOCAZIONE A 3 (Legacy)
# ==========================================
def clean_asset_name(name): return re.sub(r'\s*\(.*\)', '', str(name)).strip()

def get_advanced_stats(weights, returns, annual_factor):
    port_series = returns.dot(np.array(weights))
    mean_ret = port_series.mean() * annual_factor
    volatility = port_series.std() * np.sqrt(annual_factor)
    sharpe = mean_ret / volatility if volatility != 0 else 0
    downside_std = port_series[port_series < 0].std() * np.sqrt(annual_factor)
    sortino = mean_ret / downside_std if downside_std != 0 else 0
    cum = (1 + port_series).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return mean_ret, volatility, sharpe, sortino, mdd

def get_avg_correlation(data, assets):
    if len(assets) < 2: return 1.0
    corr = data[list(assets)].corr().values
    return corr[np.triu_indices_from(corr, k=1)].mean()

def find_best_optimized_combination(data, k, annual_factor, max_corr, min_w):
    assets = data.columns.tolist()
    if len(assets) < k or k * min_w > 1.0: return None, None, (0,0,0,0,0)
    best_sharpe, best_combo, best_weights, best_stats = -np.inf, None, None, None
    
    # Prevenzione esplosione combinatoria imposta nel refactoring
    if len(assets) > 15 and k > 2: st.warning("⚠️ Troppi asset. L'analisi combinatoria a 3 tier potrebbe essere lenta.")
    
    for combo in itertools.combinations(assets, k):
        if get_avg_correlation(data, combo) <= max_corr:
            subset = data[list(combo)].pct_change().dropna()
            w = get_optimal_weights(subset.mean() * annual_factor, subset.cov() * annual_factor, min_w, 1.0, 0.0)
            if w is not None:
                stats = get_advanced_stats(w, subset, annual_factor)
                if stats[2] > best_sharpe:
                    best_sharpe, best_combo, best_weights, best_stats = stats[2], combo, w, stats
    return best_combo, best_weights, best_stats

# ==========================================
# UI RENDERING: ALLOCAZIONE AUTO
# ==========================================
def render_auto_allocation():
    st.title("Allocazione Multi-Modello (Auto)")
    df = st.session_state.df_historical
    all_assets = df.columns.tolist()
    
    # Parametri UI (simulati dalla sidebar originale dell'Auto)
    col1, col2, col3 = st.columns(3)
    lookback = col1.slider("Orizzonte Rolling (Anni)", 1, 10, 3)
    min_weight = col2.slider("Peso Minimo", 0.0, 0.2, 0.0)
    max_weight = col3.slider("Peso Massimo", 0.1, 1.0, 0.4)
    rf = 0.03 # Fisso per brevità
    freq = 'D' if st.session_state.data_freq == "Giornaliero" else 'W' if st.session_state.data_freq == "Settimanale" else 'M'
    
    mu_strat, sigma_strat, meta = prep_data(df, all_assets, lookback, freq)
    if mu_strat is None:
        st.error(meta)
        return
    
    tab1, tab2, tab3 = st.tabs(["📊 Dati", "📈 Markowitz", "🛡️ Antifragile (CVaR)"])
    
    with tab1:
        st.dataframe(df.tail())
        st.line_chart((df / df.iloc[0]) * 100)
        
    with tab2:
        w_mk = get_optimal_weights(mu_strat, sigma_strat, min_weight, max_weight, rf)
        if w_mk is not None:
            c1, c2 = st.columns(2)
            c1.dataframe(pd.DataFrame({"Asset": all_assets, "Peso": w_mk}).sort_values("Peso", ascending=False))
            fig = px.pie(values=w_mk, names=all_assets, title="Allocazione Markowitz", hole=0.5)
            c2.plotly_chart(fig)
            
    with tab3:
        w_cvar = get_cvar_weights(meta[1].values, min_weight, max_weight)
        if w_cvar is not None:
            c1, c2 = st.columns(2)
            c1.dataframe(pd.DataFrame({"Asset": all_assets, "Peso": w_cvar}).sort_values("Peso", ascending=False))
            fig = px.pie(values=w_cvar, names=all_assets, title="Allocazione Min-CVaR", hole=0.5)
            c2.plotly_chart(fig)

# ==========================================
# UI RENDERING: ALLOCAZIONE A 3 TIER
# ==========================================
def render_tier_allocation():
    st.title("Allocazione 3-Tier Model")
    df = st.session_state.df_historical
    assets = df.columns.tolist()
    ann_factor = st.session_state.ann_factor
    
    c1, c2 = st.columns(2)
    max_corr = c1.slider("Max Correlazione Ammessa", 0.0, 1.0, 1.0)
    min_w = c2.slider("Peso Minimo Asset (%)", 0, 33, 10) / 100.0
    
    with st.spinner('Calcolo Ottimizzazione Combinatoria... (potrebbe richiedere tempo se > 15 asset)'):
        # L1: Best Single
        sharpes = {a: get_advanced_stats([1], df[[a]].pct_change().dropna(), ann_factor)[2] for a in assets}
        best_single = max(sharpes, key=sharpes.get)
        l1_stats = get_advanced_stats([1], df[[best_single]].pct_change().dropna(), ann_factor)
        
        # L2 & L3
        l2_assets, l2_w, l2_stats = find_best_optimized_combination(df, 2, ann_factor, max_corr, max(min_w, 0.01))
        l3_assets, l3_w, l3_stats = find_best_optimized_combination(df, 3, ann_factor, max_corr, max(min_w, 0.01))
        
    st.subheader("Performance Tier")
    cols = st.columns(3)
    def draw_box(col, title, stats, assets_names):
        if stats is None: return col.warning(f"{title}: Nessun asset soddisfa i vincoli.")
        r, v, s, sort, mdd = stats
        col.info(f"**{title}**\n\nAsset: {assets_names}\n\nSharpe: {s:.2f}\n\nRend: {r*100:.1f}%\n\nMDD: {mdd*100:.1f}%")
        
    draw_box(cols[0], "Linea 1", l1_stats, best_single)
    draw_box(cols[1], "Linea 2", l2_stats, ", ".join(l2_assets) if l2_assets else "N/A")
    draw_box(cols[2], "Linea 3", l3_stats, ", ".join(l3_assets) if l3_assets else "N/A")

# ==========================================
# UI RENDERING: METHODOLOGY
# ==========================================
def render_methodology():
    st.title("Nota Metodologica & Assunzioni del Modello")
    st.markdown("""
    ### 1. Generazione e Trattamento Dati
    Il Data Engine acquisisce serie storiche in formato **Total Return (Adjusted Close)** tramite API (Yahoo Finance) o scraping NAV (Morningstar). L'uso di dati Total Return è critico per internalizzare l'effetto capitalizzato di dividendi e cedole. I dati vengono allineati temporalmente e i missing values riempiti tramite metodo *forward-fill* per evitare bias di look-ahead. I rendimenti sono calcolati in logica discreta (prospettiva multi-periodo).

    ### 2. Ottimizzazione del Portafoglio
    L'architettura unifica due filosofie quantitative:
    * **Motore Auto (Markowitz & CVaR):** Implementa l'ottimizzazione Media-Varianza classica (SLSQP) soggetta a vincoli lineari. Per mitigare la fragilità statistica di Markowitz (error-maximization), il modulo integra l'ottimizzazione **Min-CVaR (Expected Shortfall a coda 5%)**, spostando il focus dalla varianza simmetrica al Tail Risk puro.
    * **Motore a 3-Tier:** Utilizza un approccio combinatorio a forza bruta. Esplora il subspazio di tutte le combinazioni possibili di $k \in \{1, 2, 3\}$ asset, scartando i cluster che superano la soglia di correlazione imposta dall'utente, e applicando l'ottimizzatore sui subset filtrati.

    ### 3. Assunzioni Statistiche
    * **Stazionarietà:** I modelli classici assumono implicitamente che i rendimenti passati (vettori $\mu$ e matrici $\Sigma$) siano stimatori non distorti del futuro.
    * **Distribuzione:** L'ottimizzazione basata sulla varianza assume normalità dei rendimenti, assioma fallace nei mercati reali (leptocurtosi). Il modulo CVaR rilassa questa assunzione operando sui quantili storici empirici.

    ### 4. Limiti del Modello e Rischi (Overfitting)
    Il framework è profondamente esposto al rischio di **Curve-Fitting**. In particolare, il Motore a 3-Tier agisce come un selettore ex-post: estraendo la combinazione storica ottimale su un campione limitato, tende a sovrastimare massicciamente le performance out-of-sample. Si raccomanda l'uso di questi strumenti a fini diagnostici e di stress-testing, non come oracoli allocativi.
    """)

# ==========================================
# SIDEBAR & ROUTING MASTER
# ==========================================
with st.sidebar:
    st.header("⚙️ Data Engine Input")
    
    input_type = st.radio("Sorgente Dati", ["API (Ticker/ISIN)", "Upload File (CSV/Excel)"])
    tickers_input = []
    uploaded_file = None
    
    if input_type == "API (Ticker/ISIN)":
        raw_text = st.text_area("Inserisci Ticker/ISIN", "^GSPC\nSWDA.MI\nGC=F")
        tickers_input = [t if t not in ALIAS_MAP else ALIAS_MAP[t] for t in re.findall(r"[\w\.\-\^\=]+", raw_text.upper())]
        years = st.selectbox("Anni Storico", [1, 3, 5, 10], index=1)
    else:
        uploaded_file = st.file_uploader("Carica File", type=["csv", "xlsx"])
        
    timeframe = st.selectbox("Frequenza Dati", ["Giornaliero", "Settimanale", "Mensile"])
    
    if st.button("🚀 Genera Serie Storiche", use_container_width=True):
        with st.spinner("Acquisizione Dati in corso..."):
            df_temp = None
            if input_type == "API (Ticker/ISIN)" and tickers_input:
                df_temp = fetch_historical_data(tickers_input, years, timeframe)
            elif input_type == "Upload File (CSV/Excel)" and uploaded_file:
                # Logica semplificata upload (Ereditata da Allocazione Auto)
                try:
                    df_temp = pd.read_csv(uploaded_file, index_col=0, parse_dates=True) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
                    df_temp = df_temp.select_dtypes(include=[np.number]).dropna()
                except: st.error("Errore lettura file")
            
            if df_temp is not None and not df_temp.empty:
                st.session_state.df_historical = df_temp
                st.session_state.data_freq = timeframe
                st.session_state.ann_factor = 252 if timeframe == "Giornaliero" else 52 if timeframe == "Settimanale" else 12
                st.success("Dati Acquisiti e Condivisi in Memoria!")
            else:
                st.error("Fallimento estrazione dati.")
                
    st.divider()
    st.header("🧭 Navigazione Moduli")
    
    # Custom Router
    if st.button("📊 Modulo Auto (Multi-Model)", use_container_width=True): st.session_state.current_view = "auto"
    if st.button("🧬 Modulo 3-Tier Combinatorio", use_container_width=True): st.session_state.current_view = "tier"
    if st.button("📘 Nota Metodologica", use_container_width=True): st.session_state.current_view = "method"

# ==========================================
# RENDERER PRINCIPALE
# ==========================================
if st.session_state.df_historical is None and st.session_state.current_view != "method":
    st.info("👈 Compila i parametri nella sidebar e clicca su 'Genera Serie Storiche' per avviare il motore quantitativo condiviso.")
else:
    if st.session_state.current_view == "auto":
        render_auto_allocation()
    elif st.session_state.current_view == "tier":
        render_tier_allocation()
    elif st.session_state.current_view == "method":
        render_methodology()