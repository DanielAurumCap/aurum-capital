import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
import statsmodels.formula.api as smf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Aurum Capital",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=DM+Sans:wght@300;400;500&family=IBM+Plex+Mono:wght@300;400;500&display=swap');
:root {
    --gold:#c9a84c;--gold2:#e8c96a;--em:#1a5c3a;--em2:#246b45;
    --bg:#060a08;--bg2:#0a100d;--border:#141f17;--border2:#1e2e22;
    --text:#e2ddd4;--muted:#435447;--muted2:#2a3a2e;
    --green:#2ecc71;--red:#e05252;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg);color:var(--text);}
.main{background-color:var(--bg);}
section[data-testid="stSidebar"]{background-color:var(--bg2);border-right:1px solid var(--border);}
.aurum-brand{font-family:'Cormorant Garamond',serif;font-size:32px;font-weight:300;letter-spacing:0.06em;line-height:1;margin-bottom:4px;}
.aurum-brand span{background:linear-gradient(135deg,#c9a84c,#e8c96a);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:600;}
.aurum-sub{font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:0.3em;text-transform:uppercase;color:var(--muted);margin-bottom:16px;}
.aurum-rule{height:1px;background:linear-gradient(90deg,var(--gold),transparent);opacity:0.4;margin-bottom:24px;}
.kpi-card{background:var(--bg2);border:1px solid var(--border);border-top:1.5px solid var(--gold);padding:18px 20px;margin-bottom:4px;}
.kpi-label{font-family:'IBM Plex Mono',monospace;font-size:8px;letter-spacing:0.22em;text-transform:uppercase;color:var(--muted);margin-bottom:10px;}
.kpi-value{font-family:'IBM Plex Mono',monospace;font-size:26px;font-weight:400;letter-spacing:-0.02em;line-height:1;}
.kpi-sub{font-family:'IBM Plex Mono',monospace;font-size:9px;color:var(--muted);margin-top:6px;}
.gold-text{color:var(--gold);}.green-text{color:var(--green);}.red-text{color:var(--red);}.white-text{color:#f0ebe0;}
.section-header{font-family:'IBM Plex Mono',monospace;font-size:8px;letter-spacing:0.28em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);padding-bottom:8px;margin:24px 0 16px;}
.metric-row{background:var(--bg2);border:1px solid var(--border);padding:14px 18px;margin-bottom:1px;}
.metric-name{font-family:'IBM Plex Mono',monospace;font-size:8px;letter-spacing:0.18em;text-transform:uppercase;color:var(--muted);margin-bottom:6px;}
.metric-port{font-family:'IBM Plex Mono',monospace;font-size:18px;color:#f0ebe0;}
.metric-port.win{color:var(--green);}
.stButton>button{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.14em;text-transform:uppercase;background:linear-gradient(135deg,#3d2e0a,#2a1f06);color:var(--gold);border:1px solid var(--gold);padding:10px 20px;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TEXTOS BILINGÜE
# ══════════════════════════════════════════════════════════════════
TEXTOS = {
    "ES": {
        "monitor":"Monitor de Portafolio","betas":"Betas Cuantílicas",
        "optimizador":"Optimizador","backtesting":"Backtesting",
        "configuracion":"Configuración","indice":"Índice de referencia",
        "fecha_inicio":"Fecha inicio","fecha_fin":"Fecha fin",
        "correr":"▶  Correr modelo completo","actualizar":"⟳  Actualizar precios",
        "mercado_abierto":"Mercado Abierto","mercado_cerrado":"Mercado Cerrado",
        "ret_acum":"Retorno acumulado","alpha_jensen":"Alpha de Jensen",
        "sharpe":"Sharpe ratio","max_dd":"Max drawdown",
        "vs_benchmark":"vs benchmark","pesos":"Pesos del portafolio",
        "evolucion":"Evolución del portafolio","riesgo":"Riesgo y rendimiento",
        "ret_geom":"Retorno geométrico anual","volatilidad":"Volatilidad anual",
        "cvar":"CVaR 95% diario","calmar":"Calmar ratio",
        "info_ratio":"Information ratio","beta_real":"Beta realizada",
        "corriendo":"Corriendo el modelo...","descargando":"Descargando datos de mercado...",
        "calculando":"Calculando betas cuantílicas...","optimizando":"Optimizando portafolio...",
        "backtesting_msg":"Corriendo backtesting...","listo":"Modelo ejecutado correctamente",
        "anual_chart":"Retorno anual · Años completos","dd_chart":"Drawdown histórico",
        "val_acum":"Valor acumulado vs Benchmark · Base 100","rf":"Tasa libre de riesgo (Rf)",
        "portafolio":"Portafolio","benchmark":"Benchmark",
    },
    "EN": {
        "monitor":"Portfolio Monitor","betas":"Quantile Betas",
        "optimizador":"Optimizer","backtesting":"Backtesting",
        "configuracion":"Settings","indice":"Reference index",
        "fecha_inicio":"Start date","fecha_fin":"End date",
        "correr":"▶  Run full model","actualizar":"⟳  Refresh prices",
        "mercado_abierto":"Market Open","mercado_cerrado":"Market Closed",
        "ret_acum":"Cumulative return","alpha_jensen":"Jensen's Alpha",
        "sharpe":"Sharpe ratio","max_dd":"Max drawdown",
        "vs_benchmark":"vs benchmark","pesos":"Portfolio weights",
        "evolucion":"Portfolio evolution","riesgo":"Risk & return",
        "ret_geom":"Annual geometric return","volatilidad":"Annual volatility",
        "cvar":"CVaR 95% daily","calmar":"Calmar ratio",
        "info_ratio":"Information ratio","beta_real":"Realized beta",
        "corriendo":"Running the model...","descargando":"Downloading market data...",
        "calculando":"Computing quantile betas...","optimizando":"Optimizing portfolio...",
        "backtesting_msg":"Running backtesting...","listo":"Model executed successfully",
        "anual_chart":"Annual return · Full years","dd_chart":"Historical drawdown",
        "val_acum":"Cumulative value vs Benchmark · Base 100","rf":"Risk-free rate (Rf)",
        "portafolio":"Portfolio","benchmark":"Benchmark",
    }
}

# ══════════════════════════════════════════════════════════════════
# UNIVERSOS
# ══════════════════════════════════════════════════════════════════
TICKERS_IPC = [
    "AMXB.MX","WALMEX.MX","GMEXICOB.MX","FEMSAUBD.MX","GFINBURO.MX",
    "GFNORTEO.MX","CEMEXCPO.MX","TLEVISACPO.MX","BIMBOA.MX","ALSEA.MX",
    "ASURB.MX","OMAB.MX","GAPB.MX","KIMBERA.MX","LABB.MX","MEGACPO.MX",
    "PINFRA.MX","GRUMAB.MX","GCARSOA1.MX","ORBIA.MX","PE&OLES.MX",
    "VESTA.MX","BOLSAA.MX","GENTERA.MX","CUERVO.MX","CHDRAUIB.MX",
    "BBAJIOO.MX","Q.MX","GCC.MX","AC.MX","RA.MX","LACOMERUBC.MX"
]

TICKERS_SP500 = [
    "A","AAPL","ABBV","ABT","ACGL","ACN","ADBE","ADI","ADM","ADP","ADSK",
    "AEE","AEP","AES","AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN","ALL",
    "ALLE","AMAT","AMCR","AMD","AME","AMGN","AMP","AMT","AMZN","ANET",
    "AON","AOS","APA","APD","APH","APO","APTV","ARE","ATO","AVB","AVGO",
    "AVY","AWK","AXON","AXP","AZO","BA","BAC","BALL","BAX","BBY","BDX",
    "BEN","BF-B","BG","BIIB","BK","BKNG","BKR","BLDR","BLK","BMY","BR",
    "BRK-B","BRO","BSX","BX","BXP","C","CAG","CAH","CARR","CAT","CB",
    "CBOE","CBRE","CCI","CCL","CDNS","CDW","CF","CFG","CHD","CHRW","CHTR",
    "CI","CINF","CL","CLX","CMCSA","CME","CMG","CMI","CMS","CNC","CNP",
    "COF","COO","COP","COR","COST","CPAY","CPB","CPRT","CPT","CRL","CRM",
    "CRWD","CSCO","CSGP","CSX","CTAS","CTRA","CTSH","CTVA","CVS","CVX",
    "CZR","D","DAL","DAY","DD","DDOG","DE","DECK","DELL","DG","DGX","DHI",
    "DHR","DIS","DLR","DLTR","DOC","DOV","DOW","DPZ","DRI","DTE","DUK",
    "DVA","DVN","DXCM","EA","EBAY","ECL","ED","EFX","EG","EIX","EL","ELV",
    "EMN","EMR","ENPH","EOG","EPAM","EQIX","EQR","EQT","ERIE","ES","ESS",
    "ETN","ETR","EVRG","EW","EXC","EXPD","EXPE","EXR","F","FANG","FAST",
    "FCX","FDS","FDX","FE","FFIV","FI","FICO","FIS","FITB","FOX","FOXA",
    "FRT","FSLR","FTNT","FTV","GD","GDDY","GE","GEN","GILD","GIS","GL",
    "GLW","GM","GNRC","GOOG","GOOGL","GPC","GPN","GRMN","GS","GWW","HAL",
    "HAS","HBAN","HCA","HD","HIG","HII","HLT","HOLX","HON","HPE","HPQ",
    "HRL","HSIC","HST","HSY","HUBB","HUM","HWM","IBM","ICE","IDXX","IEX",
    "IFF","INCY","INTC","INTU","INVH","IP","IPG","IQV","IR","IRM","ISRG",
    "IT","ITW","IVZ","J","JBHT","JBL","JCI","JKHY","JNJ","JPM","K","KDP",
    "KEY","KEYS","KHC","KIM","KKR","KLAC","KMB","KMI","KMX","KO","KR","L",
    "LDOS","LEN","LH","LHX","LII","LIN","LKQ","LLY","LMT","LNT","LOW",
    "LRCX","LUV","LVS","LW","LYB","LYV","MA","MAA","MAR","MAS","MCD",
    "MCHP","MCK","MCO","MDLZ","MDT","MET","META","MGM","MHK","MKC","MKTX",
    "MLM","MMC","MMM","MNST","MO","MOH","MOS","MPC","MPWR","MRK","MRNA",
    "MS","MSCI","MSFT","MSI","MTB","MTCH","MTD","MU","NCLH","NDAQ","NDSN",
    "NEE","NEM","NFLX","NI","NKE","NOC","NOW","NRG","NSC","NTAP","NTRS",
    "NUE","NVDA","NVR","NWS","NWSA","NXPI","O","ODFL","OKE","OMC","ON",
    "ORCL","ORLY","OTIS","OXY","PANW","PAYC","PAYX","PCAR","PCG","PEG",
    "PEP","PFE","PFG","PG","PGR","PH","PHM","PKG","PLD","PM","PNC","PNR",
    "PNW","PODD","POOL","PPG","PPL","PRU","PSA","PSKY","PSX","PTC","PWR",
    "PYPL","QCOM","RCL","REG","REGN","RF","RJF","RL","RMD","ROK","ROL",
    "ROP","ROST","RSG","RTX","RVTY","SBAC","SBUX","SCHW","SHW","SJM","SLB",
    "SMCI","SNA","SNPS","SO","SPG","SPGI","SRE","STE","STLD","STT","STX",
    "STZ","SW","SWK","SWKS","SYF","SYK","SYY","T","TAP","TDG","TDY","TECH",
    "TEL","TER","TFC","TGT","TJX","TKO","TMO","TMUS","TPL","TPR","TRGP",
    "TRMB","TROW","TRV","TSCO","TSLA","TSN","TT","TTD","TTWO","TXN","TXT",
    "TYL","UAL","UBER","UDR","UHS","ULTA","UNH","UNP","UPS","URI","USB",
    "V","VICI","VLO","VMC","VRSK","VRSN","VRTX","VST","VTR","VTRS","VZ",
    "WAB","WAT","WBA","WBD","WDAY","WDC","WEC","WELL","WFC","WM","WMB",
    "WMT","WRB","WSM","WST","WTW","WY","WYNN","XEL","XOM","XYL","XYZ",
    "YUM","ZBH","ZBRA","ZTS"
]

UNIVERSOS = {
    "IPC (BMV — México)": {
        "indice": "^MXX",
        "tickers": TICKERS_IPC,
        "rf_default": 0.065,
        "moneda": "MXN",
    },
    "S&P 500 (USA)": {
        "indice": "^GSPC",
        "tickers": TICKERS_SP500,
        "rf_default": 0.053,
        "moneda": "USD",
    },
    "Personalizado": {
        "indice": "",
        "tickers": [],
        "rf_default": 0.05,
        "moneda": "USD",
    }
}

# ══════════════════════════════════════════════════════════════════
# FUNCIONES
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def descargar_datos(tickers, indice, fecha_inicio, fecha_fin):
    try:
        raw_idx = yf.download(indice, start=fecha_inicio, end=fecha_fin,
                              auto_adjust=True, progress=False)
        raw_stk = yf.download(tickers, start=fecha_inicio, end=fecha_fin,
                              auto_adjust=True, progress=False)
        precios_idx = raw_idx["Close"].squeeze()
        if isinstance(raw_stk["Close"], pd.Series):
            precios_stk = raw_stk["Close"].to_frame()
        else:
            precios_stk = raw_stk["Close"]
        ret_idx = np.log(precios_idx / precios_idx.shift(1)).dropna()
        ret_stk = np.log(precios_stk / precios_stk.shift(1)).dropna()
        ret_stk = ret_stk.reindex(ret_idx.index).dropna(how="all")
        ret_idx  = ret_idx.reindex(ret_stk.index)
        return precios_idx, precios_stk, ret_idx, ret_stk
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return None, None, None, None

def calcular_betas_cuantilicas(ret_stk, ret_idx, quantiles=[0.10,0.90]):
    resultados = {}
    for ticker in ret_stk.columns:
        serie = ret_stk[ticker].dropna()
        idx_c = ret_idx.index.intersection(serie.index)
        if len(idx_c) < 100:
            continue
        df_r = pd.DataFrame({
            "r_accion": serie.loc[idx_c].values,
            "r_ipc":    ret_idx.loc[idx_c].values
        }).dropna()
        betas_q = {}
        for q in quantiles:
            try:
                m = smf.quantreg("r_accion ~ r_ipc", data=df_r).fit(q=q, max_iter=500)
                betas_q[f"beta_q{int(q*100)}"] = m.params["r_ipc"]
            except:
                betas_q[f"beta_q{int(q*100)}"] = np.nan
        resultados[ticker] = betas_q
    return pd.DataFrame(resultados).T

def calcular_scores(df_betas, ret_stk, ret_idx):
    scores = {}
    for ticker in df_betas.index:
        if ticker not in ret_stk.columns:
            continue
        serie = ret_stk[ticker].dropna()
        idx_c = ret_idx.index.intersection(serie.index)
        if len(idx_c) < 100:
            continue
        exceso = serie.loc[idx_c].values - ret_idx.loc[idx_c].values
        p = np.mean(exceso > 0)
        ganancias = exceso[exceso > 0]
        perdidas  = exceso[exceso < 0]
        b = ganancias.mean() / abs(perdidas.mean()) if len(perdidas) > 0 and len(ganancias) > 0 else 1.0
        f = max(0.03, (p * b - (1-p)) / b)
        b10 = df_betas.loc[ticker, "beta_q10"] if "beta_q10" in df_betas.columns else 1.0
        b90 = df_betas.loc[ticker, "beta_q90"] if "beta_q90" in df_betas.columns else 1.0
        asimetria = float(b90 - b10) if not (np.isnan(b90) or np.isnan(b10)) else 0
        scores[ticker] = {
            "p_kelly": round(p, 3),
            "b_kelly": round(b, 3),
            "f_kelly": round(f, 4),
            "asimetria": round(asimetria, 3),
        }
    return pd.DataFrame(scores).T

def optimizar_kelly(scores_df, ret_stk, caps=None, mins=None):
    tickers = [t for t in scores_df.index if t in ret_stk.columns]
    if len(tickers) < 2:
        return pd.Series(dtype=float), np.array([])
    f_raw = scores_df.loc[tickers, "f_kelly"].values.astype(float)
    f_raw = np.nan_to_num(f_raw, nan=0.0)
    if caps is None:
        caps = {t: 0.25 for t in tickers}
    if mins is None:
        mins = {t: 0.03 for t in tickers}
    MINS = [mins.get(t, 0.03) for t in tickers]
    CAPS = [caps.get(t, 0.25) for t in tickers]
    f_norm = np.clip(f_raw, MINS, CAPS)
    if f_norm.sum() > 0:
        f_norm = f_norm / f_norm.sum()
    else:
        f_norm = np.array([1/len(tickers)] * len(tickers))
    return pd.Series(f_norm, index=tickers), f_norm

def correr_backtesting(pesos_series, precios_stk, precios_idx, capital=1_000_000):
    emisoras = pesos_series.index.tolist()
    w = pesos_series.values
    precios_bt  = precios_stk[emisoras].copy()
    precios_bt  = precios_bt.reindex(precios_idx.index).dropna(how="all")
    precios_ipc = precios_idx.reindex(precios_bt.index)
    fecha_ini   = precios_bt.dropna().index[0]
    precios_bt  = precios_bt.loc[fecha_ini:]
    precios_ipc = precios_ipc.loc[fecha_ini:]
    acciones = {}
    for i, em in enumerate(emisoras):
        px = precios_bt[em].iloc[0]
        if not np.isnan(float(px)):
            acciones[em] = (capital * w[i]) / float(px)
    fechas = precios_bt.index
    meses_reb = pd.Series(fechas).groupby(
        pd.Series(fechas).dt.to_period("M")
    ).last().values
    vals_port = []
    vals_ipc  = []
    ipc_ini   = float(precios_ipc.iloc[0])
    for fecha in fechas:
        valor = sum(
            acciones.get(em, 0) * (float(precios_bt.loc[fecha, em])
             if not np.isnan(float(precios_bt.loc[fecha, em])) else 0)
            for em in emisoras
        )
        vals_port.append(valor)
        vals_ipc.append(capital * float(precios_ipc.loc[fecha]) / ipc_ini)
        if fecha in meses_reb and fecha != fechas[-1]:
            for i, em in enumerate(emisoras):
                px = precios_bt.loc[fecha, em]
                if not np.isnan(float(px)) and valor > 0:
                    acciones[em] = (valor * w[i]) / float(px)
    return pd.Series(vals_port, index=fechas), pd.Series(vals_ipc, index=fechas)

def calcular_metricas(serie_port, serie_ipc, rf):
    ret_p = serie_port.pct_change().dropna()
    ret_i = serie_ipc.pct_change().dropna()
    idx_c = ret_p.index.intersection(ret_i.index)
    ret_p, ret_i = ret_p.loc[idx_c], ret_i.loc[idx_c]
    T = len(ret_p)
    if T < 10:
        return {k: 0.0 for k in ["ret_geom","vol","sharpe","cvar","max_dd",
                                   "calmar","ir","beta","ret_acum","alpha_j"]}, ret_p, ret_i, ret_p*0
    ret_geom  = (np.prod(1 + ret_p) ** (252/T) - 1) * 100
    vol       = ret_p.std() * np.sqrt(252) * 100
    sharpe    = (ret_geom - rf*100) / vol if vol > 0 else 0
    va        = (1 + ret_p).cumprod()
    dd        = (va - va.cummax()) / va.cummax()
    max_dd    = dd.min() * 100
    var95     = np.percentile(ret_p, 5)
    cvar95    = ret_p[ret_p <= var95].mean() * 100
    calmar    = ret_geom / abs(max_dd) if max_dd != 0 else 0
    exceso    = ret_p.values - ret_i.values
    ir        = exceso.mean() / exceso.std() * np.sqrt(252) if exceso.std() > 0 else 0
    cov_m     = np.cov(ret_p.values, ret_i.values)
    beta_r    = cov_m[0,1] / cov_m[1,1] if cov_m[1,1] != 0 else 1.0
    ret_acum  = (serie_port.iloc[-1] / serie_port.iloc[0] - 1) * 100
    ret_geom_i = (np.prod(1 + ret_i) ** (252/T) - 1) * 100
    alpha_j   = (ret_geom/100) - (rf + beta_r * (ret_geom_i/100 - rf))
    return {
        "ret_geom":  round(ret_geom, 2),
        "vol":       round(vol, 2),
        "sharpe":    round(sharpe, 3),
        "cvar":      round(cvar95, 3),
        "max_dd":    round(max_dd, 2),
        "calmar":    round(calmar, 3),
        "ir":        round(ir, 3),
        "beta":      round(beta_r, 3),
        "ret_acum":  round(ret_acum, 2),
        "alpha_j":   round(alpha_j * 100, 2),
    }, ret_p, ret_i, dd

def mercado_abierto():
    ahora = datetime.now()
    hora  = ahora.hour + ahora.minute / 60
    return ahora.weekday() < 5 and 8.5 <= hora <= 15.0

# ══════════════════════════════════════════════════════════════════
# COLORES Y LAYOUT PLOTLY
# ══════════════════════════════════════════════════════════════════
C = {"gold":"#c9a84c","gold2":"#e8c96a","em":"#1a5c3a","em2":"#2ecc71",
     "bg":"#060a08","bg2":"#0a100d","border":"#141f17","muted":"#435447",
     "red":"#e05252","text":"#e2ddd4"}

LAY = dict(
    paper_bgcolor=C["bg2"], plot_bgcolor=C["bg"],
    font=dict(family="IBM Plex Mono", color=C["muted"], size=10),
    margin=dict(l=40,r=20,t=30,b=40),
    xaxis=dict(gridcolor=C["border"],showgrid=True,zeroline=False,
               linecolor=C["border"],tickfont=dict(size=9)),
    yaxis=dict(gridcolor=C["border"],showgrid=True,zeroline=False,
               linecolor=C["border"],tickfont=dict(size=9)),
    legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=9)),
    hovermode="x unified",
)

def chart_valor(serie_port, serie_ipc, t):
    bp = serie_port / serie_port.iloc[0] * 100
    bi = serie_ipc  / serie_ipc.iloc[0]  * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bi.index,y=bi.values,name=t["benchmark"],
        line=dict(color=C["em"],width=1.5,dash="dot"),opacity=0.7))
    fig.add_trace(go.Scatter(x=bp.index,y=bp.values,name="Aurum Portfolio",
        line=dict(color=C["gold"],width=2.5),
        fill="tonexty",fillcolor="rgba(201,168,76,0.05)"))
    fig.update_layout(**LAY,
        title=dict(text=t["val_acum"],font=dict(size=10,color=C["muted"]),x=0),
        height=300)
    return fig

def chart_dd(dd, ret_i, t):
    va_i = (1+ret_i).cumprod()
    dd_i = (va_i-va_i.cummax())/va_i.cummax()*100
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=dd_i.index,y=dd_i.values,name=t["benchmark"],
        line=dict(color=C["em"],width=1,dash="dot"),
        fill="tozeroy",fillcolor="rgba(26,92,58,0.08)"))
    fig.add_trace(go.Scatter(x=dd.index,y=dd.values*100,name="Aurum Portfolio",
        line=dict(color=C["gold"],width=1.5),
        fill="tozeroy",fillcolor="rgba(201,168,76,0.08)"))
    fig.update_layout(**LAY,
        title=dict(text=t["dd_chart"],font=dict(size=10,color=C["muted"]),x=0),
        height=220,yaxis=dict(**LAY["yaxis"],ticksuffix="%"))
    return fig

def chart_anual(ret_p, ret_i, t):
    años = [a for a in sorted(ret_p.index.year.unique())
            if ret_p[ret_p.index.year==a].count()>200]
    rp_a = [((1+ret_p[ret_p.index.year==a]).prod()-1)*100 for a in años]
    ri_a = [((1+ret_i[ret_i.index.year==a]).prod()-1)*100 for a in años]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=años,y=rp_a,name="Aurum Portfolio",
        marker_color=[C["gold"] if r>0 else C["red"] for r in rp_a],
        opacity=0.85,width=0.35,offset=-0.18))
    fig.add_trace(go.Bar(x=años,y=ri_a,name=t["benchmark"],
        marker_color=[C["em"] if r>0 else "#4a1a1a" for r in ri_a],
        opacity=0.7,width=0.35,offset=0.18))
    fig.update_layout(**LAY,
        title=dict(text=t["anual_chart"],font=dict(size=10,color=C["muted"]),x=0),
        height=220,barmode="overlay",
        yaxis=dict(**LAY["yaxis"],ticksuffix="%"))
    return fig

def chart_pesos(pesos_series):
    nombres = [tk.replace(".MX","") for tk in pesos_series.index]
    valores = pesos_series.values * 100
    colores = [C["gold"] if v>=12 else C["em2"] if v>=8 else C["muted"] for v in valores]
    fig = go.Figure(go.Bar(
        x=valores, y=nombres, orientation="h",
        marker_color=colores, opacity=0.9,
        text=[f"{v:.1f}%" for v in valores],
        textposition="outside",
        textfont=dict(size=9,color=C["text"]),
    ))
    fig.update_layout(**LAY,
        title=dict(text="Pesos Kelly por emisora",font=dict(size=10,color=C["muted"]),x=0),
        height=max(280,len(nombres)*28),
        xaxis=dict(**LAY["xaxis"],ticksuffix="%"),
        yaxis=dict(**LAY["yaxis"],categoryorder="total ascending"))
    return fig

def chart_betas(df_betas):
    cols = [c for c in ["beta_q10","beta_q25","beta_q75","beta_q90"] if c in df_betas.columns]
    q_labels = [c.replace("beta_q","Q") for c in cols]
    fig = go.Figure()
    palette = [C["gold"],C["em2"],"#3498db",C["red"],"#9b59b6","#e67e22",
               "#1abc9c","#e74c3c","#2980b9","#27ae60"]
    for i, ticker in enumerate(df_betas.index[:10]):
        vals = [df_betas.loc[ticker,c] for c in cols]
        if any(np.isnan(v) for v in vals):
            continue
        fig.add_trace(go.Scatter(
            x=q_labels, y=vals, name=ticker.replace(".MX",""),
            mode="lines+markers",
            line=dict(color=palette[i%len(palette)],width=1.8),
            marker=dict(size=5)))
    fig.add_hline(y=1.0,line_dash="dash",line_color=C["muted"],
                  annotation_text="Beta=1",annotation_font_size=9)
    fig.update_layout(**LAY,
        title=dict(text="Curva de betas cuantílicas",font=dict(size=10,color=C["muted"]),x=0),
        height=320)
    return fig

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class='aurum-brand'><span>Aurum</span></div>
    <div class='aurum-brand' style='font-size:18px;letter-spacing:0.18em;color:#435447;margin-top:2px;'>Capital</div>
    <div class='aurum-sub'>Portfolio Intelligence · by Daniel</div>
    <div class='aurum-rule'></div>
    """, unsafe_allow_html=True)

    lang = st.radio("", ["ES","EN"], horizontal=True, label_visibility="collapsed")
    t = TEXTOS[lang]

    st.markdown("---")

    pagina = st.radio("",
        [t["monitor"],t["betas"],t["optimizador"],t["backtesting"],t["configuracion"]],
        label_visibility="collapsed")

    st.markdown("---")

    universo_sel = st.selectbox(t["indice"], list(UNIVERSOS.keys()))
    config_u = UNIVERSOS[universo_sel]

    if universo_sel == "Personalizado":
        custom_idx     = st.text_input("Ticker del índice", "^GSPC")
        custom_tickers = st.text_area("Tickers separados por coma", "AAPL,MSFT,GOOGL")
        config_u["indice"]  = custom_idx
        config_u["tickers"] = [x.strip() for x in custom_tickers.split(",") if x.strip()]

    # Fechas — fin automático a hoy, inicio 5 años atrás
    hoy           = datetime.today().date()
    fecha_ini_def = hoy - timedelta(days=365*5)
    fecha_inicio  = st.date_input(t["fecha_inicio"], fecha_ini_def)
    fecha_fin     = st.date_input(t["fecha_fin"], hoy)

    # Rf slider 0% a 20% con default según universo
    rf_default = float(config_u["rf_default"])
    rf = st.slider(t["rf"], 0.0, 0.20, rf_default, 0.005, format="%.2f%%")

    st.markdown("---")
    st.markdown(f'<div style="font-family:IBM Plex Mono;font-size:8px;color:#2a3a2e;letter-spacing:0.1em;">v2.2.0 · Quantile-Kelly Model</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
if "modelo_corrido" not in st.session_state:
    st.session_state.modelo_corrido = False
if "resultados" not in st.session_state:
    st.session_state.resultados = {}

# ══════════════════════════════════════════════════════════════════
# MONITOR
# ══════════════════════════════════════════════════════════════════
if pagina == t["monitor"]:

    col_title, col_badge = st.columns([3,1])
    with col_title:
        st.markdown(f"<h1 style='font-family:Cormorant Garamond,serif;font-size:28px;font-weight:300;letter-spacing:0.06em;color:#fff;margin-bottom:4px;'>{t['monitor']}</h1>", unsafe_allow_html=True)
    with col_badge:
        estado = t["mercado_abierto"] if mercado_abierto() else t["mercado_cerrado"]
        color  = "#3db87a" if mercado_abierto() else "#e05252"
        st.markdown(f"<div style='font-family:IBM Plex Mono;font-size:9px;letter-spacing:0.14em;color:{color};border:1px solid {color}44;padding:6px 10px;margin-top:8px;text-align:center;'>● {estado}</div>", unsafe_allow_html=True)

    if st.button(t["correr"], use_container_width=False):
        with st.spinner(t["corriendo"]):
            prog = st.progress(0, text=t["descargando"])

            px_idx, px_stk, ret_idx, ret_stk = descargar_datos(
                config_u["tickers"], config_u["indice"],
                str(fecha_inicio), str(fecha_fin)
            )

            if px_idx is None:
                st.error("Error descargando datos. Verifica tu conexión.")
                st.stop()

            prog.progress(25, text=t["calculando"])
            df_betas = calcular_betas_cuantilicas(ret_stk, ret_idx)
            scores   = calcular_scores(df_betas, ret_stk, ret_idx)

            prog.progress(50, text=t["optimizando"])
            pesos_s, _ = optimizar_kelly(scores, ret_stk)

            # Inicializar variables
            serie_port = None
            serie_ipc  = None
            metricas   = None
            ret_p      = None
            ret_i      = None
            dd         = None

            prog.progress(75, text=t["backtesting_msg"])
            if len(pesos_s) > 0:
                serie_port, serie_ipc = correr_backtesting(pesos_s, px_stk, px_idx)
                metricas, ret_p, ret_i, dd = calcular_metricas(serie_port, serie_ipc, rf)

            prog.progress(100, text=t["listo"])

            if serie_port is not None:
                st.session_state.modelo_corrido = True
                st.session_state.resultados = {
                    "px_idx":px_idx,"px_stk":px_stk,
                    "ret_idx":ret_idx,"ret_stk":ret_stk,
                    "df_betas":df_betas,"scores":scores,
                    "pesos":pesos_s,"serie_port":serie_port,
                    "serie_ipc":serie_ipc,"metricas":metricas,
                    "ret_p":ret_p,"ret_i":ret_i,"dd":dd,
                }
            else:
                st.warning("No se generaron pesos. Verifica el universo y las fechas.")

    if st.session_state.modelo_corrido:
        r = st.session_state.resultados
        m = r["metricas"]

        st.markdown(f"<div class='section-header'>{t['riesgo'].upper()}</div>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)

        with c1:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>{t['ret_acum']}</div>
                <div class='kpi-value green-text'>+{m['ret_acum']:.1f}%</div>
                <div class='kpi-sub'>{t['vs_benchmark']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='kpi-card' style='border-top-color:var(--gold)'>
                <div class='kpi-label'>{t['alpha_jensen']}</div>
                <div class='kpi-value gold-text'>{m['alpha_j']:.2f}%</div>
                <div class='kpi-sub'>Rf={rf*100:.1f}% · β={m['beta']:.3f}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>{t['sharpe']}</div>
                <div class='kpi-value white-text'>{m['sharpe']:.3f}</div>
                <div class='kpi-sub'>Rf={rf*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class='kpi-card' style='border-top-color:#2a3a2e'>
                <div class='kpi-label'>{t['max_dd']}</div>
                <div class='kpi-value white-text'>{m['max_dd']:.2f}%</div>
                <div class='kpi-sub'>vs benchmark</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-header'>{t['evolucion'].upper()}</div>", unsafe_allow_html=True)

        col_chart, col_pesos = st.columns([7,3])
        with col_chart:
            st.plotly_chart(chart_valor(r["serie_port"],r["serie_ipc"],t), use_container_width=True)
        with col_pesos:
            st.plotly_chart(chart_pesos(r["pesos"]), use_container_width=True)

        st.markdown(f"<div class='section-header'>{t['riesgo'].upper()}</div>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        metricas_display = [
            (t["ret_geom"],  f"{m['ret_geom']}%",  "win"),
            (t["volatilidad"],f"{m['vol']}%",        ""),
            (t["cvar"],      f"{m['cvar']}%",        ""),
            (t["calmar"],    f"{m['calmar']}",       "win"),
            (t["info_ratio"],f"{m['ir']}",           "win"),
            (t["beta_real"], f"{m['beta']}",         ""),
        ]
        for i,(label,val,cls) in enumerate(metricas_display):
            col = [c1,c2,c3][i%3]
            with col:
                st.markdown(f"""<div class='metric-row'>
                    <div class='metric-name'>{label.upper()}</div>
                    <div class='metric-port {cls}'>{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"<div class='section-header'>DRAWDOWN & RETORNOS ANUALES</div>", unsafe_allow_html=True)
        col_dd, col_an = st.columns(2)
        with col_dd:
            st.plotly_chart(chart_dd(r["dd"],r["ret_i"],t), use_container_width=True)
        with col_an:
            st.plotly_chart(chart_anual(r["ret_p"],r["ret_i"],t), use_container_width=True)

    else:
        st.markdown(f"""
        <div style='text-align:center;padding:80px 40px;'>
            <div style='font-family:Cormorant Garamond,serif;font-size:48px;font-weight:300;color:#141f17;letter-spacing:0.06em;margin-bottom:16px;'>Aurum Capital</div>
            <div style='font-family:IBM Plex Mono;font-size:10px;letter-spacing:0.24em;text-transform:uppercase;color:#1e2e22;margin-bottom:32px;'>Portfolio Intelligence System</div>
            <div style='font-family:IBM Plex Mono;font-size:10px;color:#2a4a30;letter-spacing:0.1em;'>Configura los parámetros y presiona<br><span style='color:#c9a84c;'>▶ {t["correr"]}</span></div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# BETAS
# ══════════════════════════════════════════════════════════════════
elif pagina == t["betas"]:
    st.markdown(f"<h1 style='font-family:Cormorant Garamond,serif;font-size:28px;font-weight:300;color:#fff;margin-bottom:20px;'>{t['betas']}</h1>", unsafe_allow_html=True)
    if not st.session_state.modelo_corrido:
        st.info("Corre el modelo primero desde el Monitor.")
    else:
        r = st.session_state.resultados
        st.plotly_chart(chart_betas(r["df_betas"]), use_container_width=True)
        st.markdown("<div class='section-header'>SCORES KELLY POR EMISORA</div>", unsafe_allow_html=True)
        df_show = r["scores"].copy()
        df_show.index = [i.replace(".MX","") for i in df_show.index]
        df_show = df_show.sort_values("f_kelly", ascending=False)
        df_show.columns = ["P (prob ganancia)","B (ratio gan/pérd)","F* Kelly","Asimetría β"]
        st.dataframe(df_show.style.background_gradient(cmap="Greens", subset=["F* Kelly"]),
                    use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# OPTIMIZADOR
# ══════════════════════════════════════════════════════════════════
elif pagina == t["optimizador"]:
    st.markdown(f"<h1 style='font-family:Cormorant Garamond,serif;font-size:28px;font-weight:300;color:#fff;margin-bottom:20px;'>{t['optimizador']}</h1>", unsafe_allow_html=True)
    if not st.session_state.modelo_corrido:
        st.info("Corre el modelo primero desde el Monitor.")
    else:
        r = st.session_state.resultados
        st.plotly_chart(chart_pesos(r["pesos"]), use_container_width=True)
        st.markdown("<div class='section-header'>TABLA DE PESOS</div>", unsafe_allow_html=True)
        df_pesos = pd.DataFrame({
            "Ticker": [tk.replace(".MX","") for tk in r["pesos"].index],
            "Peso %": (r["pesos"].values * 100).round(2),
        }).sort_values("Peso %", ascending=False)
        st.dataframe(df_pesos, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# BACKTESTING
# ══════════════════════════════════════════════════════════════════
elif pagina == t["backtesting"]:
    st.markdown(f"<h1 style='font-family:Cormorant Garamond,serif;font-size:28px;font-weight:300;color:#fff;margin-bottom:20px;'>{t['backtesting']}</h1>", unsafe_allow_html=True)
    if not st.session_state.modelo_corrido:
        st.info("Corre el modelo primero desde el Monitor.")
    else:
        r = st.session_state.resultados
        m = r["metricas"]
        st.plotly_chart(chart_valor(r["serie_port"],r["serie_ipc"],t), use_container_width=True)
        col_dd,col_an = st.columns(2)
        with col_dd:
            st.plotly_chart(chart_dd(r["dd"],r["ret_i"],t), use_container_width=True)
        with col_an:
            st.plotly_chart(chart_anual(r["ret_p"],r["ret_i"],t), use_container_width=True)

        st.markdown("<div class='section-header'>TABLA DE MÉTRICAS</div>", unsafe_allow_html=True)
        df_met = pd.DataFrame({
            "Métrica": [
                t["ret_geom"],t["ret_acum"],t["volatilidad"],
                f"Sharpe (Rf={rf*100:.1f}%)",t["cvar"],t["max_dd"],
                t["calmar"],t["info_ratio"],t["beta_real"],t["alpha_jensen"]
            ],
            "Portafolio": [
                f"{m['ret_geom']}%",f"{m['ret_acum']}%",f"{m['vol']}%",
                f"{m['sharpe']}",f"{m['cvar']}%",f"{m['max_dd']}%",
                f"{m['calmar']}",f"{m['ir']}",f"{m['beta']}",f"{m['alpha_j']}%"
            ],
        })
        st.dataframe(df_met, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════
elif pagina == t["configuracion"]:
    st.markdown(f"<h1 style='font-family:Cormorant Garamond,serif;font-size:28px;font-weight:300;color:#fff;margin-bottom:20px;'>{t['configuracion']}</h1>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>UNIVERSO ACTIVO</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family:IBM Plex Mono;font-size:11px;color:#435447;line-height:2;'>
        <b style='color:#c9a84c;'>Índice:</b> {universo_sel}<br>
        <b style='color:#c9a84c;'>Benchmark:</b> {config_u['indice']}<br>
        <b style='color:#c9a84c;'>Moneda:</b> {config_u['moneda']}<br>
        <b style='color:#c9a84c;'>Emisoras:</b> {len(config_u['tickers'])}<br>
        <b style='color:#c9a84c;'>Periodo:</b> {fecha_inicio} → {fecha_fin}<br>
        <b style='color:#c9a84c;'>Rf:</b> {rf*100:.2f}%
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>METODOLOGÍA</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:IBM Plex Mono;font-size:10px;color:#435447;line-height:2;'>
        1 · Descarga de datos via yfinance (log-retornos diarios)<br>
        2 · Regresión cuantílica (Q10, Q25, Q75, Q90) vs benchmark<br>
        3 · Score Kelly por emisora: f* = (p·b − q) / b<br>
        4 · Optimización de pesos con restricciones de caps<br>
        5 · Backtesting con rebalanceo mensual<br>
        6 · Alpha de Jensen: α = Rp − [Rf + β(Rm − Rf)]<br>
        7 · CVaR histórico al 95% (sin supuesto de normalidad)
    </div>""", unsafe_allow_html=True)
