import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pytz

_EST = pytz.timezone("America/New_York")

def now_est():
    """Return current datetime in US/Eastern (EST/EDT)."""
    return datetime.now(pytz.utc).astimezone(_EST)
import warnings, io, json, zipfile
import plotly.io as pio
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Options Greek Exposure", layout="wide", initial_sidebar_state="collapsed")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-box {
    background: #1e2130; border-radius: 8px; padding: 14px 18px;
    margin: 4px 0; border-left: 3px solid #4a9eff;
}
.metric-label { color: #8899aa; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { color: #ffffff; font-size: 22px; font-weight: 700; margin-top: 2px; }
.metric-sub   { color: #8899aa; font-size: 12px; margin-top: 2px; }
.regime-box   { border-radius: 8px; padding: 16px 20px; margin-bottom: 12px; }
.dampening    { background: #0d2b1a; border-left: 4px solid #00c853; }
.amplifying   { background: #2b0d0d; border-left: 4px solid #ff3d00; }
.level-row    { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #2a2f3f; }
.stTabs [data-baseweb="tab-list"] { gap: 0px; width: 100%; }
.stTabs [data-baseweb="tab"] {
    background: #1e2130; border-radius: 6px; color: #8899aa;
    flex: 1; justify-content: center; text-align: center;
    padding: 10px 4px; font-size: 14px; font-weight: 600;
    min-width: 0; white-space: nowrap;
}
.stTabs [aria-selected="true"] { background: #4a9eff; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Black-Scholes Greeks ─────────────────────────────────────────────────────

def bs_greeks(S, K, T, r, sigma, flag):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0, 0.0, 0.0
    try:
        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        pdf_d1 = norm.pdf(d1)
        gamma = pdf_d1 / (S * sigma * sqrtT)
        vanna = -(pdf_d1 * d2) / sigma if sigma != 0 else 0.0
        charm = -pdf_d1 * (2 * r * T - d2 * sigma * sqrtT) / (2 * T * sigma * sqrtT)
        if flag == "call":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        return delta, gamma, vanna, charm
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def fetch_chain(ticker: str):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="3mo")   # 3 months for momentum + prev close
    if hist.empty:
        return None, None, None, None, None

    S      = hist["Close"].iloc[-1]
    S_prev = hist["Close"].iloc[-2] if len(hist) > 1 else S
    # Keep last 30 trading days of closes for momentum model
    hist30 = hist["Close"].iloc[-30:].copy()
    r = 0.05

    expirations = tk.options
    if not expirations:
        return None, None, None

    rows = []
    for exp in expirations:
        exp_date = datetime.strptime(exp, "%Y-%m-%d")
        T = max((exp_date - now_est().replace(tzinfo=None)).days / 365, 1 / 365)
        dte = max((exp_date - now_est().replace(tzinfo=None)).days, 0)

        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue

        for flag, df in [("call", chain.calls), ("put", chain.puts)]:
            for _, row in df.iterrows():
                K = row.strike
                iv = row.impliedVolatility if not np.isnan(row.impliedVolatility) else 0
                oi = row.openInterest if not np.isnan(row.openInterest) else 0
                vol = row.volume if not np.isnan(row.volume) else 0
                if iv <= 0:
                    continue

                delta, gamma, vanna, charm = bs_greeks(S, K, T, r, iv, flag)
                mult = 100
                sign = 1 if flag == "call" else -1

                # GEX ($): gamma × OI × 100 × S² × 0.01 × sign
                #   = dollar dealers must buy/sell per 1% spot move
                #   sign separates calls(+) from puts(-); gamma is always positive
                gex_oi  = gamma * oi  * mult * S * S * 0.01 * sign
                gex_vol = gamma * vol * mult * S * S * 0.01 * sign

                # DEX ($): delta × OI × 100 × S
                #   = notional dollar delta exposure; no sign multiplier because
                #   call delta is naturally +, put delta is naturally - (already encodes side)
                dex_oi  = delta * oi  * mult * S

                # VEX ($): vanna × OI × 100 × S × sign
                #   = dollar delta change per 1-vol-point move
                #   vanna is mathematically identical for calls and puts at same strike
                #   → sign required to visually split call(+) from put(-) contribution
                vex_oi  = vanna * oi  * mult * S * sign

                # CEX ($): charm × OI × 100 × S / 252 × sign
                #   = dollar delta decay per trading day
                #   our BS charm is ∂delta/∂year, divide by 252 → per trading day
                #   multiply by S → dollar-denominated; sign splits call/put as with VEX
                cex_oi  = charm * oi  * mult * S / 252 * sign

                rows.append({
                    "strike": K, "flag": flag, "expiry": exp, "dte": dte,
                    "oi": oi, "volume": vol, "iv": iv,
                    "GEX_oi": gex_oi, "GEX_vol": gex_vol,
                    "DEX": dex_oi, "VEX": vex_oi, "CEX": cex_oi,
                    "gamma_raw": gamma, "delta_raw": delta,
                })

    if not rows:
        return None, None, None, None, None

    df_all = pd.DataFrame(rows)
    fetched_at = now_est().strftime("%Y-%m-%d %H:%M:%S EST")
    return S, S_prev, df_all, fetched_at, hist30


# ── Chart helpers ─────────────────────────────────────────────────────────────

COLORS = {"call": "rgba(0,200,100,0.85)", "put": "rgba(220,50,50,0.85)", "net": "rgba(100,160,255,0.85)"}
DARK_BG = "plotly_dark"

def fmt_num(v, prefix="$"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    av = abs(v)
    sign = "-" if v < 0 else ""
    if av >= 1e9:  return f"{sign}{prefix}{av/1e9:.2f}B"
    if av >= 1e6:  return f"{sign}{prefix}{av/1e6:.2f}M"
    if av >= 1e3:  return f"{sign}{prefix}{av/1e3:.2f}K"
    return f"{sign}{prefix}{av:.2f}"

def fmt_col(series, prefix="$"):
    """Format a pandas Series of numbers into readable strings."""
    return series.apply(lambda x: fmt_num(x, prefix))

def hbar_chart(df_strikes, col, title, spot, call_wall=None, put_wall=None,
               gamma_flip=None, pct_range=25):
    """Horizontal bar chart filtered to ±pct_range% around spot."""
    calls = df_strikes[df_strikes.flag == "call"].groupby("strike")[col].sum()
    puts  = df_strikes[df_strikes.flag == "put"].groupby("strike")[col].sum()

    lo = spot * (1 - pct_range / 100)
    hi = spot * (1 + pct_range / 100)
    all_strikes = sorted(set(calls.index) | set(puts.index))
    strikes = [s for s in all_strikes if lo <= s <= hi]

    # Always include key level strikes even if outside range
    for lvl in [call_wall, put_wall, gamma_flip]:
        if lvl and lvl not in strikes:
            strikes.append(lvl)
    strikes = sorted(strikes)

    if not strikes:
        strikes = all_strikes  # fallback

    c_vals = [calls.get(k, 0) for k in strikes]
    p_vals = [puts.get(k, 0) for k in strikes]

    # Drop strikes with negligible exposure (< 0.5% of the max bar)
    max_abs = max((abs(c) + abs(p) for c, p in zip(c_vals, p_vals)), default=1) or 1
    threshold = max_abs * 0.005
    filtered = [(s, c, p) for s, c, p in zip(strikes, c_vals, p_vals)
                if abs(c) + abs(p) >= threshold]
    if filtered:
        strikes, c_vals, p_vals = zip(*filtered)
        strikes, c_vals, p_vals = list(strikes), list(c_vals), list(p_vals)

    # Dynamic chart height: 22px per strike, min 400, max 900
    chart_h = min(900, max(400, len(strikes) * 22))

    dtick = 1  # show every single strike price

    fig = go.Figure()
    fig.add_bar(y=strikes, x=c_vals, name="Calls", orientation="h",
                marker_color=COLORS["call"],
                hovertemplate="Strike %{y}<br>Call: %{x:,.0f}<extra></extra>")
    fig.add_bar(y=strikes, x=p_vals, name="Puts", orientation="h",
                marker_color=COLORS["put"],
                hovertemplate="Strike %{y}<br>Put: %{x:,.0f}<extra></extra>")

    # Collect all level lines, then resolve overlapping labels
    level_lines = [
        (spot,       f"Spot ${spot:.2f}",          "white",   "dash",    2.0),
    ]
    if call_wall and lo <= call_wall <= hi:
        level_lines.append((call_wall, f"Call Wall ${call_wall:.0f}", "#4a9eff", "dot",     1.5))
    if put_wall and lo <= put_wall <= hi:
        level_lines.append((put_wall,  f"Put Wall ${put_wall:.0f}",  "#ff6b6b", "dot",     1.5))
    if gamma_flip and lo <= gamma_flip <= hi:
        level_lines.append((gamma_flip, f"γ Flip ${gamma_flip:.2f}", "#ffd700", "dashdot", 1.5))

    # Draw lines
    for y_val, _, color, dash, width in level_lines:
        fig.add_hline(y=y_val, line_dash=dash, line_color=color, line_width=width)

    # Resolve label overlap: cascade yshifts for any levels within min_gap of each other
    min_gap = max((hi - lo) * 0.05, 1.0)
    level_lines_sorted = sorted(level_lines, key=lambda x: x[0])
    n = len(level_lines_sorted)
    yshifts = [0] * n
    STEP = 22  # px per overlapping label
    for i in range(1, n):
        gap = abs(level_lines_sorted[i][0] - level_lines_sorted[i - 1][0])
        if gap < min_gap:
            yshifts[i] = yshifts[i - 1] + STEP

    # Center the group so labels are distributed above and below the actual line
    mid = (max(yshifts) + min(yshifts)) / 2
    yshifts = [y - mid for y in yshifts]

    for (y_val, label, color, _, _), yshift in zip(level_lines_sorted, yshifts):
        fig.add_annotation(
            x=1.0, xref="paper", y=y_val, yref="y",
            text=f" {label}",
            showarrow=False, xanchor="left",
            font=dict(color=color, size=11),
            yshift=yshift,
            bgcolor="rgba(14,17,23,0.85)",
            borderpad=3,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        barmode="relative", template=DARK_BG, height=chart_h,
        xaxis_title="Exposure ($)", yaxis_title="Strike Price",
        xaxis=dict(tickformat=",.0f"),
        yaxis=dict(dtick=dtick, tickformat=".2f", tickfont=dict(size=11)),
        legend=dict(orientation="h", y=1.04, font=dict(size=12)),
        margin=dict(l=70, r=160, t=60, b=50),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
    )
    return fig


def vbar_chart(df_expiry, col, title, spot):
    """Vertical bar chart by expiry."""
    calls = df_expiry[df_expiry.flag == "call"].groupby("expiry")[col].sum()
    puts  = df_expiry[df_expiry.flag == "put"].groupby("expiry")[col].sum()
    expiries = sorted(set(calls.index) | set(puts.index))

    c_vals = [calls.get(e, 0) for e in expiries]
    p_vals = [puts.get(e, 0) for e in expiries]

    fig = go.Figure()
    fig.add_bar(x=expiries, y=c_vals, name="Calls", marker_color=COLORS["call"],
                hovertemplate="Expiry: %{x}<br>Call: %{customdata}<extra></extra>",
                customdata=[fmt_num(v) for v in c_vals])
    fig.add_bar(x=expiries, y=p_vals, name="Puts", marker_color=COLORS["put"],
                hovertemplate="Expiry: %{x}<br>Put: %{customdata}<extra></extra>",
                customdata=[fmt_num(v) for v in p_vals])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        barmode="relative", template=DARK_BG, height=440,
        xaxis_title="Expiry", yaxis_title="Exposure ($)",
        xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
        yaxis=dict(tickformat=",.0s"),   # e.g. 1M, 500K
        legend=dict(orientation="h", y=1.04, font=dict(size=12)),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        margin=dict(l=70, r=40, t=60, b=80),
    )
    return fig


def metric_html(label, value, sub=None, color="#4a9eff"):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-box" style="border-left-color:{color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>"""


# ── Key level calculations ────────────────────────────────────────────────────

def compute_levels(df, spot):
    by_strike = df.groupby(["strike", "flag"])["GEX_oi"].sum().reset_index()
    calls_s = by_strike[by_strike.flag == "call"].set_index("strike")["GEX_oi"]
    puts_s  = by_strike[by_strike.flag == "put"].set_index("strike")["GEX_oi"]
    all_strikes = sorted(set(calls_s.index) | set(puts_s.index))
    net = pd.Series({k: calls_s.get(k, 0) + puts_s.get(k, 0) for k in all_strikes})

    # ── Call Wall (SpotGamma convention): highest call-side GEX at or above spot
    #   Only looks above spot because the resistance mechanic requires dealers selling
    #   into rallies — a call concentration BELOW spot creates a different dynamic.
    #   Source: SpotGamma support docs; Squeezemetrics white paper.
    calls_above = calls_s[calls_s.index >= spot]
    call_wall = calls_above.idxmax() if not calls_above.empty else calls_s.idxmax()

    # ── Put Wall (SpotGamma convention): most negative put-side GEX at or below spot
    puts_below = puts_s[puts_s.index <= spot]
    put_wall = puts_below.idxmin() if not puts_below.empty else puts_s.idxmin()

    # ── Absolute Call/Put Wall (GEXBoard convention): global extrema, no spot restriction
    #   Useful signal: if abs_call_wall < spot, the largest gamma concentration is BELOW
    #   current price — options market structure has shifted (bearish structural signal).
    #   Source: GEXBoard "GEX Levels Explained"; FlashAlpha dealer positioning guide.
    abs_call_wall = calls_s.idxmax() if not calls_s.empty else None
    abs_put_wall  = puts_s.idxmin()  if not puts_s.empty  else None

    # ── Peak GEX: strike with highest absolute net GEX (call+put combined)
    #   Acts as a price magnet in positive-gamma regimes; distinct from Call Wall
    #   because a heavily put-sided strike can have higher |net| than any pure call strike
    near = net[(net.index >= spot * 0.7) & (net.index <= spot * 1.3)]
    peak_gex_strike = near.abs().idxmax() if not near.empty else net.abs().idxmax()

    # ── Gamma Flip (Zero Gamma): strike where cumulative net GEX crosses zero
    #   Linear interpolation between adjacent sign-change strikes (Perfiliev canonical method).
    #   Search range ±20% of spot — Perfiliev uses this range; far-OTM strikes have
    #   near-zero gamma from model noise, not real dealer positioning.
    #   When multiple crossings exist, pick the one closest to current spot.
    #   Source: perfiliev.com canonical implementation; Barbon & Buraschi (2021, SSRN 3725454).
    gamma_flip = None
    near_net = net[(net.index >= spot * 0.80) & (net.index <= spot * 1.20)]
    crossings = []
    signs = np.sign(near_net.values)
    for i in range(len(signs) - 1):
        if signs[i] != signs[i + 1] and signs[i] != 0 and signs[i + 1] != 0:
            k1, k2 = near_net.index[i], near_net.index[i + 1]
            v1, v2 = near_net.iloc[i], near_net.iloc[i + 1]
            crossing = k1 + (k2 - k1) * (-v1) / (v2 - v1)
            crossings.append(crossing)
    if crossings:
        # Select the crossing closest to spot price
        gamma_flip = min(crossings, key=lambda x: abs(x - spot))

    # ── Acceleration levels: secondary GEX peaks between spot and the primary walls
    #   These are local maxima in the net GEX profile, NOT just the global max/min.
    #   They represent secondary support/resistance where dealers' hedging flow
    #   temporarily accelerates or decelerates before reaching the main walls.
    def find_secondary_peak(series, above=True):
        """Find the local max/min between spot and the primary wall, excluding the wall."""
        if above:
            seg = series[series.index > spot]
            if len(seg) < 3:
                return None
            # Exclude the primary call wall strike
            seg = seg[seg.index != call_wall]
            if seg.empty:
                return None
            # Find local maxima (value > both neighbors)
            vals = seg.values; idxs = seg.index.tolist()
            for j in range(1, len(vals) - 1):
                if vals[j] > vals[j - 1] and vals[j] > vals[j + 1] and vals[j] > 0:
                    return idxs[j]
            return seg.idxmax() if not seg.empty else None
        else:
            seg = series[series.index < spot]
            if len(seg) < 3:
                return None
            seg = seg[seg.index != put_wall]
            if seg.empty:
                return None
            vals = seg.values; idxs = seg.index.tolist()
            for j in range(1, len(vals) - 1):
                if vals[j] < vals[j - 1] and vals[j] < vals[j + 1] and vals[j] < 0:
                    return idxs[j]
            return seg.idxmin() if not seg.empty else None

    accel_up   = find_secondary_peak(net, above=True)
    accel_down = find_secondary_peak(net, above=False)

    # ── 0DTE concentration: % of total GEX sitting in same-day / next-day expiry
    gex_0dte = df[df.dte <= 1]["GEX_oi"].abs().sum()
    gex_total_abs = df["GEX_oi"].abs().sum()
    pct_0dte = (gex_0dte / gex_total_abs * 100) if gex_total_abs > 0 else 0.0

    net_total  = df["GEX_oi"].sum()
    call_total = df[df.flag == "call"]["GEX_oi"].sum()
    put_total  = df[df.flag == "put"]["GEX_oi"].sum()

    return {
        "call_wall": call_wall, "put_wall": put_wall,
        "abs_call_wall": abs_call_wall, "abs_put_wall": abs_put_wall,
        "gamma_flip": gamma_flip,
        "peak_gex_strike": peak_gex_strike,
        "accel_up": accel_up, "accel_down": accel_down,
        "net_total": net_total, "call_total": call_total, "put_total": put_total,
        "net_by_strike": net,
        "pct_0dte": pct_0dte,
    }


# ── Gamma decay by expiry ─────────────────────────────────────────────────────

def gamma_decay_table(df):
    grp = df.groupby(["expiry", "dte"])["GEX_oi"].sum().reset_index()
    grp["normalized"] = (grp["GEX_oi"].abs() / grp["GEX_oi"].abs().sum() * 100).round(2)
    grp = grp.sort_values("dte")
    return grp



# ════════════════════════════════════════════════════════
# DEX SCENARIO HEATMAP
# ════════════════════════════════════════════════════════

def build_dex_heatmap(df, spot, ticker, hist30=None, levels=None, net_gex=0):
    from scipy.ndimage import gaussian_filter

    # ── filter: ±15% of spot, ≤60 DTE ────────────────────────────────────────
    df_h = df[
        (df["strike"] >= spot * 0.85) & (df["strike"] <= spot * 1.15) &
        (df["dte"] <= 60) & (df["dte"] >= 0)
    ].copy()
    if df_h.empty:
        return None

    expiries = sorted(df_h["expiry"].unique(),
                      key=lambda e: datetime.strptime(e, "%Y-%m-%d"))
    if len(expiries) < 2:
        return None

    # ── numeric axes ──────────────────────────────────────────────────────────
    # Y: fine price grid (ascending internally, displayed high→low)
    step = 0.5 if spot < 20 else 1.0 if spot < 100 else 2.5 if spot < 500 else 5.0
    lo   = np.floor(spot * 0.85 / step) * step
    hi   = np.ceil (spot * 1.15 / step) * step
    price_grid = np.arange(lo, hi + step, step)   # ascending

    # X: DTE values (numeric) — gives continuous axis, no "Today" artefact
    today    = now_est().replace(tzinfo=None)
    dte_vals = []
    for exp in expiries:
        dte_vals.append(max((datetime.strptime(exp, "%Y-%m-%d") - today).days, 1))

    n_prices = len(price_grid)
    n_expiry = len(expiries)

    # ── aggregate net DEX onto price-grid × expiry matrix ────────────────────
    raw = np.zeros((n_prices, n_expiry))
    for j, exp in enumerate(expiries):
        net_by_k = df_h[df_h["expiry"] == exp].groupby("strike")["DEX"].sum()
        for k_val, dex_val in net_by_k.items():
            idx = int(np.argmin(np.abs(price_grid - k_val)))
            raw[idx, j] += dex_val

    # ── gaussian smooth (vertical spread > horizontal to keep expiry columns
    #    distinct while blending adjacent strikes into smooth blobs) ──────────
    smoothed = gaussian_filter(raw, sigma=[3.0, 0.5])

    # display: flip Y so high price is at top
    y_vals    = price_grid[::-1]       # descending numeric array
    z_display = smoothed[::-1, :]

    abs_max = max(float(np.abs(smoothed).max()), 1.0)

    # ── build figure with go.Heatmap + zsmooth for continuous colour ─────────
    # zsmooth='best' applies bilinear interpolation → smooth gradient between
    # discrete grid cells; no discrete colour blocks.
    # Red = negative DEX (put-dominated / dealers short delta)
    # Green = positive DEX (call-dominated / dealers long delta)
    colorscale = [
        [0.00, "rgb(160, 20,  20)"],   # deep red  — most negative
        [0.40, "rgb( 70,  5,   5)"],   # dark red
        [0.50, "rgb( 14, 17,  23)"],   # near-zero ≈ background (invisible)
        [0.60, "rgb(  5, 60,  10)"],   # dark green
        [1.00, "rgb( 20, 160,  30)"],  # bright green — most positive
    ]

    fig = go.Figure(go.Heatmap(
        z=z_display,
        x=dte_vals,
        y=y_vals,
        zsmooth="best",               # bilinear interpolation → smooth gradients
        colorscale=colorscale,
        zmid=0, zmin=-abs_max, zmax=abs_max,
        showscale=True,
        colorbar=dict(
            title=dict(text="Net DEX ($)", side="right"),
            tickformat=",.0s", thickness=14, len=0.8,
            tickfont=dict(size=10),
        ),
        hovertemplate=(
            "Strike: $%{y:.2f}<br>DTE: %{x}d<br>"
            "Net DEX: %{z:,.0f}<extra></extra>"
        ),
    ))

    # ── spot horizontal line ──────────────────────────────────────────────────
    fig.add_hline(y=spot, line_dash="dash", line_color="white", line_width=1.5)
    fig.add_annotation(
        x=1.01, xref="paper", y=spot, yref="y",
        text=f" ${spot:.2f}", showarrow=False, xanchor="left",
        font=dict(color="white", size=11), bgcolor="rgba(14,17,23,0.8)",
    )

    # ── prediction model ──────────────────────────────────────────────────────
    try:
        if hist30 is not None and len(hist30) >= 5:
            lb = min(10, len(hist30) - 1)
            mom_daily = (float(hist30.iloc[-1]) / float(hist30.iloc[-lb - 1]) - 1) / lb
        else:
            mom_daily = 0.0

        is_long_g   = net_gex >= 0
        regime_mult = 0.65 if is_long_g else 1.35
        regime_lbl  = "Long γ (dampened)" if is_long_g else "Short γ (amplified)"
        peak_k      = (levels or {}).get("peak_gex_strike", spot) or spot
        GRAVITY_K   = 0.04

        atm_mask = df["strike"].sub(spot).abs() < spot * 0.05
        atm_iv   = float(df[atm_mask]["iv"].mean()) if atm_mask.any() else 0.25
        if not np.isfinite(atm_iv) or atm_iv <= 0:
            atm_iv = 0.25
        daily_vol = atm_iv / np.sqrt(252)

        mu_list, up1, lo1, up2, lo2 = [], [], [], [], []
        for days in dte_vals:
            mu = float(np.clip(
                spot + spot * mom_daily * days * regime_mult
                     + (peak_k - spot) * (1 - np.exp(-GRAVITY_K * days)),
                spot * 0.75, spot * 1.25
            ))
            s = daily_vol * np.sqrt(days)
            mu_list.append(mu)
            up1.append(mu * np.exp( s));  lo1.append(mu * np.exp(-s))
            up2.append(mu * np.exp(2*s)); lo2.append(mu * np.exp(-2*s))

        # prediction x starts at DTE=0 (today, before first expiry)
        pred_x = [0] + dte_vals

        # centre line (solid gold)
        fig.add_scatter(
            x=pred_x, y=[spot] + mu_list,
            mode="lines",
            line=dict(color="gold", width=2),
            name=f"Predicted ({regime_lbl})",
            hovertemplate="DTE: %{x}d<br>Expected: $%{y:.2f}<extra></extra>",
        )
        # 1σ bounds (dashed, semi-transparent gold)
        for band, name in [(up1, "1σ upper"), (lo1, "1σ lower")]:
            fig.add_scatter(
                x=pred_x, y=[spot] + band,
                mode="lines",
                line=dict(color="rgba(255,215,0,0.55)", width=1, dash="dash"),
                name=name, showlegend=False, hoverinfo="skip",
            )
        # 2σ bounds (dotted, more transparent)
        for band, name in [(up2, "2σ upper"), (lo2, "2σ lower")]:
            fig.add_scatter(
                x=pred_x, y=[spot] + band,
                mode="lines",
                line=dict(color="rgba(255,215,0,0.28)", width=1, dash="dot"),
                name=name, showlegend=False, hoverinfo="skip",
            )
    except Exception:
        pass

    # ── layout ────────────────────────────────────────────────────────────────
    height = min(760, max(440, n_prices * 16 + 100))

    tick_step = max(1, n_prices // 14)
    tick_vals = sorted(y_vals[::tick_step].tolist(), reverse=True)

    # X ticks: map DTE → "MMM DD (Xd)" labels
    x_tickvals = dte_vals
    x_ticktext = []
    for exp, dv in zip(expiries, dte_vals):
        d = datetime.strptime(exp, "%Y-%m-%d")
        x_ticktext.append(f"{d.strftime('%b %d')} ({dv}d)")

    fig.update_layout(
        title=dict(text=f"{ticker} — Delta Exposure Heatmap", font=dict(size=13)),
        template=DARK_BG, height=height,
        xaxis=dict(
            title="Days to Expiry",
            tickmode="array", tickvals=x_tickvals, ticktext=x_ticktext,
            tickangle=-30, tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Strike / Price ($)",
            tickmode="array", tickvals=tick_vals,
            tickformat="$.2f", tickfont=dict(size=10),
            range=[y_vals[-1] - step, y_vals[0] + step],
        ),
        legend=dict(orientation="h", y=1.05, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=75, r=110, t=55, b=80),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    )
    return fig


# ════════════════════════════════════════════════════════
# GEX chart builders (used by Gamma tab AND export)
# ════════════════════════════════════════════════════════

def build_gex_strike_profile(df_g, spot, levels, strike_view="Near"):
    """Vertical bar chart of GEX by strike with key-level lines."""
    gf = levels["gamma_flip"]
    cw = levels["call_wall"]
    pw = levels["put_wall"]

    calls_s = df_g[df_g.flag=="call"].groupby("strike")["GEX_oi"].sum()
    puts_s  = df_g[df_g.flag=="put"].groupby("strike")["GEX_oi"].sum()
    all_k   = sorted(set(calls_s.index) | set(puts_s.index))

    if strike_view == "Near":
        lo, hi = spot * 0.80, spot * 1.20
        all_k = [k for k in all_k if lo <= k <= hi]

    c_vals = [calls_s.get(k, 0) for k in all_k]
    p_vals = [puts_s.get(k,  0) for k in all_k]
    max_abs = max((abs(c)+abs(p) for c,p in zip(c_vals, p_vals)), default=1) or 1
    thresh  = max_abs * 0.005
    filtered = [(k,c,p) for k,c,p in zip(all_k,c_vals,p_vals) if abs(c)+abs(p) >= thresh]
    if filtered:
        all_k, c_vals, p_vals = zip(*filtered)
        all_k, c_vals, p_vals = list(all_k), list(c_vals), list(p_vals)

    net_vals = [c+p for c,p in zip(c_vals, p_vals)]

    fig = go.Figure()
    fig.add_bar(x=all_k, y=c_vals, name="Call Gamma",
                marker_color=COLORS["call"],
                hovertemplate="Strike %{x}<br>Call GEX: %{customdata}<extra></extra>",
                customdata=[fmt_num(v) for v in c_vals])
    fig.add_bar(x=all_k, y=p_vals, name="Put Gamma",
                marker_color=COLORS["put"],
                hovertemplate="Strike %{x}<br>Put GEX: %{customdata}<extra></extra>",
                customdata=[fmt_num(v) for v in p_vals])
    fig.add_scatter(x=all_k, y=net_vals, name="Net GEX",
                    mode="lines", line=dict(color="white", width=1.5, dash="dot"),
                    hovertemplate="Strike %{x}<br>Net GEX: %{customdata}<extra></extra>",
                    customdata=[fmt_num(v) for v in net_vals])

    level_lines_v = [(spot, f"Spot ${spot:.2f}", "white", "dash", 2)]
    if cw: level_lines_v.append((cw, f"Call Wall ${cw:.0f}", "#4a9eff", "dot", 1.5))
    if pw: level_lines_v.append((pw, f"Put Wall ${pw:.0f}",  "#ff6b6b", "dot", 1.5))
    if gf: level_lines_v.append((gf, f"Zero γ ${gf:.2f}",   "#ffd700", "dashdot", 1.5))

    for x_val, _, color, dash, width in level_lines_v:
        fig.add_vline(x=x_val, line_dash=dash, line_color=color, line_width=width)

    x_range = (max(all_k) - min(all_k)) if len(all_k) > 1 else 10
    min_gap_x = x_range * 0.04
    lvl_sorted = sorted(level_lines_v, key=lambda x: x[0])
    y_positions = [1.0] * len(lvl_sorted)
    for i in range(1, len(lvl_sorted)):
        if abs(lvl_sorted[i][0] - lvl_sorted[i-1][0]) < min_gap_x:
            y_positions[i] = y_positions[i-1] - 0.10

    for (x_val, label, color, _, _), y_pos in zip(lvl_sorted, y_positions):
        fig.add_annotation(
            x=x_val, xref="x", y=y_pos, yref="paper",
            text=label, showarrow=False, yanchor="bottom", xanchor="left",
            font=dict(color=color, size=11), xshift=4,
            bgcolor="rgba(14,17,23,0.85)", borderpad=3,
        )

    chart_h = min(550, max(380, len(all_k) * 8))
    fig.update_layout(
        barmode="relative", template=DARK_BG, height=chart_h,
        xaxis=dict(title="Strike", dtick=1, tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(title="GEX ($)", tickformat=",.0s"),
        legend=dict(orientation="h", y=1.06),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        margin=dict(l=70, r=40, t=80, b=60),
    )
    return fig


def build_gex_heatmap_fig(df_g, spot):
    """Strike × Expiry GEX heatmap (colour-coded green/red)."""
    pivot = df_g.groupby(["strike","expiry"])["GEX_oi"].sum().unstack(fill_value=0)
    max_cell = pivot.abs().max().max() or 1
    pivot = pivot[pivot.abs().max(axis=1) > max_cell * 0.005]
    pivot = pivot[(pivot.index >= spot * 0.70) & (pivot.index <= spot * 1.30)]
    pivot = pivot.sort_index(ascending=False)

    if pivot.empty:
        return None

    expiry_cols = list(pivot.columns)
    strike_rows = list(pivot.index)
    col_labels = []
    for exp in expiry_cols:
        d = datetime.strptime(exp, "%Y-%m-%d")
        dte_v = max((d - now_est().replace(tzinfo=None)).days, 0)
        col_labels.append(f"{d.strftime('%b %d')} ({dte_v}d)")

    z = pivot.values.tolist()
    text = [[fmt_num(v, "$") if abs(v) > max_cell * 0.005 else "" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=col_labels, y=[f"${k:.0f}" for k in strike_rows],
        text=text, texttemplate="%{text}", textfont=dict(size=10, color="white"),
        colorscale=[
            [0.0,  "rgb(160,20,20)"], [0.45, "rgb(60,10,10)"],
            [0.5,  "rgb(20,20,30)"],  [0.55, "rgb(10,60,10)"],
            [1.0,  "rgb(20,160,20)"],
        ],
        zmid=0, showscale=True,
        colorbar=dict(title="GEX ($)", tickformat=",.0s", thickness=12, len=0.8),
    ))

    spot_label = f"${round(spot):.0f}"
    y_labels = [f"${k:.0f}" for k in strike_rows]
    if spot_label in y_labels:
        fig.add_hline(
            y=y_labels.index(spot_label), line_dash="dash",
            line_color="white", line_width=1.5,
        )

    heatmap_h = min(800, max(300, len(strike_rows) * 24 + 80))
    fig.update_layout(
        template=DARK_BG, height=heatmap_h,
        xaxis=dict(side="top", tickangle=-30, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        margin=dict(l=60, r=80, t=60, b=20),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    )
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("## Options Greek Exposure")

col_ticker, col_btn = st.columns([3, 1])
with col_ticker:
    ticker = st.text_input("Ticker", value="INTC", label_visibility="collapsed").upper().strip()
with col_btn:
    refresh = st.button("⟳  Refresh Data", use_container_width=True)

# ── Quick-pick buttons ────────────────────────────────────────────────────────
QUICK_TICKERS = ["INTC", "TEM", "NVTS", "NVDA", "NBIS"]
qcols = st.columns(len(QUICK_TICKERS))
for i, qt in enumerate(QUICK_TICKERS):
    if qcols[i].button(qt, key=f"quick_{qt}", use_container_width=True):
        st.query_params["t"] = qt
        st.rerun()

# Allow quick-pick navigation via query param
_qp = st.query_params.get("t", "")
if _qp and _qp != ticker:
    ticker = _qp.upper().strip()

if refresh:
    st.cache_data.clear()

with st.spinner(f"Loading {ticker} options chain…"):
    spot, spot_prev, df, fetched_at, hist30 = fetch_chain(ticker)

if df is None or df.empty:
    st.error(f"No options data found for **{ticker}**. Try a different ticker.")
    st.stop()

levels = compute_levels(df, spot)
net_gex = levels["net_total"]
is_long_gamma = net_gex >= 0
regime_label = "Dampening" if is_long_gamma else "Amplifying"
regime_color = "dampening" if is_long_gamma else "amplifying"
dealer_pos   = "LONG" if is_long_gamma else "SHORT"
hedge_per_pct = abs(net_gex) * 0.01

n_contracts = len(df)
n_expiries  = df["expiry"].nunique()

hdr_col, export_col = st.columns([5, 1])
with hdr_col:
    st.caption(
        f"**{ticker}** · Spot **${spot:.2f}** · {n_contracts:,} contracts · "
        f"{n_expiries} expiries · Data as of **{fetched_at}** (~15min delay)"
    )
with export_col:
    export_clicked = st.button("⬇ Export Analysis", use_container_width=True)

# ── Export ───────────────────────────────────────────────────────────────────
if export_clicked:

    with st.spinner("Building HTML report — rendering charts…"):

        # ── helpers ──────────────────────────────────────────────────────────
        def fig_to_div(fig, height=None):
            """Return an interactive Plotly div string (no kaleido needed)."""
            if fig is None:
                return None
            try:
                if height:
                    fig.update_layout(height=height)
                return fig.to_html(full_html=False, include_plotlyjs=False,
                                   config={"displayModeBar": True, "responsive": True})
            except Exception:
                return None

        def chart_tag(div, alt="chart", caption=""):
            if div is None:
                return f"<p style='color:#888;font-style:italic'>Chart unavailable ({alt})</p>"
            cap = f'<p style="color:#8899aa;font-size:12px;margin:2px 0 12px">{caption}</p>' if caption else ""
            return f'<div style="width:100%;margin:8px 0 4px">{div}</div>{cap}'

        def expiry_table(col):
            c = df[df.flag=="call"].groupby(["expiry","dte"])[col].sum().reset_index().rename(columns={col:"Call"})
            p = df[df.flag=="put"].groupby(["expiry","dte"])[col].sum().reset_index().rename(columns={col:"Put"})
            t = c.merge(p, on=["expiry","dte"], how="outer").fillna(0)
            t["Net"] = t["Call"] + t["Put"]
            return t.sort_values("dte")

        def df_to_html(tbl):
            display = tbl.copy()
            for c in display.columns:
                if display[c].dtype in [float, "float64"]:
                    display[c] = display[c].apply(fmt_num)
            return display.to_html(classes="data-table", border=0)

        # ── collect key-level vars ────────────────────────────────────────────
        gf  = levels["gamma_flip"];    cw  = levels["call_wall"]
        pw  = levels["put_wall"];      pg  = levels["peak_gex_strike"]
        au  = levels["accel_up"];      ad  = levels["accel_down"]
        pct_0dte = levels["pct_0dte"]

        ts    = now_est().strftime("%Y-%m-%d %H:%M:%S EST")
        fn_ts = now_est().strftime("%Y%m%d_%H%M")

        # ── render all charts ─────────────────────────────────────────────────
        # 1. GEX Strike Profile (All strikes)
        div_gex_profile = fig_to_div(
            build_gex_strike_profile(df, spot, levels, strike_view="All"))

        # 2. GEX Heatmap (Expiry × Strike)
        div_gex_heat = fig_to_div(build_gex_heatmap_fig(df, spot), height=700)

        # 3. DEX Heatmap with prediction
        div_dex_heat = fig_to_div(
            build_dex_heatmap(df, spot, ticker, hist30=hist30, levels=levels, net_gex=net_gex),
            height=720)

        # 4. Per-greek by-strike (horizontal bar) and by-expiry (vertical bar)
        greek_defs = [
            ("GEX_oi", "Gamma Exposure (GEX)",
             "gamma × OI × 100 × S² × 0.01 × sign — $ dealers buy/sell per 1% spot move"),
            ("DEX",    "Delta Exposure (DEX)",
             "delta × OI × 100 × S — notional $ delta; delta is already signed"),
            ("VEX",    "Vanna Exposure (VEX)",
             "vanna × OI × 100 × S × sign — $ delta shift per +1 IV-point move"),
            ("CEX",    "Charm Exposure (CEX)",
             "charm × OI × 100 × S / 252 × sign — $ delta decay per trading day"),
        ]
        greek_divs = {}
        for col, title, _formula in greek_defs:
            cw_arg = levels["call_wall"] if col == "GEX_oi" else None
            pw_arg = levels["put_wall"]  if col == "GEX_oi" else None
            gf_arg = levels["gamma_flip"] if col == "GEX_oi" else None
            greek_divs[col] = {
                "strike": fig_to_div(
                    hbar_chart(df, col, f"{ticker} — {title} by Strike (±25%)",
                               spot, cw_arg, pw_arg, gf_arg, pct_range=25)),
                "expiry": fig_to_div(
                    vbar_chart(df, col, f"{ticker} — {title} by Expiry", spot)),
            }

        # ── build HTML ───────────────────────────────────────────────────────
        regime_bg  = "#0d2b1a" if is_long_gamma else "#2b0d0d"
        regime_clr = "#00c853" if is_long_gamma else "#ff3d00"

        sections = []

        # ── Summary header ────────────────────────────────────────────────────
        sections.append(f"""
<div style="background:{regime_bg};border-left:4px solid {regime_clr};
            padding:14px 18px;border-radius:8px;margin-bottom:16px">
  <div style="font-size:12px;color:#8899aa;text-transform:uppercase">VOLATILITY REGIME</div>
  <div style="font-size:20px;font-weight:700;color:{regime_clr}">{regime_label}</div>
  <div style="color:#ccc;font-size:13px">
    {"Dealers LONG gamma — dampening. Sell rallies, buy dips." if is_long_gamma else
     "Dealers SHORT gamma — amplifying. Buy rallies, sell dips."}
  </div>
</div>
<table class="stat-table"><tr>
  <td><b>Ticker</b><br>{ticker}</td>
  <td><b>Spot</b><br>${spot:.2f}</td>
  <td><b>Data as of</b><br>{fetched_at} (~15 min delay)</td>
  <td><b>Exported</b><br>{ts}</td>
  <td><b>Contracts</b><br>{n_contracts:,}</td>
  <td><b>Expiries</b><br>{n_expiries}</td>
  <td><b>0DTE GEX%</b><br>{pct_0dte:.1f}%</td>
</tr></table>
<table class="stat-table" style="margin-top:10px"><tr>
  <td><b>Net GEX</b><br>{fmt_num(net_gex)}</td>
  <td><b>Call GEX</b><br>{fmt_num(levels["call_total"])}</td>
  <td><b>Put GEX</b><br>{fmt_num(levels["put_total"])}</td>
  <td><b>Net DEX</b><br>{fmt_num(df["DEX"].sum())}</td>
  <td><b>Net VEX</b><br>{fmt_num(df["VEX"].sum())}</td>
  <td><b>Net CEX</b><br>{fmt_num(df["CEX"].sum())}</td>
  <td><b>Hedge / ±1%</b><br>{fmt_num(hedge_per_pct)}</td>
</tr></table>
<table class="stat-table" style="margin-top:10px"><tr>
  <td><b>Call Wall</b><br>${cw:.2f} ({(cw-spot)/spot*100:+.1f}%)</td>
  <td><b>Put Wall</b><br>${pw:.2f} ({(pw-spot)/spot*100:+.1f}%)</td>
  <td><b>Zero Gamma</b><br>{"${:.2f} ({:+.1f}%)".format(gf,(gf-spot)/spot*100) if gf else "N/A"}</td>
  <td><b>Peak GEX</b><br>{"${:.2f} ({:+.1f}%)".format(pg,(pg-spot)/spot*100) if pg else "N/A"}</td>
  <td><b>Accel Up</b><br>{"${:.2f}".format(au) if au else "N/A"}</td>
  <td><b>Accel Down</b><br>{"${:.2f}".format(ad) if ad else "N/A"}</td>
</tr></table>
""")

        # ── Gamma Decay table ─────────────────────────────────────────────────
        decay_tbl = gamma_decay_table(df)
        sections.append("<h2>Gamma Decay by Expiry</h2>")
        decay_disp = decay_tbl.copy()
        decay_disp["GEX_oi"]     = decay_disp["GEX_oi"].apply(fmt_num)
        decay_disp["normalized"] = decay_disp["normalized"].apply(lambda x: f"{x:.2f}%")
        decay_disp.columns = ["Expiry","DTE","Net GEX","% of Total"]
        sections.append(df_to_html(decay_disp))

        # ── GEX charts ────────────────────────────────────────────────────────
        sections.append("<h2>Gamma Exposure (GEX) — Strike Profile</h2>")
        sections.append(chart_tag(div_gex_profile, "GEX Strike Profile",
            "Call GEX (green) · Put GEX (red) · Net GEX line (white) · Key levels annotated"))

        sections.append("<h2>GEX Heatmap — Expiry × Strike</h2>")
        sections.append(chart_tag(div_gex_heat, "GEX Heatmap",
            "Green = positive (call-heavy) · Red = negative (put-heavy) · Spot ±30%"))

        # ── DEX heatmap ───────────────────────────────────────────────────────
        sections.append("<h2>Delta Exposure Heatmap + Price Prediction</h2>")
        sections.append(chart_tag(div_dex_heat, "DEX Heatmap",
            "Blue = positive DEX (dealers long delta) · Red = negative DEX · "
            "Gold line = mechanical predicted price path · Dashed = 1σ/2σ vol cone"))

        # ── Per-greek sections ────────────────────────────────────────────────
        for col, title, formula in greek_defs:
            sections.append(f"<h2>{title}</h2>")
            sections.append(f'<p style="color:#8899aa;font-size:13px"><b>Formula:</b> {formula}</p>')

            sections.append(f"<h3>By Strike (±25% of spot)</h3>")
            sections.append(chart_tag(greek_divs[col]["strike"], f"{title} by Strike"))

            # Strike data table
            calls_s = df[df.flag=="call"].groupby("strike")[col].sum()
            puts_s  = df[df.flag=="put"].groupby("strike")[col].sum()
            stk = pd.concat([calls_s.rename("Call"), puts_s.rename("Put")], axis=1).fillna(0)
            stk["Net"] = stk["Call"] + stk["Put"]
            stk.index.name = "Strike"
            lo25, hi25 = spot * 0.75, spot * 1.25
            stk = stk[(stk.index >= lo25) & (stk.index <= hi25)].sort_index(ascending=False)
            sections.append(df_to_html(stk))

            sections.append(f"<h3>By Expiry</h3>")
            sections.append(chart_tag(greek_divs[col]["expiry"], f"{title} by Expiry"))
            sections.append(df_to_html(expiry_table(col)))

        # ── Full raw chain ────────────────────────────────────────────────────
        sections.append("<h2>Full Options Chain Data</h2>")
        sections.append(
            "<p style='color:#8899aa;font-size:12px'>Filtered to ±30% of spot, DTE ≤ 30. "
            "All values computed via Black-Scholes.</p>")
        raw_tbl = df[["strike","flag","expiry","dte","oi","volume","iv",
                       "GEX_oi","DEX","VEX","CEX"]].copy()
        raw_tbl.columns = ["Strike","Flag","Expiry","DTE","OI","Volume","IV",
                            "GEX($)","DEX($)","VEX($)","CEX($)"]
        lo30, hi30 = spot * 0.70, spot * 1.30
        raw_near = raw_tbl[
            (raw_tbl["Strike"] >= lo30) & (raw_tbl["Strike"] <= hi30) &
            (raw_tbl["DTE"] <= 30)
        ]
        sections.append(df_to_html(raw_near))

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{ticker} Greek Exposure — {ts}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background:#0e1117; color:#e0e0e0; font-family:'Segoe UI',Arial,sans-serif;
          font-size:14px; margin:0; padding:24px; }}
  h1   {{ color:#4a9eff; border-bottom:1px solid #2a2f3f; padding-bottom:8px; }}
  h2   {{ color:#4a9eff; margin-top:32px; border-left:3px solid #4a9eff;
          padding-left:10px; }}
  h3   {{ color:#8899aa; margin-top:20px; }}
  .stat-table {{ border-collapse:collapse; width:100%; margin-top:8px; }}
  .stat-table td {{ background:#1e2130; border:1px solid #2a2f3f;
                    padding:10px 14px; text-align:center; }}
  .stat-table b {{ color:#8899aa; font-size:11px; text-transform:uppercase;
                   display:block; margin-bottom:4px; }}
  .data-table {{ border-collapse:collapse; width:100%; font-size:12px; margin-top:8px; }}
  .data-table th {{ background:#1e2130; color:#8899aa; padding:8px 10px;
                    border:1px solid #2a2f3f; text-align:right; font-weight:600; }}
  .data-table td {{ background:#12151e; color:#ccc; padding:6px 10px;
                    border:1px solid #1a1f2e; text-align:right; }}
  .data-table tr:hover td {{ background:#1a2035; }}
  img {{ border-radius:6px; }}
  p   {{ color:#b0b8c8; line-height:1.6; }}
</style>
</head>
<body>
<h1>Options Greek Exposure — {ticker}</h1>
{"".join(sections)}
</body>
</html>"""

        fname = f"{ticker}_greek_exposure_{fn_ts}.html"

    st.success("Report ready — click below to download.")
    st.download_button(
        label=f"⬇ Download {fname}",
        data=html,
        file_name=fname,
        mime="text/html",
        use_container_width=True,
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_sum, tab_spot, tab_gamma, tab_delta, tab_vanna, tab_charm = st.tabs(
    ["Summary", "Spot", "Gamma", "Delta", "Vanna", "Charm"]
)

# ════════════════════════════════════════════════════════
# SUMMARY TAB
# ════════════════════════════════════════════════════════
with tab_sum:
    # Regime banner
    regime_text = (
        "Dealers are positioned to SUPPRESS moves. Expect mean-reverting, range-bound action. "
        "Dealers sell rallies and buy dips."
        if is_long_gamma else
        "Dealers are positioned to AMPLIFY moves. Expect trending, volatile action. "
        "Dealers sell dips and buy rallies."
    )
    icon = "🟢" if is_long_gamma else "🔴"
    st.markdown(f"""
    <div class="regime-box {regime_color}">
        <div style="font-size:11px;color:#8899aa;text-transform:uppercase;letter-spacing:.08em">
            VOLATILITY REGIME
        </div>
        <div style="font-size:20px;font-weight:700;margin:6px 0">{icon} {regime_label}</div>
        <div style="color:#ccc;font-size:13px">{regime_text}</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    gf = levels["gamma_flip"]
    cw = levels["call_wall"]
    pw = levels["put_wall"]
    pg = levels["peak_gex_strike"]
    pct_0dte = levels["pct_0dte"]
    flip_dist = ((gf - spot) / spot * 100) if gf else None

    with m1:
        st.markdown(metric_html("Net Dealer Gamma", fmt_num(net_gex),
            f"{'Long' if is_long_gamma else 'Short'} gamma — {'suppresses' if is_long_gamma else 'amplifies'} moves",
            "#00c853" if is_long_gamma else "#ff3d00"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_html("Hedge per ±1% Move", fmt_num(hedge_per_pct),
            "Dealers must buy/sell this amount"), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_html("Zero Gamma (Flip)", f"${gf:.2f}" if gf else "N/A",
            f"{flip_dist:+.1f}% from spot — regime boundary" if flip_dist else "No crossing found near spot",
            "#ffd700"), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_html("0DTE Gamma Concentration", f"{pct_0dte:.1f}%",
            f"of total GEX in 0–1 DTE options {'⚠ High' if pct_0dte > 40 else ''}",
            "#fb923c"), unsafe_allow_html=True)

    st.markdown("---")

    # Key levels
    st.markdown("#### Key Levels")
    au = levels["accel_up"]
    ad = levels["accel_down"]
    acw = levels["abs_call_wall"]; apw = levels["abs_put_wall"]
    # Flag when absolute walls diverge from spot-restricted walls (structural signal)
    cw_diverges = acw and abs(acw - cw) > 0.5 if acw and cw else False
    pw_diverges = apw and abs(apw - pw) > 0.5 if apw and pw else False

    levels_data = []
    if au:  levels_data.append(("⇡ Accel ↑",   f"${au:.2f}",  f"{(au-spot)/spot*100:+.1f}% above spot · secondary resistance", "#00c853"))
    if cw:  levels_data.append(("⬆ Call Wall",  f"${cw:.2f}",  f"{(cw-spot)/spot*100:+.1f}% above spot · dealer sell resistance", "#4a9eff"))
    if cw_diverges and acw:
            levels_data.append(("⬆ Abs Call Wall", f"${acw:.2f}", f"{(acw-spot)/spot*100:+.1f}% · global call GEX peak {'⚠ below spot' if acw < spot else ''}", "#7bc8ff"))
    if pg:  levels_data.append(("⊕ Peak GEX",   f"${pg:.2f}",  f"{(pg-spot)/spot*100:+.1f}% from spot · highest |net GEX|", "#c084fc"))
    if gf:  levels_data.append(("⇄ Zero Gamma", f"${gf:.2f}",  f"{(gf-spot)/spot*100:+.1f}% from spot · regime boundary", "#ffd700"))
    if pw:  levels_data.append(("⬇ Put Wall",   f"${pw:.2f}",  f"{(pw-spot)/spot*100:+.1f}% below spot · dealer buy support", "#ff6b6b"))
    if pw_diverges and apw:
            levels_data.append(("⬇ Abs Put Wall",  f"${apw:.2f}", f"{(apw-spot)/spot*100:+.1f}% · global put GEX trough {'⚠ above spot' if apw > spot else ''}", "#ff9999"))
    if ad:  levels_data.append(("⇣ Accel ↓",   f"${ad:.2f}",  f"{(ad-spot)/spot*100:+.1f}% below spot · secondary support", "#ff3d00"))

    cols = st.columns(len(levels_data)) if levels_data else [st]
    for i, (lbl, val, sub, clr) in enumerate(levels_data):
        with cols[i]:
            st.markdown(metric_html(lbl, val, sub, clr), unsafe_allow_html=True)

    st.markdown("---")

    # Gamma Decay by expiry
    st.markdown("#### Gamma Decay Over Time (by Expiry)")
    decay = gamma_decay_table(df)
    decay_display = decay.copy()
    decay_display["GEX_oi"] = decay_display["GEX_oi"].apply(lambda x: fmt_num(x))
    decay_display["normalized"] = decay_display["normalized"].apply(lambda x: f"{x:.2f}%")
    decay_display.columns = ["Expiry", "DTE", "Net GEX", "% of Total"]
    st.dataframe(decay_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Greeks overview bar — net by greek
    st.markdown("#### Net Greek Exposure Overview")
    net_dex = df["DEX"].sum()
    net_vex = df["VEX"].sum()
    net_cex = df["CEX"].sum()
    ov1, ov2, ov3, ov4 = st.columns(4)
    with ov1: st.markdown(metric_html("Net GEX", fmt_num(net_gex), "$ dealer buys/sells per 1% move"), unsafe_allow_html=True)
    with ov2: st.markdown(metric_html("Net DEX", fmt_num(net_dex, "$"), "$ notional delta — directional bias"), unsafe_allow_html=True)
    with ov3: st.markdown(metric_html("Net VEX", fmt_num(net_vex, "$"), "$ delta shift per +1 IV point", "#c084fc"), unsafe_allow_html=True)
    with ov4: st.markdown(metric_html("Net CEX", fmt_num(net_cex, "$"), "$ delta decay per trading day", "#fb923c"), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# SPOT TAB
# ════════════════════════════════════════════════════════
with tab_spot:
    st.markdown("#### Live Spot Gamma Footprint")

    src_mode = st.radio("Data source", ["Open Interest", "Volume"], horizontal=True)
    col_use = "GEX_oi" if src_mode == "Open Interest" else "GEX_vol"

    # Near-term expiry (first 2)
    near_expiries = sorted(df["expiry"].unique())[:2]
    df_near = df[df["expiry"].isin(near_expiries)]

    sp1, sp2, sp3 = st.columns(3)
    total_gex = df_near[col_use].sum()
    call_gex  = df_near[df_near.flag == "call"][col_use].sum()
    put_gex   = df_near[df_near.flag == "put"][col_use].sum()
    with sp1: st.markdown(metric_html("Net GEX (Near-term)", fmt_num(total_gex),
        f"Expiries: {', '.join(near_expiries)}"), unsafe_allow_html=True)
    with sp2: st.markdown(metric_html("Call GEX", fmt_num(call_gex), "Dealer short calls"), unsafe_allow_html=True)
    with sp3: st.markdown(metric_html("Put GEX",  fmt_num(put_gex),  "Dealer short puts", "#ff6b6b"), unsafe_allow_html=True)

    st.plotly_chart(
        hbar_chart(df_near, col_use, f"Spot GEX by Strike — {src_mode} ({', '.join(near_expiries)})",
                   spot, levels["call_wall"], levels["put_wall"], levels["gamma_flip"], pct_range=20),
        use_container_width=True)

    # Above / below spot
    above = df_near[df_near.strike > spot][col_use].sum()
    below = df_near[df_near.strike <= spot][col_use].sum()
    ab1, ab2 = st.columns(2)
    with ab1: st.markdown(metric_html("GEX Above Spot", fmt_num(above), "From upper strikes", "#4a9eff"), unsafe_allow_html=True)
    with ab2: st.markdown(metric_html("GEX Below Spot", fmt_num(below), "From lower strikes", "#ff6b6b"), unsafe_allow_html=True)




# ════════════════════════════════════════════════════════
# GREEK TAB BUILDER
# ════════════════════════════════════════════════════════
def render_greek_tab(df, col, label, spot, levels, prefix="$"):
    net_total  = df[col].sum()
    call_total = df[df.flag == "call"][col].sum()
    put_total  = df[df.flag == "put"][col].sum()
    above_spot = df[df.strike > spot][col].sum()
    below_spot = df[df.strike <= spot][col].sum()

    cw = levels["call_wall"] if col == "GEX_oi" else None
    pw = levels["put_wall"]  if col == "GEX_oi" else None
    gf = levels["gamma_flip"] if col == "GEX_oi" else None

    # Summary
    s1, s2, s3, s4 = st.columns(4)
    with s1: st.markdown(metric_html(f"Net {label}", fmt_num(net_total, prefix),
        f"{'Long' if net_total >= 0 else 'Short'} {label.split()[0]}"), unsafe_allow_html=True)
    with s2: st.markdown(metric_html("Call Exposure", fmt_num(call_total, prefix)), unsafe_allow_html=True)
    with s3: st.markdown(metric_html("Put Exposure",  fmt_num(put_total, prefix), color="#ff6b6b"), unsafe_allow_html=True)
    with s4:
        if gf:
            flip_dist = (gf - spot) / spot * 100
            st.markdown(metric_html("Flip Distance", f"{flip_dist:+.1f}%", f"Flip @ ${gf:.2f}", "#ffd700"), unsafe_allow_html=True)
        else:
            st.markdown(metric_html("Above / Below Spot",
                f"{fmt_num(above_spot, prefix)} / {fmt_num(below_spot, prefix)}", "↑ / ↓"), unsafe_allow_html=True)

    st.markdown("---")

    # ── By Strike ──
    st.markdown(f"#### {label} by Strike")
    pct_range = st.slider(
        "Strike range around spot (±%)", min_value=5, max_value=60, value=25, step=5,
        key=f"pct_{col}", help="Show strikes within this % above/below current spot price"
    )
    st.plotly_chart(
        hbar_chart(df, col, f"{label} by Strike  (spot ±{pct_range}%)", spot, cw, pw, gf,
                   pct_range=pct_range),
        use_container_width=True)

    # Strike table
    with st.expander("Strike table (all strikes, sorted by |Net|)"):
        calls_s = df[df.flag == "call"].groupby("strike")[col].sum().rename("Call_raw")
        puts_s  = df[df.flag == "put"].groupby("strike")[col].sum().rename("Put_raw")
        tbl = pd.concat([calls_s, puts_s], axis=1).fillna(0)
        tbl["Net_raw"] = tbl["Call_raw"] + tbl["Put_raw"]
        tbl["% Total"] = (tbl["Net_raw"].abs() / tbl["Net_raw"].abs().sum() * 100).round(2).astype(str) + "%"
        tbl = tbl.sort_values("Net_raw", key=abs, ascending=False)
        tbl_display = pd.DataFrame({
            "Call":    fmt_col(tbl["Call_raw"], prefix),
            "Put":     fmt_col(tbl["Put_raw"], prefix),
            "Net":     fmt_col(tbl["Net_raw"], prefix),
            "% Total": tbl["% Total"],
        }, index=tbl.index)
        tbl_display.index.name = "Strike"
        st.dataframe(tbl_display, use_container_width=True)

    st.markdown("---")

    # ── By Expiry ──
    st.markdown(f"#### {label} by Expiry")
    st.plotly_chart(vbar_chart(df, col, f"{label} by Expiry", spot), use_container_width=True)
    if col in ("VEX", "CEX"):
        st.caption(
            "ℹ️ **Why calls and puts sometimes appear on the same side:** "
            + ("Vanna (∂delta/∂σ) changes sign with moneyness — it is positive for OTM and negative for ITM options. "
               "When an expiry is dominated by ITM strikes, summed call VEX turns negative; summed put VEX (×−1) can also be negative. "
               "Both bars landing on the same side is correct — it reflects the real net dealer vanna positioning for that expiry."
               if col == "VEX" else
               "Charm (∂delta/∂t) changes sign with moneyness and time — it is positive for deep ITM and negative for OTM/near-expiry options. "
               "When an expiry's call book is ITM-heavy, summed call CEX turns positive; the put CEX (×−1) can also be positive. "
               "Both bars on the same side is correct — it reflects the actual delta-decay flow direction for that expiry."
            )
        )

    # Expiry table
    with st.expander("Expiry table"):
        calls_e = df[df.flag == "call"].groupby(["expiry","dte"])[col].sum().reset_index().rename(columns={col:"Call_raw"})
        puts_e  = df[df.flag == "put"].groupby(["expiry","dte"])[col].sum().reset_index().rename(columns={col:"Put_raw"})
        etbl = calls_e.merge(puts_e, on=["expiry","dte"], how="outer").fillna(0)
        etbl["Net_raw"] = etbl["Call_raw"] + etbl["Put_raw"]
        etbl["% Total"] = (etbl["Net_raw"].abs() / etbl["Net_raw"].abs().sum() * 100).round(2).astype(str) + "%"
        etbl = etbl.sort_values("dte")
        etbl_display = pd.DataFrame({
            "Expiry":  etbl["expiry"].values,
            "DTE":     etbl["dte"].values,
            "Call":    [fmt_num(v, prefix) for v in etbl["Call_raw"]],
            "Put":     [fmt_num(v, prefix) for v in etbl["Put_raw"]],
            "Net":     [fmt_num(v, prefix) for v in etbl["Net_raw"]],
            "% Total": etbl["% Total"].values,
        })
        st.dataframe(etbl_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── By Strike & Expiry ──
    st.markdown(f"#### {label} by Strike — Per Expiry")
    expiry_opts = sorted(df["expiry"].unique())
    sel_exp = st.selectbox("Select Expiry", expiry_opts, key=f"exp_{col}")
    df_sel = df[df.expiry == sel_exp]
    dte_sel = int(df_sel["dte"].iloc[0]) if not df_sel.empty else 0
    st.plotly_chart(
        hbar_chart(df_sel, col, f"{label} by Strike — {sel_exp} ({dte_sel} DTE)",
                   spot, cw, pw, gf, pct_range=pct_range),
        use_container_width=True)


# ════════════════════════════════════════════════════════
# GAMMA TAB  (InsiderFinance style)
# ════════════════════════════════════════════════════════
with tab_gamma:

    # ── Expiry filter ──────────────────────────────────
    exp_filter = st.radio(
        "Expiry filter", ["0DTE", "Weekly (≤7d)", "Monthly (≤30d)", "All"],
        index=3, horizontal=True, key="gex_exp_filter"
    )
    if exp_filter == "0DTE":
        df_g = df[df.dte == 0]
    elif exp_filter == "Weekly (≤7d)":
        df_g = df[df.dte <= 7]
    elif exp_filter == "Monthly (≤30d)":
        df_g = df[df.dte <= 30]
    else:
        df_g = df

    if df_g.empty:
        st.warning(f"No contracts for filter: {exp_filter}")
        st.stop()

    # ── Recompute levels from filtered data so walls/flip reflect the chosen expiry window ──
    levels_g = compute_levels(df_g, spot)

    # ── Stats bar ──────────────────────────────────────
    gex_net   = df_g["GEX_oi"].sum()
    gex_call  = df_g[df_g.flag=="call"]["GEX_oi"].sum()
    gex_put   = df_g[df_g.flag=="put"]["GEX_oi"].sum()
    gex_total = abs(gex_call) + abs(gex_put)
    oi_call   = df_g[df_g.flag=="call"]["oi"].sum()
    oi_put    = df_g[df_g.flag=="put"]["oi"].sum()
    oi_total  = oi_call + oi_put
    ratio     = abs(gex_put / gex_call) if gex_call != 0 else 0
    gf  = levels_g["gamma_flip"]
    cw  = levels_g["call_wall"]
    pw  = levels_g["put_wall"]
    gf_pct = (gf  - spot) / spot * 100 if gf  else None
    cw_pct = (cw  - spot) / spot * 100 if cw  else None
    pw_pct = (pw  - spot) / spot * 100 if pw  else None

    pg = levels_g["peak_gex_strike"]
    pct_0dte = levels_g["pct_0dte"]
    pg_pct = (pg - spot) / spot * 100 if pg else None

    c1,c2,c3,c4,c5,c6,c7,c8,c9,c10 = st.columns(10)
    def sm(col, label, val, sub=None, color="#4a9eff"):
        col.markdown(metric_html(label, val, sub, color), unsafe_allow_html=True)

    sm(c1,  "Spot",       f"${spot:.2f}", ticker)
    sm(c2,  "Net GEX",    fmt_num(gex_net),  f"Ratio put/call: {ratio:.2f}",
        "#00c853" if gex_net >= 0 else "#ff3d00")
    sm(c3,  "Call GEX",   fmt_num(gex_call), f"{oi_call:,.0f} OI", "#00c853")
    sm(c4,  "Put GEX",    fmt_num(gex_put),  f"{oi_put:,.0f} OI",  "#ff6b6b")
    sm(c5,  "Total GEX",  fmt_num(gex_total),f"{oi_total:,.0f} OI")
    sm(c6,  "Call Wall",  f"${cw:.0f}" if cw else "N/A",
        f"{cw_pct:+.1f}% · dealer sell wall" if cw_pct else "", "#4a9eff")
    sm(c7,  "Put Wall",   f"${pw:.0f}" if pw else "N/A",
        f"{pw_pct:+.1f}% · dealer buy wall" if pw_pct else "", "#ff6b6b")
    sm(c8,  "Zero Gamma", f"${gf:.2f}" if gf else "N/A",
        f"{gf_pct:+.1f}% · regime boundary" if gf_pct else "", "#ffd700")
    sm(c9,  "Peak GEX",   f"${pg:.0f}" if pg else "N/A",
        f"{pg_pct:+.1f}% · price magnet" if pg_pct else "", "#c084fc")
    sm(c10, "0DTE GEX %", f"{pct_0dte:.1f}%",
        "⚠ dominant" if pct_0dte > 40 else "of total abs GEX", "#fb923c")

    # Show absolute (unrestricted) walls if they diverge from spot-restricted walls
    acw = levels_g["abs_call_wall"]; apw = levels_g["abs_put_wall"]
    abs_notes = []
    if acw and cw and abs(acw - cw) > 0.5:
        flag = " ⚠ below spot — bearish structure" if acw < spot else ""
        abs_notes.append(f"**Abs Call Wall ${acw:.0f}** ({(acw-spot)/spot*100:+.1f}%){flag}")
    if apw and pw and abs(apw - pw) > 0.5:
        flag = " ⚠ above spot — unusual structure" if apw > spot else ""
        abs_notes.append(f"**Abs Put Wall ${apw:.0f}** ({(apw-spot)/spot*100:+.1f}%){flag}")
    if abs_notes:
        st.caption("Global (unrestricted) walls differ from spot-restricted walls: " + " · ".join(abs_notes))

    st.markdown("---")

    # ── Strike Profile (vertical bars) ────────────────
    st.markdown("#### Strike Profile")
    near_col, _ = st.columns([1, 4])
    with near_col:
        strike_view = st.radio("Strikes", ["Near", "All"], horizontal=True, key="gex_strike_view")

    fig_gex = build_gex_strike_profile(df_g, spot, levels_g, strike_view)
    st.plotly_chart(fig_gex, use_container_width=True)

    st.markdown("---")

    # ── GEX Heatmap by Expiry × Strike ────────────────
    st.markdown("#### GEX Heatmap by Expiration")

    fig_heat = build_gex_heatmap_fig(df_g, spot)
    if fig_heat is not None:
        st.caption("Green = Call-heavy (positive GEX) · Red = Put-heavy (negative GEX) · Spot ±30%")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No heatmap data for current filter.")

# ════════════════════════════════════════════════════════
# DELTA TAB
# ════════════════════════════════════════════════════════
with tab_delta:
    st.markdown("### Delta Exposure (DEX)")
    render_greek_tab(df, "DEX", "Delta Exposure (DEX $)", spot, levels, prefix="$")

    st.markdown("---")
    st.markdown("#### Delta Exposure Heatmap")
    st.caption(
        "**Y-axis** = strike / price level (high on top). "
        "**X-axis** = expiry date (nearest → farthest, ≤60 DTE). "
        "**Blue** = positive net DEX (calls dominant — dealers long delta, buy dips). "
        "**Red** = negative net DEX (puts dominant — dealers short delta, sell rallies). "
        "Gaussian-smoothed to show concentration blobs. "
        "Dashed white line = current spot price."
    )
    with st.spinner("Computing delta exposure heatmap + prediction…"):
        fig_dex_heat = build_dex_heatmap(
            df, spot, ticker,
            hist30=hist30,
            levels=levels,
            net_gex=net_gex,
        )
    if fig_dex_heat:
        st.plotly_chart(fig_dex_heat, use_container_width=True)
        st.caption(
            "**Gold line** = mechanical price path estimate. "
            "**Model inputs**: 10-day momentum × GEX regime multiplier "
            "(long γ → ×0.65 dampened, short γ → ×1.35 amplified) + "
            "gravitational pull toward Peak GEX strike (gamma magnet, k=0.04/day). "
            "**Bands** = ATM-IV implied 1σ / 2σ log-normal probability cone. "
            "⚠ Illustrative only — not investment advice."
        )
    else:
        st.info("Not enough near-term option data to build heatmap.")

# ════════════════════════════════════════════════════════
# VANNA TAB
# ════════════════════════════════════════════════════════
with tab_vanna:
    st.markdown("### Vanna Exposure (VEX)")
    st.info("Vanna measures how delta changes with IV. High vanna = large delta hedging flows when IV moves.")
    render_greek_tab(df, "VEX", "Vanna Exposure", spot, levels, prefix="")

# ════════════════════════════════════════════════════════
# CHARM TAB
# ════════════════════════════════════════════════════════
with tab_charm:
    st.markdown("### Charm Exposure (CEX)")
    st.info("Charm measures delta decay over time. High charm = large delta hedging flows as options expire.")
    render_greek_tab(df, "CEX", "Charm Exposure", spot, levels, prefix="")
