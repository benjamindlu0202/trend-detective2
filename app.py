import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    analyze, block_bootstrap_ci, years_to_detect, seasonal_trends, grid_trends,
)

st.set_page_config(page_title="Earth System Trend Detective", page_icon="🌍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap');
html, body, p, div, label, span { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Serif', serif !important; letter-spacing: -0.01em; }
.verdict { padding: 1.1rem 1.3rem; border-radius: 10px; border-left: 4px solid;
           font-size: 1.1rem; line-height: 1.5; margin: 0.4rem 0 1.4rem; }
.verdict.sig { background: rgba(78, 205, 196, 0.10); border-color: #4ecdc4; }
.verdict.ns  { background: rgba(255, 255, 255, 0.04); border-color: #8892a6; }
.subtitle { color: #9aa3b5; font-size: 1.05rem; margin-top: -0.6rem; }
</style>
""", unsafe_allow_html=True)

VARIABLES = {
    "Air temperature": {"code": "T2M", "unit": "°C", "cmap": "RdBu_r",
                        "desc": "Average air temperature 2 meters above the ground."},
    "Precipitation": {"code": "PRECTOTCORR", "unit": "mm/day", "cmap": "BrBG",
                      "desc": "Average daily rain and snow, measured as liquid water."},
    "Relative humidity": {"code": "RH2M", "unit": "%", "cmap": "BrBG",
                          "desc": "How close the air is to saturation, 2 meters above the ground."},
    "Wind speed": {"code": "WS10M", "unit": "m/s", "cmap": "PuOr",
                   "desc": "Average wind speed 10 meters above the ground."},
}

PLACES = {
    "Lowell, MA": (42.63, -71.32),
    "Phoenix, AZ": (33.45, -112.07),
    "Miami, FL": (25.76, -80.19),
    "Seattle, WA": (47.61, -122.33),
    "Death Valley, CA": (36.46, -116.87),
    "Custom location": None,
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def style(fig, ytitle="", height=420):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.1, x=0),
        yaxis_title=ytitle,
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def run_point(lat, lon, code, start, end):
    raw, anom, res = analyze(lat, lon, code, start, end)
    lo, hi = block_bootstrap_ci(anom)
    yrs = years_to_detect(anom, res["per_decade"])
    seas = seasonal_trends(raw)
    return raw, anom, res, lo, hi, yrs, seas


@st.cache_data(show_spinner=False)
def run_grid(lat, lon, code):
    return grid_trends(lat - 2, lat + 2, lon - 2, lon + 2, step=1.0, variable=code)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### Investigation")
    place = st.selectbox("Place", list(PLACES))
    if PLACES[place] is None:
        lat = st.number_input("Latitude", value=42.63, min_value=-90.0, max_value=90.0)
        lon = st.number_input("Longitude", value=-71.32, min_value=-180.0, max_value=180.0)
        place_label = f"{lat:.2f}, {lon:.2f}"
    else:
        lat, lon = PLACES[place]
        place_label = place
        st.caption(f"{lat:.2f}° N, {abs(lon):.2f}° W")

    var_name = st.selectbox("Variable", list(VARIABLES))
    st.caption(VARIABLES[var_name]["desc"])

    start, end = st.slider("Years", 1985, 2025, (1985, 2025))

    if st.button("Investigate", type="primary"):
        st.session_state["query"] = (lat, lon, var_name, start, end, place_label)

    st.markdown("---")
    st.caption("Data: NASA POWER monthly satellite and reanalysis records.")

# ---------- Header ----------
st.title("Earth System Trend Detective")
st.markdown('<p class="subtitle">Is it changing? Where? How fast? And is the change real, '
            'or just noise?</p>', unsafe_allow_html=True)

if "query" not in st.session_state:
    st.markdown("")
    st.info("Pick a place and a variable in the sidebar, then click **Investigate**.")
    st.stop()

lat, lon, var_name, start, end, place_label = st.session_state["query"]
v = VARIABLES[var_name]
unit = v["unit"]

with st.spinner("Downloading NASA data and running the analysis..."):
    raw, anom, res, lo, hi, yrs, seas = run_point(lat, lon, v["code"], start, end)

rate = float(res["per_decade"])
p = float(res["p_value"])
direction = "rising" if rate > 0 else "falling"
yrs_text = f"{yrs:.0f}" if np.isfinite(yrs) else "∞"

# ---------- Verdict ----------
if res["significant"]:
    verdict = (f'<div class="verdict sig"><b>{var_name}</b> in <b>{place_label}</b> is {direction} '
               f'by <b>{abs(rate):.2f} {unit} per decade</b>. This trend is statistically '
               f'significant, even after accounting for month-to-month correlation.</div>')
else:
    verdict = (f'<div class="verdict ns"><b>{var_name}</b> in <b>{place_label}</b> may be {direction} '
               f'by about {abs(rate):.2f} {unit} per decade, but the change can\'t yet be '
               f'separated from natural variability. At this rate, detecting it would take '
               f'about <b>{yrs_text} years</b> of data.</div>')
st.markdown(verdict, unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rate per decade", f"{rate:+.2f} {unit}")
c2.metric("95% interval", f"{lo:+.2f} to {hi:+.2f}",
          help="Block bootstrap: resamples 12-month chunks to respect autocorrelation.")
c3.metric("p-value", f"{p:.1e}" if p < 0.001 else f"{p:.3f}",
          help="Mann-Kendall test with Hamed-Rao autocorrelation correction.")
c4.metric("Years to detect", yrs_text,
          help="Weatherhead et al. (1998): years of data needed to detect this trend.")

st.markdown("")
tab1, tab2, tab3, tab4 = st.tabs(["Over time", "By season", "Regional map", "How it works"])

# ---------- Tab 1: time series ----------
with tab1:
    a = anom.dropna()
    tt = np.arange(len(a))
    trend = res["intercept"] + res["slope_per_month"] * tt

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.index, y=a, mode="lines", name="Monthly anomaly",
                             line=dict(color="rgba(130,170,230,0.35)", width=1)))
    fig.add_trace(go.Scatter(x=a.index, y=a.rolling(12, center=True).mean(), mode="lines",
                             name="12-month average", line=dict(color="#f4a261", width=2.5)))
    fig.add_trace(go.Scatter(x=a.index, y=trend, mode="lines",
                             name=f"Trend ({rate:+.2f} {unit}/decade)",
                             line=dict(color="#4ecdc4", width=3)))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_width=1)
    st.plotly_chart(style(fig, f"Difference from normal ({unit})"))
    st.caption("Each month is compared with the long-term average for that calendar month, "
               "so the seasonal cycle is removed and only the long-term change remains.")

# ---------- Tab 2: seasons ----------
with tab2:
    colors = ["#4ecdc4" if pv < 0.05 else "#4a5368" for pv in seas["p_value"]]
    fig2 = go.Figure(go.Bar(
        x=MONTHS, y=seas["per_decade"], marker_color=colors,
        customdata=seas["p_value"],
        hovertemplate="%{x}: %{y:.2f} " + unit + " per decade<br>p = %{customdata:.3f}<extra></extra>",
    ))
    st.plotly_chart(style(fig2, f"{unit} per decade", height=380))

    i = seas["per_decade"].abs().idxmax()
    st.markdown(f"Fastest change: **{MONTHS[i]}**, at {seas.loc[i, 'per_decade']:+.2f} "
                f"{unit} per decade. Highlighted bars are individually significant.")
    st.caption("Months with similar rates can differ in significance because some months "
               "vary much more from year to year than others.")

# ---------- Tab 3: regional map ----------
with tab3:
    st.markdown("Scan a 5 × 5 grid of locations around this point and test each one. "
                "Testing many places at once guarantees some false alarms, so results are "
                "corrected with the Benjamini-Hochberg false discovery rate procedure.")

    key = (lat, lon, v["code"])
    if st.button("Scan surrounding region"):
        st.session_state["grid_for"] = key

    if st.session_state.get("grid_for") == key:
        with st.spinner("Testing 25 locations (about 30 seconds the first time)..."):
            df = run_grid(lat, lon, v["code"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Locations tested", len(df))
        m2.metric("Significant before correction", int(df["sig_raw"].sum()))
        m3.metric("Significant after FDR", int(df["sig_fdr"].sum()))

        mx = float(df["per_decade"].abs().max()) or 1.0
        fig3 = go.Figure(go.Scattermap(
            lat=df["lat"], lon=df["lon"], mode="markers",
            marker=dict(
                size=np.where(df["sig_fdr"], 26, 15),
                color=df["per_decade"], colorscale=v["cmap"], cmin=-mx, cmax=mx,
                opacity=np.where(df["sig_fdr"], 1.0, 0.45),
                colorbar=dict(title=f"{unit}/decade"),
            ),
            text=[f"{r.lat:.0f}, {r.lon:.0f}<br>{r.per_decade:+.2f} {unit}/decade<br>"
                  f"q = {r.q:.3f}" for r in df.itertuples()],
            hoverinfo="text",
        ))
        fig3.update_layout(
            map=dict(style="carto-darkmatter", center=dict(lat=lat, lon=lon), zoom=5),
            margin=dict(l=0, r=0, t=0, b=0), height=520,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig3)
        st.caption("Large, solid dots are significant after FDR correction. "
                   f"With {len(df)} tests at the 5% level, about {0.05 * len(df):.1f} "
                   "false positives would be expected by chance alone.")

# ---------- Tab 4: methods ----------
with tab4:
    st.markdown("#### 1. Remove the seasonal cycle")
    st.markdown("Each month's value is compared with the long-term average for that calendar month.")
    st.latex(r"a_t = x_t - \bar{x}_{m(t)}")

    st.markdown("#### 2. Test for a trend")
    st.markdown("The Mann-Kendall test checks whether later values tend to exceed earlier ones. "
                "Because neighboring months are correlated, the variance is corrected with the "
                "Hamed-Rao method. The rate is Sen's slope, the median of all pairwise slopes, "
                "which resists outliers.")
    st.latex(r"\hat\beta = \operatorname{median}\left(\frac{a_j - a_i}{j - i}\right), \quad i < j")

    st.markdown("#### 3. Quantify uncertainty")
    st.markdown("A block bootstrap resamples 12-month chunks of residuals, preserving their "
                "correlation, and refits the slope 500 times. The middle 95% of those slopes "
                "is the interval.")

    st.markdown("#### 4. How long until a trend is detectable")
    st.markdown("Following Weatherhead et al. (1998), with noise σ, lag-1 autocorrelation φ, "
                "and trend ω per year:")
    st.latex(r"n^* \approx \left[\frac{3.3\,\sigma}{|\omega|}\sqrt{\frac{1+\phi}{1-\phi}}\right]^{2/3}")

    st.markdown("#### 5. Testing many places at once")
    st.markdown("The Benjamini-Hochberg procedure controls the expected share of false "
                "discoveries among all significant results. It remains valid when nearby "
                "locations are positively correlated, which they are.")
