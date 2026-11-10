import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from analysis import analyze

VARIABLES = {
    "Temperature (°C)": "T2M",
    "Precipitation (mm/day)": "PRECTOTCORR",
    "Relative humidity (%)": "RH2M",
    "Wind speed (m/s)": "WS10M",
}

st.title("Earth System Trend Detective")
st.write("Pick a place and a variable to see whether it's changing, and whether the change is significant.")

lat = st.number_input("Latitude", value=42.63, min_value=-90.0, max_value=90.0)
lon = st.number_input("Longitude", value=-71.32, min_value=-180.0, max_value=180.0)
label = st.selectbox("Variable", list(VARIABLES.keys()))
variable = VARIABLES[label]


@st.cache_data
def run(lat, lon, variable):
    return analyze(lat, lon, variable)


if st.button("Investigate"):
    with st.spinner("Downloading NASA data..."):
        raw, anom, stats = run(lat, lon, variable)

    direction = "rising" if stats["per_decade"] > 0 else "falling"
    if stats["significant"]:
        st.success(f"{label} is {direction} by {abs(stats['per_decade']):.2f} per decade. This trend is statistically significant (p = {stats['p_value']:.1e}).")
    else:
        st.info(f"{label} is {direction} by {abs(stats['per_decade']):.2f} per decade, but this is not statistically significant (p = {stats['p_value']:.2f}).")

    t = np.arange(len(anom))
    trend_line = stats["intercept"] + stats["slope_per_month"] * t

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(anom.index, anom, alpha=0.4, label="Monthly anomaly")
    ax.plot(anom.index, anom.rolling(12, center=True).mean(), linewidth=2, label="12-month average")
    ax.plot(anom.index, trend_line, color="red", linewidth=2, label="Trend")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Difference from normal")
    ax.legend()
    st.pyplot(fig)