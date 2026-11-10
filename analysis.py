import requests
import pandas as pd
import pymannkendall as mk


def get_power_monthly(lat, lon, variable="T2M", start=1985, end=2025):
    """Download a monthly time series for one NASA POWER variable at one location."""
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": variable,
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    values = r.json()["properties"]["parameter"][variable]
    s = pd.Series(values)
    s = s[~s.index.str.endswith("13")]
    s.index = pd.to_datetime(s.index, format="%Y%m")
    s = s.replace(-999, float("nan"))
    s.name = variable
    return s.sort_index()


def deseasonalize(s):
    """Subtract each calendar month's long-term average."""
    monthly_avg = s.groupby(s.index.month).transform("mean")
    return s - monthly_avg


def trend_test(anom):
    """Mann-Kendall test corrected for autocorrelation. Returns a summary dict."""
    result = mk.hamed_rao_modification_test(anom.dropna())
    return {
        "trend": result.trend,
        "p_value": result.p,
        "significant": result.p < 0.05,
        "per_decade": result.slope * 120,
        "slope_per_month": result.slope,
        "intercept": result.intercept,
    }


def analyze(lat, lon, variable="T2M", start=1985, end=2025):
    """Run the full pipeline: download, deseasonalize, test."""
    raw = get_power_monthly(lat, lon, variable, start, end)
    anom = deseasonalize(raw)
    stats = trend_test(anom)
    return raw, anom, stats
    