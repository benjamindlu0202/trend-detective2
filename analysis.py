import time

import numpy as np
import pandas as pd
import pymannkendall as mk
import requests
from scipy import stats
from statsmodels.stats.multitest import multipletests


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
    result = trend_test(anom)
    return raw, anom, result


def block_bootstrap_ci(anom, n_boot=500, block=12, seed=0):
    """95% CI for the trend per decade, resampling 12-month blocks of residuals."""
    y = anom.dropna().values
    t = np.arange(len(y))
    slope, intercept, _, _ = stats.theilslopes(y, t)
    fit = intercept + slope * t
    resid = y - fit

    rng = np.random.default_rng(seed)
    n = len(y)
    n_blocks = int(np.ceil(n / block))
    boot_slopes = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        r = np.concatenate([resid[s:s + block] for s in starts])[:n]
        boot_slopes.append(stats.theilslopes(fit + r, t)[0])

    lo, hi = np.percentile(boot_slopes, [2.5, 97.5])
    return lo * 120, hi * 120


def years_to_detect(anom, per_decade):
    """Weatherhead et al. (1998): years of data needed to detect this trend."""
    y = anom.dropna().values
    t = np.arange(len(y))
    slope, intercept, _, _ = stats.theilslopes(y, t)
    resid = y - (intercept + slope * t)

    sigma = resid.std()
    phi = pd.Series(resid).autocorr(lag=1)
    omega = abs(per_decade) / 10
    if omega == 0:
        return float("inf")
    return ((3.3 * sigma / omega) * np.sqrt((1 + phi) / (1 - phi))) ** (2 / 3)


def seasonal_trends(raw):
    """Trend per decade for each calendar month, tested separately."""
    rows = []
    for m in range(1, 13):
        s = raw[raw.index.month == m].dropna()
        r = mk.original_test(s)
        rows.append({"month": m, "per_decade": r.slope * 10, "p_value": r.p})
    return pd.DataFrame(rows)


def grid_trends(lat_min, lat_max, lon_min, lon_max, step=2.0, variable="PRECTOTCORR"):
    """Trend test on a grid of points, with Benjamini-Hochberg FDR correction."""
    rows = []
    for lat in np.arange(lat_min, lat_max + 0.01, step):
        for lon in np.arange(lon_min, lon_max + 0.01, step):
            try:
                raw = get_power_monthly(lat, lon, variable)
                s = trend_test(deseasonalize(raw))
                rows.append({"lat": lat, "lon": lon,
                             "per_decade": s["per_decade"], "p": s["p_value"]})
            except Exception as e:
                print(f"skipped {lat}, {lon}: {e}")
            time.sleep(0.3)

    df = pd.DataFrame(rows)
    df["q"] = multipletests(df["p"], method="fdr_bh")[1]
    df["sig_raw"] = df["p"] < 0.05
    df["sig_fdr"] = df["q"] < 0.05
    return df


def abc_trend(anom, n_sims=10000, keep=0.01, seed=0):
    """ABC posterior for trend + AR(1) noise. Returns accepted parameter samples."""
    rng = np.random.default_rng(seed)
    y = anom.dropna().values
    n = len(y)
    t = np.arange(n)
    tc = t - t.mean()

    def summaries(X):
        X = X - X.mean(axis=1, keepdims=True)
        slope = X @ tc / (tc @ tc)
        resid = X - slope[:, None] * tc
        ac1 = (resid[:, :-1] * resid[:, 1:]).sum(1) / (resid ** 2).sum(1)
        return np.column_stack([slope, ac1, resid.std(axis=1)])

    beta = rng.uniform(-0.01, 0.01, n_sims)
    phi = rng.uniform(0.0, 0.9, n_sims)
    sigma = rng.uniform(0.1, 3.0, n_sims)

    e = rng.standard_normal((n_sims, n))
    X = np.empty((n_sims, n))
    X[:, 0] = sigma * e[:, 0] / np.sqrt(1 - phi ** 2)
    for i in range(1, n):
        X[:, i] = phi * X[:, i - 1] + sigma * e[:, i]
    X += beta[:, None] * t

    S = summaries(X)
    s_obs = summaries(y[None, :])[0]
    dist = np.sqrt((((S - s_obs) / S.std(axis=0)) ** 2).sum(axis=1))
    idx = np.argsort(dist)[: int(keep * n_sims)]

    return pd.DataFrame({"per_decade": beta[idx] * 120,
                         "phi": phi[idx], "sigma": sigma[idx]})
