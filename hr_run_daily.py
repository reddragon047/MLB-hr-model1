
import pandas as pd
import numpy as np
from datetime import date, timedelta

from pybaseball import statcast

LOOKBACK_DAYS = 30

def get_statcast_window():
    end = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
    if df.empty:
        raise ValueError("statcast() returned no data.")
    return df

def safe_series_numeric(s, fill=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(fill)

def safe_minmax(series):
    s = safe_series_numeric(series, fill=0.0)
    smin = s.min()
    smax = s.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - smin) / (smax - smin)

def get_hitters(sc):
    df = sc.copy()
    df = df[df["type"] == "X"].copy()

    df["launch_speed"] = safe_series_numeric(df["launch_speed"])
    df["launch_angle"] = safe_series_numeric(df["launch_angle"])
    df["events"] = df["events"].fillna("")

    grouped = df.groupby("player_name", dropna=True).agg(
        avg_ev=("launch_speed", "mean"),
        launch_angle=("launch_angle", "mean"),
        hr_rate=("events", lambda x: (x == "home_run").mean()),
        bbe_count=("launch_speed", "size"),
        hard_hit_rate=("launch_speed", lambda x: (pd.to_numeric(x, errors="coerce").fillna(0) >= 95).mean()),
    ).reset_index()

    grouped["avg_ev"] = safe_series_numeric(grouped["avg_ev"])
    grouped["launch_angle"] = safe_series_numeric(grouped["launch_angle"])
    grouped["hr_rate"] = safe_series_numeric(grouped["hr_rate"])
    grouped["hard_hit_rate"] = safe_series_numeric(grouped["hard_hit_rate"])

    # proxy barrel rate from quality-of-contact components to avoid brittle missing fields
    ev_component = safe_minmax(grouped["avg_ev"])
    la_component = safe_minmax(grouped["launch_angle"].clip(lower=8, upper=32))
    grouped["barrel_rate"] = (0.7 * ev_component + 0.3 * la_component).fillna(0.0)

    return grouped

def get_pitchers(sc):
    df = sc.copy()
    df = df[df["type"] == "X"].copy()

    df["launch_speed"] = safe_series_numeric(df["launch_speed"])
    df["launch_angle"] = safe_series_numeric(df["launch_angle"])
    df["events"] = df["events"].fillna("")

    grouped = df.groupby("pitcher", dropna=True).agg(
        avg_ev_allowed=("launch_speed", "mean"),
        hr_rate_allowed=("events", lambda x: (x == "home_run").mean()),
        launch_angle_allowed=("launch_angle", "mean"),
        bbe_count=("launch_speed", "size"),
    ).reset_index()

    grouped["avg_ev_allowed"] = safe_series_numeric(grouped["avg_ev_allowed"])
    grouped["hr_rate_allowed"] = safe_series_numeric(grouped["hr_rate_allowed"])
    grouped["launch_angle_allowed"] = safe_series_numeric(grouped["launch_angle_allowed"])

    grouped["barrel_allowed"] = safe_minmax(grouped["avg_ev_allowed"])
    grouped["hr_per_9"] = (grouped["hr_rate_allowed"] * 9.0).fillna(0.0)

    # simple flyball proxy from mean LA allowed
    fly_proxy = grouped["launch_angle_allowed"].clip(lower=0, upper=30) / 30.0
    grouped["flyball_rate"] = safe_series_numeric(fly_proxy)

    return grouped

def base_score(row):
    avg_ev = 0.0 if pd.isna(row["avg_ev"]) else float(row["avg_ev"])
    launch_angle = 0.0 if pd.isna(row["launch_angle"]) else float(row["launch_angle"])
    hr_rate = 0.0 if pd.isna(row["hr_rate"]) else float(row["hr_rate"])
    barrel_rate = 0.0 if pd.isna(row["barrel_rate"]) else float(row["barrel_rate"])
    hard_hit_rate = 0.0 if pd.isna(row["hard_hit_rate"]) else float(row["hard_hit_rate"])

    return (
        0.35 * barrel_rate +
        0.25 * hard_hit_rate +
        0.20 * avg_ev +
        0.10 * launch_angle +
        0.10 * hr_rate
    )

def pitcher_mult(row):
    hr_per_9 = 0.0 if pd.isna(row["hr_per_9"]) else float(row["hr_per_9"])
    barrel_allowed = 0.0 if pd.isna(row["barrel_allowed"]) else float(row["barrel_allowed"])
    flyball_rate = 0.0 if pd.isna(row["flyball_rate"]) else float(row["flyball_rate"])

    val = (
        0.4 * hr_per_9 +
        0.3 * barrel_allowed +
        0.3 * flyball_rate
    )
    return float(np.clip(val, 0.8, 1.2))

def run():
    sc = get_statcast_window()

    hitters = get_hitters(sc)
    pitchers = get_pitchers(sc)

    hitters["base"] = hitters.apply(base_score, axis=1)

    pitcher_pool_mult = pitchers["pitcher_mult"] = pitchers.apply(pitcher_mult, axis=1)

    # v1 broad slate scoring: hitter base skill times average pitcher environment
    avg_pitcher_mult = float(pd.to_numeric(pitcher_pool_mult, errors="coerce").fillna(1.0).mean())

    results = hitters[["player_name", "base", "bbe_count"]].copy()
    results["score"] = results["base"] * avg_pitcher_mult
    results = results.rename(columns={"player_name": "player"})
    results = results.sort_values(["score", "bbe_count"], ascending=[False, False]).reset_index(drop=True)

    results.to_csv("hr_board.csv", index=False)
    print("DONE - saved hr_board.csv")

if __name__ == "__main__":
    run()
