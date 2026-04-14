
import pandas as pd
import numpy as np
from datetime import date, timedelta

from pybaseball import statcast
import statsapi

LOOKBACK_DAYS = 30

def get_schedule():
    today = date.today().isoformat()
    return statsapi.schedule(date=today)

def get_probable_pitchers(games):
    rows = []
    for g in games:
        if g.get("home_probable_pitcher") and g.get("away_probable_pitcher"):
            rows.append({
                "game_id": g["game_id"],
                "home_pitcher": g["home_probable_pitcher"],
                "away_pitcher": g["away_probable_pitcher"]
            })
    return pd.DataFrame(rows)

def get_statcast():
    end = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    return statcast(start_dt=start.isoformat(), end_dt=end.isoformat())

def get_hitters(df):
    df = df[df["type"] == "X"]

    grouped = df.groupby("player_name").agg(
        avg_ev=("launch_speed", "mean"),
        launch_angle=("launch_angle", "mean"),
        hr_rate=("events", lambda x: (x == "home_run").mean())
    ).reset_index()

    grouped["barrel_rate"] = grouped["avg_ev"] / grouped["avg_ev"].max()
    grouped["hard_hit_rate"] = grouped["avg_ev"] / grouped["avg_ev"].max()

    return grouped

def get_pitchers(df):
    df = df[df["type"] == "X"]

    grouped = df.groupby("pitcher").agg(
        avg_ev_allowed=("launch_speed", "mean"),
        hr_rate_allowed=("events", lambda x: (x == "home_run").mean())
    ).reset_index()

    grouped["barrel_allowed"] = grouped["avg_ev_allowed"] / grouped["avg_ev_allowed"].max()
    grouped["hr_per_9"] = grouped["hr_rate_allowed"] * 9
    grouped["flyball_rate"] = 0.4

    return grouped

def base_score(row):
    return (
        0.35 * row["barrel_rate"] +
        0.25 * row["hard_hit_rate"] +
        0.20 * row["avg_ev"] +
        0.10 * row["launch_angle"] +
        0.10 * row["hr_rate"]
    )

def pitcher_mult(row):
    val = (
        0.4 * row["hr_per_9"] +
        0.3 * row["barrel_allowed"] +
        0.3 * row["flyball_rate"]
    )
    return np.clip(val, 0.8, 1.2)

def run():
    games = get_schedule()
    sched = get_probable_pitchers(games)

    sc = get_statcast()

    hitters = get_hitters(sc)
    pitchers = get_pitchers(sc)

    hitters["base"] = hitters.apply(base_score, axis=1)

    results = []

    for _, p in pitchers.iterrows():
        p_mult = pitcher_mult(p)

        for _, h in hitters.iterrows():
            score = h["base"] * p_mult

            results.append({
                "player": h["player_name"],
                "score": score
            })

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    df.to_csv("hr_board.csv", index=False)
    print("DONE")

if __name__ == "__main__":
    run()
