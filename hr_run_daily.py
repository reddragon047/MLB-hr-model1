import pandas as pd
import numpy as np
from datetime import date, timedelta

from pybaseball import statcast_batter, statcast_pitcher
import statsapi

# =========================
# CONFIG
# =========================
LOOKBACK_DAYS = 30
LEAGUE_AVG_HR9 = 1.1

# =========================
# GET TODAY'S GAMES
# =========================
def get_schedule():
    today = date.today().isoformat()
    games = statsapi.schedule(date=today)
    return games

# =========================
# GET PROBABLE PITCHERS
# =========================
def get_probable_pitchers(games):
    pitchers = []
    for g in games:
        if g.get("home_probable_pitcher") and g.get("away_probable_pitcher"):
            pitchers.append({
                "game_id": g["game_id"],
                "home_team": g["home_name"],
                "away_team": g["away_name"],
                "home_pitcher": g["home_probable_pitcher"],
                "away_pitcher": g["away_probable_pitcher"]
            })
    return pd.DataFrame(pitchers)

# =========================
# GET HITTER DATA
# =========================
def get_hitter_data():
    end = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)

    df = statcast_batter(start_dt=start, end_dt=end)

    grouped = df.groupby("player_name").agg({
        "launch_speed": "mean",
        "launch_angle": "mean",
        "events": lambda x: (x == "home_run").mean()
    }).reset_index()

    grouped.columns = ["name", "avg_ev", "launch_angle", "recent_hr_rate"]

    grouped["barrel_rate"] = grouped["avg_ev"] / grouped["avg_ev"].max()
    grouped["hard_hit_rate"] = grouped["avg_ev"] / grouped["avg_ev"].max()

    return grouped

# =========================
# GET PITCHER DATA
# =========================
def get_pitcher_data():
    end = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)

    df = statcast_pitcher(start_dt=start, end_dt=end)

    grouped = df.groupby("player_name").agg({
        "events": lambda x: (x == "home_run").mean(),
        "launch_speed": "mean"
    }).reset_index()

    grouped.columns = ["name", "hr_per_event", "avg_ev_allowed"]

    grouped["hr_per_9"] = grouped["hr_per_event"] * 9
    grouped["barrel_allowed"] = grouped["avg_ev_allowed"] / grouped["avg_ev_allowed"].max()
    grouped["flyball_rate"] = 0.4  # placeholder

    return grouped

# =========================
# BASE HR SKILL
# =========================
def compute_base(row):
    return (
        0.35 * row["barrel_rate"] +
        0.25 * row["hard_hit_rate"] +
        0.20 * row["avg_ev"] +
        0.10 * row["launch_angle"] +
        0.10 * row["recent_hr_rate"]
    )

# =========================
# PITCHER MULT
# =========================
def pitcher_mult(p):
    val = (
        0.4 * p["hr_per_9"] +
        0.3 * p["barrel_allowed"] +
        0.3 * p["flyball_rate"]
    )
    return np.clip(val, 0.8, 1.2)

# =========================
# MAIN
# =========================
def run():
    games = get_schedule()
    prob_pitchers = get_probable_pitchers(games)

    hitters = get_hitter_data()
    pitchers = get_pitcher_data()

    results = []

    for _, g in prob_pitchers.iterrows():
        for _, h in hitters.iterrows():
            # naive match (can refine later)
            base = compute_base(h)

            p_row = pitchers[pitchers["name"] == g["home_pitcher"]]
            if p_row.empty:
                continue

            p_mult = pitcher_mult(p_row.iloc[0])

            score = base * p_mult

            results.append({
                "player": h["name"],
                "game_id": g["game_id"],
                "score": score
            })

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)

    df.to_csv("hr_board.csv", index=False)
    print("Done.")

if __name__ == "__main__":
    run()
