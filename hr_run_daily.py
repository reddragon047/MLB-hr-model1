
import pandas as pd
import numpy as np
from datetime import date, timedelta

from pybaseball import statcast
import statsapi

# =========================
# CONFIG
# =========================
LOOKBACK_DAYS = 30

# =========================
# GET TODAY'S GAMES
# =========================
def get_schedule():
    today = date.today().isoformat()
    return statsapi.schedule(date=today)

# =========================
# GET PROBABLE PITCHERS
# =========================
def get_probable_pitchers(games):
    rows = []
    for g in games:
        home_pitcher = g.get("home_probable_pitcher")
        away_pitcher = g.get("away_probable_pitcher")
        if not home_pitcher or not away_pitcher:
            continue

        rows.append({
            "game_id": g["game_id"],
            "home_team": g["home_name"],
            "away_team": g["away_name"],
            "home_pitcher": home_pitcher,
            "away_pitcher": away_pitcher,
        })
    return pd.DataFrame(rows)

# =========================
# GET LAST-X-DAYS STATCAST
# =========================
def get_statcast_window():
    end = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
    if df.empty:
        raise ValueError("statcast() returned no data.")
    return df

# =========================
# BUILD HITTER DATA
# =========================
def get_hitter_data(sc):
    # Only balls in play / contacted outcomes
    bbe = sc[sc["type"] == "X"].copy()

    grouped = bbe.groupby(["batter", "player_name"]).agg(
        avg_ev=("launch_speed", "mean"),
        launch_angle=("launch_angle", "mean"),
        bbe_count=("launch_speed", "size"),
        barrel_count=("launch_speed_angle", lambda x: x.isin([6]).sum()),
        hard_hit_count=("launch_speed", lambda x: (x >= 95).sum()),
        hr_count=("events", lambda x: (x == "home_run").sum()),
    ).reset_index()

    grouped["barrel_rate"] = np.where(
        grouped["bbe_count"] > 0,
        grouped["barrel_count"] / grouped["bbe_count"],
        0.0,
    )
    grouped["hard_hit_rate"] = np.where(
        grouped["bbe_count"] > 0,
        grouped["hard_hit_count"] / grouped["bbe_count"],
        0.0,
    )
    grouped["recent_hr_rate"] = np.where(
        grouped["bbe_count"] > 0,
        grouped["hr_count"] / grouped["bbe_count"],
        0.0,
    )

    grouped = grouped.rename(columns={
        "batter": "batter_id",
        "player_name": "name",
    })

    return grouped[[
        "batter_id",
        "name",
        "avg_ev",
        "launch_angle",
        "barrel_rate",
        "hard_hit_rate",
        "recent_hr_rate",
        "bbe_count",
    ]]

# =========================
# BUILD PITCHER DATA
# =========================
def get_pitcher_data(sc):
    bbe = sc[sc["type"] == "X"].copy()

    grouped = bbe.groupby(["pitcher", "pitcher.1"]).agg(
        bbe_allowed=("launch_speed", "size"),
        avg_ev_allowed=("launch_speed", "mean"),
        hr_allowed=("events", lambda x: (x == "home_run").sum()),
        hard_hit_allowed=("launch_speed", lambda x: (x >= 95).sum()),
        barrel_allowed_count=("launch_speed_angle", lambda x: x.isin([6]).sum()),
        launch_angle_allowed=("launch_angle", "mean"),
    ).reset_index()

    # Approximate HR/9 from last-30d BBE sample.
    # This is not perfect innings-based HR/9, but it is stable enough for a v1 matchup layer.
    grouped["hr_per_9_proxy"] = np.where(
        grouped["bbe_allowed"] > 0,
        (grouped["hr_allowed"] / grouped["bbe_allowed"]) * 27.0,
        0.0,
    )
    grouped["barrel_allowed"] = np.where(
        grouped["bbe_allowed"] > 0,
        grouped["barrel_allowed_count"] / grouped["bbe_allowed"],
        0.0,
    )
    grouped["flyball_rate"] = np.where(
        grouped["bbe_allowed"] > 0,
        (grouped["launch_angle_allowed"] >= 10).astype(float),
        0.0,
    )

    grouped = grouped.rename(columns={
        "pitcher": "pitcher_id",
        "pitcher.1": "name",
    })

    return grouped[[
        "pitcher_id",
        "name",
        "hr_per_9_proxy",
        "barrel_allowed",
        "flyball_rate",
        "bbe_allowed",
    ]]

# =========================
# NORMALIZATION
# =========================
def safe_minmax(series):
    series = series.fillna(0.0)
    smin = series.min()
    smax = series.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series(np.ones(len(series)) * 0.5, index=series.index)
    return (series - smin) / (smax - smin)

# =========================
# BASE HR SKILL
# =========================
def compute_base_hitter_scores(hitters):
    h = hitters.copy()

    h["barrel_n"] = safe_minmax(h["barrel_rate"])
    h["hard_hit_n"] = safe_minmax(h["hard_hit_rate"])
    h["avg_ev_n"] = safe_minmax(h["avg_ev"])
    h["launch_angle_n"] = safe_minmax(h["launch_angle"].clip(lower=0, upper=30))
    h["recent_hr_n"] = safe_minmax(h["recent_hr_rate"])

    h["base_score"] = (
        0.35 * h["barrel_n"] +
        0.25 * h["hard_hit_n"] +
        0.20 * h["avg_ev_n"] +
        0.10 * h["launch_angle_n"] +
        0.10 * h["recent_hr_n"]
    )
    return h

# =========================
# PITCHER MULTIPLIER
# =========================
def compute_pitcher_scores(pitchers):
    p = pitchers.copy()

    p["hr9_n"] = safe_minmax(p["hr_per_9_proxy"])
    p["barrel_allowed_n"] = safe_minmax(p["barrel_allowed"])
    p["flyball_n"] = safe_minmax(p["flyball_rate"])

    raw = (
        0.40 * p["hr9_n"] +
        0.35 * p["barrel_allowed_n"] +
        0.25 * p["flyball_n"]
    )

    p["pitcher_mult"] = np.clip(0.8 + (raw * 0.4), 0.8, 1.2)
    return p

# =========================
# BUILD MATCHUPS
# =========================
def build_matchup_rows(schedule_df, hitters_df, pitchers_df):
    rows = []

    # Very light v1 roster matching by team name tokens is not reliable,
    # so this version scores the broad slate against each probable pitcher
    # and tags the game context. Next upgrade should use confirmed lineups / rosters.
    for _, g in schedule_df.iterrows():
        home_pitcher = pitchers_df[pitchers_df["name"] == g["home_pitcher"]]
        away_pitcher = pitchers_df[pitchers_df["name"] == g["away_pitcher"]]

        if not home_pitcher.empty:
            hp = home_pitcher.iloc[0]
            for _, h in hitters_df.iterrows():
                rows.append({
                    "game_id": g["game_id"],
                    "opp_pitcher": g["home_pitcher"],
                    "pitcher_mult": hp["pitcher_mult"],
                    "player": h["name"],
                    "base_score": h["base_score"],
                    "matchup_side": "vs_home_pitcher",
                })

        if not away_pitcher.empty:
            ap = away_pitcher.iloc[0]
            for _, h in hitters_df.iterrows():
                rows.append({
                    "game_id": g["game_id"],
                    "opp_pitcher": g["away_pitcher"],
                    "pitcher_mult": ap["pitcher_mult"],
                    "player": h["name"],
                    "base_score": h["base_score"],
                    "matchup_side": "vs_away_pitcher",
                })

    return pd.DataFrame(rows)

# =========================
# MAIN
# =========================
def run():
    schedule = get_schedule()
    schedule_df = get_probable_pitchers(schedule)
    if schedule_df.empty:
        raise ValueError("No probable pitchers found for today's schedule.")

    sc = get_statcast_window()

    hitters = get_hitter_data(sc)
    pitchers = get_pitcher_data(sc)

    hitters = compute_base_hitter_scores(hitters)
    pitchers = compute_pitcher_scores(pitchers)

    board = build_matchup_rows(schedule_df, hitters, pitchers)
    if board.empty:
        raise ValueError("No board rows were created.")

    board["score"] = board["base_score"] * board["pitcher_mult"]
    board = board.sort_values("score", ascending=False).reset_index(drop=True)

    board.to_csv("hr_board.csv", index=False)
    print("Done. Saved hr_board.csv")

if __name__ == "__main__":
    run()
