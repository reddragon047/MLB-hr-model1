import os
import json
import argparse
from datetime import date as date_cls

import numpy as np
import pandas as pd

# Data sources
from pybaseball import statcast
import statsapi

# Modeling
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression


# ---------------------------
# Settings you can tweak later
# ---------------------------
TRAIN_SEASONS = [2023, 2024, 2025]  # change anytime
FEATURE_COLS = ["barrel_rate", "avg_ev", "avg_la", "k_rate", "bb_rate", "BBE"]

DEFAULT_SIMS = 100000
DEFAULT_TOPN = 50


# ---------------------------
# Helpers: filesystem
# ---------------------------
def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


# ---------------------------
# Helpers: stats / shrinkage
# ---------------------------
def shrink_rate(successes, trials, prior_mean, prior_strength):
    """
    Empirical-Bayes shrinkage for binomial rates.
    Posterior mean of Beta-Binomial with prior mean & strength.
    """
    a = prior_mean * prior_strength
    b = (1 - prior_mean) * prior_strength
    return (successes + a) / (trials + a + b)


def expected_pa_simple(is_home: bool) -> float:
    """
    Simple PA estimate before we add confirmed lineups.
    Good first-pass: away hitters slightly more PA on average.
    """
    base = 4.35
    if is_home:
        base -= 0.05
    return base


def sim_hr_prob(p_pa: float, exp_pa: float, n_sims: int, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    pa = rng.poisson(lam=max(exp_pa, 0.1), size=n_sims)
    hr = rng.binomial(n=pa, p=float(np.clip(p_pa, 0.0, 1.0)))
    return float((hr >= 1).mean())


# ---------------------------
# Statcast pulling + feature building
# ---------------------------
def pull_statcast_season(season: int) -> pd.DataFrame:
    # broad regular season window
    start = f"{season}-03-01"
    end = f"{season}-11-15"
    return statcast(start_dt=start, end_dt=end)


def batter_season_features(stat_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Convert pitch-level Statcast to batter-season feature rows.
    """
    df = stat_df.copy()
    df["season"] = season

    # PA-ending pitches: events not null
    pa = df[df["events"].notna()].copy()

    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_k"] = (pa["events"] == "strikeout").astype(int)
    pa["is_bb"] = (pa["events"] == "walk").astype(int)

    # Batted ball subset (needs EV/LA)
    bbe = pa[pa["launch_speed"].notna() & pa["launch_angle"].notna()].copy()

    # Barrel proxy (good enough to start; can refine later)
    bbe["is_barrel_proxy"] = ((bbe["launch_speed"] >= 98) & (bbe["launch_angle"].between(26, 30))).astype(int)

    agg_pa = pa.groupby(["batter", "season"]).agg(
        PA=("events", "size"),
        HR=("is_hr", "sum"),
        K=("is_k", "sum"),
        BB=("is_bb", "sum"),
    ).reset_index()

    agg_bbe = bbe.groupby(["batter", "season"]).agg(
        BBE=("launch_speed", "size"),
        avg_ev=("launch_speed", "mean"),
        avg_la=("launch_angle", "mean"),
        barrel_rate=("is_barrel_proxy", "mean"),
    ).reset_index()

    out = agg_pa.merge(agg_bbe, on=["batter", "season"], how="left").fillna(0)

    out["k_rate"] = out["K"] / out["PA"].clip(lower=1)
    out["bb_rate"] = out["BB"] / out["PA"].clip(lower=1)
    out["hr_rate"] = out["HR"] / out["PA"].clip(lower=1)

    return out


def get_training_table_cached() -> pd.DataFrame:
    cache_path = "data/processed/batter_season_train.parquet"
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    tables = []
    for s in TRAIN_SEASONS:
        raw_path = f"data/raw/statcast_{s}.parquet"
        if os.path.exists(raw_path):
            season_df = pd.read_parquet(raw_path)
        else:
            season_df = pull_statcast_season(s)
            season_df.to_parquet(raw_path, index=False)

        tables.append(batter_season_features(season_df, s))

    train = pd.concat(tables, ignore_index=True)

    # Multi-year weights: oldest->newest = 2/3/5 (you can tweak)
    weights = {TRAIN_SEASONS[0]: 2.0, TRAIN_SEASONS[1]: 3.0, TRAIN_SEASONS[2]: 5.0}
    train["w"] = train["season"].map(weights).fillna(1.0)

    train.to_parquet(cache_path, index=False)
    return train


# ---------------------------
# Model training + calibration
# ---------------------------
def train_or_load_model(train_df: pd.DataFrame):
    model_path = "models/hr_rate_xgb.json"
    meta_path = "models/meta.json"
    calib_path = "models/calibrator.pkl"

    if os.path.exists(model_path) and os.path.exists(meta_path) and os.path.exists(calib_path):
        model = XGBRegressor()
        model.load_model(model_path)
        meta = json.load(open(meta_path, "r"))
        calibrator = pd.read_pickle(calib_path)
        return model, calibrator, meta

    df = train_df.copy()

    # League prior for HR/PA
    league_hr_pa = float(df["HR"].sum() / df["PA"].sum()) if df["PA"].sum() > 0 else 0.032

    # Shrink HR/PA toward league average (stabilizes small samples)
    df["hr_rate_shrunk"] = shrink_rate(df["HR"], df["PA"], prior_mean=league_hr_pa, prior_strength=200)

    X = df[FEATURE_COLS].fillna(0.0)
    y = df["hr_rate_shrunk"].astype(float)
    w = df["w"].astype(float)

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, w, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=900,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    # Isotonic calibration for predicted rates (helps rare-event probability quality)
    raw_val = np.clip(model.predict(X_val), 1e-6, 0.5)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_val, sample_weight=w_val)

    model.save_model(model_path)
    json.dump({"league_hr_pa": league_hr_pa}, open(meta_path, "w"))
    pd.to_pickle(iso, calib_path)

    return model, iso, {"league_hr_pa": league_hr_pa}


# ---------------------------
# MLB schedule + roster hitters
# ---------------------------
def get_slate_games_and_probables(slate_date: str):
    sched = statsapi.schedule(date=slate_date)
    games = []
    for g in sched:
        games.append({
            "home_team": g.get("home_name"),
            "away_team": g.get("away_name"),
            "venue_name": g.get("venue_name"),
            "home_probable_pitcher": g.get("home_probable_pitcher"),
            "away_probable_pitcher": g.get("away_probable_pitcher"),
        })
    return games


def get_team_hitters(team_name: str):
    """
    Uses active roster as hitter pool (good v1).
    Later upgrade: actual confirmed lineup once posted.
    """
    teams = statsapi.lookup_team(team_name)
    if not teams:
        return []
    team_id = teams[0]["id"]

    roster = statsapi.roster(team_id, rosterType="active")
    hitters = []
    for p in roster:
        pos = (p.get("position") or "").lower()
        if "pitcher" in pos:
            continue
        hitters.append(int(p["person"]["id"]))  # MLBAM id matches Statcast 'batter'
    return hitters


# ---------------------------
# Daily scoring board
# ---------------------------
def build_daily_board(slate_date: str, n_sims: int) -> pd.DataFrame:
    train_df = get_training_table_cached()
    model, calibrator, meta = train_or_load_model(train_df)

    # Use most recent season’s feature row per batter as current skill snapshot
    feature_table = pd.read_parquet("data/processed/batter_season_train.parquet")
    latest = feature_table.sort_values("season").groupby("batter").tail(1).set_index("batter")

    games = get_slate_games_and_probables(slate_date)
    rows = []

    for g in games:
        home = g["home_team"]
        away = g["away_team"]
        venue = g["venue_name"]
        home_sp = g.get("home_probable_pitcher") or "TBD"
        away_sp = g.get("away_probable_pitcher") or "TBD"

        for side in ["home", "away"]:
            batting_team = home if side == "home" else away
            opponent = away if side == "home" else home
            opp_pitcher = away_sp if side == "home" else home_sp
            is_home = (side == "home")

            hitter_ids = get_team_hitters(batting_team)

            for pid in hitter_ids:
                if pid not in latest.index:
                    continue

                feat = latest.loc[pid, FEATURE_COLS].to_frame().T.fillna(0.0)

                p_pa_raw = float(model.predict(feat)[0])
                p_pa = float(calibrator.predict([np.clip(p_pa_raw, 1e-6, 0.5)])[0])
                p_pa = float(np.clip(p_pa, 1e-6, 0.25))

                exp_pa = expected_pa_simple(is_home=is_home)

                p_game_model = 1 - (1 - p_pa) ** exp_pa
                p_game_sim = sim_hr_prob(p_pa=p_pa, exp_pa=exp_pa, n_sims=n_sims)

                rows.append({
                    "date": slate_date,
                    "team": batting_team,
                    "opponent": opponent,
                    "venue": venue,
                    "probable_pitcher_faced": opp_pitcher,
                    "batter_id": int(pid),
                    "exp_pa": round(exp_pa, 2),
                    "p_pa": p_pa,
                    "p_hr_model": p_game_model,
                    "p_hr_sim": p_game_sim,
                })

    board = pd.DataFrame(rows)
    if board.empty:
        return board

    board = board.sort_values(["p_hr_sim", "p_hr_model"], ascending=False).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    return board


def write_outputs(board: pd.DataFrame, slate_date: str, top_n: int):
    csv_path = f"outputs/hr_board_{slate_date}.csv"
    html_path = f"outputs/hr_board_{slate_date}.html"

    board.to_csv(csv_path, index=False)

    show = board.head(top_n).copy()
    for c in ["p_hr_model", "p_hr_sim", "p_pa"]:
        if c in show.columns:
            show[c] = (show[c] * 100).round(2)

    html = f"""
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial; padding: 14px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; }}
        th {{ background: #f4f4f4; position: sticky; top: 0; }}
        tr:nth-child(even) {{ background: #fafafa; }}
      </style>
    </head>
    <body>
      <h2>HR Probability Board — {slate_date} (Top {top_n})</h2>
      <p>pHR_model / pHR_sim / pPA are percentages.</p>
      {show.to_html(index=False, escape=True)}
    </body>
    </html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=str(date_cls.today()))
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--top", type=int, default=DEFAULT_TOPN)
    args = parser.parse_args()

    ensure_dirs()

    board = build_daily_board(slate_date=args.date, n_sims=args.sims)

    if board.empty:
        print("No games found for that date (or no hitters matched features).")
        return

    write_outputs(board, slate_date=args.date, top_n=args.top)

    print("\nTop 25 (by sim probability):")
    print(board.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
