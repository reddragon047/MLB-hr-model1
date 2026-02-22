# =========================
# hr_run_daily.py (FULL)
# =========================

import os
import re
import json
import argparse
import unicodedata
from datetime import date as date_cls, datetime, timedelta

import numpy as np
import pandas as pd
import requests

from pybaseball import statcast
import statsapi

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

from inputs import market_clv

# -------------------------
# CONFIG
# -------------------------
TRAIN_SEASONS_DEFAULT = [2023, 2024, 2025]
DEFAULT_SIMS = 100000
DEFAULT_TOPN = 75

FEATURE_COLS = [
    "barrel_rate",
    "avg_ev",
    "avg_la",
    "k_rate",
    "bb_rate",
    "BBE",
]

PITCH_TYPES = ["FF", "SI", "SL", "CH", "CU", "FC", "FS"]  # common buckets

# Plate Appearance baselines by lineup spot (neutral environment).
# Used ONLY when lineup is confirmed; otherwise exp_pa falls back to proxy logic.
LINEUP_BASELINES = {
    1: 4.65,
    2: 4.55,
    3: 4.45,
    4: 4.35,
    5: 4.25,
    6: 4.15,
    7: 4.05,
    8: 3.95,
    9: 3.85,
}


# Bullpen integration (sniper-safe)
DEFAULT_W_BP = 0.40                 # expected share of hitter PAs vs bullpen
W_BP_MIN, W_BP_MAX = 0.25, 0.55     # clamp w_bp
BP_MIN, BP_MAX = 0.85, 1.15         # clamp bullpen factor
BULLPEN_PRIOR_STRENGTH = 3000       # shrinkage strength (higher = more stable)


# -------------------------
# Utils
# -------------------------
def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models_hr", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("inputs", exist_ok=True)


def shrink_rate(successes, trials, prior_mean, prior_strength):
    a = prior_mean * prior_strength
    b = (1 - prior_mean) * prior_strength
    return (successes + a) / (trials + a + b)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sim_hr_probs(p_pa: float, exp_pa: float, n_sims: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    pa = rng.poisson(lam=max(exp_pa, 0.05), size=n_sims)
    hr = rng.binomial(n=pa, p=float(np.clip(p_pa, 0.0, 1.0)))
    p1 = float((hr >= 1).mean())
    p2 = float((hr >= 2).mean())
    return p1, p2


def get_json(url: str, timeout: int = 25):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (HRBoard)"})
    r.raise_for_status()
    return r.json()


def bullpen_adjustment_multiplier(bullpen_factor: float, w_bp: float = DEFAULT_W_BP) -> float:
    """
    Sniper-safe bullpen blend:
      mult = 1 + w_bp*(bullpen_factor - 1)

    bullpen_factor: team bullpen HR/PA factor vs league (e.g., 1.08 = +8% HR allowed)
    w_bp: expected share of hitter PAs vs bullpen (0.25–0.55 typical)
    """
    w_bp = clamp(float(w_bp), W_BP_MIN, W_BP_MAX)
    bullpen_factor = clamp(float(bullpen_factor), BP_MIN, BP_MAX)
    return 1.0 + w_bp * (bullpen_factor - 1.0)


# -------------------------
# Player name cache
# -------------------------
PLAYER_NAME_CACHE: dict[int, str] = {}


def get_player_name(player_id: int) -> str:
    """
    Resolve MLBAM player_id -> full name via statsapi 'person' endpoint.
    Cached per run to avoid spamming.
    """
    if player_id in PLAYER_NAME_CACHE:
        return PLAYER_NAME_CACHE[player_id]

    name = str(player_id)
    try:
        j = statsapi.get("person", {"personId": player_id})
        people = j.get("people", [])
        if people and isinstance(people[0], dict):
            nm = people[0].get("fullName")
            if nm:
                name = nm
    except Exception:
        pass

    PLAYER_NAME_CACHE[player_id] = name
    return name


# -------------------------
# Statcast Pull + Feature Building
# -------------------------
def pull_statcast_season(season: int, end_dt: str | None = None) -> pd.DataFrame:
    """
    Pull Statcast for a season.

    If end_dt is provided (YYYY-MM-DD), pulls up to that date.
    This is used to keep the current season *fresh* every run.
    """
    start = f"{season}-03-01"
    end = end_dt or f"{season}-11-15"
    try:
        df = statcast(start_dt=start, end_dt=end)
        # Some edge cases can return an empty df without expected columns.
        # Keep it as-is; downstream guards will handle schema issues safely.
        return df
    except Exception:
        return pd.DataFrame()


def batter_pitcher_tables(stat_df: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Guard: if Statcast pull returned an empty frame or unexpected schema,
    # return empty tables instead of crashing (prevents 'KeyError: events').
    if stat_df is None or getattr(stat_df, 'empty', True) or "events" not in getattr(stat_df, 'columns', []):
        bat_cols = [
            "batter", "season", "PA", "HR", "K", "BB",
            "BBE", "avg_ev", "avg_la", "barrel_rate",
            "k_rate", "bb_rate", "hr_rate",
        ]
        pit_cols = [
            "pitcher", "season", "PA", "HR_allowed", "K", "BB",
            "BBE", "avg_ev_allowed", "avg_la_allowed", "barrel_rate_allowed",
            "k_rate_allowed", "bb_rate_allowed", "hr_rate_allowed",
        ]
        return pd.DataFrame(columns=bat_cols), pd.DataFrame(columns=pit_cols)

    df = stat_df.copy()
    df["season"] = season

    # Force numeric in case upstream cached parquet created object columns
    for col in ["launch_speed", "launch_angle"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pa = df[df["events"].notna()].copy()

    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_k"] = (pa["events"] == "strikeout").astype(int)
    pa["is_bb"] = (pa["events"] == "walk").astype(int)

    bbe = pa[pa["launch_speed"].notna() & pa["launch_angle"].notna()].copy()
    bbe["is_barrel_proxy"] = ((bbe["launch_speed"] >= 98) & (bbe["launch_angle"].between(26, 30))).astype(int)

    bat_pa = pa.groupby(["batter", "season"]).agg(
        PA=("events", "size"),
        HR=("is_hr", "sum"),
        K=("is_k", "sum"),
        BB=("is_bb", "sum"),
    ).reset_index()

    bat_bbe = bbe.groupby(["batter", "season"]).agg(
        BBE=("launch_speed", "size"),
        avg_ev=("launch_speed", "mean"),
        avg_la=("launch_angle", "mean"),
        barrel_rate=("is_barrel_proxy", "mean"),
    ).reset_index()

    bat = bat_pa.merge(bat_bbe, on=["batter", "season"], how="left").fillna(0)

    # Ensure numeric dtypes for model features
    for c in ["BBE", "avg_ev", "avg_la", "barrel_rate"]:
        if c in bat.columns:
            bat[c] = pd.to_numeric(bat[c], errors="coerce").fillna(0.0)

    bat["k_rate"] = bat["K"] / bat["PA"].clip(lower=1)
    bat["bb_rate"] = bat["BB"] / bat["PA"].clip(lower=1)
    bat["hr_rate"] = bat["HR"] / bat["PA"].clip(lower=1)

    pit_pa = pa.groupby(["pitcher", "season"]).agg(
        PA=("events", "size"),
        HR_allowed=("is_hr", "sum"),
        K=("is_k", "sum"),
        BB=("is_bb", "sum"),
    ).reset_index()

    pit_bbe = bbe.groupby(["pitcher", "season"]).agg(
        BBE=("launch_speed", "size"),
        avg_ev_allowed=("launch_speed", "mean"),
        avg_la_allowed=("launch_angle", "mean"),
        barrel_rate_allowed=("is_barrel_proxy", "mean"),
    ).reset_index()

    pit = pit_pa.merge(pit_bbe, on=["pitcher", "season"], how="left").fillna(0)

    for c in ["BBE", "avg_ev_allowed", "avg_la_allowed", "barrel_rate_allowed"]:
        if c in pit.columns:
            pit[c] = pd.to_numeric(pit[c], errors="coerce").fillna(0.0)

    pit["k_rate_allowed"] = pit["K"] / pit["PA"].clip(lower=1)
    pit["bb_rate_allowed"] = pit["BB"] / pit["PA"].clip(lower=1)
    pit["hr_rate_allowed"] = pit["HR_allowed"] / pit["PA"].clip(lower=1)

    return bat, pit


def build_pitch_mix_and_batter_damage(stat_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Guard: if Statcast schema is missing (e.g., empty pull), return empty tables.
    req = {"pitcher", "batter", "pitch_type", "events"}
    if stat_df is None or getattr(stat_df, 'empty', True) or not req.issubset(set(getattr(stat_df, 'columns', []))):
        mix_cols = ["pitcher"] + [f"pit_usage_{pt}" for pt in PITCH_TYPES]
        dmg_cols = ["batter"] + [f"bat_hr_pa_{pt}" for pt in PITCH_TYPES]
        return pd.DataFrame(columns=mix_cols), pd.DataFrame(columns=dmg_cols)

    df = stat_df.copy()
    df = df[df["pitch_type"].notna()].copy()
    df["pitch_type"] = df["pitch_type"].astype(str)

    mix = df.groupby(["pitcher", "pitch_type"]).size().reset_index(name="n")
    tot = mix.groupby("pitcher")["n"].sum().reset_index(name="tot")
    mix = mix.merge(tot, on="pitcher", how="left")
    mix["usage"] = mix["n"] / mix["tot"].clip(lower=1)
    mix = mix[mix["pitch_type"].isin(PITCH_TYPES)].copy()

    pa = df[df["events"].notna()].copy()
    pa = pa[pa["pitch_type"].isin(PITCH_TYPES)].copy()
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)

    bat_pt = pa.groupby(["batter", "pitch_type"]).agg(
        PA=("events", "size"),
        HR=("is_hr", "sum"),
    ).reset_index()

    league = bat_pt.groupby("pitch_type").agg(PA=("PA", "sum"), HR=("HR", "sum")).reset_index()
    league["league_hr_pa"] = league["HR"] / league["PA"].clip(lower=1)

    bat_pt = bat_pt.merge(league[["pitch_type", "league_hr_pa"]], on="pitch_type", how="left")
    bat_pt["hr_pa_shrunk"] = bat_pt.apply(
        lambda r: shrink_rate(r["HR"], r["PA"], r["league_hr_pa"], prior_strength=250),
        axis=1
    )

    damage = bat_pt.pivot_table(index="batter", columns="pitch_type", values="hr_pa_shrunk", aggfunc="mean").fillna(np.nan)
    damage.columns = [f"bat_hr_pa_{c}" for c in damage.columns]
    damage = damage.reset_index()

    mix_p = mix.pivot_table(index="pitcher", columns="pitch_type", values="usage", aggfunc="mean").fillna(0)
    mix_p.columns = [f"pit_usage_{c}" for c in mix_p.columns]
    mix_p = mix_p.reset_index()

    return mix_p, damage


def compute_park_factors(stat_df: pd.DataFrame, k_barrel: int = 1500, k_fb: int = 2000) -> pd.DataFrame:
    """Compute park HR conversion multipliers keyed by home_team."""
    # Guard: if Statcast schema is missing (e.g., empty pull), return empty park table.
    req = {"home_team", "events", "launch_speed", "launch_angle"}
    if stat_df is None or getattr(stat_df, 'empty', True) or not req.issubset(set(getattr(stat_df, 'columns', []))):
        return pd.DataFrame(columns=[
            "home_team", "park_barrel_mult", "park_fb_mult",
            "barrels", "flyballs", "hr_per_barrel_shrunk", "hr_per_fb_shrunk",
        ])

    df = stat_df.copy()

    df = df[df["events"].notna()].copy()
    df = df[df["launch_speed"].notna() & df["launch_angle"].notna()].copy()

    df["launch_speed"] = pd.to_numeric(df["launch_speed"], errors="coerce")
    df["launch_angle"] = pd.to_numeric(df["launch_angle"], errors="coerce")
    df = df[df["launch_speed"].notna() & df["launch_angle"].notna()].copy()

    df["is_hr"] = (df["events"] == "home_run").astype(int)

    la = df["launch_angle"].astype(float)
    ev = df["launch_speed"].astype(float)

    df["is_fb_proxy"] = ((la >= 25.0) & (la <= 50.0)).astype(int)

    if "is_barrel_proxy" not in df.columns:
        df["is_barrel_proxy"] = ((ev >= 98.0) & (la.between(26.0, 30.0))).astype(int)

    league_barrels = float(df["is_barrel_proxy"].sum())
    league_hr_on_barrels = float((df["is_hr"] * df["is_barrel_proxy"]).sum())
    league_hr_per_barrel = league_hr_on_barrels / max(1.0, league_barrels)

    league_fbs = float(df["is_fb_proxy"].sum())
    league_hr_on_fbs = float((df["is_hr"] * df["is_fb_proxy"]).sum())
    league_hr_per_fb = league_hr_on_fbs / max(1.0, league_fbs)

    park = df.groupby("home_team").apply(
        lambda x: pd.Series({
            "barrels": float(x["is_barrel_proxy"].sum()),
            "hr_on_barrels": float((x["is_hr"] * x["is_barrel_proxy"]).sum()),
            "flyballs": float(x["is_fb_proxy"].sum()),
            "hr_on_flyballs": float((x["is_hr"] * x["is_fb_proxy"]).sum()),
        })
    ).reset_index()

    park["hr_per_barrel_raw"] = park["hr_on_barrels"] / park["barrels"].clip(lower=1.0)
    park["hr_per_fb_raw"] = park["hr_on_flyballs"] / park["flyballs"].clip(lower=1.0)

    park["hr_per_barrel_shrunk"] = park.apply(
        lambda r: shrink_rate(
            successes=r["hr_on_barrels"],
            trials=r["barrels"],
            prior_mean=float(league_hr_per_barrel),
            prior_strength=int(k_barrel),
        ),
        axis=1,
    )
    park["hr_per_fb_shrunk"] = park.apply(
        lambda r: shrink_rate(
            successes=r["hr_on_flyballs"],
            trials=r["flyballs"],
            prior_mean=float(league_hr_per_fb),
            prior_strength=int(k_fb),
        ),
        axis=1,
    )

    park["park_barrel_mult"] = park["hr_per_barrel_shrunk"] / max(float(league_hr_per_barrel), 1e-6)
    park["park_fb_mult"] = park["hr_per_fb_shrunk"] / max(float(league_hr_per_fb), 1e-6)

    return park[
        [
            "home_team",
            "park_barrel_mult",
            "park_fb_mult",
            "barrels",
            "flyballs",
            "hr_per_barrel_shrunk",
            "hr_per_fb_shrunk",
        ]
    ]


def compute_bullpen_factors(stat_all: pd.DataFrame) -> pd.DataFrame:
    """
    Build bullpen HR/PA factors by team from Statcast training data.
    """
    required = {"game_pk", "inning_topbot", "home_team", "away_team", "pitcher", "events"}
    if not required.issubset(stat_all.columns):
        return pd.DataFrame({"team": [], "bullpen_factor": []})

    pa = stat_all[stat_all["events"].notna()].copy()
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)

    pa["pitching_team"] = np.where(
        pa["inning_topbot"].astype(str).str.lower().str.startswith("top"),
        pa["home_team"],
        pa["away_team"],
    )

    sort_cols = ["game_pk", "pitching_team", "inning"]
    if "at_bat_number" in pa.columns:
        sort_cols.append("at_bat_number")

    pa_sorted = pa.sort_values(sort_cols)

    starters = (
        pa_sorted.groupby(["game_pk", "pitching_team"])["pitcher"]
        .first()
        .reset_index()
        .rename(columns={"pitcher": "starter_pitcher"})
    )

    pa2 = pa.merge(starters, on=["game_pk", "pitching_team"], how="left")
    bullpen_pa = pa2[pa2["pitcher"] != pa2["starter_pitcher"]].copy()

    if bullpen_pa.empty:
        return pd.DataFrame({"team": [], "bullpen_factor": []})

    team_bp = bullpen_pa.groupby("pitching_team").agg(
        PA=("events", "size"),
        HR=("is_hr", "sum"),
    ).reset_index().rename(columns={"pitching_team": "team"})

    league_hr_pa = float(team_bp["HR"].sum() / team_bp["PA"].sum()) if team_bp["PA"].sum() > 0 else 0.032

    team_bp["bp_hr_pa_shrunk"] = team_bp.apply(
        lambda r: shrink_rate(r["HR"], r["PA"], prior_mean=league_hr_pa, prior_strength=BULLPEN_PRIOR_STRENGTH),
        axis=1
    )

    team_bp["bullpen_factor"] = team_bp["bp_hr_pa_shrunk"] / max(league_hr_pa, 1e-6)
    team_bp["bullpen_factor"] = team_bp["bullpen_factor"].clip(lower=BP_MIN, upper=BP_MAX)

    return team_bp[["team", "bullpen_factor"]]


# -------------------------
# Training Cache (with live current-season refresh)
# -------------------------
def _season_is_current(season: int, date_str: str) -> bool:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return int(season) == int(d.year)
    except Exception:
        return False


def get_training_cached(train_seasons: list[int], date_str: str):
    key = "_".join(map(str, train_seasons))
    bat_path = f"data/processed/bat_{key}.parquet"
    pit_path = f"data/processed/pit_{key}.parquet"
    mix_path = f"data/processed/mix_{key}.parquet"
    dmg_path = f"data/processed/dmg_{key}.parquet"
    park_path = f"data/processed/park_{key}.parquet"
    bp_path = f"data/processed/bullpen_{key}.parquet"

    have_all = all(os.path.exists(p) for p in [bat_path, pit_path, mix_path, dmg_path, park_path, bp_path])

    # If we have cached training, still refresh current season raw Statcast each run.
    all_stat = []
    bat_tables = []
    pit_tables = []

    for s in train_seasons:
        raw_path = f"data/raw/statcast_{s}.parquet"

        season_df = None

        if os.path.exists(raw_path):
            try:
                season_df = pd.read_parquet(raw_path)
            except Exception:
                season_df = None

        if _season_is_current(s, date_str):
            # Pull through *yesterday* relative to date_str (so morning run captures yesterday)
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
                end_dt = (d - timedelta(days=1)).isoformat()
            except Exception:
                end_dt = None

            fresh = pull_statcast_season(s, end_dt=end_dt)
            if isinstance(fresh, pd.DataFrame) and not fresh.empty:
                season_df = fresh
                try:
                    season_df.to_parquet(raw_path, index=False)
                except Exception:
                    pass
        else:
            if season_df is None:
                season_df = pull_statcast_season(s)
                try:
                    season_df.to_parquet(raw_path, index=False)
                except Exception:
                    pass

        if season_df is None:
            season_df = pd.DataFrame()

        all_stat.append(season_df)

        bat, pit = batter_pitcher_tables(season_df, s)
        bat_tables.append(bat)
        pit_tables.append(pit)

    stat_all = pd.concat(all_stat, ignore_index=True) if all_stat else pd.DataFrame()
    bat_all = pd.concat(bat_tables, ignore_index=True) if bat_tables else pd.DataFrame()
    pit_all = pd.concat(pit_tables, ignore_index=True) if pit_tables else pd.DataFrame()

    mix_p, dmg_b = build_pitch_mix_and_batter_damage(stat_all)
    park = compute_park_factors(stat_all)
    bullpen = compute_bullpen_factors(stat_all)

    # Write processed caches
    try:
        bat_all.to_parquet(bat_path, index=False)
        pit_all.to_parquet(pit_path, index=False)
        mix_p.to_parquet(mix_path, index=False)
        dmg_b.to_parquet(dmg_path, index=False)
        park.to_parquet(park_path, index=False)
        bullpen.to_parquet(bp_path, index=False)
    except Exception:
        pass

    return bat_all, pit_all, mix_p, dmg_b, park, bullpen


# -------------------------
# Model Training
# -------------------------
def train_or_load_hr_model(bat_df: pd.DataFrame, train_seasons: list[int]):
    key = "_".join(map(str, train_seasons))
    model_path = f"models_hr/hr_xgb_{key}.json"
    calib_path = f"models_hr/hr_cal_{key}.pkl"
    meta_path = f"models_hr/hr_meta_{key}.json"

    if os.path.exists(model_path) and os.path.exists(calib_path) and os.path.exists(meta_path):
        model = XGBRegressor()
        model.load_model(model_path)
        calib = pd.read_pickle(calib_path)
        meta = json.load(open(meta_path, "r"))
        return model, calib, meta

    df = bat_df.copy()
    if df.empty:
        raise RuntimeError("Training data is empty. Statcast pull returned no rows.")

    # Ensure numeric dtypes for XGBoost
    for c in FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    league_hr_pa = float(df["HR"].sum() / df["PA"].sum()) if df["PA"].sum() > 0 else 0.032

    df["hr_rate_shrunk"] = df.apply(
        lambda r: shrink_rate(r["HR"], r["PA"], prior_mean=league_hr_pa, prior_strength=250),
        axis=1
    )

    X = df[FEATURE_COLS].fillna(0.0)
    y = df["hr_rate_shrunk"].astype(float)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=900,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    raw_val = np.clip(model.predict(X_val), 1e-6, 0.5)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_val)

    model.save_model(model_path)
    pd.to_pickle(iso, calib_path)
    json.dump({"league_hr_pa": league_hr_pa, "train_seasons": train_seasons}, open(meta_path, "w"))

    return model, iso, {"league_hr_pa": league_hr_pa, "train_seasons": train_seasons}


# -------------------------
# Slate + IDs
# -------------------------
def get_games(date_str: str):
    sched = statsapi.schedule(date=date_str)
    games = []
    for g in sched:
        games.append({
            "game_pk": g.get("game_id") or g.get("game_pk") or g.get("gamePk"),
            "home_team": g.get("home_name"),
            "away_team": g.get("away_name"),
            "venue_name": g.get("venue_name"),
            "home_probable_pitcher": g.get("home_probable_pitcher"),
            "away_probable_pitcher": g.get("away_probable_pitcher"),
        })
    return games


# -------------------------
# Confirmed Lineups (for exp_pa only)
# -------------------------
# We only use lineup slots when the lineup is actually posted in the MLB boxscore feed.
# If lineups are not posted yet, we fall back to the stable exp_pa proxy logic.

LINEUP_CACHE: dict[int, dict] = {}

def _batting_order_to_slot(batting_order) -> int | None:
    """Convert MLB battingOrder field (often 100,200,...900) to lineup slot 1-9."""
    if batting_order is None:
        return None
    try:
        v = int(str(batting_order).strip())
    except Exception:
        return None

    # Common format is 100..900
    if v >= 100:
        slot = v // 100
    else:
        slot = v

    if 1 <= slot <= 9:
        return slot
    return None


def get_confirmed_lineup_slots(game_pk):
    """Return per-side lineup slot maps for a game_pk.

    Output:
      {
        "home_slots": {player_id: slot, ...},
        "away_slots": {player_id: slot, ...},
        "home_confirmed": bool,
        "away_confirmed": bool
      }
    """
    if not game_pk:
        return {"home_slots": {}, "away_slots": {}, "home_confirmed": False, "away_confirmed": False}

    try:
        gpk = int(game_pk)
    except Exception:
        return {"home_slots": {}, "away_slots": {}, "home_confirmed": False, "away_confirmed": False}

    if gpk in LINEUP_CACHE:
        return LINEUP_CACHE[gpk]

    out = {"home_slots": {}, "away_slots": {}, "home_confirmed": False, "away_confirmed": False}

    try:
        box = statsapi.boxscore_data(gpk)
    except Exception:
        LINEUP_CACHE[gpk] = out
        return out

    def parse_side(batters):
        slots = {}
        if not batters:
            return slots
        for b in batters:
            if not isinstance(b, dict):
                continue
            pid = b.get("personId") or b.get("person_id") or b.get("id")
            try:
                pid = int(pid)
            except Exception:
                continue

            slot = _batting_order_to_slot(b.get("battingOrder"))
            if slot is None:
                continue

            # If duplicates appear, keep the first (starter) record
            if pid not in slots:
                slots[pid] = slot
        return slots

    home_slots = parse_side(box.get("homeBatters", []))
    away_slots = parse_side(box.get("awayBatters", []))

    # Consider confirmed if we have a near-full batting order (>=7 distinct slots)
    out["home_slots"] = home_slots
    out["away_slots"] = away_slots
    out["home_confirmed"] = len(set(home_slots.values())) >= 7
    out["away_confirmed"] = len(set(away_slots.values())) >= 7

    LINEUP_CACHE[gpk] = out
    return out


def lookup_player_id_by_name(name: str):
    if not name or name == "TBD":
        return None
    res = statsapi.lookup_player(name)
    if not res:
        return None
    try:
        return int(res[0]["id"])
    except Exception:
        return None


def get_team_hitters(team_name: str):
    teams = statsapi.lookup_team(team_name)
    if not teams:
        return []
    team_id = teams[0]["id"]

    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
    try:
        j = get_json(url)
    except Exception:
        return []

    roster = j.get("roster", []) or []
    hitters = []

    for item in roster:
        if not isinstance(item, dict):
            continue

        person = item.get("person", {}) if isinstance(item.get("person", {}), dict) else {}
        position = item.get("position", {}) if isinstance(item.get("position", {}), dict) else {}

        pos_abbr = (position.get("abbreviation") or "").upper()
        if pos_abbr == "P":
            continue

        pid = person.get("id")
        if pid is not None:
            try:
                hitters.append(int(pid))
            except Exception:
                continue

    return hitters


# -------------------------
# Weather (optional)
# -------------------------
def load_stadium_coords():
    path = "inputs/stadium_coords.json"
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path, "r"))
    except Exception:
        return {}


def get_daily_temp_f(lat: float, lon: float, date_str: str) -> float | None:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max"
        "&temperature_unit=fahrenheit&timezone=auto"
        f"&start_date={date_str}&end_date={date_str}"
    )
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        j = r.json()
        temps = j.get("daily", {}).get("temperature_2m_max", [])
        if temps:
            return float(temps[0])
    except Exception:
        return None
    return None


def temp_multiplier(temp_f: float | None) -> float:
    if temp_f is None:
        return 1.0
    return float(np.clip(1.0 + 0.04 * ((temp_f - 70.0) / 10.0), 0.85, 1.20))


# -------------------------
# Build Board
# -------------------------
def build_board(date_str: str, n_sims: int, train_seasons: list[int], use_weather: bool):
    bat_df, pit_df, mix_df, dmg_df, park_df, bullpen_df = get_training_cached(train_seasons, date_str)
    model, calib, meta = train_or_load_hr_model(bat_df, train_seasons)

    if bat_df.empty:
        return pd.DataFrame()

    bat_latest = bat_df.sort_values("season").groupby("batter").tail(1).set_index("batter")

    barrel_map = dict(zip(park_df.get("home_team", []), park_df.get("park_barrel_mult", pd.Series(dtype=float))))
    fb_map = dict(zip(park_df.get("home_team", []), park_df.get("park_fb_mult", pd.Series(dtype=float))))
    mix_map = mix_df.set_index("pitcher").to_dict(orient="index") if not mix_df.empty else {}
    dmg_map = dmg_df.set_index("batter").to_dict(orient="index") if not dmg_df.empty else {}

    bullpen_map = dict(zip(bullpen_df.get("team", []), bullpen_df.get("bullpen_factor", []))) if not bullpen_df.empty else {}

    coords = load_stadium_coords()
    games = get_games(date_str)

    rows = []

    for g in games:
        home = g["home_team"]
        away = g["away_team"]
        venue = g["venue_name"]

        game_pk = g.get("game_pk")
        lineup_info = get_confirmed_lineup_slots(game_pk)
        home_lineup_slots = lineup_info.get("home_slots", {}) or {}
        away_lineup_slots = lineup_info.get("away_slots", {}) or {}
        home_lineup_confirmed = bool(lineup_info.get("home_confirmed", False))
        away_lineup_confirmed = bool(lineup_info.get("away_confirmed", False))


        home_sp_name = g.get("home_probable_pitcher") or "TBD"
        away_sp_name = g.get("away_probable_pitcher") or "TBD"

        home_sp_id = lookup_player_id_by_name(home_sp_name)
        away_sp_id = lookup_player_id_by_name(away_sp_name)

        barrel_mult = float(barrel_map.get(home, 1.0))
        fb_mult = float(fb_map.get(home, 1.0))
        park_contact_mult = (barrel_mult ** 0.6) * (fb_mult ** 0.4)
        park_contact_mult = float(np.clip(park_contact_mult, 0.85, 1.15))

        w_mult = 1.0
        if use_weather:
            loc = coords.get(home) or coords.get(venue)
            if loc and "lat" in loc and "lon" in loc:
                t = get_daily_temp_f(float(loc["lat"]), float(loc["lon"]), date_str)
                w_mult = temp_multiplier(t)

        for side in ["away", "home"]:
            batting_team = away if side == "away" else home
            pitcher_name = home_sp_name if side == "away" else away_sp_name
            pitcher_id = home_sp_id if side == "away" else away_sp_id
            is_home = (side == "home")

            # Confirmed lineup slots (used ONLY for exp_pa)
            if side == "away":
                lineup_slots = away_lineup_slots
                lineup_confirmed = away_lineup_confirmed
            else:
                lineup_slots = home_lineup_slots
                lineup_confirmed = home_lineup_confirmed


            pitching_team = home if side == "away" else away
            bullpen_factor = float(bullpen_map.get(pitching_team, 1.0))
            bp_mult = bullpen_adjustment_multiplier(bullpen_factor, w_bp=DEFAULT_W_BP)

            hitter_ids = get_team_hitters(batting_team)

            for hid in hitter_ids:
                if hid not in bat_latest.index:
                    continue

                player_name = get_player_name(int(hid))

                feat = bat_latest.loc[hid, FEATURE_COLS].to_frame().T.fillna(0.0)
                # Ensure numeric
                for c in FEATURE_COLS:
                    if c in feat.columns:
                        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)

                p_raw = float(model.predict(feat)[0])
                p_pa = float(calib.predict([np.clip(p_raw, 1e-6, 0.5)])[0])
                p_pa = float(np.clip(p_pa, 1e-6, 0.25))

                pt_mult = 1.0
                if pitcher_id and pitcher_id in mix_map and hid in dmg_map:
                    mix = mix_map[pitcher_id]
                    dmg = dmg_map[hid]
                    weighted = 0.0
                    wsum = 0.0
                    for pt in PITCH_TYPES:
                        u = float(mix.get(f"pit_usage_{pt}", 0.0))
                        d = dmg.get(f"bat_hr_pa_{pt}", np.nan)
                        if not np.isnan(d):
                            weighted += u * float(d)
                            wsum += u
                    if wsum > 0:
                        baseline = float(meta.get("league_hr_pa", 0.032))
                        pt_mult = float(np.clip((weighted / max(baseline, 1e-6)), 0.85, 1.20))

                env_mult = float(np.clip(park_contact_mult * w_mult, 0.80, 1.30))

                p_pa_adj = float(np.clip(p_pa * pt_mult * env_mult * bp_mult, 1e-6, 0.30))

                # ---------------------------
                # Improved exp_pa logic
                # ---------------------------
                # Default neutral if lineup unknown (future-proof: later you can wire confirmed lineups).
                lineup_slot = lineup_slots.get(int(hid)) if lineup_confirmed else None

                if lineup_slot in LINEUP_BASELINES:
                    exp_pa = float(LINEUP_BASELINES[lineup_slot])
                else:
                    pa_last = float(bat_latest.loc[hid, 'PA'])
                    games_proxy = max(pa_last / 4.2, 1.0)
                    exp_pa = float(pa_last / games_proxy)
                
                # Home team slight reduction (no guaranteed 9th-inning PA).
                if is_home:
                    exp_pa -= 0.08
                
                exp_pa = float(np.clip(exp_pa, 3.3, 5.3))

                p1, p2 = sim_hr_probs(p_pa_adj, exp_pa, n_sims=n_sims)

                rows.append({
                    "date": date_str,
                    "team": batting_team,
                    "venue": venue,
                    "probable_pitcher_faced": pitcher_name,
                    "player_name": player_name,
                    "batter_id": int(hid),
                    "exp_pa": round(exp_pa, 2),
                    "p_hr_pa": p_pa_adj,
                    "p_hr_1plus_sim": p1,
                    "p_hr_2plus_sim": p2,
                    "park_factor": round(park_contact_mult, 3),
                    "weather_mult": round(w_mult, 3),
                    "pitchtype_mult": round(pt_mult, 3),
                })

    board = pd.DataFrame(rows)
    if board.empty:
        return board

    board = board.sort_values(["p_hr_1plus_sim", "p_hr_pa"], ascending=False).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    return board


def write_outputs(board: pd.DataFrame, date_str: str, top_n: int):
    csv_path = f"outputs/hr_board_{date_str}.csv"
    html_path = f"outputs/hr_board_{date_str}.html"

    board.to_csv(csv_path, index=False)

    show = board.head(top_n).copy()
    for c in ["p_hr_1plus_sim", "p_hr_2plus_sim", "p_hr_pa"]:
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
      <h2>HR Probability Board — {date_str} (Top {top_n})</h2>
      <p>p_hr_1plus_sim / p_hr_2plus_sim / p_hr_pa are percentages.</p>
      {show.to_html(index=False, escape=True)}
    </body>
    </html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=str(date_cls.today()))
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--top", type=int, default=DEFAULT_TOPN)
    parser.add_argument("--seasons", type=str, default=",".join(map(str, TRAIN_SEASONS_DEFAULT)))
    parser.add_argument("--weather", action="store_true", help="Enable weather temp multiplier (requires inputs/stadium_coords.json)")
    args = parser.parse_args()

    ensure_dirs()

    train_seasons = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
    board = build_board(args.date, n_sims=args.sims, train_seasons=train_seasons, use_weather=args.weather)

    if board.empty:
        pd.DataFrame([{"message": f"No board produced for {args.date}"}]).to_csv(f"outputs/NO_DATA_{args.date}.csv", index=False)
        print("No board produced.")
        return

    # Attach CLV + market columns (safe if odds files missing)
    board = market_clv.attach_clv(board)

    write_outputs(board, args.date, top_n=args.top)
    print("\nTop 25 (by P(HR>=1) sim):")
    print(board.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
