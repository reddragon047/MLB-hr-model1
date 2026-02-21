import os
import json
import argparse
import re
import unicodedata
from datetime import date as date_cls

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
# Odds + Edge Helpers
# -------------------------

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

def normalize_player_name(x: str) -> str:
    """Loose, safe normalization for player_name matching."""
    if x is None:
        return ""
    s = str(x).strip().lower()

    # Strip accents (e.g. José -> Jose)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # Normalize apostrophes and remove punctuation (keep comma for "last, first")
    s = s.replace("’", "'")
    s = re.sub(r"[.\-']", "", s)

    # Flip "last, first" -> "first last"
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) >= 2:
            s = f"{parts[1]} {parts[0]}"

    tokens = [t for t in re.split(r"\s+", s) if t]
    tokens = [t for t in tokens if t not in _SUFFIXES]

    s = " ".join(tokens)
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def name_fallback_key(norm: str) -> str:
    """Fallback key: last name + first initial."""
    parts = norm.split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        return f"{last}_{first[0]}"
    return norm

def american_to_implied_prob(odds) -> float:
    """Convert American odds (+320 / -120) to implied probability."""
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return np.nan
    if isinstance(odds, str):
        s = odds.strip().replace(" ", "")
        if s == "":
            return np.nan
        try:
            odds = int(s)
        except Exception:
            return np.nan
    try:
        odds = float(odds)
    except Exception:
        return np.nan

    if odds > 0:
        return 100.0 / (odds + 100.0)
    a = abs(odds)
    return a / (a + 100.0)

def add_market_edge(board: pd.DataFrame) -> pd.DataFrame:
    """Option 1 (CLV infrastructure):
    - Supports odds at open + close, computes implied probs, edges, and CLV columns.
    - Backwards compatible with the old single-file odds_input.csv.

    Expected files (preferred):
      inputs/odds_open.csv   and/or   inputs/odds_close.csv
      (also supported at repo root: odds_open.csv / odds_close.csv)

    Back-compat file:
      inputs/odds_input.csv  (or odds_input.csv at repo root)

    CSV formats:
      player_name,odds_1plus
      Mike Trout,+320
      Byron Buxton,+410
    """

    # --- helpers ---
    def read_odds_file(fp: str):
        if not os.path.exists(fp):
            return None
        df = pd.read_csv(fp)
        if df.empty:
            return None
        # allow a couple common column name variants
        col_name = None
        for c in df.columns:
            if c.strip().lower() in ("player_name", "name", "player"):
                col_name = c
                break
        if col_name is None:
            raise ValueError(f"{fp}: missing 'player_name' column")
        col_odds = None
        for c in df.columns:
            if c.strip().lower() in ("odds_1plus", "odds", "odds1plus", "hr_odds", "odds_hr"):
                col_odds = c
                break
        if col_odds is None:
            raise ValueError(f"{fp}: missing 'odds_1plus' column")
        out = df[[col_name, col_odds]].copy()
        out.columns = ["player_name", "odds_1plus"]
        out["name_key"] = out["player_name"].map(normalize_player_name)
        out = out.dropna(subset=["name_key"]).drop_duplicates(subset=["name_key"], keep="first")
        return out

    # American odds -> implied prob (no vig)
    def american_to_implied_prob(odds) -> float:
        if pd.isna(odds):
            return float("nan")
        s = str(odds).strip()
        if not s:
            return float("nan")
        # handle strings like +320, -150, 320
        try:
            v = float(s.replace("+", ""))
        except Exception:
            return float("nan")
        if v == 0:
            return float("nan")
        if v > 0:
            return 100.0 / (v + 100.0)
        return (-v) / ((-v) + 100.0)

    board = board.copy()
    if "player_name" not in board.columns:
        return board

    board["name_key"] = board["player_name"].map(normalize_player_name)

    # --- load odds files (open/close preferred) ---
    open_df = read_odds_file("inputs/odds_open.csv") or read_odds_file("odds_open.csv")
    close_df = read_odds_file("inputs/odds_close.csv") or read_odds_file("odds_close.csv")

    # fallback: single odds file (treated as "open")
    if open_df is None and close_df is None:
        open_df = read_odds_file("inputs/odds_input.csv") or read_odds_file("odds_input.csv")

    if open_df is None and close_df is None:
        # nothing to merge
        return board.drop(columns=["name_key"], errors="ignore")

    merged = board.copy()

    if open_df is not None:
        merged = merged.merge(
            open_df[["name_key", "odds_1plus"]].rename(columns={"odds_1plus": "odds_open_1plus"}),
            on="name_key",
            how="left",
        )
    else:
        merged["odds_open_1plus"] = float("nan")

    if close_df is not None:
        merged = merged.merge(
            close_df[["name_key", "odds_1plus"]].rename(columns={"odds_1plus": "odds_close_1plus"}),
            on="name_key",
            how="left",
        )
    else:
        merged["odds_close_1plus"] = float("nan")

    # if we only had the old file, keep a friendly alias too
    if "odds_open_1plus" in merged.columns and "odds_1plus" not in merged.columns:
        merged["odds_1plus"] = merged["odds_open_1plus"]

    merged["implied_prob_open_1plus"] = merged["odds_open_1plus"].map(american_to_implied_prob)
    merged["implied_prob_close_1plus"] = merged["odds_close_1plus"].map(american_to_implied_prob)

    # model prob column (this is what we compare against the market)
    model_col = "p_hr_1plus_sim" if "p_hr_1plus_sim" in merged.columns else ("p_hr_1plus" if "p_hr_1plus" in merged.columns else None)

    if model_col is not None:
        merged["edge_open_1plus"] = merged[model_col] - merged["implied_prob_open_1plus"]
        merged["edge_close_1plus"] = merged[model_col] - merged["implied_prob_close_1plus"]
    else:
        merged["edge_open_1plus"] = float("nan")
        merged["edge_close_1plus"] = float("nan")

    # CLV (probability space): positive = the market moved toward your side (close implied prob > open implied prob)
    merged["clv_prob_1plus"] = merged["implied_prob_close_1plus"] - merged["implied_prob_open_1plus"]
    merged["clv_pct_1plus"] = (merged["implied_prob_close_1plus"] / merged["implied_prob_open_1plus"]) - 1.0

    # Keep your existing one-file columns for compatibility with your current board layout
    # (these show up when only open is present)
    merged["odds_1plus"] = merged.get("odds_1plus", merged.get("odds_open_1plus"))
    merged["implied_prob_1plus"] = merged.get("implied_prob_1plus", merged.get("implied_prob_open_1plus"))
    merged["edge_1plus"] = merged.get("edge_1plus", merged.get("edge_open_1plus"))

    # Sort preference: by edge_open if available, else edge_1plus, else leave as-is
    sort_col = "edge_open_1plus" if merged["edge_open_1plus"].notna().any() else ("edge_1plus" if "edge_1plus" in merged.columns else None)
    if sort_col is not None:
        merged = merged.sort_values(by=sort_col, ascending=False, na_position="last")

    return merged.drop(columns=["name_key"], errors="ignore")


# -------------------------
# Statcast Pull + Feature Building
# -------------------------
def pull_statcast_season(season: int, end_dt: str | None = None) -> pd.DataFrame:
    """Pull Statcast for a season.

    If end_dt is provided (YYYY-MM-DD), pulls through that date. This is used for
    *current-season live refresh* so the model captures yesterday's games when you
    run it this morning.
    """
    start = f"{season}-03-01"
    end = end_dt if end_dt else f"{season}-11-15"
    return statcast(start_dt=start, end_dt=end)


def batter_pitcher_tables(stat_df: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = stat_df.copy()
    df["season"] = season

    pa = df[df["events"].notna()].copy()

    # Statcast can sometimes come in with mixed dtypes; force numeric here
    # so downstream aggregates (avg_ev/avg_la) stay float and XGBoost never sees objects.
    pa["launch_speed"] = pd.to_numeric(pa.get("launch_speed"), errors="coerce")
    pa["launch_angle"] = pd.to_numeric(pa.get("launch_angle"), errors="coerce")

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

    # Force model feature cols to real numeric dtypes (XGBoost requirement)
    for c in ["barrel_rate", "avg_ev", "avg_la", "BBE"]:
        if c in bat.columns:
            bat[c] = pd.to_numeric(bat[c], errors="coerce").fillna(0.0).astype(float)

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
    pit["k_rate_allowed"] = pit["K"] / pit["PA"].clip(lower=1)
    pit["bb_rate_allowed"] = pit["BB"] / pit["PA"].clip(lower=1)
    pit["hr_rate_allowed"] = pit["HR_allowed"] / pit["PA"].clip(lower=1)

    return bat, pit


def build_pitch_mix_and_batter_damage(stat_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    df = stat_df.copy()

    df = df[df["events"].notna()].copy()
    df["launch_speed"] = pd.to_numeric(df.get("launch_speed"), errors="coerce")
    df["launch_angle"] = pd.to_numeric(df.get("launch_angle"), errors="coerce")
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

    return park[[
        "home_team",
        "park_barrel_mult",
        "park_fb_mult",
        "barrels",
        "flyballs",
        "hr_per_barrel_shrunk",
        "hr_per_fb_shrunk",
    ]]


def compute_bullpen_factors(stat_all: pd.DataFrame) -> pd.DataFrame:
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


def get_training_cached(
    train_seasons: list[int],
    date_str: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load (or build) training tables.

    IMPORTANT: If the *current season* is included in train_seasons, we refresh the Statcast pull
    every run through `date_str` so you always capture the most up-to-date numbers.
    """
    key = "_".join(map(str, train_seasons))
    bat_path = f"data/processed/bat_{key}.parquet"
    pit_path = f"data/processed/pit_{key}.parquet"
    mix_path = f"data/processed/mix_{key}.parquet"
    dmg_path = f"data/processed/dmg_{key}.parquet"
    park_path = f"data/processed/park_{key}.parquet"
    bp_path = f"data/processed/bullpen_{key}.parquet"

    current_season = date_cls.today().year

    # If current season is in training set, we rebuild each run (no processed cache),
    # because the underlying data changes daily.
    can_use_processed_cache = (current_season not in train_seasons)

    if can_use_processed_cache and all(os.path.exists(p) for p in [bat_path, pit_path, mix_path, dmg_path, park_path, bp_path]):
        return (
            pd.read_parquet(bat_path),
            pd.read_parquet(pit_path),
            pd.read_parquet(mix_path),
            pd.read_parquet(dmg_path),
            pd.read_parquet(park_path),
            pd.read_parquet(bp_path),
        )

    all_stat = []
    bat_tables = []
    pit_tables = []

    for s in train_seasons:
        if s == current_season:
            # LIVE refresh each run
            raw_path = f"data/raw/statcast_{s}_live.parquet"
            season_df = pull_statcast_season(s, end_dt=date_str)
            season_df.to_parquet(raw_path, index=False)
        else:
            raw_path = f"data/raw/statcast_{s}.parquet"
            if os.path.exists(raw_path):
                season_df = pd.read_parquet(raw_path)
            else:
                season_df = pull_statcast_season(s)
                season_df.to_parquet(raw_path, index=False)

        all_stat.append(season_df)
        bat, pit = batter_pitcher_tables(season_df, s)
        bat_tables.append(bat)
        pit_tables.append(pit)

    stat_all = pd.concat(all_stat, ignore_index=True)
    bat_all = pd.concat(bat_tables, ignore_index=True)
    pit_all = pd.concat(pit_tables, ignore_index=True)

    mix_p, dmg_b = build_pitch_mix_and_batter_damage(stat_all)
    park = compute_park_factors(stat_all)
    bullpen = compute_bullpen_factors(stat_all)

    # Only persist processed cache for stable (all-historical) sets
    if can_use_processed_cache:
        bat_all.to_parquet(bat_path, index=False)
        pit_all.to_parquet(pit_path, index=False)
        mix_p.to_parquet(mix_path, index=False)
        dmg_b.to_parquet(dmg_path, index=False)
        park.to_parquet(park_path, index=False)
        bullpen.to_parquet(bp_path, index=False)

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
    league_hr_pa = float(df["HR"].sum() / df["PA"].sum()) if df["PA"].sum() > 0 else 0.032

    df["hr_rate_shrunk"] = df.apply(
        lambda r: shrink_rate(r["HR"], r["PA"], prior_mean=league_hr_pa, prior_strength=250),
        axis=1
    )

    X = df[FEATURE_COLS].apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0.0).astype(float)
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
            "home_team": g.get("home_name"),
            "away_team": g.get("away_name"),
            "venue_name": g.get("venue_name"),
            "home_probable_pitcher": g.get("home_probable_pitcher"),
            "away_probable_pitcher": g.get("away_probable_pitcher"),
        })
    return games


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

    bat_latest = bat_df.sort_values("season").groupby("batter").tail(1).set_index("batter")

    barrel_map = dict(zip(park_df["home_team"], park_df.get("park_barrel_mult", pd.Series(dtype=float))))
    fb_map = dict(zip(park_df["home_team"], park_df.get("park_fb_mult", pd.Series(dtype=float))))
    mix_map = mix_df.set_index("pitcher").to_dict(orient="index")
    dmg_map = dmg_df.set_index("batter").to_dict(orient="index")

    bullpen_map = dict(zip(bullpen_df.get("team", []), bullpen_df.get("bullpen_factor", [])))

    coords = load_stadium_coords()
    games = get_games(date_str)

    rows = []

    for g in games:
        home = g["home_team"]
        away = g["away_team"]
        venue = g["venue_name"]

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

            pitching_team = home if side == "away" else away
            bullpen_factor = float(bullpen_map.get(pitching_team, 1.0))
            bp_mult = bullpen_adjustment_multiplier(bullpen_factor, w_bp=DEFAULT_W_BP)

            hitter_ids = get_team_hitters(batting_team)

            for hid in hitter_ids:
                if hid not in bat_latest.index:
                    continue

                player_name = get_player_name(int(hid))

                feat = bat_latest.loc[hid, FEATURE_COLS].to_frame().T
                feat = feat.apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0.0).astype(float)

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

                pa_last = float(bat_latest.loc[hid, "PA"])
                games_proxy = max(pa_last / 4.2, 1.0)
                exp_pa = float(pa_last / games_proxy)

                exp_pa += (0.05 if not is_home else -0.05)
                exp_pa = float(np.clip(exp_pa, 3.2, 5.2))

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
    current_season = date_cls.today().year
    if current_season not in train_seasons:
        train_seasons.append(current_season)
    train_seasons = sorted(set(train_seasons))

    board = build_board(args.date, n_sims=args.sims, train_seasons=train_seasons, use_weather=args.weather)

    if board.empty:
        pd.DataFrame([{"message": f"No board produced for {args.date}"}]).to_csv(f"outputs/NO_DATA_{args.date}.csv", index=False)
        print("No board produced.")
        return

    # Attach market odds + edge + CLV (if odds files exist)
    board = market_clv.attach_clv(board)

    write_outputs(board, args.date, top_n=args.top)
    print("\nTop 25 (by P(HR>=1) sim):")
    print(board.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
