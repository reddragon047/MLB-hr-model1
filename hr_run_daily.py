import os
import json
import argparse
import re
import unicodedata
import collections
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
from inputs import platoon

# -------------------------
# CONFIG
# -------------------------
TRAIN_SEASONS_DEFAULT = [2023, 2024, 2025]
DEFAULT_SIMS = 100000
DEFAULT_TOPN = 75

PA_DISPERSION_K_DEFAULT = 8.0  # Negative Binomial dispersion for PA sims (higher = less variance)

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


def sim_hr_probs(
    p_pa: float,
    exp_pa: float,
    n_sims: int,
    seed: int = 42,
    pa_k: float | None = None,
):
    """
    Monte Carlo for P(HR>=1) / P(HR>=2) given:

      p_pa   = per-PA HR probability
      exp_pa = expected plate appearances

    PA variance model:
      - Default: Negative Binomial via Gamma-Poisson mixture (over-dispersed vs Poisson).
      - Fallback: Poisson if pa_k is None or <= 0.

    pa_k is a dispersion parameter:
      var(PA) = mu + mu^2 / k
      Larger k -> closer to Poisson; smaller k -> fatter tails.

    This improves realism (lineup spot / game context causes extra variance) while
    keeping the mean fixed at exp_pa.
    """
    rng = np.random.default_rng(seed)

    mu = float(max(exp_pa, 0.05))
    p = float(np.clip(p_pa, 0.0, 1.0))

    # Use a global default if present, otherwise a sane default.
    k_default = globals().get("PA_DISPERSION_K_DEFAULT", 8.0)
    k = k_default if pa_k is None else float(pa_k)

    if k is not None and k > 0:
        # Gamma-Poisson mixture => Negative Binomial with mean mu and dispersion k
        lam = rng.gamma(shape=k, scale=(mu / k), size=n_sims)
        pa = rng.poisson(lam=lam)
    else:
        pa = rng.poisson(lam=mu, size=n_sims)

    hr = rng.binomial(n=pa, p=p)
    p1 = float((hr >= 1).mean())
    p2 = float((hr >= 2).mean())
    return p1, p2



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
def pull_statcast_season(season: int) -> pd.DataFrame:
    start = f"{season}-03-01"
    end = f"{season}-11-15"
    return statcast(start_dt=start, end_dt=end)


def batter_pitcher_tables(stat_df: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = stat_df.copy()
    df["season"] = season

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
    """Compute park HR conversion multipliers keyed by home_team.

    We build two conversion rates and compare them to league baselines:
      - HR per Barrel (proxy barrels via EV/LA)  -> park_barrel_mult
      - HR per Fly Ball (proxy via launch angle) -> park_fb_mult

    Both rates are shrunk toward league mean for stability.
    """
    df = stat_df.copy()

    # Only batted balls with launch metrics
    df = df[df["events"].notna()].copy()
    df = df[df["launch_speed"].notna() & df["launch_angle"].notna()].copy()

    df["is_hr"] = (df["events"] == "home_run").astype(int)

    la = df["launch_angle"].astype(float)
    ev = df["launch_speed"].astype(float)

    # Fly ball proxy (tunable)
    df["is_fb_proxy"] = ((la >= 25.0) & (la <= 50.0)).astype(int)

    # Barrel proxy (consistent with batter_pitcher_tables)
    if "is_barrel_proxy" not in df.columns:
        df["is_barrel_proxy"] = ((ev >= 98.0) & (la.between(26.0, 30.0))).astype(int)

    # League baselines
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

    # Raw rates
    park["hr_per_barrel_raw"] = park["hr_on_barrels"] / park["barrels"].clip(lower=1.0)
    park["hr_per_fb_raw"] = park["hr_on_flyballs"] / park["flyballs"].clip(lower=1.0)

    # Shrink toward league
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

    # Multipliers vs league baseline
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
    """
    Build bullpen HR/PA factors by team from Statcast training data.

    Method:
    - Infer pitching team using inning_topbot + home/away team fields.
    - For each game + pitching team: starter = first pitcher to appear.
    - Bullpen PA = all other pitchers' PA for that team in that game.
    - Compute bullpen HR/PA by team, shrink toward league avg, convert to factor vs league.
    """
    required = {"game_pk", "inning_topbot", "home_team", "away_team", "pitcher", "events"}
    if not required.issubset(stat_all.columns):
        return pd.DataFrame({"team": [], "bullpen_factor": []})

    pa = stat_all[stat_all["events"].notna()].copy()
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)

    pa["pitching_team"] = np.where(
        pa["inning_topbot"].astype(str).str.lower().str.startswith("top"),
        pa["home_team"],   # top: away bats -> home pitches
        pa["away_team"],   # bot: home bats -> away pitches
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


def get_training_cached(train_seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = "_".join(map(str, train_seasons))
    bat_path = f"data/processed/bat_{key}.parquet"
    pit_path = f"data/processed/pit_{key}.parquet"
    mix_path = f"data/processed/mix_{key}.parquet"
    dmg_path = f"data/processed/dmg_{key}.parquet"
    park_path = f"data/processed/park_{key}.parquet"
    bp_path = f"data/processed/bullpen_{key}.parquet"
    platoon_path = f"data/processed/platoon_{key}.parquet"
    if all(os.path.exists(p) for p in [bat_path, pit_path, mix_path, dmg_path, park_path, bp_path, platoon_path]):
        return (
            pd.read_parquet(bat_path),
            pd.read_parquet(pit_path),
            pd.read_parquet(mix_path),
            pd.read_parquet(dmg_path),
            pd.read_parquet(park_path),
            pd.read_parquet(bp_path),
            pd.read_parquet(platoon_path),
        )

    all_stat = []
    bat_tables = []
    pit_tables = []

    for s in train_seasons:
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

    bat_all.to_parquet(bat_path, index=False)
    pit_all.to_parquet(pit_path, index=False)
    mix_p.to_parquet(mix_path, index=False)
    dmg_b.to_parquet(dmg_path, index=False)
    park.to_parquet(park_path, index=False)
    bullpen.to_parquet(bp_path, index=False)
# ---- platoon cache ----
# Build batter/pitcher handedness & platoon splits once per training cache
    platoon_df = platoon.compute_batter_platoon_splits(stat_all)  # <-- from your platoon module
    platoon_df.to_parquet(platoon_path, index=False)
    return bat_all, pit_all, mix_p, dmg_b, park, bullpen, platoon_df


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


_TEAM_ID_MEMO: dict[str, int] = {}

def get_team_id(team_name: str) -> int | None:
    if not team_name:
        return None
    if team_name in _TEAM_ID_MEMO:
        return _TEAM_ID_MEMO[team_name]
    try:
        teams = statsapi.lookup_team(team_name)
        if teams:
            tid = int(teams[0]["id"])
            _TEAM_ID_MEMO[team_name] = tid
            return tid
    except Exception:
        pass
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
# -------------------------
# Lineups + exp_pa (sharper)
# -------------------------
_LINEUP_SLOT_MEMO: dict[tuple, int | None] = {}
_PLAYER_SEASON_MEMO: dict[tuple, dict] = {}

def _year_from_date(date_str: str) -> int:
    try:
        return int(datetime.strptime(date_str, "%Y-%m-%d").year)
    except Exception:
        return int(date_cls.today().year)

def fetch_confirmed_lineup_slot(game_pk: int | None, batter_id: int, batting_side: str) -> int | None:
    """
    Try to read a *confirmed* batting order slot (1-9) from the MLB StatsAPI feed.
    Returns None if lineups are not posted yet.
    """
    if not game_pk:
        return None
    batting_side = (batting_side or "").lower()
    if batting_side not in {"home", "away"}:
        return None

    cache_key = ("confirmed", int(game_pk), int(batter_id), batting_side)
    if cache_key in _LINEUP_SLOT_MEMO:
        return _LINEUP_SLOT_MEMO[cache_key]

    url = f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live"
    slot = None
    try:
        j = get_json(url)
        bo = (
            j.get("liveData", {})
             .get("boxscore", {})
             .get("teams", {})
             .get(batting_side, {})
             .get("battingOrder", [])
        )
        if isinstance(bo, list) and bo:
            # battingOrder is a list of batterIds in order (1..9, then repeats for subs)
            # We'll take the *first* occurrence of batter_id.
            for i, bid in enumerate(bo):
                try:
                    if int(bid) == int(batter_id):
                        slot = i + 1
                        break
                except Exception:
                    continue
            if slot is not None and slot > 9:
                # Some feeds include bench/PH/PR; clamp to 9 for slot purposes.
                slot = ((slot - 1) % 9) + 1
    except Exception:
        slot = None

    _LINEUP_SLOT_MEMO[cache_key] = slot
    return slot


def fetch_recent_lineup_slot(team_id: int | None, batter_id: int, date_str: str,
                             lookback_days: int = 14, max_games: int = 12) -> int | None:
    """
    If lineups aren't posted yet, estimate lineup slot by looking at the most recent games
    for this team and taking the modal batting order position for this batter.

    Safe + bounded: looks back `lookback_days` and reads at most `max_games` boxscores.
    """
    if not team_id:
        return None

    season = _year_from_date(date_str)
    memo_key = ("recent", int(team_id), int(batter_id), season)
    if memo_key in _LINEUP_SLOT_MEMO:
        return _LINEUP_SLOT_MEMO[memo_key]

    end_dt = None
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        end_dt = d - timedelta(days=1)
    except Exception:
        end_dt = date_cls.today() - timedelta(days=1)

    start_dt = end_dt - timedelta(days=int(lookback_days))
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&teamId={int(team_id)}"
        f"&startDate={start_dt.isoformat()}&endDate={end_dt.isoformat()}"
    )

    slots = []
    try:
        sched = get_json(url)
        dates = sched.get("dates", []) or []
        game_pks = []
        for dd in dates:
            for g in (dd.get("games", []) or []):
                pk = g.get("gamePk")
                if pk is not None:
                    game_pks.append(int(pk))
        # most recent first
        game_pks = list(reversed(game_pks))[: int(max_games)]

        for pk in game_pks:
            try:
                bx = get_json(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore")
            except Exception:
                continue

            teams = bx.get("teams", {}) or {}
            # Determine which side is the team
            side = None
            try:
                if int(teams.get("home", {}).get("team", {}).get("id")) == int(team_id):
                    side = "home"
                elif int(teams.get("away", {}).get("team", {}).get("id")) == int(team_id):
                    side = "away"
            except Exception:
                side = None

            if not side:
                continue

            bo = teams.get(side, {}).get("battingOrder", [])
            if not isinstance(bo, list) or not bo:
                continue

            for i, bid in enumerate(bo[:9]):  # starters order is first 9
                try:
                    if int(bid) == int(batter_id):
                        slots.append(i + 1)
                        break
                except Exception:
                    continue

    except Exception:
        slots = []

    slot = None
    if slots:
        try:
            slot = collections.Counter(slots).most_common(1)[0][0]
        except Exception:
            slot = None

    _LINEUP_SLOT_MEMO[memo_key] = slot
    return slot


def get_player_pa_per_game(player_id: int, season: int) -> float | None:
    """
    Pull plateAppearances + gamesPlayed for a player season using StatsAPI.
    Returns PA/G, or None if unavailable (e.g., pre-season / no stats).
    """
    key = (int(player_id), int(season))
    if key in _PLAYER_SEASON_MEMO:
        return _PLAYER_SEASON_MEMO[key].get("pa_per_g")

    pa_per_g = None
    try:
        j = statsapi.get("person", {
            "personId": int(player_id),
            "hydrate": f"stats(group=[hitting],type=[season],season={int(season)})"
        })
        stats = (j.get("people", [{}])[0].get("stats", []) or [])
        # Find the season hitting split
        for st in stats:
            if st.get("group", {}).get("displayName") == "hitting":
                splits = st.get("splits", []) or []
                if splits:
                    s0 = splits[0].get("stat", {}) or {}
                    pa = s0.get("plateAppearances")
                    gp = s0.get("gamesPlayed")
                    if pa is not None and gp:
                        pa = float(pa)
                        gp = float(gp)
                        if gp > 0:
                            pa_per_g = pa / gp
                            break
    except Exception:
        pa_per_g = None

    _PLAYER_SEASON_MEMO[key] = {"pa_per_g": pa_per_g}
    return pa_per_g


def estimate_exp_pa(
    batter_id: int,
    team_id: int | None,
    game_pk: int | None,
    is_home: bool,
    date_str: str,
    pa_last_fallback: float | None = None,
) -> float:
    """
    Best-effort exp_pa:
      1) If confirmed lineup exists -> slot baseline
      2) Else recent slot mode -> slot baseline
      3) Blend with player PA/G for that season (if available)
      4) Else fallback to Statcast PA proxy (pa_last_fallback)

    Returns a clipped expected PA in [3.3, 5.3].
    """
    season = _year_from_date(date_str)

    side = "home" if is_home else "away"
    slot = fetch_confirmed_lineup_slot(game_pk, batter_id, side)
    if slot is None:
        slot = fetch_recent_lineup_slot(team_id, batter_id, date_str)

    # Neutral baselines by lineup slot (starters) — realistic MLB means
    LINEUP_BASELINES = {
        1: 4.75,
        2: 4.65,
        3: 4.55,
        4: 4.45,
        5: 4.35,
        6: 4.25,
        7: 4.15,
        8: 4.05,
        9: 3.95,
    }

    # Player PA/G (season) if available
    pa_per_g = get_player_pa_per_game(batter_id, season)

    if slot in LINEUP_BASELINES:
        base = float(LINEUP_BASELINES[int(slot)])
        if pa_per_g is not None:
            # Blend: player's typical PA/G (captures substitutions + durability) toward slot expectation
            exp_pa = 0.65 * base + 0.35 * float(pa_per_g)
        else:
            exp_pa = base
    else:
        # No slot information: use player's PA/G if we have it, else fallback
        if pa_per_g is not None:
            exp_pa = float(pa_per_g)
        else:
            # Last resort: Statcast PA proxy or a neutral 4.25
            if pa_last_fallback is not None:
                pa_last = float(pa_last_fallback)
                games_proxy = max(pa_last / 4.2, 1.0)
                exp_pa = float(pa_last / games_proxy)
            else:
                exp_pa = 4.25

    # Home team slight reduction (no guaranteed bottom 9th)
    if is_home:
        exp_pa -= 0.08

    return float(np.clip(exp_pa, 3.3, 5.3))

# Build Board
# -------------------------
def build_board(date_str: str, n_sims: int, train_seasons: list[int], use_weather: bool):
    bat_df, pit_df, mix_df, dmg_df, park_df, bullpen_df, platoon_df = get_training_cached(train_seasons)
    platoon_map = platoon_df.set_index("batter").to_dict(orient="index") if platoon_df is not None else {}
    model, calib, meta = train_or_load_hr_model(bat_df, train_seasons)

    bat_latest = bat_df.sort_values("season").groupby("batter").tail(1).set_index("batter")

    barrel_map = dict(zip(park_df["home_team"], park_df.get("park_barrel_mult", pd.Series(dtype=float))))
    fb_map = dict(zip(park_df["home_team"], park_df.get("park_fb_mult", pd.Series(dtype=float))))
    mix_map = mix_df.set_index("pitcher").to_dict(orient="index")
    dmg_map = dmg_df.set_index("batter").to_dict(orient="index")

    bullpen_map = dict(zip(bullpen_df.get("team", []), bullpen_df.get("bullpen_factor", [])))

    coords = load_stadium_coords()
    games = get_games(date_str)
    print(f"[DEBUG] games={len(games)} for {date_str}")
    rows = []

    for g in games:
    print(f"[DEBUG] game: {g.get('away_team')} @ {g.get('home_team')} venue={g.get('venue_name')}")  
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

            # Opposing bullpen factor (pitching team = home if away bats, else away)
            pitching_team = home if side == "away" else away
            bullpen_factor = float(bullpen_map.get(pitching_team, 1.0))
            bp_mult = bullpen_adjustment_multiplier(bullpen_factor, w_bp=DEFAULT_W_BP)

            hitter_ids = get_team_hitters(batting_team)
            print(f"[DEBUG] {date_str} {batting_team} hitters={len(hitter_ids)} sample={hitter_ids[:5]}")
            for hid in hitter_ids:
                if hid not in bat_latest.index:
                    continue

                player_name = get_player_name(int(hid))

                feat = bat_latest.loc[hid, FEATURE_COLS].to_frame().T.fillna(0.0)

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

    # --- Pitcher handedness (R/L) ---
    pitch_hand = ""
    if pitcher_id:
        _, pitch_hand = platoon.get_handedness(int(pitcher_id))
    pitch_hand = (pitch_hand or "").upper()[:1]

    # --- Platoon adjustment ---
    platoon_mult = 1.0
    if pitch_hand in ("R", "L"):
        pinfo = platoon_map.get(int(hid))
        if pinfo:
            overall = float(pinfo.get("hr_pa_overall_shrunk", np.nan))
            if pitch_hand == "R":
                split = float(pinfo.get("hr_pa_vs_R_shrunk", np.nan))
            else:
                split = float(pinfo.get("hr_pa_vs_L_shrunk", np.nan))

            if np.isfinite(overall) and overall > 0 and np.isfinite(split) and split > 0:
                platoon_mult = split / overall
                platoon_mult = float(np.clip(platoon_mult, 0.80, 1.25))

    # ✅ Apply bullpen multiplier here (sniper-safe)
    p_pa_adj = float(np.clip(p_pa * pt_mult * env_mult * bp_mult * platoon_mult, 1e-6, 0.30))
                # exp_pa (expected plate appearances)
                # Uses confirmed lineup if available; otherwise estimates slot from recent games + season PA/G.
    pa_last = float(bat_latest.loc[hid, "PA"])
    team_id = get_team_id(batting_team)
    game_pk = g.get("game_pk")
    exp_pa = estimate_exp_pa(int(hid), team_id, game_pk, is_home, date_str, pa_last_fallback=pa_last)

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
    if len(rows) <= 5:
    print(f"[DEBUG] appended row #{len(rows)} player={player_name}")
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


# -------------------------
# Performance Log (daily append + auto-settle yesterday)
# -------------------------

PERF_LOG_PATH = "outputs/performance_log.csv"

def _safe_int(x):
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def _implied_prob_from_american(odds):
    """American odds (+320/-150) -> implied probability."""
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return np.nan
    s = str(odds).strip().replace(" ", "")
    if s == "":
        return np.nan
    try:
        v = float(s.replace("+", ""))
    except Exception:
        return np.nan
    if v == 0:
        return np.nan
    if v > 0:
        return 100.0 / (v + 100.0)
    return (-v) / ((-v) + 100.0)


def _ensure_perf_log_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee required columns exist."""
    required = [
        "date",
        "player_name",
        "batter_id",
        "team",
        "bet_type",
        "bet_price",
        "close_price",
        "model_prob",
        "implied_open_prob",
        "implied_close_prob",
        "edge_open_pct",
        "clv_prob",
        "clv_pct",
        "result",
        "units",
        "cumulative_units",
        "drawdown_pct",
    ]
    out = df.copy()
    for c in required:
        if c not in out.columns:
            out[c] = np.nan
    return out[required]


def _recompute_cum_and_dd(log_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute cumulative_units and drawdown_pct from 'units'. NaN units -> 0."""
    df = log_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

    df["units"] = pd.to_numeric(df.get("units", 0), errors="coerce").fillna(0.0)

    # stable sort
    sort_cols = ["date", "player_name"]
    for c in sort_cols:
        if c not in df.columns:
            sort_cols = ["date"]
            break
    df = df.sort_values(sort_cols).reset_index(drop=True)

    df["cumulative_units"] = df["units"].cumsum()
    peak = df["cumulative_units"].cummax()
    dd = df["cumulative_units"] - peak
    df["drawdown_pct"] = np.where(peak != 0, dd / peak, 0.0)
    return df


def append_performance_log(board: pd.DataFrame, run_date_str: str, log_path: str = PERF_LOG_PATH):
    """
    Append today's actionable rows (where we have an open odds price) to outputs/performance_log.csv.

    Assumptions:
    - market_clv.attach_clv(board) has already been called.
    - Uses odds_open_1plus if present; else falls back to odds_1plus.
    - model_prob uses p_hr_1plus_sim.
    """
    if board is None or board.empty:
        return

    df = board.copy()

    # Prefer explicit open odds columns (added by market_clv)
    open_col = "odds_open_1plus" if "odds_open_1plus" in df.columns else ("odds_1plus" if "odds_1plus" in df.columns else None)
    close_col = "odds_close_1plus" if "odds_close_1plus" in df.columns else None

    if open_col is None:
        # nothing to log
        return

    # Only log rows where we actually have an open price
    mask_open = df[open_col].notna()
    df = df[mask_open].copy()
    if df.empty:
        return

    # Pull implied columns if present; else compute
    implied_open_col = "implied_prob_open_1plus" if "implied_prob_open_1plus" in df.columns else None
    implied_close_col = "implied_prob_close_1plus" if "implied_prob_close_1plus" in df.columns else None

    df["bet_price"] = df[open_col]
    df["close_price"] = df[close_col] if close_col and close_col in df.columns else np.nan

    df["model_prob"] = pd.to_numeric(df.get("p_hr_1plus_sim", np.nan), errors="coerce")

    if implied_open_col:
        df["implied_open_prob"] = pd.to_numeric(df[implied_open_col], errors="coerce")
    else:
        df["implied_open_prob"] = df["bet_price"].map(_implied_prob_from_american)

    if implied_close_col:
        df["implied_close_prob"] = pd.to_numeric(df[implied_close_col], errors="coerce")
    else:
        df["implied_close_prob"] = df["close_price"].map(_implied_prob_from_american)

    df["edge_open_pct"] = df["model_prob"] - df["implied_open_prob"]

    # CLV: probability-space delta, plus a pct-change version (close/open - 1)
    df["clv_prob"] = df["implied_close_prob"] - df["implied_open_prob"]
    df["clv_pct"] = (df["implied_close_prob"] / df["implied_open_prob"]) - 1.0

    out = pd.DataFrame({
        "date": run_date_str,
        "player_name": df.get("player_name"),
        "batter_id": df.get("batter_id"),
        "team": df.get("team"),
        "bet_type": "HR_1plus",
        "bet_price": df["bet_price"],
        "close_price": df["close_price"],
        "model_prob": df["model_prob"],
        "implied_open_prob": df["implied_open_prob"],
        "implied_close_prob": df["implied_close_prob"],
        "edge_open_pct": df["edge_open_pct"],
        "clv_prob": df["clv_prob"],
        "clv_pct": df["clv_pct"],
        "result": np.nan,
        "units": np.nan,
        "cumulative_units": np.nan,
        "drawdown_pct": np.nan,
    })

    # Load existing log if exists
    if os.path.exists(log_path):
        try:
            log_df = pd.read_csv(log_path)
        except Exception:
            log_df = pd.DataFrame()
    else:
        log_df = pd.DataFrame()

    log_df = _ensure_perf_log_schema(log_df) if not log_df.empty else _ensure_perf_log_schema(pd.DataFrame())

    # Prevent duplicates: same date + batter_id + bet_type
    if not log_df.empty:
        existing_keys = set(
            (str(d), str(b), str(t))
            for d, b, t in zip(log_df["date"].astype(str), log_df["batter_id"].astype(str), log_df["bet_type"].astype(str))
        )
    else:
        existing_keys = set()

    out["__key"] = list(zip(out["date"].astype(str), out["batter_id"].astype(str), out["bet_type"].astype(str)))
    out = out[~out["__key"].map(lambda k: k in existing_keys)].drop(columns=["__key"])
    if out.empty:
        return

    merged = pd.concat([log_df, out], ignore_index=True)
    merged = _recompute_cum_and_dd(merged)

    merged.to_csv(log_path, index=False)


def auto_settle_yesterday_hr_results(run_date_str: str, log_path: str = PERF_LOG_PATH):
    """
    Auto-fill 'result' (0/1) for yesterday's logged HR_1plus bets using Statcast events.
    - Matches on batter_id.
    - Only fills blank results.
    """
    if not os.path.exists(log_path):
        return

    # Settle based on *today*, not the slate date (prevents future-date test runs from trying to settle the future)
    try:
        today = date_cls.today()
        yday = (today - timedelta(days=1)).isoformat()
    except Exception:
        return

    try:
        log_df = pd.read_csv(log_path)
    except Exception:
        return

    if log_df is None or log_df.empty:
        return

    log_df = _ensure_perf_log_schema(log_df)

    # rows to settle: yday + blank result + HR bet_type
    ymask = (log_df["date"].astype(str) == yday)
    rblank = log_df["result"].isna() | (log_df["result"].astype(str).str.strip() == "")
    hrmask = log_df["bet_type"].astype(str).str.upper().str.contains("HR")

    idxs = log_df.index[ymask & rblank & hrmask].tolist()
    if not idxs:
        return

    target_ids = set(_safe_int(log_df.loc[i, "batter_id"]) for i in idxs)
    target_ids = set(x for x in target_ids if x is not None)
    if not target_ids:
        return

    # Pull statcast for yday and get batter ids who homered
    try:
        sc = statcast(start_dt=yday, end_dt=yday)
    except Exception:
        return

    homered = set()
    if sc is not None and not sc.empty and "events" in sc.columns and "batter" in sc.columns:
        hr_df = sc[sc["events"] == "home_run"]
        if not hr_df.empty:
            homered = set(pd.to_numeric(hr_df["batter"], errors="coerce").dropna().astype(int).tolist())

    for i in idxs:
        bid = _safe_int(log_df.loc[i, "batter_id"])
        if bid is None:
            continue
        log_df.loc[i, "result"] = 1 if bid in homered else 0

    log_df = _recompute_cum_and_dd(log_df)
    log_df.to_csv(log_path, index=False)

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

    # Merge market odds + edge if inputs/odds_input.csv (or odds_input.csv) exists
    board["_name_key"] = board["player_name"].map(normalize_player_name)
    board["_fb_key"] = board["_name_key"].map(name_fallback_key)
    board = market_clv.attach_clv(board)

    # Append actionable rows to performance log (safe if odds missing)
    append_performance_log(board, args.date)

    # Auto-settle yesterday's HR results using Statcast (safe if no log / no games)
    auto_settle_yesterday_hr_results(args.date)

    

    write_outputs(board, args.date, top_n=args.top)
    print("\nTop 25 (by P(HR>=1) sim):")
    print(board.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
