import os
import json
import argparse
import re
import unicodedata
import collections
from datetime import date as date_cls, datetime, timedelta

import io
import numpy as np
import pandas as pd
import requests

# Patch pandas read_csv globally so pybaseball Statcast downloads
# use the forgiving Python parser and skip malformed rows.
_original_read_csv = pd.read_csv

def safe_global_read_csv(*args, **kwargs):
    kwargs.setdefault("on_bad_lines", "skip")
    kwargs.setdefault("engine", "python")
    return _original_read_csv(*args, **kwargs)

pd.read_csv = safe_global_read_csv

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
    "recent_barrel_rate",
    "recent_hard_hit_rate",
    "recent_ev",
]

BASE_FEATURE_COLS = [
    "barrel_rate",
    "avg_ev",
    "avg_la",
    "k_rate",
    "bb_rate",
    "BBE",
]

RECENT_WINDOW_DAYS = 14
MODEL_VERSION = "v3_power_gate_today_matchup_ranker"
FINAL_CALIBRATOR_PATH = "models_hr/final_calibrator_isotonic.pkl"
PERF_LOG_PATH = "outputs/performance_log.csv"
MIN_FINAL_CALIBRATION_ROWS = 80
MIN_FINAL_CALIBRATION_POSITIVES = 8
FINAL_CALIBRATION_LOOKBACK_DAYS = 540

POWER_SCORE_QUALIFY_THRESHOLD = 50.0
MATCHUP_SCORE_WEIGHTS = {
    "pitcher_vulnerability": 0.30,
    "pitch_mix_fit": 0.25,
    "split_advantage": 0.15,
    "opportunity": 0.15,
    "environment": 0.10,
    "recent_damage": 0.05,
}

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


def safe_read_csv(fp: str, **kwargs) -> pd.DataFrame:
    """Read CSV with forgiving parser settings for user-maintained inputs/logs."""
    return pd.read_csv(fp, on_bad_lines='skip', engine='python', **kwargs)


def shrink_rate(successes, trials, prior_mean, prior_strength):
    a = prior_mean * prior_strength
    b = (1 - prior_mean) * prior_strength
    return (successes + a) / (trials + a + b)


def get_power_aware_prior(row, league_hr_pa: float) -> tuple[float, float]:
    """
    Protect low-sample / translated power bats from being flattened all the way
    to bland league-average HR skill. This only softens baseline shrinkage; it
    does not force any player into the top of the board.
    Returns: (prior_mean, prior_strength)
    """
    pa = float(row.get("PA", 0) or 0)
    bbe = float(row.get("BBE", 0) or 0)
    barrel_rate = float(row.get("barrel_rate", 0) or 0)
    recent_barrel_rate = float(row.get("recent_barrel_rate", 0) or 0)
    avg_ev = float(row.get("avg_ev", 0) or 0)
    recent_ev = float(row.get("recent_ev", 0) or 0)
    recent_hh = float(row.get("recent_hard_hit_rate", 0) or 0)

    prior_mean = float(league_hr_pa)
    prior_strength = 250.0

    # Build a modest power-aware prior from observable quality-of-contact signals.
    if barrel_rate >= 0.10:
        prior_mean = max(prior_mean, league_hr_pa * 1.25)
    elif barrel_rate >= 0.08:
        prior_mean = max(prior_mean, league_hr_pa * 1.15)

    if recent_barrel_rate >= 0.14:
        prior_mean = max(prior_mean, league_hr_pa * 1.45)
    elif recent_barrel_rate >= 0.10:
        prior_mean = max(prior_mean, league_hr_pa * 1.25)

    if avg_ev >= 91.0 or recent_ev >= 92.0:
        prior_mean = max(prior_mean, league_hr_pa * 1.18)
    elif avg_ev >= 89.5 or recent_ev >= 90.5:
        prior_mean = max(prior_mean, league_hr_pa * 1.10)

    if recent_hh >= 0.50:
        prior_mean = max(prior_mean, league_hr_pa * 1.10)

    # Low-sample / translated-hitter protection
    if pa < 80 or bbe < 40:
        prior_strength = 125.0

    # Extra protection for very small samples with real pop signals
    if (pa < 60 or bbe < 25) and (
        recent_barrel_rate >= 0.10 or barrel_rate >= 0.08 or recent_ev >= 91.0 or avg_ev >= 90.0
    ):
        prior_strength = 100.0
        prior_mean = max(prior_mean, league_hr_pa * 1.20)

    prior_mean = float(min(prior_mean, league_hr_pa * 1.60))
    return prior_mean, prior_strength


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




def ensure_recent_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure recent-form columns always exist so training/inference never crash."""
    fallback_map = {
        "recent_barrel_rate": "barrel_rate",
        "recent_hard_hit_rate": "hard_hit_rate",
        "recent_ev": "avg_ev",
    }
    for col, fallback in fallback_map.items():
        if col not in df.columns:
            if fallback in df.columns:
                df[col] = df[fallback]
            else:
                df[col] = 0.0
    return df

def get_recent_statcast_features(batter_ids, end_date=None, days: int = RECENT_WINDOW_DAYS) -> pd.DataFrame:
    """
    Pull a recent Statcast window and aggregate short-term form features for the
    supplied batter ids. Returns an empty frame on failure so the pipeline can
    safely fall back to season-level features.
    """
    cols = ["batter", "recent_barrel_rate", "recent_hard_hit_rate", "recent_ev", "recent_bbe"]
    try:
        batter_ids = [int(x) for x in set(pd.Series(list(batter_ids)).dropna().astype(int).tolist())]
        if not batter_ids:
            return pd.DataFrame(columns=cols)

        end_dt = pd.to_datetime(end_date).date() if end_date is not None else date_cls.today()
        start_dt = end_dt - timedelta(days=max(int(days) - 1, 0))

        df_stat = statcast(start_dt=start_dt.strftime("%Y-%m-%d"), end_dt=end_dt.strftime("%Y-%m-%d"))
        if df_stat is None or df_stat.empty or "batter" not in df_stat.columns:
            return pd.DataFrame(columns=cols)

        df_stat = df_stat[df_stat["batter"].isin(batter_ids)].copy()
        if df_stat.empty:
            return pd.DataFrame(columns=cols)

        bbe = df_stat[df_stat["launch_speed"].notna() & df_stat["launch_angle"].notna()].copy()
        if bbe.empty:
            return pd.DataFrame(columns=cols)

        bbe["is_barrel_proxy"] = ((bbe["launch_speed"] >= 98) & (bbe["launch_angle"].between(26, 30))).astype(int)
        bbe["is_hard_hit"] = (bbe["launch_speed"] >= 95).astype(int)

        agg = bbe.groupby("batter").agg(
            recent_barrel_rate=("is_barrel_proxy", "mean"),
            recent_hard_hit_rate=("is_hard_hit", "mean"),
            recent_ev=("launch_speed", "mean"),
            recent_bbe=("launch_speed", "size"),
        ).reset_index()

        return agg
    except Exception as e:
        print(f"[DEBUG] recent statcast pull failed: {e}")
        return pd.DataFrame(columns=cols)


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
        df = safe_read_csv(fp)
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


def compute_pitch_matchup_multiplier(
    pitcher_id: int | None,
    batter_id: int,
    mix_map: dict,
    dmg_map: dict,
    league_pitch_map: dict[str, float],
    baseline_hr_pa: float,
    batter_bbe: float | int | None,
) -> tuple[float, str, float, float]:
    """
    Stable pitch-mix multiplier for live rankings.
    Returns: multiplier, dominant_pitch_type, dominant_pitch_usage, batter_vs_pitch_ratio

    Important fallback:
    If pitcher mix exists but batter pitch-type damage is missing, do NOT collapse to
    a fully neutral 1.0 / no dominant pitch. That was suppressing legit power bats
    with incomplete MLB pitch-type history (e.g. newer / translated hitters).
    """
    mult = 1.0
    dominant_pt = ""
    dominant_usage = 0.0
    batter_vs_pitch_ratio = 1.0

    if not pitcher_id or pitcher_id not in mix_map:
        return mult, dominant_pt, dominant_usage, batter_vs_pitch_ratio

    mix = mix_map[pitcher_id]

    usage_pairs = []
    for pt in PITCH_TYPES:
        u = float(mix.get(f"pit_usage_{pt}", 0.0) or 0.0)
        if u > 0:
            usage_pairs.append((pt, u))

    if usage_pairs:
        dominant_pt, dominant_usage = max(usage_pairs, key=lambda x: x[1])

    # Fallback path for missing batter pitch-type history.
    # Small positive bias for known playable power profiles facing concentrated pitch mixes.
    if batter_id not in dmg_map:
        base = float(baseline_hr_pa or 0.0)
        bbe = float(batter_bbe or 0.0)

        fallback_ratio = 1.0
        if dominant_usage >= 0.55:
            fallback_ratio = 1.10
        elif dominant_usage >= 0.45:
            fallback_ratio = 1.07
        elif dominant_usage >= 0.35:
            fallback_ratio = 1.04

        # If the hitter already has at least some batted-ball history and a non-trivial
        # baseline HR rate, let the fallback show up a little more.
        if base >= 0.028 and bbe >= 25:
            mult = fallback_ratio
            batter_vs_pitch_ratio = fallback_ratio
        else:
            mult = min(1.03, fallback_ratio)
            batter_vs_pitch_ratio = min(1.03, fallback_ratio)

        return float(np.clip(mult, 0.98, 1.10)), dominant_pt, float(dominant_usage), float(batter_vs_pitch_ratio)

    dmg = dmg_map[batter_id]

    weighted = 0.0
    total_usage = 0.0
    for pt in PITCH_TYPES:
        u = float(mix.get(f"pit_usage_{pt}", 0.0) or 0.0)
        d = dmg.get(f"bat_hr_pa_{pt}", np.nan)
        if u > 0 and not pd.isna(d):
            weighted += u * float(d)
            total_usage += u

    if dominant_pt:
        dpt = dmg.get(f"bat_hr_pa_{dominant_pt}", np.nan)
        league_pt = float(league_pitch_map.get(dominant_pt, baseline_hr_pa) or baseline_hr_pa)
        if not pd.isna(dpt) and league_pt > 0:
            batter_vs_pitch_ratio = float(dpt) / league_pt

    if total_usage <= 0:
        return mult, dominant_pt, dominant_usage, batter_vs_pitch_ratio

    raw_ratio = weighted / max(float(baseline_hr_pa), 1e-6)
    raw_ratio = float(np.clip(raw_ratio, 0.88, 1.18))

    bbe = float(batter_bbe or 0.0)
    confidence = float(np.clip(bbe / 160.0, 0.0, 1.0))
    mult = 1.0 + (raw_ratio - 1.0) * confidence
    mult = float(np.clip(mult, 0.93, 1.10))
    return mult, dominant_pt, float(dominant_usage), float(batter_vs_pitch_ratio)


def compute_recent_power_multiplier(
    barrel_rate: float,
    avg_ev: float,
    recent_barrel_rate: float,
    recent_hard_hit_rate: float,
    recent_ev: float,
    recent_bbe: float | int | None,
) -> tuple[float, float]:
    """
    Recent-form refinement. Small enough to sharpen rankings without hijacking them.
    Returns multiplier and a diagnostic score.
    """
    base_barrel = float(barrel_rate or 0.0)
    base_ev = float(avg_ev or 0.0)
    recent_barrel = float(recent_barrel_rate or 0.0)
    recent_hh = float(recent_hard_hit_rate or 0.0)
    recent_ev = float(recent_ev or 0.0)
    recent_bbe = float(recent_bbe or 0.0)

    raw = (
        1.0
        + 0.55 * (recent_barrel - base_barrel)
        + 0.18 * (recent_hh - 0.40)
        + 0.010 * (recent_ev - base_ev)
    )
    raw = float(np.clip(raw, 0.92, 1.12))
    confidence = float(np.clip(recent_bbe / 15.0, 0.0, 1.0))
    mult = 1.0 + (raw - 1.0) * confidence
    mult = float(np.clip(mult, 0.95, 1.08))
    form_score = float(np.clip((mult - 1.0) * 100.0, -8.0, 8.0))
    return mult, form_score




def compute_pitcher_hr_profile_multiplier(
    pitcher_id: int | None,
    pit_map: dict,
    pit_league: dict[str, float],
) -> tuple[float, float]:
    """
    Starter HR-risk refinement based on pitcher contact damage allowed.
    Uses only stable current features already in the training cache.

    Returns:
      multiplier (small bounded adjustment)
      diagnostic score (negative = suppressive, positive = attackable)
    """
    if not pitcher_id or pitcher_id not in pit_map:
        return 1.0, 0.0

    p = pit_map[pitcher_id]

    hr_rate_allowed = float(p.get("hr_rate_allowed", pit_league.get("hr_rate_allowed", 0.032)) or pit_league.get("hr_rate_allowed", 0.032))
    barrel_rate_allowed = float(p.get("barrel_rate_allowed", pit_league.get("barrel_rate_allowed", 0.07)) or pit_league.get("barrel_rate_allowed", 0.07))
    avg_ev_allowed = float(p.get("avg_ev_allowed", pit_league.get("avg_ev_allowed", 89.0)) or pit_league.get("avg_ev_allowed", 89.0))
    avg_la_allowed = float(p.get("avg_la_allowed", pit_league.get("avg_la_allowed", 12.0)) or pit_league.get("avg_la_allowed", 12.0))
    pa = float(p.get("PA", 0.0) or 0.0)
    bbe = float(p.get("BBE", 0.0) or 0.0)

    league_hr = max(float(pit_league.get("hr_rate_allowed", 0.032) or 0.032), 1e-6)
    league_barrel = max(float(pit_league.get("barrel_rate_allowed", 0.07) or 0.07), 1e-6)
    league_ev = float(pit_league.get("avg_ev_allowed", 89.0) or 89.0)
    league_la = float(pit_league.get("avg_la_allowed", 12.0) or 12.0)

    # Air-contact sweet spot: LA around low-to-mid teens tends to be more HR-friendly than pure worm-burners.
    la_center = 14.0
    la_edge = 8.0
    la_score = 1.0 + 0.06 * np.clip((avg_la_allowed - la_center) / la_edge, -1.0, 1.0)

    raw = (
        1.0
        + 0.45 * ((hr_rate_allowed / league_hr) - 1.0)
        + 0.35 * ((barrel_rate_allowed / league_barrel) - 1.0)
        + 0.012 * (avg_ev_allowed - league_ev)
    ) * la_score

    raw = float(np.clip(raw, 0.88, 1.16))

    confidence = float(np.clip(min(pa / 350.0, bbe / 120.0), 0.0, 1.0))
    mult = 1.0 + (raw - 1.0) * confidence
    mult = float(np.clip(mult, 0.95, 1.10))

    score = float(np.clip((mult - 1.0) * 100.0, -10.0, 10.0))
    return mult, score

def make_bet_flag(
    p1: float,
    dominant_pitch: str,
    dominant_usage: float,
    batter_vs_pitch_ratio: float,
    recent_barrel_rate: float,
    recent_hard_hit_rate: float,
    recent_ev: float,
) -> tuple[int, str, int, str]:
    """
    Score strong pitch-fit + recent-power spots.

    Returns:
      bet_flag         -> 1 when score is actionable
      bet_flag_reason  -> compact semicolon-separated reason list
      bet_flag_score   -> 0 to 5
      bet_flag_tier    -> none / watch / playable / strong
    """
    score = 0
    reasons = []

    # Pitch fit: dominant pitch + hitter success vs that pitch
    if dominant_pitch and dominant_usage >= 0.35:
        score += 1
        reasons.append(f"{dominant_pitch} volume")
    if dominant_pitch and dominant_usage >= 0.35 and batter_vs_pitch_ratio >= 1.08:
        score += 1
        reasons.append(f"{dominant_pitch} fit")
    if dominant_pitch and dominant_usage >= 0.40 and batter_vs_pitch_ratio >= 1.18:
        score += 1
        reasons.append("elite pitch fit")

    # Recent power quality
    if recent_barrel_rate >= 0.08:
        score += 1
        reasons.append("barrels")
    if recent_hard_hit_rate >= 0.42:
        score += 1
        reasons.append("hard-hit")
    if recent_ev >= 91.0:
        score += 1
        reasons.append("EV")

    # Probability sanity anchor
    if p1 >= 0.10:
        score += 1
        reasons.append("prob")
    if p1 >= 0.14:
        score += 1
        reasons.append("plus prob")

    # Keep score on a compact 0-5 scale
    score = int(min(score, 5))

    # Require either a pitch-fit leg or a strong probability anchor
    has_pitch_fit = (
        bool(dominant_pitch)
        and dominant_usage >= 0.35
        and batter_vs_pitch_ratio >= 1.08
    )
    has_power = (
        recent_barrel_rate >= 0.08
        or recent_hard_hit_rate >= 0.42
        or recent_ev >= 91.0
    )

    bet_flag = int((score >= 3 and has_pitch_fit) or (score >= 4 and has_power and p1 >= 0.10))

    if score >= 5:
        tier = "strong"
    elif score >= 4:
        tier = "playable"
    elif score >= 2:
        tier = "watch"
    else:
        tier = "none"

    # Deduplicate while preserving order
    seen = set()
    compact_reasons = []
    for r in reasons:
        if r not in seen:
            compact_reasons.append(r)
            seen.add(r)

    return bet_flag, "; ".join(compact_reasons[:4]), score, tier


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
    model_path = f"models_hr/hr_xgb_{key}_{MODEL_VERSION}.json"
    calib_path = f"models_hr/hr_cal_{key}_{MODEL_VERSION}.pkl"
    meta_path = f"models_hr/hr_meta_{key}_{MODEL_VERSION}.json"

    if os.path.exists(model_path) and os.path.exists(calib_path) and os.path.exists(meta_path):
        model = XGBRegressor()
        model.load_model(model_path)
        calib = pd.read_pickle(calib_path)
        meta = json.load(open(meta_path, "r"))
        return model, calib, meta

    df = bat_df.copy()
    df = ensure_recent_feature_columns(df)
    league_hr_pa = float(df["HR"].sum() / df["PA"].sum()) if df["PA"].sum() > 0 else 0.032

    def _calc_hr_rate_shrunk(r):
        prior_mean, prior_strength = get_power_aware_prior(r, league_hr_pa)
        return shrink_rate(r["HR"], r["PA"], prior_mean=prior_mean, prior_strength=prior_strength)

    df["hr_rate_shrunk"] = df.apply(_calc_hr_rate_shrunk, axis=1)

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
# Final probability calibration
# -------------------------
_FINAL_CALIBRATOR_CACHE = None

def _prepare_final_calibration_training_data(log_path: str = PERF_LOG_PATH,
                                             lookback_days: int = FINAL_CALIBRATION_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Build a clean training set for the final one-game HR probability calibrator.
    Prefers raw model probabilities when available, otherwise falls back to the
    logged model_prob column from older runs.
    """
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=["p_raw", "result"])

    try:
        df = safe_read_csv(log_path)
    except Exception:
        return pd.DataFrame(columns=["p_raw", "result"])

    if df is None or df.empty:
        return pd.DataFrame(columns=["p_raw", "result"])

    if "result" not in df.columns:
        return pd.DataFrame(columns=["p_raw", "result"])

    work = df.copy()
    work = work[work["result"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["p_raw", "result"])

    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        if lookback_days and work["date"].notna().any():
            cutoff = pd.Timestamp(date_cls.today()) - pd.Timedelta(days=int(lookback_days))
            work = work[work["date"].isna() | (work["date"] >= cutoff)].copy()

    prob_col = "raw_model_prob" if "raw_model_prob" in work.columns else ("model_prob" if "model_prob" in work.columns else None)
    if prob_col is None:
        return pd.DataFrame(columns=["p_raw", "result"])

    work["p_raw"] = pd.to_numeric(work[prob_col], errors="coerce")
    work["result"] = pd.to_numeric(work["result"], errors="coerce")
    work = work[work["p_raw"].notna() & work["result"].isin([0, 1])].copy()
    if work.empty:
        return pd.DataFrame(columns=["p_raw", "result"])

    work["p_raw"] = work["p_raw"].clip(1e-6, 0.999)

    # Deduplicate repeated rows for the same settled bet. Keep the most recent entry.
    dedupe_cols = [c for c in ["date", "batter_id", "bet_type"] if c in work.columns]
    if dedupe_cols:
        work = work.sort_values(dedupe_cols).drop_duplicates(subset=dedupe_cols, keep="last")

    return work[["p_raw", "result"]].reset_index(drop=True)


def fit_final_calibrator_from_log(log_path: str = PERF_LOG_PATH,
                                  save_path: str = FINAL_CALIBRATOR_PATH,
                                  min_rows: int = MIN_FINAL_CALIBRATION_ROWS,
                                  min_positives: int = MIN_FINAL_CALIBRATION_POSITIVES) -> tuple[object | None, dict]:
    """
    Fit a final isotonic calibrator on settled HR bets in outputs/performance_log.csv.
    Returns (calibrator_or_none, metadata).
    """
    global _FINAL_CALIBRATOR_CACHE

    train_df = _prepare_final_calibration_training_data(log_path=log_path)
    n_rows = int(len(train_df))
    n_pos = int(train_df["result"].sum()) if n_rows else 0

    meta = {
        "trained": False,
        "n_rows": n_rows,
        "n_pos": n_pos,
        "path": save_path,
        "min_rows": int(min_rows),
        "min_positives": int(min_positives),
    }

    if n_rows < int(min_rows) or n_pos < int(min_positives):
        return None, meta

    x = train_df["p_raw"].to_numpy(dtype=float)
    y = train_df["result"].to_numpy(dtype=float)

    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x, y)
        pd.to_pickle(iso, save_path)
        _FINAL_CALIBRATOR_CACHE = iso
        meta["trained"] = True
        return iso, meta
    except Exception as e:
        meta["error"] = str(e)
        return None, meta


def maybe_refresh_final_calibrator(log_path: str = PERF_LOG_PATH) -> dict:
    """Attempt to refresh the final calibrator after settled results are updated."""
    _, meta = fit_final_calibrator_from_log(log_path=log_path)
    return meta


def load_final_calibrator():
    global _FINAL_CALIBRATOR_CACHE
    if _FINAL_CALIBRATOR_CACHE is not None:
        return _FINAL_CALIBRATOR_CACHE
    if os.path.exists(FINAL_CALIBRATOR_PATH):
        try:
            _FINAL_CALIBRATOR_CACHE = pd.read_pickle(FINAL_CALIBRATOR_PATH)
            return _FINAL_CALIBRATOR_CACHE
        except Exception:
            _FINAL_CALIBRATOR_CACHE = None
    return None


def smooth_probability_cap(p: float,
                           start: float = 0.26,
                           hard_cap: float = 0.34,
                           tail_shrink: float = 0.30) -> float:
    """
    Preserve rank ordering while preventing unrealistic one-game HR probabilities.
    Anything above `start` gets compressed toward `hard_cap`.
    """
    p = float(np.clip(p, 1e-6, 0.999))
    if p <= start:
        return p
    p = start + (p - start) * tail_shrink
    return float(min(p, hard_cap))


def calibrate_final_hr_probability(raw_p1: float) -> tuple[float, str]:
    """
    Preferred path: learned isotonic calibrator fit on historical outcomes.
    Fallback path: smooth cap that keeps board order but prevents blowout values.
    """
    raw_p1 = float(np.clip(raw_p1, 1e-6, 0.999))
    cal = load_final_calibrator()
    if cal is not None:
        try:
            return float(np.clip(cal.predict([raw_p1])[0], 1e-6, 0.999)), "isotonic_final"
        except Exception:
            pass
    return smooth_probability_cap(raw_p1), "smooth_cap"

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
            "game_pk": g.get("game_id") or g.get("game_pk") or g.get("gamePk"),
            "home_probable_pitcher": g.get("home_probable_pitcher"),
            "away_probable_pitcher": g.get("away_probable_pitcher"),
            "game_datetime": g.get("game_datetime") or g.get("gameDate") or g.get("datetime") or g.get("gameDateTime"),
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
DEFAULT_OUT_TO_CF_BEARINGS = {
    "Minnesota Twins": 0.0,
    "New York Yankees": 58.0,
    "Tampa Bay Rays": 32.0,
    "Pittsburgh Pirates": 315.0,
    "Atlanta Braves": 145.0,
    "New York Mets": 20.0,
    "Chicago Cubs": 45.0,
    "Los Angeles Dodgers": 35.0,
    "Boston Red Sox": 53.0,
    "Philadelphia Phillies": 28.0,
    "St. Louis Cardinals": 45.0,
    "Milwaukee Brewers": 25.0,
    "Seattle Mariners": 24.0,
    "San Francisco Giants": 60.0,
    "San Diego Padres": 35.0,
    "Kansas City Royals": 22.0,
    "Detroit Tigers": 28.0,
    "Cleveland Guardians": 18.0,
    "Toronto Blue Jays": 45.0,
    "Texas Rangers": 17.0,
    "Houston Astros": 35.0,
    "Arizona Diamondbacks": 22.0,
    "Colorado Rockies": 15.0,
    "Los Angeles Angels": 38.0,
    "Baltimore Orioles": 50.0,
    "Washington Nationals": 48.0,
    "Miami Marlins": 20.0,
    "Oakland Athletics": 35.0,
    "Athletics": 35.0,
    "Cincinnati Reds": 35.0,
    "Chicago White Sox": 30.0,
}


def load_stadium_coords():
    path = "inputs/stadium_coords.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _parse_game_hour_local(game_datetime: str | None) -> int | None:
    if not game_datetime:
        return None
    try:
        dt = pd.to_datetime(game_datetime)
        if pd.isna(dt):
            return None
        return int(dt.hour)
    except Exception:
        return None



def get_game_weather(lat: float, lon: float, date_str: str, game_hour_local: int | None = None) -> dict:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,windspeed_10m,winddirection_10m,precipitation,rain,snowfall"
        "&daily=temperature_2m_max"
        "&wind_speed_unit=mph"
        "&temperature_unit=fahrenheit&timezone=auto"
        f"&start_date={date_str}&end_date={date_str}"
    )
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        j = r.json()
    except Exception:
        return {
            "temp_f": None,
            "wind_speed_mph": None,
            "wind_dir_deg": None,
            "precip_mm": None,
            "rain_mm": None,
            "snowfall_cm": None,
        }

    daily_temp = None
    try:
        temps = j.get("daily", {}).get("temperature_2m_max", [])
        if temps:
            daily_temp = float(temps[0])
    except Exception:
        daily_temp = None

    hourly = j.get("hourly", {}) or {}
    hours = hourly.get("time", []) or []
    temps = hourly.get("temperature_2m", []) or []
    speeds = hourly.get("windspeed_10m", []) or []
    dirs = hourly.get("winddirection_10m", []) or []
    precips = hourly.get("precipitation", []) or []
    rains = hourly.get("rain", []) or []
    snows = hourly.get("snowfall", []) or []

    if not hours:
        return {
            "temp_f": daily_temp,
            "wind_speed_mph": None,
            "wind_dir_deg": None,
            "precip_mm": None,
            "rain_mm": None,
            "snowfall_cm": None,
        }

    target_hour = 19 if game_hour_local is None else int(game_hour_local)
    best_idx = None
    best_dist = 999
    for i, ts in enumerate(hours):
        try:
            hour = pd.to_datetime(ts).hour
            dist = abs(hour - target_hour)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        except Exception:
            continue

    if best_idx is None:
        return {
            "temp_f": daily_temp,
            "wind_speed_mph": None,
            "wind_dir_deg": None,
            "precip_mm": None,
            "rain_mm": None,
            "snowfall_cm": None,
        }

    temp_f = daily_temp
    try:
        temp_f = float(temps[best_idx]) if len(temps) > best_idx else daily_temp
    except Exception:
        temp_f = daily_temp

    def _pick(arr):
        try:
            return float(arr[best_idx]) if len(arr) > best_idx else None
        except Exception:
            return None

    return {
        "temp_f": temp_f,
        "wind_speed_mph": _pick(speeds),
        "wind_dir_deg": _pick(dirs),
        "precip_mm": _pick(precips),
        "rain_mm": _pick(rains),
        "snowfall_cm": _pick(snows),
    }


def _wind_out_in_components(wind_speed_mph: float | None,
                            wind_dir_deg: float | None,
                            out_to_cf_deg: float | None) -> tuple[float, float]:
    if wind_speed_mph is None or wind_dir_deg is None or out_to_cf_deg is None:
        return 0.0, 0.0
    speed = float(max(wind_speed_mph, 0.0))
    if speed < 2.0:
        return 0.0, 0.0
    diff = abs(((float(wind_dir_deg) - float(out_to_cf_deg) + 180.0) % 360.0) - 180.0)
    inward_component = float(speed * np.cos(np.deg2rad(diff)))
    outward_component = max(0.0, -inward_component)
    inward_component = max(0.0, inward_component)
    return outward_component, inward_component


def _generic_weather_multiplier(temp_f: float | None,
                                wind_speed_mph: float | None,
                                wind_dir_deg: float | None,
                                out_to_cf_deg: float | None,
                                precip_mm: float | None = None,
                                snowfall_cm: float | None = None) -> tuple[float, float, float]:
    t_mult = temp_multiplier(temp_f)
    w_mult = wind_multiplier(wind_speed_mph, wind_dir_deg, out_to_cf_deg)
    mult = float(t_mult * w_mult)

    # precipitation / snow suppress carry
    if snowfall_cm is not None and snowfall_cm > 0:
        mult *= 0.60
    elif precip_mm is not None and precip_mm >= 2.5:
        mult *= 0.75
    elif precip_mm is not None and precip_mm >= 0.8:
        mult *= 0.90

    # cold suppression
    if temp_f is not None:
        if temp_f <= 38:
            mult *= 0.60
        elif temp_f <= 45:
            mult *= 0.75
        elif temp_f <= 50:
            mult *= 0.90

    return float(np.clip(mult, 0.55, 1.25)), float(t_mult), float(w_mult)


def _park_weather_override(home_team: str,
                           temp_f: float | None,
                           wind_speed_mph: float | None,
                           wind_dir_deg: float | None,
                           out_to_cf_deg: float | None,
                           precip_mm: float | None = None,
                           snowfall_cm: float | None = None) -> float:
    out_mph, in_mph = _wind_out_in_components(wind_speed_mph, wind_dir_deg, out_to_cf_deg)
    team = (home_team or "").strip()

    # Wrigley = weather amplifier
    if team == "Chicago Cubs":
        if temp_f is not None and temp_f >= 65 and out_mph >= 8:
            return 1.20 if out_mph < 12 else 1.25
        if in_mph >= 10:
            return 0.70 if in_mph >= 14 else 0.75

    # Coors = usually boost, but cold/snow overrides hard
    if team == "Colorado Rockies":
        if (temp_f is not None and temp_f <= 45) or ((snowfall_cm or 0) > 0):
            return 0.70
        return 1.08

    # Yankee short porch / RF carry
    if team == "New York Yankees":
        if out_mph >= 8:
            return 1.06 if out_mph < 12 else 1.10

    # Great American plays in warmth
    if team == "Cincinnati Reds":
        if temp_f is not None and temp_f >= 75:
            return 1.05

    # Suppressive parks get deader with incoming wind/cold
    if team == "San Diego Padres":
        if in_mph >= 10:
            return 0.88
        if temp_f is not None and temp_f <= 55:
            return 0.93

    if team == "Seattle Mariners":
        if in_mph >= 10:
            return 0.90
        if temp_f is not None and temp_f <= 55:
            return 0.95

    if team == "San Francisco Giants":
        if in_mph >= 10:
            return 0.90
        if temp_f is not None and temp_f <= 55:
            return 0.95

    return 1.0


def compute_weather_multiplier(loc: dict | None, home_team: str, game_datetime: str | None, date_str: str) -> tuple[float, dict]:
    if not loc or "lat" not in loc or "lon" not in loc:
        return 1.0, {
            "temp_f": None,
            "wind_speed_mph": None,
            "wind_dir_deg": None,
            "precip_mm": None,
            "rain_mm": None,
            "snowfall_cm": None,
            "temp_mult": 1.0,
            "wind_mult": 1.0,
            "park_weather_override": 1.0,
        }

    game_hour = _parse_game_hour_local(game_datetime)
    weather = get_game_weather(float(loc["lat"]), float(loc["lon"]), date_str, game_hour_local=game_hour)
    out_to_cf_deg = loc.get("out_to_cf_deg")
    if out_to_cf_deg is None:
        out_to_cf_deg = DEFAULT_OUT_TO_CF_BEARINGS.get(home_team)

    base_mult, t_mult, w_mult = _generic_weather_multiplier(
        temp_f=weather.get("temp_f"),
        wind_speed_mph=weather.get("wind_speed_mph"),
        wind_dir_deg=weather.get("wind_dir_deg"),
        out_to_cf_deg=out_to_cf_deg,
        precip_mm=weather.get("precip_mm"),
        snowfall_cm=weather.get("snowfall_cm"),
    )
    park_override = _park_weather_override(
        home_team=home_team,
        temp_f=weather.get("temp_f"),
        wind_speed_mph=weather.get("wind_speed_mph"),
        wind_dir_deg=weather.get("wind_dir_deg"),
        out_to_cf_deg=out_to_cf_deg,
        precip_mm=weather.get("precip_mm"),
        snowfall_cm=weather.get("snowfall_cm"),
    )
    final_mult = float(np.clip(base_mult * park_override, 0.55, 1.30))
    weather.update({
        "temp_mult": round(t_mult, 3),
        "wind_mult": round(w_mult, 3),
        "park_weather_override": round(park_override, 3),
        "out_to_cf_deg": out_to_cf_deg,
    })
    return final_mult, weather


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

def to_100_score(value: float | None, low: float, high: float) -> float:
    """Linear 0-100 scaling with clipping."""
    try:
        v = float(value)
    except Exception:
        return 50.0
    if high <= low:
        return 50.0
    return float(np.clip((v - low) / (high - low), 0.0, 1.0) * 100.0)


def centered_band_score(value: float | None, center: float, tolerance: float) -> float:
    """100 at center, fading linearly to 0 outside the tolerance band."""
    try:
        v = float(value)
    except Exception:
        return 50.0
    tol = max(float(tolerance), 1e-6)
    return float(np.clip(1.0 - abs(v - center) / tol, 0.0, 1.0) * 100.0)


def compute_power_profile(
    barrel_rate: float,
    avg_ev: float,
    avg_la: float,
    k_rate: float,
    bbe: float,
    recent_barrel_rate: float,
    recent_hard_hit_rate: float,
    recent_ev: float,
) -> tuple[float, int, str, float]:
    """Stage 1: stable power qualification."""
    barrel_s = to_100_score(barrel_rate, 0.03, 0.16)
    ev_s = to_100_score(avg_ev, 86.5, 93.5)
    la_s = centered_band_score(avg_la, center=18.0, tolerance=18.0)
    recent_s = (
        0.40 * to_100_score(recent_barrel_rate, 0.03, 0.16)
        + 0.30 * to_100_score(recent_hard_hit_rate, 0.28, 0.60)
        + 0.30 * to_100_score(recent_ev, 87.5, 94.0)
    )
    sample_s = to_100_score(bbe, 20.0, 220.0)
    k_penalty = to_100_score(k_rate, 0.22, 0.38)

    power_score = (
        0.38 * barrel_s
        + 0.25 * ev_s
        + 0.12 * la_s
        + 0.15 * recent_s
        + 0.10 * sample_s
        - 0.10 * k_penalty
    )
    power_score = float(np.clip(power_score, 0.0, 100.0))

    has_real_pop = (
        barrel_rate >= 0.06
        or avg_ev >= 89.0
        or recent_barrel_rate >= 0.08
        or recent_ev >= 90.5
    )
    power_qualified = int(power_score >= POWER_SCORE_QUALIFY_THRESHOLD and has_real_pop and bbe >= 20)

    if power_score >= 72:
        power_tier = "elite"
        gate_mult = 1.00
    elif power_score >= 60:
        power_tier = "plus"
        gate_mult = 0.96
    elif power_score >= POWER_SCORE_QUALIFY_THRESHOLD:
        power_tier = "qualified"
        gate_mult = 0.92
    elif power_score >= 40:
        power_tier = "fringe"
        gate_mult = 0.80
    else:
        power_tier = "thin"
        gate_mult = 0.68

    if power_qualified:
        gate_mult = max(gate_mult, 0.92)

    return power_score, power_qualified, power_tier, float(np.clip(gate_mult, 0.65, 1.00))


def compute_today_matchup_scores(
    power_score: float,
    power_gate_mult: float,
    pt_mult: float,
    dominant_pitch_usage: float,
    batter_vs_pitch_ratio: float,
    pitcher_hr_mult: float,
    pitcher_hr_score: float,
    platoon_mult: float,
    exp_pa: float,
    exp_bbe_mult: float,
    env_mult: float,
    bp_mult: float,
    weather_mult: float,
    recent_form_mult: float,
    recent_barrel_rate: float,
    recent_hard_hit_rate: float,
    recent_ev: float,
) -> dict:
    """Stage 2: matchup-first ranking for today's HR spots."""
    pitcher_vulnerability_score = (
        0.65 * to_100_score(pitcher_hr_mult, 0.94, 1.10)
        + 0.35 * to_100_score(pitcher_hr_score, -6.0, 8.0)
    )
    pitch_mix_fit_score = (
        0.45 * to_100_score(pt_mult, 0.94, 1.16)
        + 0.35 * to_100_score(batter_vs_pitch_ratio, 0.92, 1.24)
        + 0.20 * to_100_score(dominant_pitch_usage, 0.25, 0.60)
    )
    split_advantage_score = to_100_score(platoon_mult, 0.88, 1.12)
    opportunity_score = (
        0.75 * to_100_score(exp_pa, 3.65, 4.95)
        + 0.25 * to_100_score(exp_bbe_mult, 0.88, 1.10)
    )
    environment_score = (
        0.55 * to_100_score(env_mult, 0.92, 1.14)
        + 0.25 * to_100_score(bp_mult, 0.96, 1.08)
        + 0.20 * to_100_score(weather_mult, 0.92, 1.12)
    )
    recent_damage_score = (
        0.35 * to_100_score(recent_barrel_rate, 0.03, 0.16)
        + 0.25 * to_100_score(recent_hard_hit_rate, 0.28, 0.60)
        + 0.20 * to_100_score(recent_ev, 87.5, 94.0)
        + 0.20 * to_100_score(recent_form_mult, 0.96, 1.08)
    )

    matchup_score = (
        MATCHUP_SCORE_WEIGHTS["pitcher_vulnerability"] * pitcher_vulnerability_score
        + MATCHUP_SCORE_WEIGHTS["pitch_mix_fit"] * pitch_mix_fit_score
        + MATCHUP_SCORE_WEIGHTS["split_advantage"] * split_advantage_score
        + MATCHUP_SCORE_WEIGHTS["opportunity"] * opportunity_score
        + MATCHUP_SCORE_WEIGHTS["environment"] * environment_score
        + MATCHUP_SCORE_WEIGHTS["recent_damage"] * recent_damage_score
    )
    matchup_score = float(np.clip(matchup_score, 0.0, 100.0))

    daily_matchup_hr_score = (0.90 * matchup_score + 0.10 * power_score) * float(power_gate_mult)
    daily_matchup_hr_score = float(np.clip(daily_matchup_hr_score, 0.0, 100.0))

    if daily_matchup_hr_score >= 72:
        matchup_tier = "premium"
    elif daily_matchup_hr_score >= 60:
        matchup_tier = "strong"
    elif daily_matchup_hr_score >= 48:
        matchup_tier = "playable"
    else:
        matchup_tier = "thin"

    return {
        "pitcher_vulnerability_score": float(np.clip(pitcher_vulnerability_score, 0.0, 100.0)),
        "pitch_mix_fit_score": float(np.clip(pitch_mix_fit_score, 0.0, 100.0)),
        "split_advantage_score": float(np.clip(split_advantage_score, 0.0, 100.0)),
        "opportunity_score": float(np.clip(opportunity_score, 0.0, 100.0)),
        "environment_score": float(np.clip(environment_score, 0.0, 100.0)),
        "recent_damage_score": float(np.clip(recent_damage_score, 0.0, 100.0)),
        "matchup_score": matchup_score,
        "daily_matchup_hr_score": daily_matchup_hr_score,
        "matchup_tier": matchup_tier,
    }


# Build Board
# -------------------------


def compute_hybrid_matchup_score(
    matchup_mult,
    pitchtype_mult,
    recent_form_mult,
    pitcher_hr_mult,
    park_factor,
    weather_mult,
    exp_pa,
    batter_vs_pitch_ratio,
    dominant_pitch_usage,
):
    import numpy as np

    def score(v, lo, hi):
        try:
            x = float(v)
        except:
            x = (lo + hi) / 2
        return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100)

    matchup_mult = matchup_mult or 1.0
    pitchtype_mult = pitchtype_mult or 1.0
    recent_form_mult = recent_form_mult or 1.0
    pitcher_hr_mult = pitcher_hr_mult or 1.0
    park_factor = park_factor or 1.0
    weather_mult = weather_mult or 1.0
    exp_pa = exp_pa or 4.0
    batter_vs_pitch_ratio = batter_vs_pitch_ratio or 1.0
    dominant_pitch_usage = dominant_pitch_usage or 0.3

    s = (
        0.34 * score(pitchtype_mult, 0.94, 1.18) +
        0.26 * score(pitcher_hr_mult, 0.94, 1.10) +
        0.24 * score(matchup_mult, 0.94, 1.18) +
        0.16 * score(recent_form_mult, 0.96, 1.08)
    )

    s = float(np.clip(s, 35, 100))

    if s >= 72:
        tier = "premium"
    elif s >= 60:
        tier = "strong"
    elif s >= 48:
        tier = "playable"
    else:
        tier = "thin"

    return s, tier

def build_board(date_str: str, n_sims: int, train_seasons: list[int], use_weather: bool):
    bat_df, pit_df, mix_df, dmg_df, park_df, bullpen_df, platoon_df = get_training_cached(train_seasons)
    platoon_map = platoon_df.set_index("batter").to_dict(orient="index") if platoon_df is not None else {}
    model, calib, meta = train_or_load_hr_model(bat_df, train_seasons)

    bat_latest = bat_df.sort_values("season").groupby("batter").tail(1).set_index("batter")
    bat_latest = ensure_recent_feature_columns(bat_latest)
    recent_df = get_recent_statcast_features(bat_latest.index.tolist(), end_date=date_str, days=RECENT_WINDOW_DAYS)
    recent_map = recent_df.set_index("batter").to_dict(orient="index") if recent_df is not None and not recent_df.empty else {}
    bat_df = ensure_recent_feature_columns(bat_df)
    league_feat = bat_df[FEATURE_COLS].mean(numeric_only=True).to_frame().T.fillna(0.0)

    barrel_map = dict(zip(park_df["home_team"], park_df.get("park_barrel_mult", pd.Series(dtype=float))))
    fb_map = dict(zip(park_df["home_team"], park_df.get("park_fb_mult", pd.Series(dtype=float))))
    mix_map = mix_df.set_index("pitcher").to_dict(orient="index")
    dmg_map = dmg_df.set_index("batter").to_dict(orient="index")
    league_pitch_map = {}
    for pt in PITCH_TYPES:
        col = f"bat_hr_pa_{pt}"
        if col in dmg_df.columns:
            league_pitch_map[pt] = float(pd.to_numeric(dmg_df[col], errors="coerce").mean())

    bullpen_map = dict(zip(bullpen_df.get("team", []), bullpen_df.get("bullpen_factor", [])))
    pit_latest = pit_df.sort_values("season").groupby("pitcher").tail(1).set_index("pitcher") if pit_df is not None and not pit_df.empty else pd.DataFrame()
    pit_map = pit_latest.to_dict(orient="index") if isinstance(pit_latest, pd.DataFrame) and not pit_latest.empty else {}
    pit_league = {
        "hr_rate_allowed": float(pd.to_numeric(pit_df.get("hr_rate_allowed", pd.Series(dtype=float)), errors="coerce").mean()) if pit_df is not None and "hr_rate_allowed" in pit_df.columns else float(meta.get("league_hr_pa", 0.032)),
        "barrel_rate_allowed": float(pd.to_numeric(pit_df.get("barrel_rate_allowed", pd.Series(dtype=float)), errors="coerce").mean()) if pit_df is not None and "barrel_rate_allowed" in pit_df.columns else 0.07,
        "avg_ev_allowed": float(pd.to_numeric(pit_df.get("avg_ev_allowed", pd.Series(dtype=float)), errors="coerce").mean()) if pit_df is not None and "avg_ev_allowed" in pit_df.columns else 89.0,
        "avg_la_allowed": float(pd.to_numeric(pit_df.get("avg_la_allowed", pd.Series(dtype=float)), errors="coerce").mean()) if pit_df is not None and "avg_la_allowed" in pit_df.columns else 12.0,
    }

    coords = load_stadium_coords()
    games = get_games(date_str)
    print(f"[DEBUG] games={len(games)} for {date_str}")
    rows = []

    for g in games:
        print(f"[DEBUG] game: {g.get('away_team')} @ {g.get('home_team')} venue={g.get('venue_name')}")  
        home = g["home_team"]
        away = g["away_team"]
        venue = g["venue_name"]
        game_pk = g.get('game_pk')

        home_sp_name = g.get("home_probable_pitcher") or "TBD"
        away_sp_name = g.get("away_probable_pitcher") or "TBD"

        home_sp_id = lookup_player_id_by_name(home_sp_name)
        away_sp_id = lookup_player_id_by_name(away_sp_name)

        barrel_mult = float(barrel_map.get(home, 1.0))
        fb_mult = float(fb_map.get(home, 1.0))
        park_contact_mult = (barrel_mult ** 0.6) * (fb_mult ** 0.4)
        park_contact_mult = float(np.clip(park_contact_mult, 0.85, 1.15))

        w_mult = 1.0
        weather_info = {"temp_f": None, "wind_speed_mph": None, "wind_dir_deg": None, "temp_mult": 1.0, "wind_mult": 1.0}
        if use_weather:
            loc = coords.get(home) or coords.get(venue)
            w_mult, weather_info = compute_weather_multiplier(
                loc=loc,
                home_team=home,
                game_datetime=g.get("game_datetime"),
                date_str=date_str,
            )

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
                hid = int(hid)
                player_name = get_player_name(hid)

                if hid in bat_latest.index:
                    feat = bat_latest.loc[[hid], FEATURE_COLS].copy().fillna(0.0)
                else:
                    feat = league_feat.copy()

                rinfo = recent_map.get(hid)
                if rinfo:
                    feat.loc[:, "recent_barrel_rate"] = float(rinfo.get("recent_barrel_rate", feat.iloc[0].get("barrel_rate", 0.0)))
                    feat.loc[:, "recent_hard_hit_rate"] = float(rinfo.get("recent_hard_hit_rate", feat.iloc[0].get("hard_hit_rate", feat.iloc[0].get("recent_hard_hit_rate", 0.0))))
                    feat.loc[:, "recent_ev"] = float(rinfo.get("recent_ev", feat.iloc[0].get("avg_ev", 0.0)))
                else:
                    feat.loc[:, "recent_barrel_rate"] = float(feat.iloc[0].get("barrel_rate", 0.0))
                    feat.loc[:, "recent_hard_hit_rate"] = float(feat.iloc[0].get("hard_hit_rate", feat.iloc[0].get("recent_hard_hit_rate", 0.0)))
                    feat.loc[:, "recent_ev"] = float(feat.iloc[0].get("avg_ev", 0.0))

                p_raw = float(model.predict(feat)[0])
                p_pa = float(calib.predict([np.clip(p_raw, 1e-6, 0.5)])[0])
                p_pa = float(np.clip(p_pa, 1e-6, 0.25))

                baseline = float(meta.get("league_hr_pa", 0.032))
                batter_bbe = float(feat.iloc[0].get("BBE", 0.0))
                recent_bbe = float((rinfo or {}).get("recent_bbe", 0.0))
                recent_barrel_rate = float(feat.iloc[0].get("recent_barrel_rate", 0.0))
                recent_hard_hit_rate = float(feat.iloc[0].get("recent_hard_hit_rate", 0.0))
                recent_ev = float(feat.iloc[0].get("recent_ev", 0.0))
                base_barrel_rate = float(feat.iloc[0].get("barrel_rate", 0.0))
                base_ev = float(feat.iloc[0].get("avg_ev", 0.0))

                pt_mult, dominant_pitch, dominant_pitch_usage, batter_vs_pitch_ratio = compute_pitch_matchup_multiplier(
                    pitcher_id=pitcher_id,
                    batter_id=int(hid),
                    mix_map=mix_map,
                    dmg_map=dmg_map,
                    league_pitch_map=league_pitch_map,
                    baseline_hr_pa=baseline,
                    batter_bbe=batter_bbe,
                )
                pt_mult = sharpened_pitchtype_multiplier(
                    pt_mult=pt_mult,
                    dominant_pitch_usage=dominant_pitch_usage,
                    batter_vs_pitch_ratio=batter_vs_pitch_ratio,
                )

                recent_form_mult, recent_form_score = compute_recent_power_multiplier(
                    barrel_rate=base_barrel_rate,
                    avg_ev=base_ev,
                    recent_barrel_rate=recent_barrel_rate,
                    recent_hard_hit_rate=recent_hard_hit_rate,
                    recent_ev=recent_ev,
                    recent_bbe=recent_bbe,
                )

                pitcher_hr_mult, pitcher_hr_score = compute_pitcher_hr_profile_multiplier(
                    pitcher_id=pitcher_id,
                    pit_map=pit_map,
                    pit_league=pit_league,
                )

                env_mult = float(np.clip(park_contact_mult * w_mult, 0.80, 1.30))

                # ---- Pitcher handedness (R/L) ----
                pitch_hand = ""
                if pitcher_id:
                    _, pitch_hand = platoon.get_handedness(int(pitcher_id))
                pitch_hand = (pitch_hand or "").upper()[:1]

                # ---- Platoon adjustment ----
                platoon_mult = 1.0
                if pitch_hand in ("R", "L"):
                    pinfo = platoon_map.get(int(hid))
                    if pinfo:
                        overall = float(pinfo.get("hr_pa_overall_shrunk", np.nan))
                        split_key = "hr_pa_vs_R_shrunk" if pitch_hand == "R" else "hr_pa_vs_L_shrunk"
                        split = float(pinfo.get(split_key, np.nan))

                        if np.isfinite(overall) and overall > 0 and np.isfinite(split) and split > 0:
                            platoon_mult = float(np.clip(split / overall, 0.80, 1.20))

                # ---- Final per-PA probability ----
                matchup_mult = float(np.clip(pt_mult * recent_form_mult * pitcher_hr_mult, 0.90, 1.22))
                p_pa_adj = float(np.clip(p_pa * matchup_mult * env_mult * bp_mult * platoon_mult, 1e-6, 0.30))

                power_score, power_qualified, power_tier, power_gate_mult = compute_power_profile(
                    barrel_rate=base_barrel_rate,
                    avg_ev=base_ev,
                    avg_la=float(feat.iloc[0].get("avg_la", 0.0)),
                    k_rate=float(feat.iloc[0].get("k_rate", 0.0)),
                    bbe=batter_bbe,
                    recent_barrel_rate=recent_barrel_rate,
                    recent_hard_hit_rate=recent_hard_hit_rate,
                    recent_ev=recent_ev,
                )

                # ---- Expected PA for lineup slot ----
                pa_last_fallback = None
                if hid in bat_latest.index and "PA" in bat_latest.columns:
                    try:
                        pa_last_fallback = float(bat_latest.loc[hid, "PA"])
                    except Exception:
                        pa_last_fallback = None

                team_id = get_team_id(batting_team)
                exp_pa = estimate_exp_pa(
                    batter_id=hid,
                    team_id=team_id,
                    game_pk=game_pk,
                    is_home=is_home,
                    date_str=date_str,
                    pa_last_fallback=pa_last_fallback,
                )

                # ---- Extra sharpening ----
                exp_bbe_mult = exp_bbe_multiplier(
                    k_rate=float(feat.iloc[0].get("k_rate", 0.0)),
                    exp_pa=exp_pa,
                )
                power_spike_mult = recent_power_spike_multiplier(
                    recent_barrel_rate=recent_barrel_rate,
                    recent_hard_hit_rate=recent_hard_hit_rate,
                    recent_ev=recent_ev,
                )
                p_pa_adj = float(np.clip(p_pa_adj * exp_bbe_mult * power_spike_mult, 1e-6, 0.30))

                matchup_scores = compute_today_matchup_scores(
                    power_score=power_score,
                    power_gate_mult=power_gate_mult,
                    pt_mult=pt_mult,
                    dominant_pitch_usage=dominant_pitch_usage,
                    batter_vs_pitch_ratio=batter_vs_pitch_ratio,
                    pitcher_hr_mult=pitcher_hr_mult,
                    pitcher_hr_score=pitcher_hr_score,
                    platoon_mult=platoon_mult,
                    exp_pa=exp_pa,
                    exp_bbe_mult=exp_bbe_mult,
                    env_mult=env_mult,
                    bp_mult=bp_mult,
                    weather_mult=w_mult,
                    recent_form_mult=recent_form_mult,
                    recent_barrel_rate=recent_barrel_rate,
                    recent_hard_hit_rate=recent_hard_hit_rate,
                    recent_ev=recent_ev,
                )

                # ---- Simulate ----
                lam = p_pa_adj * exp_pa
                p1 = float(1.0 - np.exp(-lam))

                raw_p1 = p1
                p1, calibration_method = calibrate_final_hr_probability(raw_p1)

                # keep 2+ probability coherent after 1+ calibration
                p2_raw = float(1.0 - (1.0 + lam) * np.exp(-lam))
                p2 = float(min(p2_raw, max(1e-6, p1 * 0.55)))
                bet_flag, bet_flag_reason, bet_flag_score, bet_flag_tier = make_bet_flag(
                    p1=p1,
                    dominant_pitch=dominant_pitch,
                    dominant_usage=dominant_pitch_usage,
                    batter_vs_pitch_ratio=batter_vs_pitch_ratio,
                    recent_barrel_rate=recent_barrel_rate,
                    recent_hard_hit_rate=recent_hard_hit_rate,
                    recent_ev=recent_ev,
                )

                rows.append(
                    {
                        "date": date_str,
                        "team": batting_team,
                        "venue": venue,
                        "probable_pitcher_faced": pitcher_name,
                        "player_name": player_name,
                        "bet_flag_tier": bet_flag_tier,
                        "bet_flag_score": int(bet_flag_score),
                        "bet_flag": int(bet_flag),
                        "bet_flag_reason": bet_flag_reason,
                        "batter_id": int(hid),
                        "power_score": round(float(power_score), 2),
                        "power_qualified": int(power_qualified),
                        "power_tier": power_tier,
                        "power_gate_mult": round(float(power_gate_mult), 3),
                        "matchup_tier": matchup_scores["matchup_tier"],
                        "daily_matchup_hr_score": round(float(matchup_scores["daily_matchup_hr_score"]), 2),
                        "matchup_score_today": round(float(matchup_scores["matchup_score"]), 2),
                        "pitcher_vulnerability_score": round(float(matchup_scores["pitcher_vulnerability_score"]), 2),
                        "pitch_mix_fit_score": round(float(matchup_scores["pitch_mix_fit_score"]), 2),
                        "split_advantage_score": round(float(matchup_scores["split_advantage_score"]), 2),
                        "opportunity_score": round(float(matchup_scores["opportunity_score"]), 2),
                        "environment_score": round(float(matchup_scores["environment_score"]), 2),
                        "recent_damage_score": round(float(matchup_scores["recent_damage_score"]), 2),
                        "exp_pa": round(exp_pa, 2),
                        "p_hr_pa": p_pa_adj,
                        "p_hr_1plus_sim": p1,
                        "p_hr_2plus_sim": p2,
                        "park_factor": round(park_contact_mult, 3),
                        "weather_mult": round(w_mult, 3),
                        "pitchtype_mult": round(pt_mult, 3),
                        "recent_form_mult": round(recent_form_mult, 3),
                        "pitcher_hr_mult": round(pitcher_hr_mult, 3),
                        "matchup_mult": round(matchup_mult, 3),
                        "dominant_pitch": dominant_pitch,
                        "dominant_pitch_usage": round(float(dominant_pitch_usage), 3),
                        "batter_vs_pitch_ratio": round(float(batter_vs_pitch_ratio), 3),
                        "recent_power_score": round(float(recent_form_score), 2),
                        "pitcher_hr_score": round(float(pitcher_hr_score), 2),
                        "recent_barrel_rate": round(recent_barrel_rate, 4),
                        "recent_hard_hit_rate": round(recent_hard_hit_rate, 4),
                        "recent_ev": round(recent_ev, 2),
                        "weather_temp_f": weather_info.get("temp_f"),
                        "weather_wind_mph": weather_info.get("wind_speed_mph"),
                        "weather_wind_dir_deg": weather_info.get("wind_dir_deg"),
                        "weather_temp_mult": weather_info.get("temp_mult"),
                        "weather_wind_mult": weather_info.get("wind_mult"),
                        "exp_bbe_mult": round(float(exp_bbe_mult), 3),
                        "power_spike_mult": round(float(power_spike_mult), 3),
                        "p_hr_1plus_raw": round(float(raw_p1), 6),
                        "calibration_method": calibration_method,
                    }
                )
                if len(rows) <= 5:
                            print(f"[DEBUG] appended row #{len(rows)} player={player_name}")
    # ---- Build final board after all rows are collected ----
    board = pd.DataFrame(rows)
    if board.empty:
        return board

    if "rank_score" not in board.columns:
        if "p_hr_1plus_sim" in board.columns:
            board["rank_score"] = board["p_hr_1plus_sim"]
        else:
            board["rank_score"] = 0.0

    board = board.sort_values(["daily_matchup_hr_score", "p_hr_1plus_sim", "p_hr_pa"], ascending=False).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    preferred_cols = [
        "rank", "player_name", "power_tier", "matchup_tier", "power_score", "daily_matchup_hr_score", "matchup_score_today",
        "pitcher_vulnerability_score", "pitch_mix_fit_score", "split_advantage_score", "opportunity_score", "environment_score", "recent_damage_score",
        "bet_flag_tier", "bet_flag_score", "bet_flag", "bet_flag_reason",
        "team", "probable_pitcher_faced", "venue", "p_hr_1plus_sim", "p_hr_2plus_sim", "p_hr_pa",
        "power_qualified", "power_gate_mult", "exp_pa", "matchup_mult", "pitchtype_mult", "recent_form_mult", "pitcher_hr_mult", "park_factor", "weather_mult",
        "dominant_pitch", "dominant_pitch_usage", "batter_vs_pitch_ratio", "recent_power_score", "pitcher_hr_score",
        "recent_barrel_rate", "recent_hard_hit_rate", "recent_ev", "exp_bbe_mult", "power_spike_mult", "p_hr_1plus_raw", "calibration_method", "batter_id", "date"
    ]
    ordered = [c for c in preferred_cols if c in board.columns]
    remaining = [c for c in board.columns if c not in ordered]
    board = board[ordered + remaining]
    return board




def force_sequential_ranks(board: pd.DataFrame) -> pd.DataFrame:
    if board is None or board.empty:
        return board
    board = board.sort_values(["p_hr_1plus_sim", "p_hr_pa"], ascending=False).reset_index(drop=True)
    if "rank" in board.columns:
        board = board.drop(columns=["rank"])
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    return board

def write_outputs(board: pd.DataFrame, date_str: str, top_n: int):
    if "rank_score" not in board.columns and "p_hr_1plus_sim" in board.columns:
        board["rank_score"] = board["p_hr_1plus_sim"]
    board = force_sequential_ranks(board)
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
        "raw_model_prob",
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

    

    # Pull implied columns if present; else compute
    implied_open_col = "implied_prob_open_1plus" if "implied_prob_open_1plus" in df.columns else None
    implied_close_col = "implied_prob_close_1plus" if "implied_prob_close_1plus" in df.columns else None

    df["bet_price"] = df[open_col] if open_col else np.nan
    df["close_price"] = df[close_col] if close_col and close_col in df.columns else np.nan

    df["raw_model_prob"] = pd.to_numeric(df.get("p_hr_1plus_raw", np.nan), errors="coerce")
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
        "raw_model_prob": df["raw_model_prob"],
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
            log_df = safe_read_csv(log_path)
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
        log_df = safe_read_csv(log_path)
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

    # Refresh final probability calibrator from settled historical results when enough data exists.
    cal_meta = maybe_refresh_final_calibrator()
    if cal_meta.get("trained"):
        print(f"[calibration] refreshed final calibrator on {cal_meta.get('n_rows')} settled rows ({cal_meta.get('n_pos')} HRs).")
    else:
        print(f"[calibration] not refreshed yet: {cal_meta.get('n_rows')} settled rows / {cal_meta.get('n_pos')} HRs available.")

    
    # --- Output filter: keep real power bats and obvious premium probabilities ---
    PROB_FLOOR = 0.08
    board = board[(board["power_gate_mult"] >= 0.80) & (board["p_hr_1plus_sim"] >= PROB_FLOOR)].copy()
    board = board.sort_values(["daily_matchup_hr_score", "p_hr_1plus_sim"], ascending=False)

    write_outputs(board, args.date, top_n=args.top)
    print("\nTop 25 (by daily matchup HR score):")
    print(board.head(25).to_string(index=False))


if __name__ == "__main__":
    main()