import os
import re
import unicodedata
import pandas as pd
import numpy as np

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _normalize_name(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()

    # strip accents (josé -> jose)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # normalize punctuation
    s = s.replace("’", "'")
    s = re.sub(r"[.\-']", "", s)

    # flip "last, first" -> "first last"
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


def _fallback_key(norm: str) -> str:
    parts = norm.split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        return f"{last}_{first[0]}"
    return norm


def _american_to_implied(odds) -> float:
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


def _read_odds_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df is None or df.empty:
        return None

    # expected: player_name, odds_1plus (allow common variants)
    cols = {c.lower().strip(): c for c in df.columns}

    name_col = None
    for k in ("player_name", "name", "player"):
        if k in cols:
            name_col = cols[k]
            break
    if name_col is None:
        raise ValueError(f"{path} must contain a player_name column")

    odds_col = None
    for k in ("odds_1plus", "odds", "odds1plus", "hr_odds", "odds_hr"):
        if k in cols:
            odds_col = cols[k]
            break
    if odds_col is None:
        raise ValueError(f"{path} must contain an odds_1plus column")

    out = df[[name_col, odds_col]].copy()
    out.columns = ["player_name", "odds_1plus"]
    out["_name_key"] = out["player_name"].apply(_normalize_name)
    out["_fb_key"] = out["_name_key"].apply(_fallback_key)
    out = out.dropna(subset=["_name_key"]).drop_duplicates(subset=["_name_key"], keep="first")
    return out


def attach_clv(
    board: pd.DataFrame,
    open_path: str = "inputs/odds_open.csv",
    close_path: str = "inputs/odds_close.csv",
    fallback_path: str = "inputs/odds_input.csv",
) -> pd.DataFrame:
    """
    Adds columns to your HR board:
      odds_open_1plus, implied_prob_open_1plus, edge_open_1plus
      odds_close_1plus, implied_prob_close_1plus, edge_close_1plus
      clv_prob_1plus, clv_pct_1plus

    If open/close files missing, uses fallback_path as "open".
    Never crashes if files are missing.

    NOTE: This function now creates _name_key/_fb_key itself if missing,
    so hr_run_daily.py doesn't need to.
    """
    if board is None or board.empty or "player_name" not in board.columns:
        return board

    out = board.copy()

    # Ensure keys exist (this prevents the KeyError you hit)
    if "_name_key" not in out.columns:
        out["_name_key"] = out["player_name"].apply(_normalize_name)
    if "_fb_key" not in out.columns:
        out["_fb_key"] = out["_name_key"].apply(_fallback_key)

    # Load odds (prefer open/close, fallback is treated as open)
    open_df = _read_odds_csv(open_path) or _read_odds_csv("odds_open.csv")
    close_df = _read_odds_csv(close_path) or _read_odds_csv("odds_close.csv")

    if open_df is None and close_df is None:
        open_df = _read_odds_csv(fallback_path) or _read_odds_csv("odds_input.csv")

    if open_df is None and close_df is None:
        # nothing to attach
        return out

    # Join on strong key first; fallback key is there if you ever want it later
    if open_df is not None:
        out = out.merge(
            open_df[["_name_key", "odds_1plus"]].rename(columns={"odds_1plus": "odds_open_1plus"}),
            on="_name_key",
            how="left",
        )
    else:
        out["odds_open_1plus"] = np.nan

    if close_df is not None:
        out = out.merge(
            close_df[["_name_key", "odds_1plus"]].rename(columns={"odds_1plus": "odds_close_1plus"}),
            on="_name_key",
            how="left",
        )
    else:
        out["odds_close_1plus"] = np.nan

    # implied probabilities
    out["implied_prob_open_1plus"] = out["odds_open_1plus"].map(_american_to_implied)
    out["implied_prob_close_1plus"] = out["odds_close_1plus"].map(_american_to_implied)

    # model probability column
    model_col = None
    if "p_hr_1plus_sim" in out.columns:
        model_col = "p_hr_1plus_sim"
    elif "p_hr_1plus" in out.columns:
        model_col = "p_hr_1plus"

    if model_col is not None:
        out["edge_open_1plus"] = out[model_col] - out["implied_prob_open_1plus"]
        out["edge_close_1plus"] = out[model_col] - out["implied_prob_close_1plus"]
    else:
        out["edge_open_1plus"] = np.nan
        out["edge_close_1plus"] = np.nan

    # CLV
    out["clv_prob_1plus"] = out["implied_prob_close_1plus"] - out["implied_prob_open_1plus"]
    out["clv_pct_1plus"] = (out["implied_prob_close_1plus"] / out["implied_prob_open_1plus"]) - 1.0

    return out
