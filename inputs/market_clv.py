import os
import pandas as pd
import numpy as np
import re
import unicodedata


# -------------------------
# Name normalization
# -------------------------

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _normalize_name(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()

    # strip accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # normalize punctuation
    s = s.replace("’", "'")
    s = re.sub(r"[.\-']", "", s)

    # flip "last, first"
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


# -------------------------
# Odds helpers
# -------------------------

def _american_to_implied(odds):
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


def _read_odds_csv(path: str):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if df is None or df.empty:
        return None

    cols = {c.lower().strip(): c for c in df.columns}

    if "player_name" not in cols or "odds_1plus" not in cols:
        raise ValueError(f"{path} must contain columns: player_name, odds_1plus")

    out = df[[cols["player_name"], cols["odds_1plus"]]].copy()
    out.columns = ["player_name", "odds_1plus"]

    out["_name_key"] = out["player_name"].apply(_normalize_name)
    out["_fb_key"] = out["_name_key"].apply(_fallback_key)

    out = out.drop_duplicates(subset=["_fb_key"])
    return out


# -------------------------
# Main attach function
# -------------------------

def attach_clv(
    board: pd.DataFrame,
    open_path: str = "inputs/odds_open.csv",
    close_path: str = "inputs/odds_close.csv",
    fallback_path: str = "inputs/odds_input.csv",
) -> pd.DataFrame:

    if board is None or board.empty or "player_name" not in board.columns:
        return board

    out = board.copy()

    # ensure matching keys exist
    if "_name_key" not in out.columns:
        out["_name_key"] = out["player_name"].apply(_normalize_name)

    if "_fb_key" not in out.columns:
        out["_fb_key"] = out["_name_key"].apply(_fallback_key)

    # read files safely
    open_df = _read_odds_csv(open_path)
    if open_df is None:
        open_df = _read_odds_csv("odds_open.csv")

    close_df = _read_odds_csv(close_path)
    if close_df is None:
        close_df = _read_odds_csv("odds_close.csv")

    # fallback logic (no DataFrame truth testing)
    if open_df is None and close_df is None:
        open_df = _read_odds_csv(fallback_path)
        if open_df is None:
            open_df = _read_odds_csv("odds_input.csv")

    if open_df is None and close_df is None:
        return out

    # -------------------------
    # Merge OPEN
    # -------------------------

    if open_df is not None:
        out = out.merge(
            open_df[["_fb_key", "odds_1plus"]].rename(
                columns={"odds_1plus": "odds_open_1plus"}
            ),
            on="_fb_key",
            how="left",
        )
    else:
        out["odds_open_1plus"] = np.nan

    # -------------------------
    # Merge CLOSE
    # -------------------------

    if close_df is not None:
        out = out.merge(
            close_df[["_fb_key", "odds_1plus"]].rename(
                columns={"odds_1plus": "odds_close_1plus"}
            ),
            on="_fb_key",
            how="left",
        )
    else:
        out["odds_close_1plus"] = np.nan

    # -------------------------
    # Compute implied + edges
    # -------------------------

    out["implied_prob_open_1plus"] = out["odds_open_1plus"].apply(
        _american_to_implied
    )
    out["implied_prob_close_1plus"] = out["odds_close_1plus"].apply(
        _american_to_implied
    )

    model_col = None
    if "p_hr_1plus_sim" in out.columns:
        model_col = "p_hr_1plus_sim"
    elif "p_hr_1plus" in out.columns:
        model_col = "p_hr_1plus"

    if model_col is not None:
        out["edge_open_1plus"] = (
            out[model_col] - out["implied_prob_open_1plus"]
        )
        out["edge_close_1plus"] = (
            out[model_col] - out["implied_prob_close_1plus"]
        )
    else:
        out["edge_open_1plus"] = np.nan
        out["edge_close_1plus"] = np.nan

    # CLV in probability space
    out["clv_prob_1plus"] = (
        out["implied_prob_close_1plus"]
        - out["implied_prob_open_1plus"]
    )

    out["clv_pct_1plus"] = (
        out["implied_prob_close_1plus"]
        / out["implied_prob_open_1plus"]
        - 1.0
    )

    # remove internal keys from final output
    return out.drop(columns=["_name_key", "_fb_key"], errors="ignore")
