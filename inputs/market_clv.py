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

    # strip accents: josé -> jose
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
    # "mike trout" -> "trout_m"
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
        odds = int(s)
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        a = abs(odds)
        return a / (a + 100.0)

def _read_odds_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df is None or df.empty:
        return None

    # expected columns: player_name, odds_1plus
    cols = {c.lower().strip(): c for c in df.columns}
    if "player_name" not in cols or "odds_1plus" not in cols:
        raise ValueError(f"{path} must contain columns: player_name, odds_1plus")

    out = df[[cols["player_name"], cols["odds_1plus"]]].copy()
    out.columns = ["player_name", "odds_1plus"]
    out["_name_key"] = out["player_name"].apply(_normalize_name)
    out["_fb_key"] = out["_name_key"].apply(_fallback_key)
    out = out.drop_duplicates(subset=["_name_key"])
    return out

def attach_clv(board: pd.DataFrame,
               open_path: str = "inputs/odds_open.csv",
               close_path: str = "inputs/odds_close.csv",
               fallback_path: str = "inputs/odds_input.csv") -> pd.DataFrame:
    """
    Adds columns to your HR board:
      odds_open_1plus, implied_prob_open_1plus, edge_open_1plus
      odds_close_1plus, implied_prob_close_1plus, edge_close_1plus
      clv_prob_1plus, clv_pct_1plus

    If open/close files missing, uses fallback_path as "open".
    Never crashes if files are missing.
    """
    if board is None or board.empty or "player_name" not in board.columns:
        return board

    out = board.copy()
    out["_name_key"] = out["player_name"].apply(_normalize_name)
    out["_fb_key"] = out["_name_key"].apply(_fallback_key)

    open_df = _read_odds_csv(open_path)
    close_df = _read_odds_csv(close_path)

    if (open_df is None or open_df.empty) and (close_df is None or close_df.empty):
        open_df = _read_odds_csv(fallback_path)

    if (open_df is None or open_df.empty) and (close_df is None or close_df.empty):
        return out.drop(columns=["_name_key", "_fb_key"], errors="ignore")

    # merge open
    if open_df is not None and not open_df.empty:
        out = out.merge(open_df[["_name_key", "_fb_key", "odds_1plus"]],
                        on="_name_key", how="left")
        # fallback fill via fb key
        miss = out["odds_1plus"].isna()
        if miss.any():
            tmp = out.loc[miss, ["_fb_key"]].merge(
                open_df[["_fb_key", "odds_1plus"]].drop_duplicates("_fb_key"),
                on="_fb_key", how="left"
            )
            out.loc[miss, "odds_1plus"] = tmp["odds_1plus"].values
        out = out.rename(columns={"odds_1plus": "odds_open_1plus"})
    else:
        out["odds_open_1plus"] = np.nan

    # merge close
    if close_df is not None and not close_df.empty:
        out = out.merge(close_df[["_name_key", "_fb_key", "odds_1plus"]],
                        on="_name_key", how="left")
        miss = out["odds_1plus"].isna()
        if miss.any():
            tmp = out.loc[miss, ["_fb_key"]].merge(
                close_df[["_fb_key", "odds_1plus"]].drop_duplicates("_fb_key"),
                on="_fb_key", how="left"
            )
            out.loc[miss, "odds_1plus"] = tmp["odds_1plus"].values
        out = out.rename(columns={"odds_1plus": "odds_close_1plus"})
    else:
        out["odds_close_1plus"] = np.nan

    # model prob column
    model_col = None
    for c in ["p_hr_1plus_sim", "p_hr_1plus", "p_hr_pa"]:
        if c in out.columns:
            model_col = c
            break

    out["implied_prob_open_1plus"] = out["odds_open_1plus"].apply(_american_to_implied)
    out["implied_prob_close_1plus"] = out["odds_close_1plus"].apply(_american_to_implied)

    if model_col is not None:
        out["edge_open_1plus"] = out[model_col] - out["implied_prob_open_1plus"]
        out["edge_close_1plus"] = out[model_col] - out["implied_prob_close_1plus"]
    else:
        out["edge_open_1plus"] = np.nan
        out["edge_close_1plus"] = np.nan

    out["clv_prob_1plus"] = out["implied_prob_close_1plus"] - out["implied_prob_open_1plus"]
    out["clv_pct_1plus"] = out["clv_prob_1plus"] * 100.0

    return out.drop(columns=["_name_key", "_fb_key"], errors="ignore")
