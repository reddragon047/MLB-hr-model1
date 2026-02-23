# inputs/platoon.py

import numpy as np
import pandas as pd
import statsapi

# cache handedness per run
_HAND_CACHE: dict[int, tuple[str, str]] = {}

def get_handedness(player_id: int) -> tuple[str, str]:
    """
    Returns (bat_side, pitch_hand) like ('L','R').
    bat_side: R/L/S/'' ; pitch_hand: R/L/''
    """
    pid = int(player_id)
    if pid in _HAND_CACHE:
        return _HAND_CACHE[pid]

    bat_side, pitch_hand = "", ""
    try:
        j = statsapi.get("person", {"personId": pid})
        people = j.get("people", [])
        if people and isinstance(people[0], dict):
            p = people[0]
            bat_side = ((p.get("batSide") or {}).get("code") or "").upper()[:1]
            pitch_hand = ((p.get("pitchHand") or {}).get("code") or "").upper()[:1]
    except Exception:
        pass

    _HAND_CACHE[pid] = (bat_side, pitch_hand)
    return bat_side, pitch_hand


def shrink_rate(successes: float, trials: float, prior_mean: float, prior_strength: int) -> float:
    a = prior_mean * prior_strength
    b = (1 - prior_mean) * prior_strength
    return (successes + a) / (trials + a + b)


def compute_batter_platoon_splits(stat_all: pd.DataFrame, prior_strength: int = 300) -> pd.DataFrame:
    """
    Build shrunk batter HR/PA splits vs RHP/LHP from Statcast training data.
    Requires columns: batter, events, p_throws
    Output columns:
      batter, hr_pa_overall_shrunk, hr_pa_vs_R_shrunk, hr_pa_vs_L_shrunk
    """
    req = {"batter", "events", "p_throws"}
    if stat_all is None or stat_all.empty or not req.issubset(set(stat_all.columns)):
        return pd.DataFrame(columns=["batter","hr_pa_overall_shrunk","hr_pa_vs_R_shrunk","hr_pa_vs_L_shrunk"])

    pa = stat_all[stat_all["events"].notna()].copy()
    if pa.empty:
        return pd.DataFrame(columns=["batter","hr_pa_overall_shrunk","hr_pa_vs_R_shrunk","hr_pa_vs_L_shrunk"])

    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    league_hr_pa = float(pa["is_hr"].sum() / max(1, len(pa)))

    overall = pa.groupby("batter").agg(PA=("events","size"), HR=("is_hr","sum")).reset_index()
    overall["hr_pa_overall_shrunk"] = overall.apply(
        lambda r: shrink_rate(r["HR"], r["PA"], prior_mean=league_hr_pa, prior_strength=prior_strength),
        axis=1,
    )

    pa["p_throws"] = pa["p_throws"].astype(str).str.upper().str[0]
    pa = pa[pa["p_throws"].isin(["R","L"])].copy()
    if pa.empty:
        out = overall[["batter","hr_pa_overall_shrunk"]].copy()
        out["hr_pa_vs_R_shrunk"] = np.nan
        out["hr_pa_vs_L_shrunk"] = np.nan
        return out

    split = pa.groupby(["batter","p_throws"]).agg(PA=("events","size"), HR=("is_hr","sum")).reset_index()

    league_split = split.groupby("p_throws").agg(PA=("PA","sum"), HR=("HR","sum")).reset_index()
    league_split["league_hr_pa"] = league_split["HR"] / league_split["PA"].clip(lower=1)
    league_map = dict(zip(league_split["p_throws"], league_split["league_hr_pa"]))

    split["hr_pa_shrunk"] = split.apply(
        lambda r: shrink_rate(
            r["HR"],
            r["PA"],
            prior_mean=float(league_map.get(r["p_throws"], league_hr_pa)),
            prior_strength=prior_strength,
        ),
        axis=1,
    )

    piv = split.pivot_table(index="batter", columns="p_throws", values="hr_pa_shrunk", aggfunc="mean").reset_index()
    piv = piv.rename(columns={"R": "hr_pa_vs_R_shrunk", "L": "hr_pa_vs_L_shrunk"})

    out = overall[["batter","hr_pa_overall_shrunk"]].merge(piv, on="batter", how="left")
    return out


def platoon_multiplier(batter_id: int, pitcher_id: int | None, platoon_map: dict) -> float:
    """
    Returns a clamped multiplier ~ [0.85, 1.15] based on batter split vs pitcher hand.
    platoon_map[batter_id] should contain:
      hr_pa_overall_shrunk, hr_pa_vs_R_shrunk, hr_pa_vs_L_shrunk
    """
    try:
        pid = int(pitcher_id) if pitcher_id else None
        pit_hand = "R"
        if pid:
            pit_hand = get_handedness(pid)[1] or "R"

        rec = platoon_map.get(int(batter_id), {})
        overall = float(rec.get("hr_pa_overall_shrunk", np.nan))
        vs_r = float(rec.get("hr_pa_vs_R_shrunk", np.nan))
        vs_l = float(rec.get("hr_pa_vs_L_shrunk", np.nan))

        if np.isnan(overall) or overall <= 0:
            return 1.0

        split = vs_r if pit_hand == "R" else vs_l
        if np.isnan(split) or split <= 0:
            return 1.0

        mult = split / overall
        mult = float(np.clip(mult, 0.85, 1.15))

        bat_side = get_handedness(int(batter_id))[0]
        if bat_side == "S":
            mult = float(np.clip(mult * 1.02, 0.85, 1.15))

        return mult
    except Exception:
        return 1.0
