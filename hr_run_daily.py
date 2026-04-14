
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pybaseball import statcast

LOOKBACK_DAYS = 30

def get_statcast_window():
    end = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
    if df is None or df.empty:
        raise ValueError("statcast() returned no data.")
    return df

def num(s, fill=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(fill)

def minmax(s):
    s = num(s, 0.0)
    if len(s) == 0:
        return s
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - lo) / (hi - lo)

def build_board(sc):
    debug = {}

    # Keep only contacted balls if available
    if "type" in sc.columns:
        x = sc[sc["type"] == "X"].copy()
    else:
        x = sc.copy()

    debug["raw_rows"] = len(sc)
    debug["contact_rows"] = len(x)

    if x.empty:
        raise ValueError("No contacted-ball rows found in statcast window.")

    x["player_name"] = x["player_name"].fillna("UNKNOWN")
    x["launch_speed"] = num(x.get("launch_speed", 0))
    x["launch_angle"] = num(x.get("launch_angle", 0))
    x["events"] = x.get("events", pd.Series([""] * len(x))).fillna("")

    hitters = x.groupby("player_name", dropna=True).agg(
        bbe_count=("launch_speed", "size"),
        avg_ev=("launch_speed", "mean"),
        avg_la=("launch_angle", "mean"),
        hr_count=("events", lambda s: (s == "home_run").sum()),
        hard_hit_rate=("launch_speed", lambda s: (num(s) >= 95).mean()),
    ).reset_index()

    debug["grouped_hitters"] = len(hitters)

    if hitters.empty:
        raise ValueError("Grouped hitters dataframe is empty.")

    hitters["recent_hr_rate"] = np.where(hitters["bbe_count"] > 0, hitters["hr_count"] / hitters["bbe_count"], 0.0)
    hitters["ev_n"] = minmax(hitters["avg_ev"])
    hitters["la_n"] = minmax(hitters["avg_la"].clip(lower=5, upper=30))
    hitters["hh_n"] = minmax(hitters["hard_hit_rate"])
    hitters["hr_n"] = minmax(hitters["recent_hr_rate"])

    # Barrel proxy so we don't depend on brittle Savant fields
    hitters["barrel_proxy"] = 0.7 * hitters["ev_n"] + 0.3 * hitters["la_n"]

    hitters["score"] = (
        0.35 * hitters["barrel_proxy"] +
        0.25 * hitters["hh_n"] +
        0.20 * hitters["ev_n"] +
        0.10 * hitters["la_n"] +
        0.10 * hitters["hr_n"]
    )

    hitters = hitters.sort_values(["score", "bbe_count"], ascending=[False, False]).reset_index(drop=True)
    board = hitters.rename(columns={"player_name": "player"})[
        ["player", "score", "bbe_count", "avg_ev", "avg_la", "hard_hit_rate", "recent_hr_rate"]
    ]

    debug["final_rows"] = len(board)
    return board, debug

def run():
    sc = get_statcast_window()
    board, debug = build_board(sc)

    board.to_csv("hr_board.csv", index=False)
    pd.DataFrame([debug]).to_csv("hr_debug.csv", index=False)

    print("DONE")
    print(debug)

if __name__ == "__main__":
    run()
