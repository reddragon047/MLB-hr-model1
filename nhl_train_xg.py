import os, math, json, argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

NHL_BASE = "https://api-web.nhle.com"

def ensure_dirs():
    os.makedirs("nhl_models", exist_ok=True)

def get_json(url: str, timeout: int = 30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def shot_distance_angle(x: float, y: float):
    ax, ay = abs(float(x)), abs(float(y))
    dx, dy = 89.0 - ax, ay
    dist = math.sqrt(dx*dx + dy*dy)
    ang = math.degrees(math.atan2(dy, max(dx, 1e-6)))
    return dist, ang

def extract_shots_from_pbp(pbp: dict) -> pd.DataFrame:
    plays = pbp.get("plays", []) or pbp.get("playsByPeriod", [])
    rows = []
    for p in plays:
        t = (p.get("typeDescKey") or p.get("typeCode") or "").lower()
        if "blocked" in t:
            continue
        if not any(k in t for k in ["shot", "goal", "missed"]):
            continue
        details = p.get("details", {}) or {}
        shooter_id = details.get("shootingPlayerId") or details.get("scoringPlayerId")
        x, y = details.get("xCoord"), details.get("yCoord")
        if shooter_id is None or x is None or y is None:
            continue
        is_goal = 1 if ("goal" in t and "no-goal" not in t) else 0
        dist, ang = shot_distance_angle(x, y)
        shot_type = (details.get("shotType") or "unknown").lower()
        is_rebound = int(bool(details.get("isRebound")))
        is_rush = int(bool(details.get("isRush")))
        situation = (details.get("situationCode") or details.get("strength") or "unknown").lower()
        rows.append({
            "is_goal": is_goal,
            "dist": dist,
            "angle": ang,
            "shot_type": shot_type,
            "is_rebound": is_rebound,
            "is_rush": is_rush,
            "situation": situation,
        })
    return pd.DataFrame(rows)

def get_schedule(date_str: str):
    return get_json(f"{NHL_BASE}/v1/schedule/{date_str}")

def find_game_ids(schedule_json: dict):
    ids = []
    for day in schedule_json.get("gameWeek", []):
        for g in day.get("games", []):
            if g.get("id"):
                ids.append(int(g["id"]))
    return sorted(set(ids))

def get_pbp(game_id: int):
    return get_json(f"{NHL_BASE}/v1/gamecenter/{game_id}/play-by-play")

def build_training_shots(days_back: int) -> pd.DataFrame:
    end_dt = datetime.utcnow()
    all_rows = []
    for i in range(days_back):
        d = (end_dt - timedelta(days=i)).date()
        sched = get_schedule(str(d))
        for gid in find_game_ids(sched):
            pbp = get_pbp(gid)
            df = extract_shots_from_pbp(pbp)
            if not df.empty:
                all_rows.append(df)
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)

def train_xg(shots: pd.DataFrame):
    shots = shots.copy()
    shots["shot_type"] = shots["shot_type"].fillna("unknown").astype(str)
    shots["situation"] = shots["situation"].fillna("unknown").astype(str)

    X_num = shots[["dist","angle","is_rebound","is_rush"]].astype(float)
    X_cat = pd.get_dummies(shots[["shot_type","situation"]], drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
    y = shots["is_goal"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    base = LogisticRegression(max_iter=2000, n_jobs=1)
    base.fit(X_train, y_train)

    raw_val = np.clip(base.predict_proba(X_val)[:,1], 1e-6, 1-1e-6)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_val)

    meta = {
        "trained_utc": datetime.utcnow().isoformat() + "Z",
        "n_shots": int(len(shots)),
        "goal_rate": float(y.mean()),
        "columns": list(X.columns)
    }
    return base, iso, meta, list(X.columns)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days_back", type=int, default=120)
    args = parser.parse_args()

    ensure_dirs()
    shots = build_training_shots(args.days_back)
    if shots.empty:
        raise RuntimeError("No shots collected. Try increasing days_back or run during season.")
    base, iso, meta, cols = train_xg(shots)

    dump(base, "nhl_models/xg_base.joblib")
    dump(iso, "nhl_models/xg_calibrator.joblib")
    with open("nhl_models/xg_meta.json","w") as f:
        json.dump(meta, f, indent=2)
    with open("nhl_models/xg_columns.json","w") as f:
        json.dump(cols, f)

    print("Saved xG model to nhl_models/")

if __name__ == "__main__":
    main()
