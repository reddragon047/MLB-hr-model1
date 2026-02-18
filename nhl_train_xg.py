import os
import math
import json
import argparse
import time
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

# Base domain (schedule uses /v1/schedule/...; gamecenter pbp uses /v1/gamecenter/... )
NHL_BASE = "https://api-web.nhle.com"


def ensure_dirs():
    os.makedirs("nhl_models", exist_ok=True)


def get_json(url: str, timeout: int = 20, retries: int = 8):
    """
    Robust fetch with:
    - User-Agent header
    - retry/backoff for 429/5xx
    - jitter to avoid request bursts
    - throws a clear RuntimeError on exhaustion
    """
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (GitHubActions; NHLxGTrainer)"},
            )

            # Rate limit / transient server issues
            if r.status_code in (429, 502, 503, 504):
                sleep_time = (10 + i * 6) + random.random() * 2
                print(f"[{r.status_code}] {url} -> sleep {sleep_time:.1f}s, retry {i+1}/{retries}")
                time.sleep(sleep_time)
                continue

            r.raise_for_status()

            # Small throttle between successful calls
            time.sleep(0.6 + random.random() * 0.4)
            return r.json()

        except Exception as e:
            last_err = e
            sleep_time = (6 + i * 5) + random.random() * 2
            print(f"[ERR] {url} -> {e} | sleep {sleep_time:.1f}s, retry {i+1}/{retries}")
            time.sleep(sleep_time)

    raise RuntimeError(f"Failed to fetch after {retries} retries: {url}. Last error: {last_err}")


def shot_distance_angle(x: float, y: float):
    """
    NHL coords: center ice near (0,0); nets near x=±89.
    Use abs(x), abs(y) so we always measure to the nearest net.
    """
    ax = abs(float(x))
    ay = abs(float(y))
    dx = 89.0 - ax
    dy = ay
    dist = math.sqrt(dx * dx + dy * dy)
    ang = math.degrees(math.atan2(dy, max(dx, 1e-6)))  # 0..90
    return dist, ang


def extract_shots_from_pbp(pbp: dict) -> pd.DataFrame:
    """
    Extract unblocked shot attempts + goals (labeled) with coords.
    Excludes blocked shots to avoid shooter/coord ambiguity.
    """
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
        x = details.get("xCoord")
        y = details.get("yCoord")
        if shooter_id is None or x is None or y is None:
            continue

        is_goal = 1 if ("goal" in t and "no-goal" not in t) else 0

        dist, ang = shot_distance_angle(x, y)
        shot_type = (details.get("shotType") or "unknown").lower()
        is_rebound = int(bool(details.get("isRebound")))
        is_rush = int(bool(details.get("isRush")))
        situation = (details.get("situationCode") or details.get("strength") or "unknown").lower()

        rows.append(
            {
                "is_goal": is_goal,
                "dist": dist,
                "angle": ang,
                "shot_type": shot_type,
                "is_rebound": is_rebound,
                "is_rush": is_rush,
                "situation": situation,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["is_goal", "dist", "angle", "shot_type", "is_rebound", "is_rush", "situation"])

    return pd.DataFrame(rows)


def get_schedule(date_str: str):
    return get_json(f"{NHL_BASE}/v1/schedule/{date_str}")


def find_game_ids(schedule_json: dict):
    ids = []
    for day in schedule_json.get("gameWeek", []):
        for g in day.get("games", []):
            gid = g.get("id")
            if gid:
                ids.append(int(gid))
    return sorted(set(ids))


def get_pbp(game_id: int):
    return get_json(f"{NHL_BASE}/v1/gamecenter/{game_id}/play-by-play")


def build_training_shots(days_back: int) -> pd.DataFrame:
    """
    Pull shots from last N days (rolling window).
    Skips games that fail to fetch (rate limit / transient).
    """
    end_dt = datetime.utcnow()
    all_rows = []
    games_total = 0
    games_ok = 0

    for i in range(days_back):
        d = (end_dt - timedelta(days=i)).date()
        ds = str(d)
        try:
            sched = get_schedule(ds)
        except Exception as e:
            print(f"Skipping schedule {ds} due to fetch error: {e}")
            continue

        game_ids = find_game_ids(sched)
        if not game_ids:
            continue

        print(f"{ds}: {len(game_ids)} games")
        for gid in game_ids:
            games_total += 1
            try:
                pbp = get_pbp(gid)
            except Exception as e:
                print(f"  - skip game {gid} pbp fetch error: {e}")
                continue

            df = extract_shots_from_pbp(pbp)
            if not df.empty:
                all_rows.append(df)
                games_ok += 1

            # extra throttle every few games
            if games_total % 8 == 0:
                time.sleep(1.5 + random.random() * 1.0)

    print(f"Games attempted: {games_total}, games with shots: {games_ok}")

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    return out


def train_xg_model(shots: pd.DataFrame):
    """
    Train shot-level xG:
      P(goal | dist, angle, rebound, rush, shot_type, situation)
    """
    df = shots.copy()
    if df.empty:
        raise RuntimeError("No shot data collected. Try a smaller window during active season or re-run later.")

    df["shot_type"] = df["shot_type"].fillna("unknown").astype(str)
    df["situation"] = df["situation"].fillna("unknown").astype(str)

    X_num = df[["dist", "angle", "is_rebound", "is_rush"]].astype(float)
    X_cat = pd.get_dummies(df[["shot_type", "situation"]], drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df["is_goal"].astype(int)

    # Guard: if y has only 0s or 1s, model can't train
    if y.nunique() < 2:
        raise RuntimeError("Training data has only one class (all goals or all non-goals). Increase days_back.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic regression is fast and stable; calibrate with isotonic regression
    base = LogisticRegression(max_iter=2000, n_jobs=1)
    base.fit(X_train, y_train)

    raw_val = np.clip(base.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_val)

    meta = {
        "trained_utc": datetime.utcnow().isoformat() + "Z",
        "n_shots": int(len(df)),
        "goal_rate": float(y.mean()),
        "columns": list(X.columns),
    }
    return base, iso, meta, list(X.columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days_back", type=int, default=30, help="Days of games to use for training")
    args = parser.parse_args()

    ensure_dirs()

    print(f"Training window: last {args.days_back} days")
    shots = build_training_shots(args.days_back)

    print(f"Collected shots: {len(shots)}")
    if shots.empty:
        raise RuntimeError("No shots collected. Try re-running later, or reduce days_back, or run during season.")

    base, iso, meta, cols = train_xg_model(shots)

    dump(base, "nhl_models/xg_base.joblib")
    dump(iso, "nhl_models/xg_calibrator.joblib")
    with open("nhl_models/xg_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open("nhl_models/xg_columns.json", "w", encoding="utf-8") as f:
        json.dump(cols, f)

    print("Saved xG model artifacts to nhl_models/")


if __name__ == "__main__":
    main()
