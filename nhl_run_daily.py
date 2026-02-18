import os
import math
import json
import time
import random
import argparse
from datetime import date as date_cls

import numpy as np
import pandas as pd
import requests
from joblib import load

NHL_BASE = "https://api-web.nhle.com/v1"


def ensure_dirs():
    os.makedirs("outputs_nhl", exist_ok=True)


def get_json(url: str, timeout: int = 20, retries: int = 6):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (NHLGoalBoard)"})
            if r.status_code in (429, 502, 503, 504):
                sleep_time = (8 + i * 5) + random.random() * 2
                print(f"[{r.status_code}] {url} -> sleep {sleep_time:.1f}s then retry")
                time.sleep(sleep_time)
                continue
            r.raise_for_status()
            time.sleep(0.4 + random.random() * 0.3)
            return r.json()
        except Exception as e:
            last_err = e
            sleep_time = (4 + i * 4) + random.random() * 2
            print(f"[ERR] {url} -> {e} | sleep {sleep_time:.1f}s then retry")
            time.sleep(sleep_time)
    raise RuntimeError(f"Failed to fetch after {retries} retries: {url}. Last error: {last_err}")


def schedule_game_ids(date_str: str):
    """
    NHL schedule JSON can appear in a few shapes depending on endpoint version.
    We search for game IDs in multiple places.
    """
    sched = get_json(f"{NHL_BASE}/schedule/{date_str}")

    ids = set()

    # Common: sched["gameWeek"][...]["games"][...]["id"]
    for day in sched.get("gameWeek", []) or []:
        for g in day.get("games", []) or []:
            gid = g.get("id") or g.get("gameId")
            if gid:
                ids.add(int(gid))

    # Sometimes: sched["games"] is directly present
    for g in sched.get("games", []) or []:
        gid = g.get("id") or g.get("gameId")
        if gid:
            ids.add(int(gid))

    # Sometimes: sched["gameDay"] / "days"
    for day in sched.get("days", []) or []:
        for g in day.get("games", []) or []:
            gid = g.get("id") or g.get("gameId")
            if gid:
                ids.add(int(gid))

    # Debug help (shows you what came back if no games found)
    if not ids:
        print("Schedule keys:", list(sched.keys())[:25])

    return sorted(ids)


def boxscore(game_id: int):
    return get_json(f"{NHL_BASE}/gamecenter/{game_id}/boxscore")


def shot_distance_angle(x: float, y: float):
    ax, ay = abs(float(x)), abs(float(y))
    dx, dy = 89.0 - ax, ay
    dist = math.sqrt(dx * dx + dy * dy)
    ang = math.degrees(math.atan2(dy, max(dx, 1e-6)))
    return dist, ang


def play_by_play(game_id: int):
    return get_json(f"{NHL_BASE}/gamecenter/{game_id}/play-by-play")


def extract_team_players(box: dict, side_key: str) -> pd.DataFrame:
    """
    Pull skaters from boxscore. Field names vary, so we handle lots of variants.
    """
    team = box.get(side_key, {}) or {}
    players = team.get("players", {}) or {}

    rows = []
    for _, p in players.items():
        pos = (p.get("position") or "").upper()
        if pos == "G":
            continue

        pid = p.get("playerId") or p.get("id")
        if not pid:
            continue

        # Robust SOG extraction (field names vary)
        sog = p.get("shots")
        if sog is None:
            sog = p.get("shotsOnGoal")
        if sog is None:
            sog = p.get("sog")
        if sog is None:
            sog = 0

        # Name fields vary
        name = p.get("name")
        if not name:
            fn = p.get("firstName", "") or ""
            ln = p.get("lastName", "") or ""
            name = (fn + " " + ln).strip() or str(pid)

        rows.append(
            {
                "player_id": int(pid),
                "name": name,
                "pos": pos,
                "sog": float(sog),
            }
        )

    return pd.DataFrame(rows)


def select_candidates(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Candidate pool: Top 6 forwards + PP1/PP2 forwards proxy + PP defensemen proxy.

    With only boxscore info, we can't perfectly reconstruct PP units.
    For v1, we:
      - take first 6 forwards by appearance order (usually top of box list)
      - take first 8 forwards as PP proxy
      - take first 3 D as PP D proxy
    (We can upgrade to proper TOI/PPTOI inference later.)
    """
    if team_df.empty:
        return team_df

    fwd = team_df[team_df["pos"].isin(["C", "LW", "RW", "F"])].copy()
    d = team_df[team_df["pos"] == "D"].copy()

    top6 = fwd.head(6)
    pp_f = fwd.head(8)
    pp_d = d.head(3)

    out = pd.concat([top6, pp_f, pp_d], ignore_index=True).drop_duplicates("player_id")
    return out


def xg_for_shots(pbp: dict, player_id: int, xg_base, xg_iso, xg_cols: list):
    """
    Use trained xG model to score a player's shots from the game pbp.
    If pbp is pregame/empty, returns [].
    """
    plays = pbp.get("plays", []) or pbp.get("playsByPeriod", [])
    rows = []
    for p in plays:
        t = (p.get("typeDescKey") or p.get("typeCode") or "").lower()
        if "blocked" in t:
            continue
        if not any(k in t for k in ["shot", "goal", "missed"]):
            continue

        det = p.get("details", {}) or {}
        shooter = det.get("shootingPlayerId") or det.get("scoringPlayerId")
        if shooter is None or int(shooter) != int(player_id):
            continue

        x, y = det.get("xCoord"), det.get("yCoord")
        if x is None or y is None:
            continue

        dist, ang = shot_distance_angle(x, y)
        rows.append(
            {
                "dist": dist,
                "angle": ang,
                "is_rebound": int(bool(det.get("isRebound"))),
                "is_rush": int(bool(det.get("isRush"))),
                "shot_type": (det.get("shotType") or "unknown").lower(),
                "situation": (det.get("situationCode") or det.get("strength") or "unknown").lower(),
            }
        )

    if not rows:
        return []

    df = pd.DataFrame(rows)
    X_num = df[["dist", "angle", "is_rebound", "is_rush"]].astype(float)
    X_cat = pd.get_dummies(df[["shot_type", "situation"]], drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)

    for c in xg_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[xg_cols]

    raw = np.clip(xg_base.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    cal = xg_iso.predict(raw)
    return list(map(float, cal))


def monte_carlo_goal_prob(exp_shots: float, mean_xg_per_shot: float, n_sims: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    shots = rng.poisson(lam=max(exp_shots, 0.05), size=n_sims)
    p = float(np.clip(mean_xg_per_shot, 1e-6, 0.9))
    goals = rng.binomial(n=shots, p=p)
    p1 = float((goals >= 1).mean())
    p2 = float((goals >= 2).mean())
    return p1, p2, float(goals.mean())


def write_outputs(board: pd.DataFrame, slate_date: str, top_n: int = 75):
    csv_path = f"outputs_nhl/goal_board_{slate_date}.csv"
    html_path = f"outputs_nhl/goal_board_{slate_date}.html"
    board.to_csv(csv_path, index=False)

    show = board.head(top_n).copy()
    show["p_goal_sim"] = (show["p_goal_sim"] * 100).round(2)
    show["p_2plus_sim"] = (show["p_2plus_sim"] * 100).round(2)

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
      <h2>NHL Goal Probability Board — {slate_date} (Top {top_n})</h2>
      <p>p_goal_sim and p_2plus_sim are percentages (Monte Carlo). Candidate pool: Top 6 forwards + PP forwards + PP defensemen.</p>
      {show.to_html(index=False, escape=True)}
    </body>
    </html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=str(date_cls.today()))
    parser.add_argument("--sims", type=int, default=50000)
    args = parser.parse_args()

    ensure_dirs()

    # Require trained xG model
    if not os.path.exists("nhl_models/xg_base.joblib"):
        raise RuntimeError("Missing nhl_models/xg_base.joblib. Run NHL Train xG Weekly first.")

    xg_base = load("nhl_models/xg_base.joblib")
    xg_iso = load("nhl_models/xg_calibrator.joblib")
    xg_cols = json.load(open("nhl_models/xg_columns.json", "r", encoding="utf-8"))

    game_ids = schedule_game_ids(args.date)

    if not game_ids:
        # This could mean: date is valid but parsing missed it, or schedule not available yet
        pd.DataFrame([{"message": f"No games parsed for {args.date}. (Date may be valid; check schedule parsing.)"}]).to_csv(
            f"outputs_nhl/NO_GAMES_{args.date}.csv", index=False
        )
        print(f"No games parsed for {args.date}")
        return

    rows = []
    for gid in game_ids:
        box = boxscore(gid)

        # NOTE: play-by-play may be empty pregame; we handle that with fallback mean_xg
        pbp = {}
        try:
            pbp = play_by_play(gid)
        except Exception as e:
            print(f"Skipping pbp for game {gid}: {e}")

        for side_key, side_label in [("homeTeam", "HOME"), ("awayTeam", "AWAY")]:
            team_df = extract_team_players(box, side_key)
            if team_df.empty:
                continue

            cands = select_candidates(team_df)

            league_sog = float(team_df["sog"].mean()) if "sog" in team_df.columns else 2.0

            for _, p in cands.iterrows():
                pid = int(p["player_id"])
                name = str(p["name"])
                pos = str(p["pos"])

                # Expected shots: regress to team mean
                exp_shots = 0.65 * float(p["sog"]) + 0.35 * league_sog

                # Mean xG/shot: use pbp shots if available; otherwise fallback by position
                xg_list = xg_for_shots(pbp, pid, xg_base, xg_iso, xg_cols) if pbp else []
                if len(xg_list) >= 3:
                    mean_xg = float(np.mean(xg_list))
                else:
                    mean_xg = 0.06 if pos in ["C", "LW", "RW", "F"] else 0.045

                p1, p2, mean_goals = monte_carlo_goal_prob(exp_shots, mean_xg, n_sims=args.sims)

                rows.append(
                    {
                        "date": args.date,
                        "game_id": gid,
                        "side": side_label,
                        "player": name,
                        "player_id": pid,
                        "pos": pos,
                        "exp_shots": round(exp_shots, 2),
                        "mean_xg_per_shot": round(mean_xg, 4),
                        "mean_goals_sim": round(mean_goals, 4),
                        "p_goal_sim": p1,
                        "p_2plus_sim": p2,
                    }
                )

    if not rows:
        pd.DataFrame([{"message": f"No player rows produced for {args.date} (games parsed: {len(game_ids)})."}]).to_csv(
            f"outputs_nhl/NO_DATA_{args.date}.csv", index=False
        )
        print("No player rows produced.")
        return

    board = pd.DataFrame(rows).sort_values("p_goal_sim", ascending=False).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))

    write_outputs(board, slate_date=args.date, top_n=75)
    print(board.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
