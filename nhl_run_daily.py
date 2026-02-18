import os
import math
import json
import argparse
from datetime import date as date_cls
import numpy as np
import pandas as pd
import requests
from joblib import load

NHL_BASE = "https://api-web.nhle.com/v1"

def ensure_dirs():
    os.makedirs("outputs_nhl", exist_ok=True)

def get_json(url: str, timeout: int = 30):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

def schedule_game_ids(date_str: str):
    sched = get_json(f"{NHL_BASE}/schedule/{date_str}")
    ids = []
    for day in sched.get("gameWeek", []):
        for g in day.get("games", []):
            if g.get("id"):
                ids.append(int(g["id"]))
    return sorted(set(ids))

def boxscore(game_id: int):
    return get_json(f"{NHL_BASE}/gamecenter/{game_id}/boxscore")

def extract_team_players(box: dict, side_key: str):
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

        # Robust SOG extraction
        sog = (
            p.get("shots")
            or p.get("sog")
            or p.get("shotsOnGoal")
            or 0
        )

        rows.append({
            "player_id": int(pid),
            "name": p.get("name") or f"{pid}",
            "pos": pos,
            "toi_min": 0.0,
            "pp_toi_min": 0.0,
            "sog": float(sog)
        })

    return pd.DataFrame(rows)

def select_candidates(team_df: pd.DataFrame):
    if team_df.empty:
        return team_df

    fwd = team_df[team_df["pos"].isin(["C","LW","RW","F"])].copy()
    d   = team_df[team_df["pos"] == "D"].copy()

    top6 = fwd.head(6)
    pp_f = fwd.head(8)
    pp_d = d.head(3)

    return pd.concat([top6, pp_f, pp_d], ignore_index=True).drop_duplicates("player_id")

def mc_goal(exp_shots: float, mean_xg: float, n_sims: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    shots = rng.poisson(lam=max(exp_shots, 0.1), size=n_sims)
    p = float(np.clip(mean_xg, 1e-6, 0.9))
    goals = rng.binomial(n=shots, p=p)
    return float((goals>=1).mean()), float((goals>=2).mean()), float(goals.mean())

def write_outputs(board: pd.DataFrame, date_str: str):
    csv_path = f"outputs_nhl/goal_board_{date_str}.csv"
    html_path = f"outputs_nhl/goal_board_{date_str}.html"

    board.to_csv(csv_path, index=False)

    show = board.head(75).copy()
    show["p_goal_sim"] = (show["p_goal_sim"]*100).round(2)

    html = f"""
    <html><head><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
    body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto; padding:14px; }}
    table {{ border-collapse:collapse; width:100%; }}
    th,td {{ border:1px solid #ddd; padding:8px; font-size:14px; }}
    th {{ background:#f4f4f4; position:sticky; top:0; }}
    </style></head><body>
    <h2>NHL Goal Probability Board — {date_str}</h2>
    {show.to_html(index=False, escape=True)}
    </body></html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=str(date_cls.today()))
    parser.add_argument("--sims", type=int, default=50000)
    args = parser.parse_args()

    ensure_dirs()

    gids = schedule_game_ids(args.date)

    if not gids:
        pd.DataFrame([{"message": f"No NHL games found for {args.date}"}]).to_csv(
            f"outputs_nhl/NO_GAMES_{args.date}.csv", index=False
        )
        return

    rows = []

    for gid in gids:
        box = boxscore(gid)

        for side_key, side_label in [("homeTeam","HOME"), ("awayTeam","AWAY")]:
            team_df = extract_team_players(box, side_key)
            cands = select_candidates(team_df)

            if team_df.empty:
                continue

            league_sog = float(team_df["sog"].mean()) if "sog" in team_df.columns else 2.0

            for _, p in cands.iterrows():
                exp_shots = max(p["sog"], league_sog)
                mean_xg = 0.08 if p["pos"] in ["C","LW","RW","F"] else 0.05

                p1, p2, mean_g = mc_goal(exp_shots, mean_xg, args.sims)

                rows.append({
                    "date": args.date,
                    "game_id": gid,
                    "side": side_label,
                    "player": p["name"],
                    "pos": p["pos"],
                    "exp_shots": round(exp_shots,2),
                    "p_goal_sim": p1,
                })

    board = pd.DataFrame(rows).sort_values("p_goal_sim", ascending=False).reset_index(drop=True)
    write_outputs(board, args.date)

if __name__ == "__main__":
    main()
