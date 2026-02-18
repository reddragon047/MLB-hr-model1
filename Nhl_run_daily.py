import os, math, json, argparse
from datetime import date as date_cls
import numpy as np
import pandas as pd
import requests
from joblib import load

NHL_BASE = "https://api-web.nhle.com/v1"

def ensure_dirs():
    os.makedirs("outputs_nhl", exist_ok=True)

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

def play_by_play(game_id: int):
    return get_json(f"{NHL_BASE}/gamecenter/{game_id}/play-by-play")

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
        def toi_to_min(v):
            if v is None:
                return 0.0
            if isinstance(v, (int,float)):
                return float(v)
            if isinstance(v, str) and ":" in v:
                mm, ss = v.split(":")
                return float(mm) + float(ss)/60.0
            return 0.0
        toi = toi_to_min(p.get("toi") or p.get("timeOnIce") or p.get("timeOnIceTotal"))
        pp_toi = toi_to_min(p.get("powerPlayTimeOnIce") or p.get("ppToi") or p.get("ppTimeOnIce"))
        sog = p.get("shots") or p.get("sog") or p.get("shotsOnGoal")
        rows.append({
            "player_id": int(pid),
            "name": p.get("name") or f"{pid}",
            "pos": pos,
            "toi_min": toi,
            "pp_toi_min": pp_toi,
            "sog": float(sog) if sog is not None else np.nan
        })
    return pd.DataFrame(rows)

def select_candidates(team_df: pd.DataFrame):
    if team_df.empty:
        return team_df
    fwd = team_df[team_df["pos"].isin(["C","LW","RW","F"])].copy()
    d   = team_df[team_df["pos"] == "D"].copy()
    top6 = fwd.sort_values("toi_min", ascending=False).head(6)
    pp_f = fwd.sort_values("pp_toi_min", ascending=False).head(8)  # PP1/PP2 forwards proxy
    pp_d = d.sort_values("pp_toi_min", ascending=False).head(3)    # PP defensemen
    out = pd.concat([top6, pp_f, pp_d], ignore_index=True).drop_duplicates("player_id")
    return out

def xg_list_for_player(pbp: dict, player_id: int, base, iso, cols: list):
    plays = pbp.get("plays", []) or pbp.get("playsByPeriod", [])
    rows = []
    for p in plays:
        t = (p.get("typeDescKey") or p.get("typeCode") or "").lower()
        if "blocked" in t:
            continue
        if not any(k in t for k in ["shot","goal","missed"]):
            continue
        det = p.get("details", {}) or {}
        shooter = det.get("shootingPlayerId") or det.get("scoringPlayerId")
        if shooter is None or int(shooter) != int(player_id):
            continue
        x, y = det.get("xCoord"), det.get("yCoord")
        if x is None or y is None:
            continue
        dist, ang = shot_distance_angle(x, y)
        rows.append({
            "dist": dist,
            "angle": ang,
            "is_rebound": int(bool(det.get("isRebound"))),
            "is_rush": int(bool(det.get("isRush"))),
            "shot_type": (det.get("shotType") or "unknown").lower(),
            "situation": (det.get("situationCode") or det.get("strength") or "unknown").lower(),
        })
    if not rows:
        return []
    df = pd.DataFrame(rows)
    X_num = df[["dist","angle","is_rebound","is_rush"]].astype(float)
    X_cat = pd.get_dummies(df[["shot_type","situation"]], drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    X = X[cols]
    raw = np.clip(base.predict_proba(X)[:,1], 1e-6, 1-1e-6)
    cal = iso.predict(raw)
    return list(map(float, cal))

def mc_goal(exp_shots: float, mean_xg: float, n_sims: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    shots = rng.poisson(lam=max(exp_shots, 0.05), size=n_sims)
    p = float(np.clip(mean_xg, 1e-6, 0.9))
    goals = rng.binomial(n=shots, p=p)
    return float((goals>=1).mean()), float((goals>=2).mean()), float(goals.mean())

def write_outputs(board: pd.DataFrame, date_str: str):
    csv_path = f"outputs_nhl/goal_board_{date_str}.csv"
    html_path = f"outputs_nhl/goal_board_{date_str}.html"
    board.to_csv(csv_path, index=False)
    show = board.head(75).copy()
    show["p_goal_sim"] = (show["p_goal_sim"]*100).round(2)
    show["p_2plus_sim"] = (show["p_2plus_sim"]*100).round(2)
    html = f"""
    <html><head><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
    body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial; padding:14px; }}
    table {{ border-collapse:collapse; width:100%; }}
    th,td {{ border:1px solid #ddd; padding:8px; font-size:14px; }}
    th {{ background:#f4f4f4; position:sticky; top:0; }}
    tr:nth-child(even) {{ background:#fafafa; }}
    </style></head><body>
    <h2>NHL Goal Probability Board — {date_str}</h2>
    <p>Candidate pool: Top 6 forwards + PP forwards + PP defensemen. Probabilities are Monte Carlo.</p>
    {show.to_html(index=False, escape=True)}
    </body></html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=str(date_cls.today()))
    parser.add_argument("--sims", type=int, default=100000)
    args = parser.parse_args()

    ensure_dirs()

    if not os.path.exists("nhl_models/xg_base.joblib"):
        raise RuntimeError("xG model missing. Run nhl_train_xg.py first.")

    base = load("nhl_models/xg_base.joblib")
    iso = load("nhl_models/xg_calibrator.joblib")
    cols = json.load(open("nhl_models/xg_columns.json","r"))

    gids = schedule_game_ids(args.date)
    if not gids:
        pd.DataFrame([{"message": f"No NHL games found for {args.date}"}]).to_csv(
            f"outputs_nhl/NO_GAMES_{args.date}.csv", index=False
        )
        print("No games today.")
        return

    rows = []
    for gid in gids:
        box = boxscore(gid)
        pbp = play_by_play(gid)

        for side_key, side_label in [("homeTeam","HOME"), ("awayTeam","AWAY")]:
            team_df = extract_team_players(box, side_key)
            cands = select_candidates(team_df)

            league_sog = float(team_df["sog"].mean()) if team_df["sog"].notna().any() else 2.0

            for _, p in cands.iterrows():
                pid = int(p["player_id"])
                name = str(p["name"])
                pos = p["pos"]

                sog = p["sog"]
                exp_shots = (0.65*float(sog) + 0.35*league_sog) if not np.isnan(sog) else (0.9*league_sog)

                xgs = xg_list_for_player(pbp, pid, base, iso, cols)
                mean_xg = float(np.mean(xgs)) if len(xgs) >= 3 else (0.06 if pos in ["C","LW","RW","F"] else 0.045)

                p1, p2, mean_g = mc_goal(exp_shots, mean_xg, n_sims=args.sims)

                rows.append({
                    "date": args.date,
                    "game_id": gid,
                    "side": side_label,
                    "player": name,
                    "player_id": pid,
                    "pos": pos,
                    "exp_shots": round(exp_shots, 2),
                    "mean_xg_per_shot": round(mean_xg, 4),
                    "mean_goals_sim": round(mean_g, 4),
                    "p_goal_sim": p1,
                    "p_2plus_sim": p2,
                })

    board = pd.DataFrame(rows).sort_values("p_goal_sim", ascending=False).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board)+1))
    write_outputs(board, args.date)
    print(board.head(25).to_string(index=False))

if __name__ == "__main__":
    main()
