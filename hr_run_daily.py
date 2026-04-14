
import argparse
import collections
import unicodedata
from datetime import date as date_cls, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from pybaseball import statcast
import statsapi

LOOKBACK_DAYS = 30
MIN_PA = 20
MIN_BBE = 8

TEAM_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI", "Diamondbacks": "ARI", "D-backs": "ARI",
    "Atlanta Braves": "ATL", "Braves": "ATL",
    "Baltimore Orioles": "BAL", "Orioles": "BAL",
    "Boston Red Sox": "BOS", "Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Cubs": "CHC",
    "Chicago White Sox": "CWS", "White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Reds": "CIN",
    "Cleveland Guardians": "CLE", "Guardians": "CLE",
    "Colorado Rockies": "COL", "Rockies": "COL",
    "Detroit Tigers": "DET", "Tigers": "DET",
    "Houston Astros": "HOU", "Astros": "HOU",
    "Kansas City Royals": "KC", "Royals": "KC",
    "Los Angeles Angels": "LAA", "Angels": "LAA",
    "Los Angeles Dodgers": "LAD", "Dodgers": "LAD",
    "Miami Marlins": "MIA", "Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Brewers": "MIL",
    "Minnesota Twins": "MIN", "Twins": "MIN",
    "New York Mets": "NYM", "Mets": "NYM",
    "New York Yankees": "NYY", "Yankees": "NYY",
    "Athletics": "ATH", "Oakland Athletics": "ATH", "A's": "ATH",
    "Philadelphia Phillies": "PHI", "Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "Pirates": "PIT",
    "San Diego Padres": "SD", "Padres": "SD",
    "San Francisco Giants": "SF", "Giants": "SF",
    "Seattle Mariners": "SEA", "Mariners": "SEA",
    "St. Louis Cardinals": "STL", "Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Rays": "TB",
    "Texas Rangers": "TEX", "Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Blue Jays": "TOR",
    "Washington Nationals": "WSH", "Nationals": "WSH",
}

PLAYER_NAME_CACHE = {}

def num(series, fill=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(fill)

def minmax(series):
    s = num(series, 0.0)
    if len(s) == 0:
        return s
    lo = s.min()
    hi = s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - lo) / (hi - lo)

def normalize_name(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace(".", "").replace("'", "")
    s = " ".join(s.split())
    return s

def parse_args():
    parser = argparse.ArgumentParser(description="Build daily HR board.")
    parser.add_argument("--date", type=str, default=str(date_cls.today()))
    return parser.parse_args()

def get_json(url: str):
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

def resolve_team_abbr(game_dict, side: str):
    candidates = [
        game_dict.get(f"{side}_abbr"),
        game_dict.get(f"{side}_team_abbr"),
        game_dict.get(f"{side}_name"),
        game_dict.get(f"{side}_team_name"),
    ]
    for c in candidates:
        if not c:
            continue
        if c in TEAM_NAME_TO_ABBR:
            return TEAM_NAME_TO_ABBR[c]
        if isinstance(c, str) and len(c) in (2, 3):
            return c.upper()
    return None

def get_schedule_rows(slate_date: str) -> pd.DataFrame:
    games = statsapi.schedule(date=slate_date)
    rows = []
    for g in games:
        home_abbr = resolve_team_abbr(g, "home")
        away_abbr = resolve_team_abbr(g, "away")
        if not home_abbr or not away_abbr:
            continue
        rows.append({
            "game_id": g.get("game_id"),
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_name": g.get("home_name"),
            "away_name": g.get("away_name"),
            "home_pitcher_raw": g.get("home_probable_pitcher") or "",
            "away_pitcher_raw": g.get("away_probable_pitcher") or "",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No scheduled games found for {slate_date}.")
    return df

def get_team_hitters(team_name_or_abbr: str):
    teams = statsapi.lookup_team(team_name_or_abbr)
    if not teams:
        teams = statsapi.lookup_team(TEAM_NAME_TO_ABBR.get(team_name_or_abbr, team_name_or_abbr))
    if not teams:
        return []
    team_id = teams[0]["id"]
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
    j = get_json(url)
    roster = j.get("roster", []) or []
    hitters = []
    for item in roster:
        if not isinstance(item, dict):
            continue
        person = item.get("person", {}) if isinstance(item.get("person", {}), dict) else {}
        position = item.get("position", {}) if isinstance(item.get("position", {}), dict) else {}
        pos_abbr = str(position.get("abbreviation") or "").upper()
        if pos_abbr == "P":
            continue
        pid = person.get("id")
        if pid is None:
            continue
        try:
            hitters.append(int(pid))
        except Exception:
            pass
    return sorted(set(hitters))

def get_player_name(player_id: int) -> str:
    if player_id in PLAYER_NAME_CACHE:
        return PLAYER_NAME_CACHE[player_id]
    try:
        info = statsapi.get("person", {"personId": int(player_id)})
        people = info.get("people", []) or []
        if people:
            name = people[0].get("fullName") or ""
            PLAYER_NAME_CACHE[player_id] = name
            return name
    except Exception:
        pass
    PLAYER_NAME_CACHE[player_id] = ""
    return ""

def lookup_player_id_by_name(name: str):
    if not name:
        return None
    try:
        candidates = statsapi.lookup_player(name)
    except Exception:
        candidates = []
    key = normalize_name(name)
    for cand in candidates:
        full = normalize_name(cand.get("fullName", ""))
        if full == key:
            return int(cand["id"])
    if candidates:
        try:
            return int(candidates[0]["id"])
        except Exception:
            return None
    return None

def get_statcast_window(slate_date: str) -> pd.DataFrame:
    end = datetime.strptime(slate_date, "%Y-%m-%d").date()
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
    if df is None or df.empty:
        raise ValueError("statcast() returned no data.")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def add_batting_team(sc: pd.DataFrame) -> pd.DataFrame:
    x = sc.copy()
    x["home_team"] = x.get("home_team", "").fillna("")
    x["away_team"] = x.get("away_team", "").fillna("")
    x["inning_topbot"] = x.get("inning_topbot", "").fillna("")
    x["batting_team"] = np.where(
        x["inning_topbot"].astype(str).str.lower().eq("top"),
        x["away_team"],
        x["home_team"],
    )
    return x

def build_hitter_pool(sc: pd.DataFrame, slate_teams: set, slate_hitter_ids: set) -> pd.DataFrame:
    x = add_batting_team(sc)
    x = x[x["batting_team"].isin(slate_teams)].copy()
    x["batter"] = num(x["batter"], fill=np.nan)
    x = x[x["batter"].notna()].copy()
    x["batter"] = x["batter"].astype(int)
    x = x[x["batter"].isin(slate_hitter_ids)].copy()

    if x.empty:
        raise ValueError("No Statcast rows matched today's slate hitters.")

    x["launch_speed"] = num(x["launch_speed"])
    x["launch_angle"] = num(x["launch_angle"])
    x["events"] = x.get("events", "").fillna("")
    x["type"] = x.get("type", "").fillna("")
    x["game_date"] = pd.to_datetime(x.get("game_date"), errors="coerce")

    pa = (
        x[["batter", "batting_team", "game_pk", "at_bat_number"]]
        .dropna()
        .drop_duplicates()
        .groupby(["batter", "batting_team"])
        .size()
        .reset_index(name="pa_count")
    )

    bbe = x[x["type"] == "X"].copy()
    if bbe.empty:
        raise ValueError("No contacted-ball rows found for today's slate hitters.")

    if "launch_speed_angle" in bbe.columns:
        bbe["launch_speed_angle"] = num(bbe["launch_speed_angle"], fill=np.nan)
        barrel_rate = ("launch_speed_angle", lambda s: (num(s, fill=-1) == 6).mean())
    else:
        barrel_rate = ("launch_speed", lambda s: 0.0)

    agg = (
        bbe.groupby(["batter", "batting_team"])
        .agg(
            bbe_count=("launch_speed", "size"),
            avg_ev=("launch_speed", "mean"),
            avg_la=("launch_angle", "mean"),
            hr_count=("events", lambda s: (s == "home_run").sum()),
            hard_hit_rate=("launch_speed", lambda s: (num(s) >= 95).mean()),
            barrel_rate=barrel_rate,
            recent_game_date=("game_date", "max"),
        )
        .reset_index()
    )

    hitters = agg.merge(pa, on=["batter", "batting_team"], how="left")
    hitters["pa_count"] = num(hitters["pa_count"])
    hitters["recent_hr_rate"] = np.where(hitters["bbe_count"] > 0, hitters["hr_count"] / hitters["bbe_count"], 0.0)

    hitters = hitters[(hitters["pa_count"] >= MIN_PA) & (hitters["bbe_count"] >= MIN_BBE)].copy()
    if hitters.empty:
        raise ValueError("All hitters were filtered out by MIN_PA / MIN_BBE.")

    hitters["team_bbe_rank"] = hitters.groupby("batting_team")["bbe_count"].rank(method="first", ascending=False)
    hitters = hitters[hitters["team_bbe_rank"] <= 11].copy()

    hitters["player_name"] = hitters["batter"].map(get_player_name)
    hitters = hitters[hitters["player_name"].astype(str).str.len() > 0].copy()

    hitters["ev_n"] = minmax(hitters["avg_ev"])
    hitters["la_n"] = minmax(hitters["avg_la"].clip(lower=5, upper=30))
    hitters["hh_n"] = minmax(hitters["hard_hit_rate"])
    hitters["hr_n"] = minmax(hitters["recent_hr_rate"])
    hitters["barrel_n"] = minmax(hitters["barrel_rate"])

    hitters["base_score"] = (
        0.35 * hitters["barrel_n"] +
        0.25 * hitters["hh_n"] +
        0.20 * hitters["ev_n"] +
        0.10 * hitters["la_n"] +
        0.10 * hitters["hr_n"]
    )
    return hitters

def build_pitcher_table(sc: pd.DataFrame) -> pd.DataFrame:
    x = sc.copy()
    x["pitcher"] = num(x["pitcher"], fill=np.nan)
    x = x[x["pitcher"].notna()].copy()
    x["pitcher"] = x["pitcher"].astype(int)
    x["launch_speed"] = num(x["launch_speed"])
    x["launch_angle"] = num(x["launch_angle"])
    x["events"] = x.get("events", "").fillna("")
    x["type"] = x.get("type", "").fillna("")
    bbe = x[x["type"] == "X"].copy()

    if bbe.empty:
        return pd.DataFrame(columns=["pitcher_id", "pitcher_name", "pitcher_mult"])

    if "launch_speed_angle" in bbe.columns:
        bbe["launch_speed_angle"] = num(bbe["launch_speed_angle"], fill=np.nan)
        barrel_allowed = ("launch_speed_angle", lambda s: (num(s, fill=-1) == 6).mean())
    else:
        barrel_allowed = ("launch_speed", lambda s: 0.0)

    p = (
        bbe.groupby("pitcher")
        .agg(
            bbe_allowed=("launch_speed", "size"),
            avg_ev_allowed=("launch_speed", "mean"),
            avg_la_allowed=("launch_angle", "mean"),
            hr_allowed=("events", lambda s: (s == "home_run").sum()),
            hard_hit_allowed=("launch_speed", lambda s: (num(s) >= 95).mean()),
            barrel_allowed=barrel_allowed,
        )
        .reset_index()
    )

    p["hr_rate_allowed"] = np.where(p["bbe_allowed"] > 0, p["hr_allowed"] / p["bbe_allowed"], 0.0)
    p["hr_n"] = minmax(p["hr_rate_allowed"])
    p["hh_n"] = minmax(p["hard_hit_allowed"])
    p["barrel_n"] = minmax(p["barrel_allowed"])
    p["la_n"] = minmax(p["avg_la_allowed"].clip(lower=5, upper=30))
    raw = 0.40 * p["hr_n"] + 0.30 * p["barrel_n"] + 0.20 * p["hh_n"] + 0.10 * p["la_n"]
    p["pitcher_mult"] = np.clip(0.85 + 0.30 * raw, 0.85, 1.15)
    p["pitcher_id"] = p["pitcher"].astype(int)
    p["pitcher_name"] = p["pitcher_id"].map(get_player_name)
    return p[["pitcher_id", "pitcher_name", "pitcher_mult"]]

def build_daily_board(hitters: pd.DataFrame, pitchers: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    pitcher_mult_by_id = dict(zip(pitchers["pitcher_id"], pitchers["pitcher_mult"]))
    rows = []

    for _, game in schedule_df.iterrows():
        home_pid = lookup_player_id_by_name(game["home_pitcher_raw"])
        away_pid = lookup_player_id_by_name(game["away_pitcher_raw"])
        away_mult = float(pitcher_mult_by_id.get(home_pid, 1.0))
        home_mult = float(pitcher_mult_by_id.get(away_pid, 1.0))

        away_hitters = hitters[hitters["batting_team"] == game["away_team"]].copy()
        for _, h in away_hitters.iterrows():
            rows.append({
                "player_name": h["player_name"],
                "batter_id": int(h["batter"]),
                "team": h["batting_team"],
                "opp_team": game["home_team"],
                "probable_pitcher_faced": game["home_pitcher_raw"],
                "pa_count": int(h["pa_count"]),
                "bbe_count": int(h["bbe_count"]),
                "avg_ev": round(float(h["avg_ev"]), 2),
                "avg_la": round(float(h["avg_la"]), 2),
                "hard_hit_rate": round(float(h["hard_hit_rate"]), 4),
                "barrel_rate": round(float(h["barrel_rate"]), 4),
                "recent_hr_rate": round(float(h["recent_hr_rate"]), 4),
                "base_score": round(float(h["base_score"]), 4),
                "pitcher_mult": round(float(away_mult), 4),
                "rank_score": float(h["base_score"] * away_mult),
            })

        home_hitters = hitters[hitters["batting_team"] == game["home_team"]].copy()
        for _, h in home_hitters.iterrows():
            rows.append({
                "player_name": h["player_name"],
                "batter_id": int(h["batter"]),
                "team": h["batting_team"],
                "opp_team": game["away_team"],
                "probable_pitcher_faced": game["away_pitcher_raw"],
                "pa_count": int(h["pa_count"]),
                "bbe_count": int(h["bbe_count"]),
                "avg_ev": round(float(h["avg_ev"]), 2),
                "avg_la": round(float(h["avg_la"]), 2),
                "hard_hit_rate": round(float(h["hard_hit_rate"]), 4),
                "barrel_rate": round(float(h["barrel_rate"]), 4),
                "recent_hr_rate": round(float(h["recent_hr_rate"]), 4),
                "base_score": round(float(h["base_score"]), 4),
                "pitcher_mult": round(float(home_mult), 4),
                "rank_score": float(h["base_score"] * home_mult),
            })

    board = pd.DataFrame(rows)
    if board.empty:
        raise ValueError("No board rows were created.")
    board = board.sort_values(["rank_score", "bbe_count", "pa_count"], ascending=[False, False, False]).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    return board

def write_outputs(board: pd.DataFrame, debug: pd.DataFrame, slate_date: str):
    board.to_csv("hr_board.csv", index=False)
    debug.to_csv("hr_debug.csv", index=False)
    # Back-compat outputs folder if it exists
    try:
        import os
        os.makedirs("outputs", exist_ok=True)
        board.to_csv(f"outputs/hr_board_{slate_date}.csv", index=False)
        debug.to_csv(f"outputs/hr_debug_{slate_date}.csv", index=False)
    except Exception:
        pass

def main():
    args = parse_args()
    slate_date = args.date.strip() or str(date_cls.today())
    schedule_df = get_schedule_rows(slate_date)
    slate_teams = set(schedule_df["home_team"]).union(set(schedule_df["away_team"]))

    slate_hitter_ids = set()
    for t in sorted(slate_teams):
        slate_hitter_ids.update(get_team_hitters(t))
    if not slate_hitter_ids:
        raise ValueError("Could not build today's hitter pool from team rosters.")

    sc = get_statcast_window(slate_date)
    hitters = build_hitter_pool(sc, slate_teams, slate_hitter_ids)
    pitchers = build_pitcher_table(sc)
    board = build_daily_board(hitters, pitchers, schedule_df)

    debug = pd.DataFrame([{
        "slate_date": slate_date,
        "games": len(schedule_df),
        "slate_teams": len(slate_teams),
        "slate_hitter_ids": len(slate_hitter_ids),
        "raw_statcast_rows": len(sc),
        "qualified_hitters": len(hitters),
        "pitchers_in_table": len(pitchers),
        "board_rows": len(board),
        "min_pa_filter": MIN_PA,
        "min_bbe_filter": MIN_BBE,
    }])

    write_outputs(board, debug, slate_date)
    print("DONE")
    print(debug.to_dict(orient="records")[0])
    print(board.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
