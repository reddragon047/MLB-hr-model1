import argparse
import os
from datetime import date as date_cls, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from pybaseball import statcast
import statsapi

LOOKBACK_DAYS = 30
HOT_WINDOW_DAYS = 7
MIN_PA = 20
MIN_BBE = 8
TOP_TEAM_BATS = 11
USER_AGENT = "Mozilla/5.0"

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
    "Kansas City Royals": "KC", "KC": "KC", "Royals": "KC",
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

# Park layer: HR factor is the main signal; cozy_score is an extra park-dimensions nudge.
PARK_DATA = {
    "ARI": {"park": "Chase Field", "lat": 33.4455, "lon": -112.0667, "tz": "America/Phoenix", "hr_factor": 1.01, "cozy_score": 0.50, "roof": "retractable"},
    "ATL": {"park": "Truist Park", "lat": 33.8908, "lon": -84.4677, "tz": "America/New_York", "hr_factor": 1.02, "cozy_score": 0.56, "roof": "open"},
    "BAL": {"park": "Oriole Park at Camden Yards", "lat": 39.2840, "lon": -76.6217, "tz": "America/New_York", "hr_factor": 0.97, "cozy_score": 0.45, "roof": "open"},
    "BOS": {"park": "Fenway Park", "lat": 42.3467, "lon": -71.0972, "tz": "America/New_York", "hr_factor": 1.01, "cozy_score": 0.62, "roof": "open"},
    "CHC": {"park": "Wrigley Field", "lat": 41.9484, "lon": -87.6553, "tz": "America/Chicago", "hr_factor": 1.00, "cozy_score": 0.52, "roof": "open"},
    "CWS": {"park": "Rate Field", "lat": 41.8300, "lon": -87.6338, "tz": "America/Chicago", "hr_factor": 1.04, "cozy_score": 0.57, "roof": "open"},
    "CIN": {"park": "Great American Ball Park", "lat": 39.0979, "lon": -84.5082, "tz": "America/New_York", "hr_factor": 1.10, "cozy_score": 0.72, "roof": "open"},
    "CLE": {"park": "Progressive Field", "lat": 41.4962, "lon": -81.6852, "tz": "America/New_York", "hr_factor": 0.98, "cozy_score": 0.48, "roof": "open"},
    "COL": {"park": "Coors Field", "lat": 39.7559, "lon": -104.9942, "tz": "America/Denver", "hr_factor": 1.12, "cozy_score": 0.67, "roof": "open"},
    "DET": {"park": "Comerica Park", "lat": 42.3390, "lon": -83.0485, "tz": "America/Detroit", "hr_factor": 0.95, "cozy_score": 0.35, "roof": "open"},
    "HOU": {"park": "Daikin Park", "lat": 29.7573, "lon": -95.3555, "tz": "America/Chicago", "hr_factor": 1.02, "cozy_score": 0.55, "roof": "retractable"},
    "KC": {"park": "Kauffman Stadium", "lat": 39.0517, "lon": -94.4803, "tz": "America/Chicago", "hr_factor": 0.94, "cozy_score": 0.33, "roof": "open"},
    "LAA": {"park": "Angel Stadium", "lat": 33.8003, "lon": -117.8827, "tz": "America/Los_Angeles", "hr_factor": 0.99, "cozy_score": 0.46, "roof": "open"},
    "LAD": {"park": "Dodger Stadium", "lat": 34.0739, "lon": -118.2400, "tz": "America/Los_Angeles", "hr_factor": 1.00, "cozy_score": 0.49, "roof": "open"},
    "MIA": {"park": "loanDepot park", "lat": 25.7781, "lon": -80.2207, "tz": "America/New_York", "hr_factor": 0.93, "cozy_score": 0.32, "roof": "retractable"},
    "MIL": {"park": "American Family Field", "lat": 43.0280, "lon": -87.9712, "tz": "America/Chicago", "hr_factor": 1.01, "cozy_score": 0.50, "roof": "retractable"},
    "MIN": {"park": "Target Field", "lat": 44.9817, "lon": -93.2783, "tz": "America/Chicago", "hr_factor": 0.98, "cozy_score": 0.43, "roof": "open"},
    "NYM": {"park": "Citi Field", "lat": 40.7571, "lon": -73.8458, "tz": "America/New_York", "hr_factor": 0.96, "cozy_score": 0.38, "roof": "open"},
    "NYY": {"park": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262, "tz": "America/New_York", "hr_factor": 1.08, "cozy_score": 0.68, "roof": "open"},
    "ATH": {"park": "Sutter Health Park", "lat": 38.5806, "lon": -121.5136, "tz": "America/Los_Angeles", "hr_factor": 1.00, "cozy_score": 0.50, "roof": "open"},
    "PHI": {"park": "Citizens Bank Park", "lat": 39.9061, "lon": -75.1665, "tz": "America/New_York", "hr_factor": 1.08, "cozy_score": 0.70, "roof": "open"},
    "PIT": {"park": "PNC Park", "lat": 40.4469, "lon": -80.0057, "tz": "America/New_York", "hr_factor": 0.92, "cozy_score": 0.31, "roof": "open"},
    "SD": {"park": "Petco Park", "lat": 32.7073, "lon": -117.1566, "tz": "America/Los_Angeles", "hr_factor": 0.92, "cozy_score": 0.29, "roof": "open"},
    "SF": {"park": "Oracle Park", "lat": 37.7786, "lon": -122.3893, "tz": "America/Los_Angeles", "hr_factor": 0.89, "cozy_score": 0.22, "roof": "open"},
    "SEA": {"park": "T-Mobile Park", "lat": 47.5914, "lon": -122.3325, "tz": "America/Los_Angeles", "hr_factor": 0.93, "cozy_score": 0.34, "roof": "retractable"},
    "STL": {"park": "Busch Stadium", "lat": 38.6226, "lon": -90.1928, "tz": "America/Chicago", "hr_factor": 0.95, "cozy_score": 0.37, "roof": "open"},
    "TB": {"park": "George M. Steinbrenner Field", "lat": 27.9809, "lon": -82.5067, "tz": "America/New_York", "hr_factor": 1.02, "cozy_score": 0.52, "roof": "open"},
    "TEX": {"park": "Globe Life Field", "lat": 32.7473, "lon": -97.0847, "tz": "America/Chicago", "hr_factor": 1.01, "cozy_score": 0.48, "roof": "retractable"},
    "TOR": {"park": "Rogers Centre", "lat": 43.6414, "lon": -79.3894, "tz": "America/Toronto", "hr_factor": 1.03, "cozy_score": 0.58, "roof": "retractable"},
    "WSH": {"park": "Nationals Park", "lat": 38.8730, "lon": -77.0074, "tz": "America/New_York", "hr_factor": 0.99, "cozy_score": 0.47, "roof": "open"},
}

PLAYER_NAME_CACHE = {}
PLAYER_ID_CACHE = {}


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


def clamp(val, lo, hi):
    return float(max(lo, min(hi, val)))


def parse_args():
    parser = argparse.ArgumentParser(description="Build a sharp daily HR board.")
    parser.add_argument("--date", type=str, default=str(date_cls.today()))
    return parser.parse_args()


def get_json(url: str):
    r = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
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


def lookup_player_id_by_name(name: str):
    if not name:
        return None
    if name in PLAYER_ID_CACHE:
        return PLAYER_ID_CACHE[name]
    try:
        candidates = statsapi.lookup_player(name)
    except Exception:
        candidates = []
    if candidates:
        try:
            pid = int(candidates[0]["id"])
            PLAYER_ID_CACHE[name] = pid
            return pid
        except Exception:
            return None
    return None


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


def get_schedule_rows(slate_date: str) -> pd.DataFrame:
    games = statsapi.schedule(date=slate_date)
    rows = []
    for g in games:
        home_abbr = resolve_team_abbr(g, "home")
        away_abbr = resolve_team_abbr(g, "away")
        if not home_abbr or not away_abbr:
            continue
        game_dt_utc = None
        raw_dt = g.get("game_datetime")
        if raw_dt:
            try:
                game_dt_utc = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
            except Exception:
                game_dt_utc = None
        rows.append({
            "game_id": g.get("game_id"),
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_name": g.get("home_name"),
            "away_name": g.get("away_name"),
            "home_pitcher_raw": g.get("home_probable_pitcher") or "",
            "away_pitcher_raw": g.get("away_probable_pitcher") or "",
            "game_datetime_utc": game_dt_utc,
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
        person = item.get("person", {}) if isinstance(item, dict) else {}
        position = item.get("position", {}) if isinstance(item, dict) else {}
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


def get_statcast_window(slate_date: str) -> pd.DataFrame:
    end = datetime.strptime(slate_date, "%Y-%m-%d").date()
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
    if df is None or df.empty:
        raise ValueError("statcast() returned no data.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def add_teams(sc: pd.DataFrame) -> pd.DataFrame:
    x = sc.copy()
    x["home_team"] = x.get("home_team", "").fillna("")
    x["away_team"] = x.get("away_team", "").fillna("")
    x["inning_topbot"] = x.get("inning_topbot", "").fillna("")
    top_mask = x["inning_topbot"].astype(str).str.lower().eq("top")
    x["batting_team"] = np.where(top_mask, x["away_team"], x["home_team"])
    x["pitching_team"] = np.where(top_mask, x["home_team"], x["away_team"])
    return x


def build_hitter_pool(sc: pd.DataFrame, slate_teams: set, slate_hitter_ids: set) -> pd.DataFrame:
    x = add_teams(sc)
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

    hot_cut = pd.Timestamp(datetime.strptime(str(x["game_date"].max().date()), "%Y-%m-%d") - timedelta(days=HOT_WINDOW_DAYS - 1))
    recent = bbe[bbe["game_date"] >= hot_cut].copy()

    agg = (
        bbe.groupby(["batter", "batting_team"])
        .agg(
            bbe_count=("launch_speed", "size"),
            avg_ev=("launch_speed", "mean"),
            avg_la=("launch_angle", "mean"),
            hr_count=("events", lambda s: (s == "home_run").sum()),
            hard_hit_rate=("launch_speed", lambda s: (num(s) >= 95).mean()),
            barrel_rate=barrel_rate,
        )
        .reset_index()
    )
    hot = (
        recent.groupby(["batter", "batting_team"])
        .agg(
            hot_bbe=("launch_speed", "size"),
            hot_ev=("launch_speed", "mean"),
            hot_hr_count=("events", lambda s: (s == "home_run").sum()),
            hot_hh_rate=("launch_speed", lambda s: (num(s) >= 95).mean()),
            hot_barrel_rate=barrel_rate,
        )
        .reset_index()
    ) if not recent.empty else pd.DataFrame(columns=["batter", "batting_team", "hot_bbe", "hot_ev", "hot_hr_count", "hot_hh_rate", "hot_barrel_rate"])

    hitters = agg.merge(pa, on=["batter", "batting_team"], how="left").merge(hot, on=["batter", "batting_team"], how="left")
    hitters["pa_count"] = num(hitters["pa_count"])
    hitters["recent_hr_rate"] = np.where(hitters["bbe_count"] > 0, hitters["hr_count"] / hitters["bbe_count"], 0.0)
    hitters["hot_hr_rate"] = np.where(num(hitters.get("hot_bbe", 0)) > 0, num(hitters.get("hot_hr_count", 0)) / num(hitters.get("hot_bbe", 0)), 0.0)
    hitters["hot_hh_rate"] = num(hitters.get("hot_hh_rate", 0))
    hitters["hot_barrel_rate"] = num(hitters.get("hot_barrel_rate", 0))
    hitters["hot_ev"] = num(hitters.get("hot_ev", 0))
    hitters["hot_bbe"] = num(hitters.get("hot_bbe", 0))

    hitters = hitters[(hitters["pa_count"] >= MIN_PA) & (hitters["bbe_count"] >= MIN_BBE)].copy()
    if hitters.empty:
        raise ValueError("All hitters were filtered out by MIN_PA / MIN_BBE.")

    hitters["team_bbe_rank"] = hitters.groupby("batting_team")["bbe_count"].rank(method="first", ascending=False)
    hitters = hitters[hitters["team_bbe_rank"] <= TOP_TEAM_BATS].copy()
    hitters["player_name"] = hitters["batter"].map(get_player_name)
    hitters = hitters[hitters["player_name"].astype(str).str.len() > 0].copy()

    hitters["ev_n"] = minmax(hitters["avg_ev"])
    hitters["la_n"] = minmax(hitters["avg_la"].clip(lower=5, upper=30))
    hitters["hh_n"] = minmax(hitters["hard_hit_rate"])
    hitters["hr_n"] = minmax(hitters["recent_hr_rate"])
    hitters["barrel_n"] = minmax(hitters["barrel_rate"])

    hitters["hot_ev_n"] = minmax(hitters["hot_ev"])
    hitters["hot_hh_n"] = minmax(hitters["hot_hh_rate"])
    hitters["hot_hr_n"] = minmax(hitters["hot_hr_rate"])
    hitters["hot_barrel_n"] = minmax(hitters["hot_barrel_rate"])

    hitters["base_score"] = (
        0.32 * hitters["barrel_n"] +
        0.24 * hitters["hh_n"] +
        0.18 * hitters["ev_n"] +
        0.08 * hitters["la_n"] +
        0.18 * hitters["hr_n"]
    )
    hitters["hot_bat_score"] = (
        0.35 * hitters["hot_barrel_n"] +
        0.30 * hitters["hot_hh_n"] +
        0.20 * hitters["hot_hr_n"] +
        0.15 * hitters["hot_ev_n"]
    )
    hitters["hot_bat_score"] = hitters["hot_bat_score"].fillna(0.0)
    return hitters


def build_pitcher_table(sc: pd.DataFrame) -> pd.DataFrame:
    x = add_teams(sc)
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
    p["pitcher_id"] = p["pitcher"].astype(int)
    p["pitcher_name"] = p["pitcher_id"].map(get_player_name)
    p["hr_rate_allowed"] = np.where(p["bbe_allowed"] > 0, p["hr_allowed"] / p["bbe_allowed"], 0.0)
    p["hr_n"] = minmax(p["hr_rate_allowed"])
    p["hh_n"] = minmax(p["hard_hit_allowed"])
    p["barrel_n"] = minmax(p["barrel_allowed"])
    p["la_n"] = minmax(p["avg_la_allowed"].clip(lower=5, upper=30))
    p["pitcher_vuln_score"] = 0.40 * p["hr_n"] + 0.30 * p["barrel_n"] + 0.20 * p["hh_n"] + 0.10 * p["la_n"]
    p["pitcher_mult"] = p["pitcher_vuln_score"].apply(lambda x: clamp(0.85 + 0.30 * x, 0.85, 1.15))
    return p[["pitcher_id", "pitcher_name", "bbe_allowed", "barrel_allowed", "hr_rate_allowed", "pitcher_vuln_score", "pitcher_mult"]]


def build_bullpen_table(sc: pd.DataFrame, starter_ids_by_team: dict) -> pd.DataFrame:
    x = add_teams(sc)
    x["pitcher"] = num(x["pitcher"], fill=np.nan)
    x = x[x["pitcher"].notna()].copy()
    x["pitcher"] = x["pitcher"].astype(int)
    x["launch_speed"] = num(x["launch_speed"])
    x["launch_angle"] = num(x["launch_angle"])
    x["events"] = x.get("events", "").fillna("")
    x["type"] = x.get("type", "").fillna("")
    x = x[x["pitching_team"].isin(starter_ids_by_team.keys())].copy()
    x["starter_id_for_team"] = x["pitching_team"].map(starter_ids_by_team)
    x = x[x["pitcher"] != x["starter_id_for_team"]].copy()
    bbe = x[x["type"] == "X"].copy()
    if bbe.empty:
        return pd.DataFrame(columns=["team", "bullpen_mult", "bullpen_score"])

    if "launch_speed_angle" in bbe.columns:
        bbe["launch_speed_angle"] = num(bbe["launch_speed_angle"], fill=np.nan)
        barrel_allowed = ("launch_speed_angle", lambda s: (num(s, fill=-1) == 6).mean())
    else:
        barrel_allowed = ("launch_speed", lambda s: 0.0)

    bp = (
        bbe.groupby("pitching_team")
        .agg(
            bbe_allowed=("launch_speed", "size"),
            avg_ev_allowed=("launch_speed", "mean"),
            hr_allowed=("events", lambda s: (s == "home_run").sum()),
            hh_allowed=("launch_speed", lambda s: (num(s) >= 95).mean()),
            barrel_allowed=barrel_allowed,
        )
        .reset_index()
        .rename(columns={"pitching_team": "team"})
    )
    bp["hr_rate_allowed"] = np.where(bp["bbe_allowed"] > 0, bp["hr_allowed"] / bp["bbe_allowed"], 0.0)
    bp["hr_n"] = minmax(bp["hr_rate_allowed"])
    bp["hh_n"] = minmax(bp["hh_allowed"])
    bp["barrel_n"] = minmax(bp["barrel_allowed"])
    bp["bullpen_score"] = 0.45 * bp["hr_n"] + 0.30 * bp["barrel_n"] + 0.25 * bp["hh_n"]
    bp["bullpen_mult"] = bp["bullpen_score"].apply(lambda x: clamp(0.92 + 0.16 * x, 0.92, 1.08))
    return bp[["team", "bbe_allowed", "hr_rate_allowed", "barrel_allowed", "bullpen_score", "bullpen_mult"]]


def get_hourly_weather(lat: float, lon: float, game_dt_utc: datetime, tz_name: str, roof_type: str):
    if roof_type == "retractable":
        # keep retractable roofs mostly neutral; avoids bad guesses on open/closed state.
        return {
            "temperature_2m": np.nan,
            "wind_speed_10m": np.nan,
            "wind_direction_10m": np.nan,
            "weather_score": 0.5,
            "weather_mult": 1.00,
            "weather_note": "retractable_roof_neutral",
        }
    if not game_dt_utc:
        return {
            "temperature_2m": np.nan,
            "wind_speed_10m": np.nan,
            "wind_direction_10m": np.nan,
            "weather_score": 0.5,
            "weather_mult": 1.00,
            "weather_note": "missing_game_time",
        }
    try:
        tz = ZoneInfo(tz_name)
        local_dt = game_dt_utc.astimezone(tz)
        target_hour = local_dt.replace(minute=0, second=0, microsecond=0)
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,wind_speed_10m,wind_direction_10m"
            f"&timezone={tz_name}"
            f"&start_date={local_dt.date().isoformat()}&end_date={local_dt.date().isoformat()}"
        )
        j = get_json(url)
        hourly = j.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            raise ValueError("No hourly times returned")
        target_iso = target_hour.strftime("%Y-%m-%dT%H:00")
        if target_iso in times:
            idx = times.index(target_iso)
        else:
            dt_times = [datetime.fromisoformat(t) for t in times]
            idx = min(range(len(dt_times)), key=lambda i: abs((dt_times[i] - target_hour.replace(tzinfo=None)).total_seconds()))
        temp = float(hourly.get("temperature_2m", [np.nan])[idx])
        wind = float(hourly.get("wind_speed_10m", [np.nan])[idx])
        wdir = float(hourly.get("wind_direction_10m", [np.nan])[idx])

        # Mild weather score: warm + moderate wind bumps HR environment.
        temp_score = clamp((temp - 55.0) / 30.0, 0.0, 1.0)
        wind_score = clamp(wind / 20.0, 0.0, 1.0)
        weather_score = 0.65 * temp_score + 0.35 * wind_score
        weather_mult = clamp(0.95 + 0.10 * weather_score, 0.95, 1.05)
        return {
            "temperature_2m": temp,
            "wind_speed_10m": wind,
            "wind_direction_10m": wdir,
            "weather_score": weather_score,
            "weather_mult": weather_mult,
            "weather_note": "forecast",
        }
    except Exception as e:
        return {
            "temperature_2m": np.nan,
            "wind_speed_10m": np.nan,
            "wind_direction_10m": np.nan,
            "weather_score": 0.5,
            "weather_mult": 1.00,
            "weather_note": f"weather_fallback:{type(e).__name__}",
        }


def get_park_context(home_team: str, game_dt_utc: datetime):
    park = PARK_DATA.get(home_team)
    if not park:
        return {
            "park_name": home_team,
            "park_hr_factor": 1.0,
            "park_dim_score": 0.5,
            "park_mult": 1.0,
            "roof": "unknown",
            **get_hourly_weather(0, 0, None, "UTC", "unknown"),
        }
    park_raw = 0.70 * clamp((park["hr_factor"] - 0.90) / 0.22, 0.0, 1.0) + 0.30 * park["cozy_score"]
    park_mult = clamp(0.94 + 0.12 * park_raw, 0.94, 1.06)
    weather = get_hourly_weather(park["lat"], park["lon"], game_dt_utc, park["tz"], park["roof"])
    return {
        "park_name": park["park"],
        "park_hr_factor": park["hr_factor"],
        "park_dim_score": park["cozy_score"],
        "park_mult": park_mult,
        "roof": park["roof"],
        **weather,
    }


def grade_from_score(x: float) -> str:
    if x >= 0.86:
        return "A+"
    if x >= 0.75:
        return "A"
    if x >= 0.64:
        return "B"
    if x >= 0.53:
        return "C"
    return "D"


def build_daily_board(hitters: pd.DataFrame, pitchers: pd.DataFrame, bullpens: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    pitcher_mult_by_id = dict(zip(pitchers["pitcher_id"], pitchers["pitcher_mult"]))
    pitcher_vuln_by_id = dict(zip(pitchers["pitcher_id"], pitchers["pitcher_vuln_score"]))
    bullpen_mult_by_team = dict(zip(bullpens["team"], bullpens["bullpen_mult"]))
    bullpen_score_by_team = dict(zip(bullpens["team"], bullpens["bullpen_score"]))
    rows = []

    for _, game in schedule_df.iterrows():
        home_pid = lookup_player_id_by_name(game["home_pitcher_raw"])
        away_pid = lookup_player_id_by_name(game["away_pitcher_raw"])
        park_ctx = get_park_context(game["home_team"], game.get("game_datetime_utc"))

        # Away hitters face home starter + home bullpen.
        away_hitters = hitters[hitters["batting_team"] == game["away_team"]].copy()
        away_pitcher_mult = float(pitcher_mult_by_id.get(home_pid, 1.0))
        away_pitcher_score = float(pitcher_vuln_by_id.get(home_pid, 0.5))
        away_bullpen_mult = float(bullpen_mult_by_team.get(game["home_team"], 1.0))
        away_bullpen_score = float(bullpen_score_by_team.get(game["home_team"], 0.5))
        for _, h in away_hitters.iterrows():
            env_score = 0.60 * park_ctx["park_dim_score"] + 0.40 * park_ctx["weather_score"]
            trust_score = clamp((0.55 * min(h["pa_count"] / 120.0, 1.0) + 0.45 * min(h["bbe_count"] / 40.0, 1.0)), 0.0, 1.0)
            bet_grade_score = (
                0.35 * float(h["hot_bat_score"]) +
                0.30 * away_pitcher_score +
                0.15 * away_bullpen_score +
                0.10 * env_score +
                0.10 * trust_score
            )
            rows.append({
                "player_name": h["player_name"],
                "batter_id": int(h["batter"]),
                "team": h["batting_team"],
                "opp_team": game["home_team"],
                "probable_pitcher_faced": game["home_pitcher_raw"],
                "park_name": park_ctx["park_name"],
                "roof": park_ctx["roof"],
                "temp_f": round(park_ctx["temperature_2m"] * 9 / 5 + 32, 1) if pd.notna(park_ctx["temperature_2m"]) else np.nan,
                "wind_speed_mph": round(park_ctx["wind_speed_10m"] * 0.621371, 1) if pd.notna(park_ctx["wind_speed_10m"]) else np.nan,
                "pa_count": int(h["pa_count"]),
                "bbe_count": int(h["bbe_count"]),
                "avg_ev": round(float(h["avg_ev"]), 2),
                "avg_la": round(float(h["avg_la"]), 2),
                "hard_hit_rate": round(float(h["hard_hit_rate"]), 4),
                "barrel_rate": round(float(h["barrel_rate"]), 4),
                "recent_hr_rate": round(float(h["recent_hr_rate"]), 4),
                "hot_bat_score": round(float(h["hot_bat_score"]), 4),
                "starter_vuln_score": round(away_pitcher_score, 4),
                "bullpen_score": round(away_bullpen_score, 4),
                "park_mult": round(float(park_ctx["park_mult"]), 4),
                "weather_mult": round(float(park_ctx["weather_mult"]), 4),
                "bullpen_mult": round(away_bullpen_mult, 4),
                "pitcher_mult": round(away_pitcher_mult, 4),
                "base_score": round(float(h["base_score"]), 4),
                "final_score": float(h["base_score"] * away_pitcher_mult * park_ctx["park_mult"] * park_ctx["weather_mult"] * away_bullpen_mult),
                "bet_grade_score": round(float(bet_grade_score), 4),
                "bet_grade": grade_from_score(float(bet_grade_score)),
            })

        # Home hitters face away starter + away bullpen.
        home_hitters = hitters[hitters["batting_team"] == game["home_team"]].copy()
        home_pitcher_mult = float(pitcher_mult_by_id.get(away_pid, 1.0))
        home_pitcher_score = float(pitcher_vuln_by_id.get(away_pid, 0.5))
        home_bullpen_mult = float(bullpen_mult_by_team.get(game["away_team"], 1.0))
        home_bullpen_score = float(bullpen_score_by_team.get(game["away_team"], 0.5))
        for _, h in home_hitters.iterrows():
            env_score = 0.60 * park_ctx["park_dim_score"] + 0.40 * park_ctx["weather_score"]
            trust_score = clamp((0.55 * min(h["pa_count"] / 120.0, 1.0) + 0.45 * min(h["bbe_count"] / 40.0, 1.0)), 0.0, 1.0)
            bet_grade_score = (
                0.35 * float(h["hot_bat_score"]) +
                0.30 * home_pitcher_score +
                0.15 * home_bullpen_score +
                0.10 * env_score +
                0.10 * trust_score
            )
            rows.append({
                "player_name": h["player_name"],
                "batter_id": int(h["batter"]),
                "team": h["batting_team"],
                "opp_team": game["away_team"],
                "probable_pitcher_faced": game["away_pitcher_raw"],
                "park_name": park_ctx["park_name"],
                "roof": park_ctx["roof"],
                "temp_f": round(park_ctx["temperature_2m"] * 9 / 5 + 32, 1) if pd.notna(park_ctx["temperature_2m"]) else np.nan,
                "wind_speed_mph": round(park_ctx["wind_speed_10m"] * 0.621371, 1) if pd.notna(park_ctx["wind_speed_10m"]) else np.nan,
                "pa_count": int(h["pa_count"]),
                "bbe_count": int(h["bbe_count"]),
                "avg_ev": round(float(h["avg_ev"]), 2),
                "avg_la": round(float(h["avg_la"]), 2),
                "hard_hit_rate": round(float(h["hard_hit_rate"]), 4),
                "barrel_rate": round(float(h["barrel_rate"]), 4),
                "recent_hr_rate": round(float(h["recent_hr_rate"]), 4),
                "hot_bat_score": round(float(h["hot_bat_score"]), 4),
                "starter_vuln_score": round(home_pitcher_score, 4),
                "bullpen_score": round(home_bullpen_score, 4),
                "park_mult": round(float(park_ctx["park_mult"]), 4),
                "weather_mult": round(float(park_ctx["weather_mult"]), 4),
                "bullpen_mult": round(home_bullpen_mult, 4),
                "pitcher_mult": round(home_pitcher_mult, 4),
                "base_score": round(float(h["base_score"]), 4),
                "final_score": float(h["base_score"] * home_pitcher_mult * park_ctx["park_mult"] * park_ctx["weather_mult"] * home_bullpen_mult),
                "bet_grade_score": round(float(bet_grade_score), 4),
                "bet_grade": grade_from_score(float(bet_grade_score)),
            })

    board = pd.DataFrame(rows)
    if board.empty:
        raise ValueError("No board rows were created.")
    board = board.sort_values(["final_score", "bet_grade_score", "bbe_count", "pa_count"], ascending=[False, False, False, False]).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    return board


def write_outputs(board: pd.DataFrame, debug: pd.DataFrame, slate_date: str):
    board.to_csv("hr_board.csv", index=False)
    debug.to_csv("hr_debug.csv", index=False)
    os.makedirs("outputs", exist_ok=True)
    board.to_csv(f"outputs/hr_board_{slate_date}.csv", index=False)
    debug.to_csv(f"outputs/hr_debug_{slate_date}.csv", index=False)


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

    starter_ids_by_team = {}
    for _, g in schedule_df.iterrows():
        starter_ids_by_team[g["home_team"]] = lookup_player_id_by_name(g["home_pitcher_raw"])
        starter_ids_by_team[g["away_team"]] = lookup_player_id_by_name(g["away_pitcher_raw"])
    bullpens = build_bullpen_table(sc, starter_ids_by_team)

    board = build_daily_board(hitters, pitchers, bullpens, schedule_df)

    debug = pd.DataFrame([{
        "slate_date": slate_date,
        "games": len(schedule_df),
        "slate_teams": len(slate_teams),
        "slate_hitter_ids": len(slate_hitter_ids),
        "raw_statcast_rows": len(sc),
        "qualified_hitters": len(hitters),
        "pitchers_in_table": len(pitchers),
        "bullpen_teams": len(bullpens),
        "board_rows": len(board),
        "lookback_days": LOOKBACK_DAYS,
        "hot_window_days": HOT_WINDOW_DAYS,
        "min_pa_filter": MIN_PA,
        "min_bbe_filter": MIN_BBE,
    }])

    write_outputs(board, debug, slate_date)
    print("DONE")
    print(debug.to_dict(orient="records")[0])
    print(board.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
