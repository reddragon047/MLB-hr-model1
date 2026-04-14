
import argparse
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from pybaseball import statcast
import statsapi

LOOKBACK_DAYS = 30
MIN_BBE = 8

def num(s, fill=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(fill)

def minmax(s):
    s = num(s, 0.0)
    if len(s)==0: return s
    lo,hi=s.min(),s.max()
    if hi==lo: return pd.Series([0.5]*len(s))
    return (s-lo)/(hi-lo)

def run():
    slate_date = date.today().isoformat()

    end = datetime.strptime(slate_date,"%Y-%m-%d").date()
    start = end - timedelta(days=LOOKBACK_DAYS)
    sc = statcast(start_dt=start.isoformat(), end_dt=end.isoformat())

    sc["launch_speed"]=num(sc["launch_speed"])
    sc["launch_angle"]=num(sc["launch_angle"])
    sc["events"]=sc["events"].fillna("")
    sc["type"]=sc["type"].fillna("")

    bbe = sc[sc["type"]=="X"]

    hitters = bbe.groupby("player_name").agg(
        bbe=("launch_speed","size"),
        ev=("launch_speed","mean"),
        la=("launch_angle","mean"),
        hr=("events",lambda s:(s=="home_run").sum()),
        hh=("launch_speed",lambda s:(s>=95).mean())
    ).reset_index()

    hitters = hitters[hitters["bbe"]>=MIN_BBE]
    hitters["score"]=(
        0.4*minmax(hitters["ev"]) +
        0.3*minmax(hitters["hh"]) +
        0.3*minmax(hitters["hr"]/hitters["bbe"])
    )

    hitters = hitters.sort_values("score",ascending=False)
    hitters.to_csv("hr_board.csv",index=False)

    pd.DataFrame([{"rows":len(hitters)}]).to_csv("hr_debug.csv",index=False)

if __name__=="__main__":
    run()
