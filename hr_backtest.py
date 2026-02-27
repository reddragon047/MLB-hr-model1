import os
import argparse
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

# We reuse your model code by importing build_board()
# Make sure hr_run_daily.py is in the repo root (same folder).
from hr_run_daily import ensure_dirs, build_board


def daterange(start_date: str, end_date: str):
    d0 = datetime.strptime(start_date, "%Y-%m-%d").date()
    d1 = datetime.strptime(end_date, "%Y-%m-%d").date()
    cur = d0
    while cur <= d1:
        yield str(cur)
        cur += timedelta(days=1)


def american_to_implied(odds: float) -> float:
    # odds like +350 or -120
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def score_predictions(df: pd.DataFrame) -> dict:
    """
    df requires columns: p (model prob), y (0/1 outcome)
    """
    p = np.clip(df["p"].astype(float).values, 1e-6, 1 - 1e-6)
    y = df["y"].astype(int).values

    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    # AUC (only valid if both classes present)
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = np.nan
    # simple calibration: bucket probabilities
    bins = np.linspace(0, 1, 11)
    dfc = df.copy()
    dfc["bin"] = pd.cut(dfc["p"], bins=bins, include_lowest=True)
    cal = (
        dfc.groupby("bin", observed=True)
        .agg(n=("y", "size"), avg_p=("p", "mean"), hr_rate=("y", "mean"))
        .reset_index()
    )

    out = {
        "n": int(len(df)),
        "brier": brier,
        "logloss": logloss,
        "auc": auc,
    }
    return out, cal

# =============================
# Lift + Top-N Diagnostics
# =============================

def decile_lift_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Bucket predictions into quantile bins (deciles by default)
    and compare predicted vs actual HR rate.
    """
    x = df.dropna(subset=["p", "y"]).copy()
    x["p"] = x["p"].astype(float)
    x["y"] = x["y"].astype(int)

    baseline = x["y"].mean()

    x["bin"] = pd.qcut(x["p"], q=n_bins, duplicates="drop")

    g = (
        x.groupby("bin", observed=True)
         .agg(
             n=("y", "size"),
             avg_p=("p", "mean"),
             hr_rate=("y", "mean"),
             min_p=("p", "min"),
             max_p=("p", "max"),
         )
         .reset_index()
         .sort_values("avg_p", ascending=False)
    )   

    g["baseline_hr_rate"] = baseline
    g["lift_vs_baseline"] = g["hr_rate"] / baseline if baseline > 0 else np.nan

    return g

def topn_hit_table(df: pd.DataFrame, topn_list=(10, 25, 50, 75)) -> pd.DataFrame:
    """
    For each date:
        - take Top N by p
        - compute HR rate in Top N
        - compute capture rate
        - compute hit-any rate
    """
    x = df.dropna(subset=["date", "p", "y"]).copy()
    x["p"] = x["p"].astype(float)
    x["y"] = x["y"].astype(int)
    x["date"] = x["date"].astype(str)

    out_rows = []

    for d, g in x.groupby("date"):
        g = g.sort_values("p", ascending=False)
        total_hrs = int(g["y"].sum())

        for N in topn_list:
            top= g.head(int(N))
            top_hrs = int(top["y"].sum())

            out_rows.append({
                "date": d,
                "topN": int(N),
                "n_in_topN": int(len(top)),
                "topN_hr_rate": float(top["y"].mean()) if len(top) else np.nan,
                "topN_hrs": top_hrs,
                "total_hrs": total_hrs,
                "capture_rate": float(top_hrs / total_hrs) if total_hrs > 0 else np.nan,
                "hit_any": int(top_hrs > 0),
            })

    daily = pd.DataFrame(out_rows)

    if daily.empty:
        return daily

    summary = (
        daily.groupby("topN", as_index=False)
             .agg(
                 days=("date", "nunique"),
                 avg_topN_hr_rate=("topN_hr_rate", "mean"),
                 total_topN_hrs=("topN_hrs", "sum"),
                 total_hrs=("total_hrs", "sum"),
                 avg_capture_rate=("capture_rate", "mean"),
                 hit_any_rate=("hit_any", "mean"),
             )
    )
    summary["overall_capture_rate"] = (
        summary["total_topN_hrs"] /
        summary["total_hrs"].replace(0, np.nan)
    )

    return summary
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--sims", type=int, default=50000)
    parser.add_argument("--seasons", type=str, default="2023,2024,2025")
    parser.add_argument("--weather", action="store_true")
    parser.add_argument("--odds_csv", type=str, default="", help="Optional: inputs/odds_history.csv")
    args = parser.parse_args()

    ensure_dirs()

    train_seasons = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]

    all_preds = []
    daily_summary = []

    # Optional odds history
    odds_df = None
    if args.odds_csv and os.path.exists(args.odds_csv):
        odds_df = pd.read_csv(args.odds_csv)
        # expected columns: date, player_name, odds_american
        odds_df["date"] = odds_df["date"].astype(str)

    for d in daterange(args.start, args.end):
        # Build board (this also pulls schedule, etc.)
        board = build_board(d, n_sims=args.sims, train_seasons=train_seasons, use_weather=args.weather)
        if board.empty:
            continue

        # We backtest on "did the player homer (>=1) on that date?"
        # We can derive outcomes from Statcast events by reusing cached Statcast pulls from your pipeline
        # Simplest robust proxy: Use pybaseball statcast for that day and mark batters who homered.
        # NOTE: This is heavier; keep date ranges small at first.
        try:
            from pybaseball import statcast
            day = statcast(start_dt=d, end_dt=d)

            # --- INSERT PA BLOCK HERE (LINE 99)---
            pa_df = (
                day.dropna(subset=["batter", "game_pk", "at_bat_number"])
                    .drop_duplicates(subset=["game_pk", "at_bat_number", "batter"])
                    .groupby("batter", as_index=False)
                    .size()
                    .rename(columns={"size": "actual_pa"})
            )

            pa_ids = set(pa_df["batter"].astype(int).tolist())
            #----------------------------------------
        
        except Exception:
            continue

        homers = day[day["events"] == "home_run"]
        # batter IDs that homered that day
        homer_ids = set()
        if "batter" in homers.columns:
            homer_ids = set(int(x) for x in homers["batter"].dropna().astype(int).unique())

        tmp = board.copy()
        tmp = tmp[tmp["batter_id"].astype(int).isin(pa_ids)].copy()
        tmp["y"] = tmp["batter_id"].apply(lambda x: 1 if int(x) in homer_ids else 0)
        tmp["p"] = tmp["p_hr_1plus_sim"].astype(float)
        tmp["date"] = d

        # optional odds merge
        if odds_df is not None and "player_name" in tmp.columns:
            m = odds_df[odds_df["date"] == d].copy()
            tmp = tmp.merge(m[["date", "player_name", "odds_american"]], on=["date", "player_name"], how="left")
            tmp["implied_p"] = tmp["odds_american"].apply(lambda o: american_to_implied(float(o)) if pd.notna(o) else np.nan)
            tmp["edge"] = tmp["p"] - tmp["implied_p"]

        all_preds.append(tmp)

        # quick daily top-N hit rate snapshots
        top10 = tmp.sort_values("p", ascending=False).head(10)
        top25 = tmp.sort_values("p", ascending=False).head(25)
        daily_summary.append({
            "date": d,
            "n_players": int(len(tmp)),
            "top10_any_hr": int(top10["y"].sum() > 0),
            "top25_any_hr": int(top25["y"].sum() > 0),
            "top10_hr_count": int(top10["y"].sum()),
            "top25_hr_count": int(top25["y"].sum()),
        })

    if not all_preds:
        os.makedirs("outputs", exist_ok=True)
        pd.DataFrame([{"message": "No backtest results. Try a smaller date range."}]).to_csv("outputs/hr_backtest_empty.csv", index=False)
        return

    results = pd.concat(all_preds, ignore_index=True)
    summary = pd.DataFrame(daily_summary)

    # Core scoring
    metrics, cal = score_predictions(results[["p", "y"]].dropna())

    os.makedirs("outputs", exist_ok=True)
    results.to_csv("outputs/hr_backtest_rows.csv", index=False)
    summary.to_csv("outputs/hr_backtest_daily.csv", index=False)
    pd.DataFrame([metrics]).to_csv("outputs/hr_backtest_metrics.csv", index=False)
    cal.to_csv("outputs/hr_backtest_calibration.csv", index=False)

    # Optional “betting” report if odds were provided
    if "odds_american" in results.columns:
        bet_df = results[pd.notna(results["odds_american"])].copy()
        if not bet_df.empty:
            # simple: bet when model edge >= 0.03 (3%)
            bet_df["bet"] = bet_df["edge"] >= 0.03
            plays = bet_df[bet_df["bet"]].copy()
            if not plays.empty:
                # $1 flat stake; profit is odds payout if win else -1
                def profit_row(r):
                    o = float(r["odds_american"])
                    if r["y"] == 1:
                        if o >= 0:
                            return o / 100.0
                        else:
                            return 100.0 / (-o)
                    return -1.0

                plays["profit"] = plays.apply(profit_row, axis=1)
                roi = float(plays["profit"].sum() / len(plays))
                out = {
                    "n_odds_rows": int(len(bet_df)),
                    "n_plays_edge_ge_3pct": int(len(plays)),
                    "roi_per_bet_flat_1u": roi,
                    "avg_edge": float(plays["edge"].mean()),
                    "avg_p": float(plays["p"].mean()),
                    "hit_rate": float(plays["y"].mean()),
                    "avg_odds": float(plays["odds_american"].mean()),
                }
                pd.DataFrame([out]).to_csv("outputs/hr_backtest_betting.csv", index=False)

    print("Backtest complete.")
    print(metrics)
    print("\nTop-N Summary:")
    topn_summary = topn_hit_table(bet_df)
    print(topn_summary.to_string(index=False))

    #save Top-N summary as artifact
    topn_summary.to_csv("outputs/hr_topn_summary.csv",


if __name__ == "__main__":
    main()
