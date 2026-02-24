import os
import argparse
import math
import pandas as pd
import numpy as np
from datetime import date as date_cls


def ensure_dirs():
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 1.0 + (odds / 100.0)
    else:
        return 1.0 + (100.0 / (-odds))


def american_to_implied_prob(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return (-odds) / ((-odds) + 100.0)


def kelly_fraction(p: float, decimal_odds: float) -> float:
    """
    Binary Kelly for win prob p and decimal odds.
    f* = (b*p - q) / b, where b = decimal_odds - 1, q = 1-p
    """
    b = decimal_odds - 1.0
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f = (b * p - q) / b
    return float(max(0.0, f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=str(date_cls.today()))
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--frac_kelly", type=float, default=0.25)
    parser.add_argument("--max_bet_frac", type=float, default=0.005)  # 0.5% bankroll cap per play
    parser.add_argument("--edge_min", type=float, default=0.03)       # 3% edge
    parser.add_argument("--max_bets", type=int, default=6)
    args = parser.parse_args()

    ensure_dirs()

    board_path = f"outputs/hr_board_{args.date}.csv"
    odds_path = f"inputs/odds_input.csv"

    if not os.path.exists(board_path):
        raise FileNotFoundError(f"Missing {board_path}. Run hr_run_daily.py first.")
    if not os.path.exists(odds_path):
        raise FileNotFoundError(
            f"Missing {odds_path}. Create it with columns: player, odds_american (and optional team/opponent)."
        )

    board = pd.read_csv(board_path)
    odds = pd.read_csv(odds_path)

    # Normalize merge on batter_id if user provides it, otherwise on player name
    if "batter_id" in odds.columns:
        merged = board.merge(odds, on="batter_id", how="inner", suffixes=("", "_odds"))
    else:
        if "player" not in odds.columns:
            raise ValueError("odds CSV must have either batter_id OR player column.")
        # board doesn't have player names (MLBAM ids only). We merge by batter_id if possible.
        # So require batter_id for sharp workflow.
        raise ValueError("For best results, include batter_id in odds CSV. (Your board outputs batter_id.)")

    if "odds_american" not in merged.columns:
        raise ValueError("odds CSV must include odds_american column.")

    merged["implied_prob"] = merged["odds_american"].apply(american_to_implied_prob)
    merged["decimal_odds"] = merged["odds_american"].apply(american_to_decimal)

    # Use sim probability as our bet prob
    merged["model_prob"] = merged["p_hr_1plus_sim"].astype(float)
    merged["edge"] = merged["model_prob"] - merged["implied_prob"]

    merged["kelly_full"] = merged.apply(lambda r: kelly_fraction(float(r["model_prob"]), float(r["decimal_odds"])), axis=1)
    merged["kelly_used"] = merged["kelly_full"] * float(args.frac_kelly)

    merged["bet_frac"] = merged["kelly_used"].clip(lower=0.0, upper=float(args.max_bet_frac))
    merged["bet_amount"] = (merged["bet_frac"] * float(args.bankroll)).round(2)

    # Filter to edges
    bets = merged[merged["edge"] >= float(args.edge_min)].copy()
    bets = bets.sort_values(["edge", "model_prob"], ascending=False).head(int(args.max_bets)).reset_index(drop=True)
    bets.insert(0, "rank", np.arange(1, len(bets) + 1))

    out_csv = f"outputs/bets_{args.date}.csv"
    out_html = f"outputs/bets_{args.date}.html"

    bets.to_csv(out_csv, index=False)

    show = bets.copy()
    for c in ["model_prob", "implied_prob", "edge"]:
        if c in show.columns:
            show[c] = (show[c] * 100).round(2)

    html = f"""
    <html><head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial; padding: 14px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; }}
        th {{ background: #f4f4f4; position: sticky; top: 0; }}
        tr:nth-child(even) {{ background: #fafafa; }}
      </style>
    </head><body>
      <h2>HR Bets — {args.date}</h2>
      <p>model_prob, implied_prob, edge are percentages. Stake uses {args.frac_kelly:.2f} Kelly capped at {(args.max_bet_frac*100):.2f}% bankroll per play.</p>
      {show.to_html(index=False, escape=True)}
    </body></html>
    """
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote {out_csv} and {out_html}")
    if bets.empty:
        print("No bets passed the edge threshold.")


if __name__ == "__main__":
    main()
