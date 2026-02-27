import argparse
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows_csv", required=True, help="outputs/hr_backtest_rows.csv")
    ap.add_argument("--out_json", default="outputs/calibrator_isotonic.json")
    ap.add_argument("--min_p", type=float, default=1e-6)
    ap.add_argument("--max_p", type=float, default=1 - 1e-6)
    args = ap.parse_args()

    df = pd.read_csv(args.rows_csv)
    df = df.dropna(subset=["p", "y"]).copy()
    p = np.clip(df["p"].astype(float).values, args.min_p, args.max_p)
    y = df["y"].astype(int).values

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)

    # Save as a piecewise-linear mapping so we can apply it without sklearn later
    x = iso.X_thresholds_.tolist()
    yhat = iso.y_thresholds_.tolist()

    payload = {
        "type": "isotonic_piecewise",
        "min_p": args.min_p,
        "max_p": args.max_p,
        "x": x,
        "y": yhat,
        "n": int(len(df)),
        "base_rate": float(np.mean(y)),
    }

    with open(args.out_json, "w") as f:
        json.dump(payload, f)

    print(f"Saved calibrator -> {args.out_json}")
    print(f"n={payload['n']} base_rate={payload['base_rate']:.6f} knots={len(x)}")

if __name__ == "__main__":
    main()
