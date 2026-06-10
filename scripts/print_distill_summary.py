#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def tracking_name(row):
    return row.get("tracking_metric_name") or (
        "inner_loss"
        if row.get("objective")
        in ("feature_matching", "gradient_matching", "trajectory_matching")
        else "outer_loss"
    )


def tracking_value(row):
    if "tracking_value" in row:
        return row["tracking_value"]
    name = tracking_name(row)
    return row.get(name, float("nan"))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: print_distill_summary.py <run_dir_or_metrics_jsonl>")
        return 2

    path = Path(sys.argv[1])
    metrics_path = (
        path
        if path.name == "distill_metrics.jsonl"
        else path / "distill_metrics.jsonl"
    )
    if not metrics_path.exists():
        print(f"metrics file not found: {metrics_path}")
        return 1

    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        print(f"no rows in: {metrics_path}")
        return 1

    last = rows[-1]
    metric_name = tracking_name(last)
    best = min(rows, key=tracking_value)

    print(f"run: {metrics_path.parent.name}")
    print(
        f"last step={last['step']} "
        f"{metric_name}={tracking_value(last):.4f} "
        f"outer_loss={last['outer_loss']:.4f} "
        f"outer_ppl={last['outer_ppl']:.2f} "
        f"entropy={last['entropy']:.4f}"
    )
    print(
        f"best step={best['step']} "
        f"{metric_name}={tracking_value(best):.4f} "
        f"outer_loss={best['outer_loss']:.4f} "
        f"outer_ppl={best['outer_ppl']:.2f} "
        f"entropy={best['entropy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
