from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ANALYSIS_FIELDS = [
    "outcome",
    "reason",
    "status",
    "dataset",
    "horizon",
    "val_static_mse",
    "val_refined_mse",
    "val_mse_delta",
    "val_mse_delta_pct",
    "val_static_mae",
    "val_refined_mae",
    "val_mae_delta",
    "val_mae_delta_pct",
    "learnable_adopted",
    "learnable_adopted_channels",
    "test_mse",
    "test_mae",
    "config_path",
    "out_dir",
    "error",
]


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def delta_pct(delta: float, baseline: float | None) -> float | None:
    if baseline is None or baseline == 0:
        return None
    return round(delta / baseline * 100.0, 6)


def analyze_rows(
    rows: list[dict[str, Any]],
    *,
    mse_tolerance: float = 0.0,
    mae_tolerance: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    analysis = [
        analyze_row(row, mse_tolerance=mse_tolerance, mae_tolerance=mae_tolerance)
        for row in rows
    ]
    summary = {
        "total": len(analysis),
        "pass": 0,
        "regressed": 0,
        "incomplete": 0,
        "missing_learnable_metrics": 0,
    }
    for row in analysis:
        outcome = str(row["outcome"])
        summary[outcome] = summary.get(outcome, 0) + 1
    return analysis, summary


def analyze_row(
    row: dict[str, Any],
    *,
    mse_tolerance: float,
    mae_tolerance: float,
) -> dict[str, Any]:
    status = str(row.get("status", "ok")).strip() or "ok"
    base = {
        "status": status,
        "dataset": row.get("dataset", ""),
        "horizon": row.get("horizon", ""),
        "learnable_adopted": row.get("learnable_adopted", ""),
        "learnable_adopted_channels": row.get("learnable_adopted_channels", ""),
        "test_mse": row.get("test_mse", ""),
        "test_mae": row.get("test_mae", ""),
        "config_path": row.get("config_path", ""),
        "out_dir": row.get("out_dir", ""),
        "error": row.get("error", ""),
    }
    if status not in {"ok", "skipped"}:
        return {
            **base,
            "outcome": "incomplete",
            "reason": f"status={status or 'missing'}",
        }

    static_mse = parse_optional_float(first_present(row, ("learnable_val_static_mse", "val_static_mse")))
    refined_mse = parse_optional_float(first_present(row, ("learnable_val_refined_mse", "val_refined_mse")))
    static_mae = parse_optional_float(first_present(row, ("learnable_val_static_mae", "val_static_mae")))
    refined_mae = parse_optional_float(first_present(row, ("learnable_val_refined_mae", "val_refined_mae")))
    if None in {static_mse, refined_mse, static_mae, refined_mae}:
        return {
            **base,
            "outcome": "missing_learnable_metrics",
            "reason": "missing static/refined val metrics",
        }

    mse_delta = float(refined_mse - static_mse)
    mae_delta = float(refined_mae - static_mae)
    regressions: list[str] = []
    if mse_delta > mse_tolerance:
        regressions.append(f"mse_delta={mse_delta:.9g} > tolerance={mse_tolerance:.9g}")
    if mae_delta > mae_tolerance:
        regressions.append(f"mae_delta={mae_delta:.9g} > tolerance={mae_tolerance:.9g}")

    return {
        **base,
        "outcome": "regressed" if regressions else "pass",
        "reason": "; ".join(regressions),
        "val_static_mse": static_mse,
        "val_refined_mse": refined_mse,
        "val_mse_delta": round(mse_delta, 12),
        "val_mse_delta_pct": delta_pct(mse_delta, static_mse),
        "val_static_mae": static_mae,
        "val_refined_mae": refined_mae,
        "val_mae_delta": round(mae_delta, 12),
        "val_mae_delta_pct": delta_pct(mae_delta, static_mae),
    }


def read_summary_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_analysis_outputs(
    out_dir: Path,
    analysis: list[dict[str, Any]],
    summary: dict[str, int],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "analysis.csv"
    json_path = out_dir / "analysis.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANALYSIS_FIELDS)
        writer.writeheader()
        for row in analysis:
            writer.writerow({field: row.get(field, "") for field in ANALYSIS_FIELDS})
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": analysis}, f, indent=2, ensure_ascii=False)
    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="outputs/full_learnable_anchor_matrix_20260627/summary.csv")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--mse-tolerance", type=float, default=0.0)
    parser.add_argument("--mae-tolerance", type=float, default=0.0)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent
    rows = read_summary_csv(summary_path)
    analysis, summary = analyze_rows(
        rows,
        mse_tolerance=float(args.mse_tolerance),
        mae_tolerance=float(args.mae_tolerance),
    )
    csv_path, json_path = write_analysis_outputs(out_dir, analysis, summary)
    print(f"Analysis CSV: {csv_path}")
    print(f"Analysis JSON: {json_path}")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    failed = summary.get("regressed", 0) + summary.get("missing_learnable_metrics", 0)
    if not args.allow_incomplete:
        failed += summary.get("incomplete", 0)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
