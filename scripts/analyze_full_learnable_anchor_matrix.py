from __future__ import annotations

import argparse
import csv
import json
import re
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
    "baseline_val_mse",
    "baseline_val_mae",
    "val_vs_baseline_mse_delta",
    "val_vs_baseline_mse_delta_pct",
    "val_vs_baseline_mae_delta",
    "val_vs_baseline_mae_delta_pct",
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


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    dataset = str(row.get("dataset", "")).strip().lower()
    horizon = str(row.get("horizon", "")).strip()
    if horizon == "":
        horizon = infer_horizon(row)
    return dataset, horizon


def display_horizon(row: dict[str, Any]) -> str:
    horizon = str(row.get("horizon", "")).strip()
    return horizon if horizon else infer_horizon(row)


def infer_horizon(row: dict[str, Any]) -> str:
    for key in ("config_path", "summary_path", "out_dir"):
        text = str(row.get(key, "")).strip()
        match = re.search(r"(?:^|[_/\\-])H(\d+)(?:\D|$)", text)
        if match:
            return match.group(1)
    return ""


def delta_pct(delta: float, baseline: float | None) -> float | None:
    if baseline is None or baseline == 0:
        return None
    return round(delta / baseline * 100.0, 6)


def analyze_rows(
    rows: list[dict[str, Any]],
    *,
    baseline_rows: list[dict[str, Any]] | None = None,
    mse_tolerance: float = 0.0,
    mae_tolerance: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    baseline_lookup = build_baseline_lookup(baseline_rows) if baseline_rows is not None else None
    analysis = [
        analyze_row(
            row,
            baseline_lookup=baseline_lookup,
            mse_tolerance=mse_tolerance,
            mae_tolerance=mae_tolerance,
        )
        for row in rows
    ]
    summary = {
        "total": len(analysis),
        "pass": 0,
        "regressed": 0,
        "incomplete": 0,
        "missing_learnable_metrics": 0,
    }
    if baseline_lookup is not None:
        summary["missing_baseline"] = 0
    for row in analysis:
        outcome = str(row["outcome"])
        summary[outcome] = summary.get(outcome, 0) + 1
    return analysis, summary


def build_baseline_lookup(
    baseline_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], tuple[float | None, float | None]]:
    lookup: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for row in baseline_rows:
        key = row_key(row)
        if not all(key):
            continue
        mse = parse_optional_float(
            first_present(row, ("baseline_val_mse", "val_mse", "val_static_mse", "avg_mse", "mse"))
        )
        mae = parse_optional_float(
            first_present(row, ("baseline_val_mae", "val_mae", "val_static_mae", "avg_mae", "mae"))
        )
        lookup[key] = (mse, mae)
    return lookup


def analyze_row(
    row: dict[str, Any],
    *,
    baseline_lookup: dict[tuple[str, str], tuple[float | None, float | None]] | None,
    mse_tolerance: float,
    mae_tolerance: float,
) -> dict[str, Any]:
    status = str(row.get("status", "ok")).strip() or "ok"
    base = {
        "status": status,
        "dataset": row.get("dataset", ""),
        "horizon": display_horizon(row),
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

    baseline_fields: dict[str, Any] = {}
    if baseline_lookup is not None:
        baseline = baseline_lookup.get(row_key(row))
        if baseline is None or baseline[0] is None or baseline[1] is None:
            return {
                **base,
                "outcome": "missing_baseline",
                "reason": "missing baseline val metrics",
                "val_static_mse": static_mse,
                "val_refined_mse": refined_mse,
                "val_mse_delta": round(mse_delta, 12),
                "val_mse_delta_pct": delta_pct(mse_delta, static_mse),
                "val_static_mae": static_mae,
                "val_refined_mae": refined_mae,
                "val_mae_delta": round(mae_delta, 12),
                "val_mae_delta_pct": delta_pct(mae_delta, static_mae),
            }
        baseline_mse, baseline_mae = baseline
        baseline_mse_delta = float(refined_mse - baseline_mse)
        baseline_mae_delta = float(refined_mae - baseline_mae)
        baseline_fields = {
            "baseline_val_mse": baseline_mse,
            "baseline_val_mae": baseline_mae,
            "val_vs_baseline_mse_delta": round(baseline_mse_delta, 12),
            "val_vs_baseline_mse_delta_pct": delta_pct(baseline_mse_delta, baseline_mse),
            "val_vs_baseline_mae_delta": round(baseline_mae_delta, 12),
            "val_vs_baseline_mae_delta_pct": delta_pct(baseline_mae_delta, baseline_mae),
        }
        if baseline_mse_delta > mse_tolerance:
            regressions.append(
                f"baseline_mse_delta={baseline_mse_delta:.9g} > tolerance={mse_tolerance:.9g}"
            )
        if baseline_mae_delta > mae_tolerance:
            regressions.append(
                f"baseline_mae_delta={baseline_mae_delta:.9g} > tolerance={mae_tolerance:.9g}"
            )

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
        **baseline_fields,
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
    parser.add_argument(
        "--baseline-summary",
        default=None,
        help=(
            "Optional CSV keyed by dataset,horizon with val_mse/val_mae or "
            "baseline_val_mse/baseline_val_mae columns."
        ),
    )
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
    baseline_rows = read_summary_csv(Path(args.baseline_summary)) if args.baseline_summary else None
    analysis, summary = analyze_rows(
        rows,
        baseline_rows=baseline_rows,
        mse_tolerance=float(args.mse_tolerance),
        mae_tolerance=float(args.mae_tolerance),
    )
    csv_path, json_path = write_analysis_outputs(out_dir, analysis, summary)
    print(f"Analysis CSV: {csv_path}")
    print(f"Analysis JSON: {json_path}")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    failed = (
        summary.get("regressed", 0)
        + summary.get("missing_learnable_metrics", 0)
        + summary.get("missing_baseline", 0)
    )
    if not args.allow_incomplete:
        failed += summary.get("incomplete", 0)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
