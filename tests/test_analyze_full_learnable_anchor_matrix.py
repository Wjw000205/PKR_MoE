from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_full_learnable_anchor_matrix import analyze_rows, write_analysis_outputs


def test_analyze_rows_classifies_non_regression_and_failure_modes() -> None:
    rows = [
        {
            "status": "ok",
            "dataset": "ETTh1",
            "horizon": "96",
            "learnable_val_static_mse": "1.0",
            "learnable_val_refined_mse": "0.9",
            "learnable_val_static_mae": "0.5",
            "learnable_val_refined_mae": "0.49",
        },
        {
            "status": "ok",
            "dataset": "ETTh1",
            "horizon": "192",
            "learnable_val_static_mse": "1.0",
            "learnable_val_refined_mse": "1.01",
            "learnable_val_static_mae": "0.5",
            "learnable_val_refined_mae": "0.49",
        },
        {
            "status": "failed",
            "dataset": "weather",
            "horizon": "96",
            "error": "see log",
        },
        {
            "status": "ok",
            "dataset": "PEMS08",
            "horizon": "96",
            "learnable_val_static_mse": "",
            "learnable_val_refined_mse": "",
            "learnable_val_static_mae": "",
            "learnable_val_refined_mae": "",
        },
    ]

    analysis, summary = analyze_rows(rows, mse_tolerance=0.0, mae_tolerance=0.0)

    assert summary == {
        "total": 4,
        "pass": 1,
        "regressed": 1,
        "incomplete": 1,
        "missing_learnable_metrics": 1,
    }
    assert analysis[0]["outcome"] == "pass"
    assert analysis[0]["val_mse_delta_pct"] == -10.0
    assert analysis[1]["outcome"] == "regressed"
    assert "mse" in analysis[1]["reason"]
    assert analysis[2]["outcome"] == "incomplete"
    assert analysis[3]["outcome"] == "missing_learnable_metrics"


def test_write_analysis_outputs_csv_and_json(tmp_path: Path) -> None:
    analysis, summary = analyze_rows(
        [
            {
                "status": "ok",
                "dataset": "PEMS03",
                "horizon": "12",
                "learnable_val_static_mse": "2.0",
                "learnable_val_refined_mse": "1.8",
                "learnable_val_static_mae": "1.0",
                "learnable_val_refined_mae": "0.9",
            }
        ]
    )

    csv_path, json_path = write_analysis_outputs(tmp_path, analysis, summary)

    csv_text = csv_path.read_text(encoding="utf-8")
    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "val_mse_delta_pct" in csv_text
    assert "PEMS03" in csv_text
    assert json_payload["summary"]["pass"] == 1
    assert json_payload["rows"][0]["outcome"] == "pass"


def test_analyze_rows_accepts_existing_screen_summary_field_names() -> None:
    analysis, summary = analyze_rows(
        [
            {
                "dataset": "weather",
                "horizon": "96",
                "val_static_mse": "0.371409",
                "val_refined_mse": "0.368360",
                "val_static_mae": "0.257992",
                "val_refined_mae": "0.248983",
            }
        ]
    )

    assert summary["pass"] == 1
    assert analysis[0]["outcome"] == "pass"
    assert analysis[0]["val_mse_delta"] == -0.003049
