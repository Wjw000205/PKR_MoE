from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (96, 192, 336, 720)
DEFAULT_DEVICES = ("cuda:0", "cuda:2")
DEFAULT_WORKERS_PER_DEVICE = 2

FIELDS = [
    "status",
    "phase",
    "horizon",
    "candidate",
    "device",
    "worker",
    "predictor",
    "hidden_dim",
    "dropout",
    "batch_size",
    "lr",
    "weight_decay",
    "mse_weight",
    "mae_weight",
    "selection_metric",
    "penalties",
    "lambda_scale",
    "moe_enable",
    "pred_residual_enable",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "objective",
    "objective_source",
    "total_sec",
    "avg_epoch_sec",
    "config_path",
    "out_dir",
    "returncode",
    "error",
]


@dataclass(frozen=True)
class SearchCandidate:
    name: str
    predictor: str = "mlp"
    hidden_dim: int = 128
    dropout: float = 0.0
    batch_size: int = 64
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    mse_weight: float = 0.9
    mae_weight: float = 0.0
    selection_metric: str = "val_mse"
    penalties: tuple[str, ...] = ("amp_under", "delta", "diff_amp", "direction")
    lambda_scale: float = 0.01
    moe_enable: bool = True
    pred_residual_enable: bool = False
    pred_residual_hidden: int = 16
    pred_residual_alpha: float = 0.6
    pred_residual_policy: str = "val_mse_candidate_channel"
    topk: int = 1
    gate_temperature: float = 1.2
    gate_noise_std: float = 0.2
    skip_cost: float = 0.15
    cluster_distance_threshold: float | None = None
    merge_small_clusters: bool | None = None
    model_overrides: dict[str, Any] | None = None
    moe_overrides: dict[str, Any] | None = None


@dataclass(frozen=True)
class SearchJob:
    horizon: int
    candidate_name: str
    config_path: Path
    out_dir: Path
    device: str


def seed_candidates() -> list[SearchCandidate]:
    return [
        SearchCandidate(
            name="mlp_h128_do0_bs64_lr1e3_wd1e4_mse09",
            hidden_dim=128,
            dropout=0.0,
            batch_size=64,
            lr=1.0e-3,
            weight_decay=1.0e-4,
            mse_weight=0.9,
            mae_weight=0.0,
            lambda_scale=0.01,
            moe_enable=True,
            pred_residual_enable=False,
        ),
        SearchCandidate(
            name="mlp_h128_do0_bs128_lr1e3_wd1e4_mse10_mae03",
            hidden_dim=128,
            dropout=0.0,
            batch_size=128,
            lr=1.0e-3,
            weight_decay=1.0e-4,
            mse_weight=1.0,
            mae_weight=0.3,
            lambda_scale=0.01,
            moe_enable=True,
            pred_residual_enable=False,
        ),
        SearchCandidate(
            name="mlp_h192_do0_bs128_lr1e3_wd1e5_mse10_mae03",
            hidden_dim=192,
            dropout=0.0,
            batch_size=128,
            lr=1.0e-3,
            weight_decay=1.0e-5,
            mse_weight=1.0,
            mae_weight=0.3,
            lambda_scale=0.01,
            moe_enable=True,
            pred_residual_enable=False,
        ),
        SearchCandidate(
            name="mlp_h256_do005_bs128_lr1309_wd1e5_range_resid",
            hidden_dim=256,
            dropout=0.0468935703282562,
            batch_size=128,
            lr=0.001309395478035077,
            weight_decay=1.0644440818212169e-5,
            mse_weight=0.9,
            mae_weight=0.0,
            penalties=("amp_under", "range", "delta", "diff_amp", "direction"),
            lambda_scale=0.02693171136352346,
            moe_enable=True,
            pred_residual_enable=True,
            pred_residual_hidden=64,
            pred_residual_alpha=0.3733555458106823,
            topk=1,
            gate_temperature=0.892961150014985,
            gate_noise_std=0.15621020105136857,
            skip_cost=0.23073356986971394,
            cluster_distance_threshold=0.6855423898885412,
            merge_small_clusters=False,
            moe_overrides={
                "cluster_penalty_prior": {
                    "enable": True,
                    "topk": 1,
                    "hard_topk": True,
                    "logit_strength": 0.5296336824602615,
                    "temperature": 1.0,
                    "smoothing": 0.02,
                    "use_normalized_penalty": True,
                    "use_as_balance_target": False,
                },
                "channel_penalty_prior": {
                    "enable": True,
                    "topk": 1,
                    "hard_topk": True,
                    "temperature": 1.0,
                    "smoothing": 0.02,
                    "use_normalized_penalty": True,
                },
            },
        ),
        SearchCandidate(
            name="mlp_h224_p168_anchor_bs64_lr1309_wd1e5",
            hidden_dim=224,
            dropout=0.0,
            batch_size=64,
            lr=0.001309395478035077,
            weight_decay=1.0e-5,
            mse_weight=1.0,
            mae_weight=0.0,
            lambda_scale=0.0,
            moe_enable=False,
            model_overrides={
                "train_stat_adapter": {
                    "enable": True,
                    "period": 168,
                    "mode": "phase_mean",
                    "alpha": 1.0,
                    "blend_target": "prediction",
                    "combine_mode": "anchor_plus_prediction",
                    "input_center": True,
                }
            },
        ),
        SearchCandidate(
            name="mlp_h288_p168_anchor_bs64_lr1309_wd1e5",
            hidden_dim=288,
            dropout=0.0,
            batch_size=64,
            lr=0.001309395478035077,
            weight_decay=1.0e-5,
            mse_weight=1.0,
            mae_weight=0.0,
            lambda_scale=0.0,
            moe_enable=False,
            model_overrides={
                "train_stat_adapter": {
                    "enable": True,
                    "period": 168,
                    "mode": "phase_mean",
                    "alpha": 1.0,
                    "blend_target": "prediction",
                    "combine_mode": "anchor_plus_prediction",
                    "input_center": True,
                }
            },
        ),
    ]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def deep_update(dst: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def lambda_dict(names: tuple[str, ...], value: float) -> dict[str, float]:
    return {name: float(value) for name in names}


def configure(
    base_cfg: dict[str, Any],
    *,
    horizon: int,
    cand: SearchCandidate,
    phase: str,
    out_dir: Path,
    device: str,
    epochs: int,
    skip_test: bool,
    save_checkpoint: bool,
    lazy_windows: bool = False,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = f"electricity_H{horizon}_{cand.name}_{phase}"
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg["exp"]["device"] = str(device)

    cfg.setdefault("data", {})
    cfg["data"].setdefault("csv_path", "data/electricity.csv")
    cfg["data"].setdefault("date_col", 0)
    cfg["data"].setdefault("train_ratio", 0.6999695863746959)
    cfg["data"].setdefault("val_ratio", 0.10006082725060828)
    cfg["data"].setdefault("test_ratio", 0.19996958637469586)
    cfg["data"].setdefault("max_rows", 0)

    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = 96
    cfg["window"]["pred_len"] = int(horizon)
    cfg["window"]["past_context"] = True
    if lazy_windows:
        cfg["window"]["lazy"] = True

    cfg.setdefault("normalize", {})
    cfg["normalize"]["global_zscore"] = True
    cfg["normalize"]["train_only"] = True

    cfg.setdefault("corr", {})
    cfg["corr"]["compute"] = True
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")

    cfg.setdefault("cluster", {})
    cfg["cluster"]["train_only"] = True
    if cand.cluster_distance_threshold is not None:
        cfg["cluster"]["distance_threshold"] = float(cand.cluster_distance_threshold)
    if cand.merge_small_clusters is not None:
        cfg["cluster"]["merge_small_clusters"] = bool(cand.merge_small_clusters)

    cfg.setdefault("model", {})
    cfg["model"]["predictor"] = cand.predictor
    cfg["model"]["hidden_dim"] = int(cand.hidden_dim)
    cfg["model"]["dropout"] = float(cand.dropout)
    if cand.model_overrides:
        deep_update(cfg["model"], cand.model_overrides)

    penalties = tuple(cand.penalties)
    cfg.setdefault("penalties", {})
    cfg["penalties"]["enabled"] = list(penalties)
    cfg["penalties"]["jump_threshold"] = float(cfg["penalties"].get("jump_threshold", 0.6))

    cfg.setdefault("moe", {})
    cfg["moe"]["enable"] = bool(cand.moe_enable)
    cfg["moe"]["topk"] = int(cand.topk)
    cfg["moe"]["select_ranks"] = list(range(1, int(cand.topk) + 1))
    cfg["moe"]["lambda_init"] = lambda_dict(penalties, cand.lambda_scale)
    cfg["moe"]["lambda_min"] = lambda_dict(penalties, 0.0)
    cfg["moe"]["lambda_schedule"] = {name: "none" for name in penalties}
    cfg["moe"]["gate_temperature"] = float(cand.gate_temperature)
    cfg["moe"]["gate_noise_std"] = float(cand.gate_noise_std)
    cfg["moe"]["skip_cost"] = float(cand.skip_cost)
    cfg["moe"].setdefault("dynamic_lambda", {})["enable"] = False
    cfg["moe"].setdefault("learnable_lambda", {})["enable"] = False
    cfg["moe"].setdefault("pred_side_residual", {})
    cfg["moe"]["pred_side_residual"]["enable"] = bool(cand.pred_residual_enable)
    cfg["moe"]["pred_side_residual"]["corrector_hidden"] = int(cand.pred_residual_hidden)
    cfg["moe"]["pred_side_residual"]["alpha_scale"] = float(cand.pred_residual_alpha)
    cfg["moe"]["pred_side_residual"]["selection_policy"] = str(cand.pred_residual_policy)
    cfg["moe"]["pred_side_residual"].setdefault("feature_mode", "legacy")
    cfg["moe"]["pred_side_residual"].setdefault("residual_clip", 4.0)
    if cand.moe_overrides:
        deep_update(cfg["moe"], cand.moe_overrides)

    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["batch_size"] = int(cand.batch_size)
    cfg["train"]["lr"] = float(cand.lr)
    cfg["train"]["weight_decay"] = float(cand.weight_decay)
    cfg["train"]["mse_weight"] = float(cand.mse_weight)
    cfg["train"]["selection_metric"] = str(cand.selection_metric)
    cfg["train"]["penalty_warmup_epochs"] = min(int(cfg["train"].get("penalty_warmup_epochs", 3)), 3)
    cfg["train"].setdefault("lr_scheduler", {})
    cfg["train"]["lr_scheduler"]["name"] = "plateau"
    cfg["train"]["lr_scheduler"]["factor"] = 0.5
    cfg["train"]["lr_scheduler"]["patience"] = 3
    cfg["train"]["lr_scheduler"]["min_lr"] = 1.0e-6
    cfg["train"]["mae_objective"] = {
        "enable": bool(cand.mae_weight > 0.0),
        "kind": "l1",
        "weight": float(cand.mae_weight),
        "warmup_epochs": 3 if cand.mae_weight > 0.0 else 0,
    }

    cfg.setdefault("early_stop", {})
    cfg["early_stop"]["patience"] = int(cfg["early_stop"].get("patience", 5))
    cfg["early_stop"]["min_delta"] = float(cfg["early_stop"].get("min_delta", 1.0e-6))

    cfg["plot"] = {"enable": False}
    cfg["portrait"] = {"enable": False, "out_dir": str(out_dir / "cluster_portraits")}
    cfg["eval"] = {"skip_test": bool(skip_test)}
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": bool(save_checkpoint),
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    return cfg


def objective_from_summary(summary: dict[str, Any]) -> tuple[str, str]:
    val = summary.get("val") or {}
    value = val.get("avg_mse", "")
    if value == "" or value is None:
        return "", ""
    return str(value), "val.avg_mse"


def safe_float(value: Any) -> float:
    try:
        if value in ("", None):
            return float("inf")
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def row_from_result(
    *,
    phase: str,
    horizon: int,
    cand: SearchCandidate,
    job: SearchJob,
    worker: str,
    returncode: int,
    total_sec: float,
    output_tail: str,
) -> dict[str, Any]:
    summary = read_json(job.out_dir / "run_summary.json")
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    timing = summary.get("timing") or {}
    objective, objective_source = objective_from_summary(summary)
    status = "ok" if returncode == 0 and summary else "error"
    error = ""
    if returncode != 0:
        status = "oom" if "out of memory" in output_tail.lower() else "error"
        error = output_tail
    elif not summary:
        error = "run_summary.json not found"
    return {
        "status": status,
        "phase": phase,
        "horizon": int(horizon),
        "candidate": cand.name,
        "device": job.device,
        "worker": worker,
        "predictor": cand.predictor,
        "hidden_dim": cand.hidden_dim,
        "dropout": cand.dropout,
        "batch_size": cand.batch_size,
        "lr": cand.lr,
        "weight_decay": cand.weight_decay,
        "mse_weight": cand.mse_weight,
        "mae_weight": cand.mae_weight,
        "selection_metric": cand.selection_metric,
        "penalties": ",".join(cand.penalties),
        "lambda_scale": cand.lambda_scale,
        "moe_enable": cand.moe_enable,
        "pred_residual_enable": cand.pred_residual_enable,
        "val_mse": val.get("avg_mse", ""),
        "val_mae": val.get("avg_mae", ""),
        "test_mse": test.get("avg_mse", ""),
        "test_mae": test.get("avg_mae", ""),
        "objective": objective,
        "objective_source": objective_source,
        "total_sec": timing.get("total_sec", f"{float(total_sec):.3f}"),
        "avg_epoch_sec": timing.get("avg_epoch_sec", ""),
        "config_path": str(job.config_path),
        "out_dir": str(job.out_dir),
        "returncode": int(returncode),
        "error": error,
    }


def run_train(python_exe: str, job: SearchJob) -> tuple[int, float, str]:
    job.out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = job.out_dir / "stdout.log"
    stderr_path = job.out_dir / "stderr.log"
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        completed = subprocess.run(
            [python_exe, "-u", "-m", "src.train", "--config", str(job.config_path)],
            cwd=str(ROOT),
            text=True,
            stdout=stdout_f,
            stderr=stderr_f,
            env=env,
        )
    total_sec = time.perf_counter() - start
    tail = ""
    for path in (stderr_path, stdout_path):
        if path.exists():
            tail += path.read_text(encoding="utf-8", errors="replace")[-2000:]
    return int(completed.returncode), float(total_sec), tail[-4000:]


def parse_csv_list(raw: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def filter_candidates(candidates: list[SearchCandidate], raw_names: str) -> list[SearchCandidate]:
    names = parse_csv_list(raw_names, str)
    if not names:
        return candidates
    by_name = {cand.name: cand for cand in candidates}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown candidates: {missing}. Available: {sorted(by_name)}")
    return [by_name[name] for name in names]


def assign_jobs_to_devices(
    jobs: list[SearchJob],
    devices: tuple[str, ...],
    *,
    workers_per_device: int = 1,
) -> dict[str, list[SearchJob]]:
    if not devices:
        raise ValueError("At least one device is required.")
    if int(workers_per_device) <= 0:
        raise ValueError("workers_per_device must be positive.")
    slots: list[tuple[str, str]] = []
    for worker_idx in range(int(workers_per_device)):
        for device in devices:
            slots.append((f"{device}#{worker_idx + 1}", device))
    assigned = {slot_name: [] for slot_name, _ in slots}
    for idx, job in enumerate(jobs):
        slot_name, device = slots[idx % len(slots)]
        assigned[slot_name].append(replace(job, device=device))
    return assigned


def base_config_path(horizon: int) -> Path:
    return ROOT / "configs" / f"electricity_H{int(horizon)}.yaml"


def prepare_jobs(
    *,
    phase: str,
    horizons: list[int],
    candidates: list[SearchCandidate],
    out_root: Path,
    epochs: int,
    skip_test: bool,
    save_checkpoint: bool,
    lazy_windows: bool,
    rerun: bool,
    completed_keys: set[tuple[str, int, str]],
    candidate_by_horizon: dict[int, SearchCandidate] | None = None,
) -> tuple[list[SearchJob], dict[tuple[int, str], SearchCandidate]]:
    jobs: list[SearchJob] = []
    candidate_by_key: dict[tuple[int, str], SearchCandidate] = {}
    for horizon in horizons:
        base_path = base_config_path(int(horizon))
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        base_cfg = read_yaml(base_path)
        active_candidates = [candidate_by_horizon[int(horizon)]] if candidate_by_horizon else candidates
        for cand in active_candidates:
            key = (phase, int(horizon), cand.name)
            if key in completed_keys and not rerun:
                continue
            out_dir = out_root / phase / f"H{horizon}" / cand.name
            config_path = out_root / "configs" / phase / f"H{horizon}" / f"{cand.name}.yaml"
            cfg = configure(
                base_cfg,
                horizon=int(horizon),
                cand=cand,
                phase=phase,
                out_dir=out_dir,
                device="",
                epochs=int(epochs),
                skip_test=bool(skip_test),
                save_checkpoint=bool(save_checkpoint),
                lazy_windows=bool(lazy_windows),
            )
            write_yaml(config_path, cfg)
            jobs.append(
                SearchJob(
                    horizon=int(horizon),
                    candidate_name=cand.name,
                    config_path=config_path,
                    out_dir=out_dir,
                    device="",
                )
            )
            candidate_by_key[(int(horizon), cand.name)] = cand
    return jobs, candidate_by_key


def patch_config_device(config_path: Path, device: str) -> None:
    cfg = read_yaml(config_path)
    cfg.setdefault("exp", {})["device"] = str(device)
    write_yaml(config_path, cfg)


def worker_loop(
    *,
    worker: str,
    jobs: list[SearchJob],
    phase: str,
    candidate_by_key: dict[tuple[int, str], SearchCandidate],
    python_exe: str,
    rows: list[dict[str, Any]],
    results_path: Path,
    lock: threading.Lock,
    reuse_existing: bool,
) -> None:
    for job in jobs:
        cand = candidate_by_key[(int(job.horizon), job.candidate_name)]
        patch_config_device(job.config_path, job.device)
        print(f"[{worker}] run H{job.horizon} {job.candidate_name} on {job.device}", flush=True)
        if reuse_existing and (job.out_dir / "run_summary.json").exists():
            returncode, total_sec, tail = 0, 0.0, ""
        else:
            returncode, total_sec, tail = run_train(python_exe, job)
        row = row_from_result(
            phase=phase,
            horizon=int(job.horizon),
            cand=cand,
            job=job,
            worker=worker,
            returncode=int(returncode),
            total_sec=float(total_sec),
            output_tail=tail,
        )
        with lock:
            rows[:] = [
                existing
                for existing in rows
                if not (
                    existing.get("phase") == phase
                    and int(existing.get("horizon", -1)) == int(job.horizon)
                    and existing.get("candidate") == job.candidate_name
                )
            ]
            rows.append(row)
            write_rows(results_path, rows)
        print(
            json.dumps(
                {
                    "worker": worker,
                    "status": row["status"],
                    "horizon": row["horizon"],
                    "candidate": row["candidate"],
                    "val_mse": row["val_mse"],
                    "val_mae": row["val_mae"],
                    "test_mse": row["test_mse"],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )


def select_final_candidates(out_root: Path, candidates: list[SearchCandidate], horizons: list[int]) -> dict[int, SearchCandidate]:
    rows = [
        row
        for row in read_rows(out_root / "search_results.csv")
        if row.get("status") == "ok" and row.get("objective") not in {"", None}
    ]
    by_name = {cand.name: cand for cand in candidates}
    selected: dict[int, SearchCandidate] = {}
    for horizon in horizons:
        horizon_rows = [row for row in rows if int(row.get("horizon", -1)) == int(horizon)]
        if not horizon_rows:
            raise RuntimeError(f"No successful search rows found for H{horizon}; run --phase search first.")
        best = min(horizon_rows, key=lambda row: safe_float(row.get("objective")))
        selected[int(horizon)] = by_name[str(best["candidate"])]
    return selected


def write_rankings(out_root: Path, phase: str) -> None:
    rows = [row for row in read_rows(out_root / f"{phase}_results.csv") if row.get("status") == "ok"]
    lines = [f"# Electricity {phase} ranking", ""]
    for horizon in HORIZONS:
        horizon_rows = [row for row in rows if int(row.get("horizon", -1)) == int(horizon)]
        if not horizon_rows:
            continue
        lines.append(f"## H{horizon}")
        lines.append("")
        lines.append("| rank | candidate | val_mse | val_mae | test_mse | test_mae | device |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for rank, row in enumerate(sorted(horizon_rows, key=lambda row: safe_float(row.get("objective"))), start=1):
            lines.append(
                "| {rank} | {candidate} | {val_mse} | {val_mae} | {test_mse} | {test_mae} | {device} |".format(
                    rank=rank,
                    candidate=row.get("candidate", ""),
                    val_mse=row.get("val_mse", ""),
                    val_mae=row.get("val_mae", ""),
                    test_mse=row.get("test_mse", ""),
                    test_mae=row.get("test_mae", ""),
                    device=row.get("device", ""),
                )
            )
        lines.append("")
    (out_root / f"{phase}_ranking.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a val-only Electricity input-96 parameter search on multiple GPUs.")
    parser.add_argument("--phase", choices=["search", "final"], default="search")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "electricity_param_search")
    parser.add_argument("--horizons", default="96,192,336,720")
    parser.add_argument("--candidates", default="", help="Comma-separated candidate names. Empty uses seed candidates.")
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--devices", default=",".join(DEFAULT_DEVICES))
    parser.add_argument("--workers-per-device", type=int, default=DEFAULT_WORKERS_PER_DEVICE)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--lazy-windows", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only write configs and planned job assignment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.out_root if args.out_root.is_absolute() else (ROOT / args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    horizons = parse_csv_list(args.horizons, int)
    candidates = filter_candidates(seed_candidates(), str(args.candidates))
    if int(args.candidate_limit) > 0:
        candidates = candidates[: int(args.candidate_limit)]
    final_candidate_by_horizon = None
    if args.phase == "final" and not args.candidates:
        final_candidate_by_horizon = select_final_candidates(out_root, candidates, horizons)
    devices = tuple(parse_csv_list(args.devices, str))
    result_path = out_root / f"{args.phase}_results.csv"
    rows = read_rows(result_path)
    completed_keys = {
        (row.get("phase"), int(row.get("horizon", -1)), row.get("candidate"))
        for row in rows
        if row.get("status") == "ok"
    }
    skip_test = args.phase == "search"
    jobs, candidate_by_key = prepare_jobs(
        phase=str(args.phase),
        horizons=[int(h) for h in horizons],
        candidates=candidates,
        out_root=out_root,
        epochs=int(args.epochs),
        skip_test=bool(skip_test),
        save_checkpoint=bool(args.save_checkpoint),
        lazy_windows=bool(args.lazy_windows),
        rerun=bool(args.rerun),
        completed_keys=completed_keys,
        candidate_by_horizon=final_candidate_by_horizon,
    )
    assigned = assign_jobs_to_devices(jobs, devices, workers_per_device=int(args.workers_per_device))
    for worker_jobs in assigned.values():
        for job in worker_jobs:
            patch_config_device(job.config_path, job.device)
    plan_path = out_root / f"{args.phase}_assignment.json"
    plan_path.write_text(
        json.dumps(
            {
                worker: [
                    {
                        "horizon": job.horizon,
                        "candidate": job.candidate_name,
                        "device": job.device,
                        "config_path": str(job.config_path),
                        "out_dir": str(job.out_dir),
                    }
                    for job in worker_jobs
                ]
                for worker, worker_jobs in assigned.items()
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(f"Planned {len(jobs)} jobs across {len(assigned)} workers. Assignment: {plan_path}", flush=True)
    if args.dry_run:
        return
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=len(assigned)) as executor:
        futures = [
            executor.submit(
                worker_loop,
                worker=worker,
                jobs=worker_jobs,
                phase=str(args.phase),
                candidate_by_key=candidate_by_key,
                python_exe=str(args.python),
                rows=rows,
                results_path=result_path,
                lock=lock,
                reuse_existing=bool(args.reuse_existing),
            )
            for worker, worker_jobs in assigned.items()
            if worker_jobs
        ]
        for future in as_completed(futures):
            future.result()
    write_rows(result_path, rows)
    write_rankings(out_root, str(args.phase))
    print(f"Wrote results: {result_path}", flush=True)
    print(f"Wrote ranking: {out_root / f'{args.phase}_ranking.md'}", flush=True)


if __name__ == "__main__":
    main()
