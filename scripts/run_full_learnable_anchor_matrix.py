from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVICES = ("cuda:0", "cuda:2", "cuda:5")
DEFAULT_WORKERS_PER_DEVICE = 2
STANDARD_HORIZONS = (96, 192, 336, 720)
PEMS_HORIZONS = (12, 24, 48, 96)
STANDARD_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity")
PEMS_DATASETS = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")
ALL_DATASETS = STANDARD_DATASETS + PEMS_DATASETS

SUMMARY_FIELDS = [
    "status",
    "dataset",
    "horizon",
    "device",
    "worker",
    "config_path",
    "out_dir",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "learnable_adopted",
    "learnable_adopted_channels",
    "learnable_val_static_mse",
    "learnable_val_refined_mse",
    "learnable_val_static_mae",
    "learnable_val_refined_mae",
    "learnable_test_static_mse",
    "learnable_test_refined_mse",
    "learnable_test_static_mae",
    "learnable_test_refined_mae",
    "learnable_test_mse_gain",
    "learnable_test_mae_gain",
    "best_epoch",
    "total_sec",
    "avg_epoch_sec",
    "returncode",
    "error",
]


@dataclass(frozen=True)
class Job:
    dataset: str
    horizon: int
    base_config_path: Path
    config_path: Path
    out_dir: Path
    device: str


def learnable_anchor_config() -> dict[str, Any]:
    return {
        "enable": True,
        "hidden_dim": 16,
        "epochs": 3,
        "lr": 0.001,
        "weight_decay": 0.0,
        "mae_weight": 1.0,
        "selection_metric": "mse",
        "adoption_scope": "channel",
        "max_delta_scale": 1.0,
        "init": "zero_delta",
        "min_abs_improvement": 0.0,
        "min_rel_improvement": 0.0,
        "max_rel_mae_regression": 0.0,
    }


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def base_config_path(dataset: str, horizon: int) -> Path:
    return ROOT / "configs" / f"{dataset}_H{horizon}.yaml"


def iter_dataset_horizons(
    *,
    datasets: tuple[str, ...] | None = None,
    horizons: tuple[int, ...] | None = None,
) -> list[tuple[str, int]]:
    dataset_filter = set(datasets) if datasets is not None else None
    horizon_filter = set(int(horizon) for horizon in horizons) if horizons is not None else None
    pairs: list[tuple[str, int]] = []
    for dataset in STANDARD_DATASETS:
        if dataset_filter is not None and dataset not in dataset_filter:
            continue
        for horizon in STANDARD_HORIZONS:
            if horizon_filter is not None and horizon not in horizon_filter:
                continue
            pairs.append((dataset, horizon))
    for dataset in PEMS_DATASETS:
        if dataset_filter is not None and dataset not in dataset_filter:
            continue
        for horizon in PEMS_HORIZONS:
            if horizon_filter is not None and horizon not in horizon_filter:
                continue
            pairs.append((dataset, horizon))
    return pairs


def build_matrix(
    *,
    out_root: Path,
    devices: tuple[str, ...] = DEFAULT_DEVICES,
    datasets: tuple[str, ...] | None = None,
    horizons: tuple[int, ...] | None = None,
) -> list[Job]:
    jobs: list[Job] = []
    if not devices:
        raise ValueError("At least one device is required.")
    dataset_horizons = iter_dataset_horizons(datasets=datasets, horizons=horizons)
    if not dataset_horizons:
        raise ValueError("No dataset/horizon jobs matched the requested filters.")
    for idx, (dataset, horizon) in enumerate(dataset_horizons):
        config_path = out_root / "configs" / dataset / f"H{horizon}_stage2.yaml"
        out_dir = out_root / "runs" / dataset / f"H{horizon}"
        jobs.append(
            Job(
                dataset=dataset,
                horizon=int(horizon),
                base_config_path=base_config_path(dataset, horizon),
                config_path=config_path,
                out_dir=out_dir,
                device=devices[idx % len(devices)],
            )
        )
    return jobs


def parse_dataset_filter(value: str) -> tuple[str, ...] | None:
    value = str(value or "").strip()
    if not value or value.lower() == "all":
        return None
    canonical = {dataset.lower(): dataset for dataset in ALL_DATASETS}
    datasets: list[str] = []
    for item in value.replace(";", ",").split(","):
        key = item.strip()
        if not key:
            continue
        dataset = canonical.get(key.lower())
        if dataset is None:
            raise ValueError(f"Unknown dataset '{key}'. Valid datasets: {', '.join(ALL_DATASETS)}")
        datasets.append(dataset)
    return tuple(dict.fromkeys(datasets))


def parse_horizon_filter(value: str) -> tuple[int, ...] | None:
    value = str(value or "").strip()
    if not value or value.lower() == "all":
        return None
    horizons: list[int] = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        horizons.append(int(item))
    return tuple(dict.fromkeys(horizons))


def backbone_config_path(job: Job) -> Path:
    return job.config_path.with_name(f"H{job.horizon}_backbone.yaml")


def backbone_out_dir(job: Job) -> Path:
    return job.out_dir.with_name(f"H{job.horizon}_backbone")


def backbone_checkpoint_path(job: Job) -> Path:
    return backbone_out_dir(job) / "best_checkpoint.pt"


def configure_common_paths(cfg: dict[str, Any], *, job: Job) -> None:
    cfg.setdefault("exp", {})
    cfg["exp"]["device"] = str(job.device)

    cfg.setdefault("window", {})
    cfg["window"]["pred_len"] = int(job.horizon)
    cfg["window"].setdefault("input_len", 96)
    cfg["window"]["past_context"] = True

    cfg.setdefault("corr", {})
    cfg["corr"]["save_path"] = (job.out_dir / "corr.npy").as_posix()
    cfg.setdefault("portrait", {})
    cfg["portrait"]["out_dir"] = (job.out_dir / "cluster_portraits").as_posix()
    cfg.setdefault("memory", {})
    cfg["memory"]["path"] = (job.out_dir / "cluster_memory.pt").as_posix()
    cfg["memory"]["checkpoint_path"] = (job.out_dir / "best_checkpoint.pt").as_posix()


def disable_pred_side_residual_config(cfg: dict[str, Any]) -> None:
    cfg.setdefault("moe", {})
    cfg["moe"].setdefault("pred_side_residual", {})
    cfg["moe"]["pred_side_residual"]["enable"] = False
    cfg["moe"]["pred_side_residual"]["selection_policy"] = "none"


def configure_backbone_run(base_cfg: dict[str, Any], *, job: Job) -> dict[str, Any]:
    backbone_job = replace(
        job,
        config_path=backbone_config_path(job),
        out_dir=backbone_out_dir(job),
    )
    cfg = copy.deepcopy(base_cfg)
    configure_common_paths(cfg, job=backbone_job)
    cfg["exp"]["name"] = f"{job.dataset}_H{job.horizon}_backbone_full"
    cfg["exp"]["out_dir"] = backbone_job.out_dir.as_posix()
    cfg["finetune"] = {"enable": False}
    cfg.setdefault("train", {})
    cfg["train"]["freeze_backbone"] = False
    cfg.setdefault("moe", {})
    cfg["moe"]["enable"] = False
    cfg["moe"]["freeze_backbone"] = False
    cfg["moe"]["learnable_output_anchor_refiner"] = {"enable": False}
    disable_pred_side_residual_config(cfg)
    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = True
    cfg.setdefault("memory", {})
    cfg["memory"]["save_checkpoint"] = True
    cfg["memory"]["checkpoint_path"] = backbone_checkpoint_path(job).as_posix()
    return cfg


def configure_run(
    base_cfg: dict[str, Any],
    *,
    job: Job,
    skip_test: bool,
    disable_pred_side_residual: bool,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    configure_common_paths(cfg, job=job)
    cfg["exp"]["name"] = f"{job.dataset}_H{job.horizon}_learnable_anchor_full"
    cfg["exp"]["out_dir"] = job.out_dir.as_posix()
    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = bool(skip_test)
    cfg["finetune"] = {
        "enable": True,
        "checkpoint_path": backbone_checkpoint_path(job).as_posix(),
        "strict_window": True,
        "strict_model": True,
        "cluster_map": "index",
        "load_model": True,
        "load_gate": False,
        "load_dynamic_lambda": False,
        "load_learnable_lambda": False,
    }
    cfg.setdefault("train", {})
    cfg["train"]["freeze_backbone"] = True

    cfg.setdefault("moe", {})
    cfg["moe"]["enable"] = True
    cfg["moe"]["freeze_backbone"] = True
    if disable_pred_side_residual:
        disable_pred_side_residual_config(cfg)
    cfg["moe"]["learnable_output_anchor_refiner"] = learnable_anchor_config()
    return cfg


def assign_jobs_to_devices(
    jobs: list[Job],
    devices: tuple[str, ...],
    workers_per_device: int,
) -> dict[str, list[Job]]:
    if workers_per_device <= 0:
        raise ValueError("workers_per_device must be positive.")
    worker_keys = [
        f"{device}#{worker_idx}"
        for worker_idx in range(1, int(workers_per_device) + 1)
        for device in devices
    ]
    assigned: dict[str, list[Job]] = {key: [] for key in worker_keys}
    for idx, job in enumerate(jobs):
        key = worker_keys[idx % len(worker_keys)]
        device = key.split("#", 1)[0]
        assigned[key].append(replace(job, device=device))
    return assigned


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_progress_line(
    *,
    completed: int,
    total: int,
    job: Job,
    worker_key: str,
    status: str,
    elapsed_s: float,
    error: str = "",
) -> str:
    percent = 0.0 if total <= 0 else completed / total * 100.0
    parts = [
        f"[{completed}/{total} {percent:.1f}%]",
        status.upper(),
        f"{job.dataset}_H{job.horizon}",
        f"device={job.device}",
        f"worker={worker_key}",
        f"elapsed={format_duration(elapsed_s)}",
    ]
    if error:
        parts.append(f"error={error}")
    return " ".join(parts)


def print_progress(line: str) -> None:
    print(line, flush=True)


def completed_summary(path: Path) -> bool:
    summary = read_json(path)
    val = summary.get("val") or {}
    if val.get("avg_mse") is None or val.get("avg_mae") is None:
        return False
    return True


def row_from_summary(job: Job, *, status: str, returncode: int = 0, error: str = "") -> dict[str, Any]:
    summary_path = job.out_dir / "run_summary.json"
    summary = read_json(summary_path)
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    learnable = summary.get("learnable_output_anchor_refiner") or {}
    timing = summary.get("timing") or {}
    return {
        "status": status,
        "dataset": job.dataset,
        "horizon": int(job.horizon),
        "device": job.device,
        "worker": "",
        "config_path": str(job.config_path),
        "out_dir": str(job.out_dir),
        "val_mse": val.get("avg_mse"),
        "val_mae": val.get("avg_mae"),
        "test_mse": test.get("avg_mse"),
        "test_mae": test.get("avg_mae"),
        "learnable_adopted": learnable.get("adopted"),
        "learnable_adopted_channels": learnable.get("adopted_channel_count"),
        "learnable_val_static_mse": learnable.get("val_static_mse"),
        "learnable_val_refined_mse": learnable.get("val_refined_mse"),
        "learnable_val_static_mae": learnable.get("val_static_mae"),
        "learnable_val_refined_mae": learnable.get("val_refined_mae"),
        "learnable_test_static_mse": learnable.get("test_static_mse"),
        "learnable_test_refined_mse": learnable.get("test_refined_mse"),
        "learnable_test_static_mae": learnable.get("test_static_mae"),
        "learnable_test_refined_mae": learnable.get("test_refined_mae"),
        "learnable_test_mse_gain": learnable.get("test_mse_gain"),
        "learnable_test_mae_gain": learnable.get("test_mae_gain"),
        "best_epoch": json.dumps(summary.get("best_epoch", ""), ensure_ascii=False),
        "total_sec": timing.get("total_time_s"),
        "avg_epoch_sec": timing.get("avg_epoch_time_s"),
        "returncode": returncode,
        "error": error,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def prepare_configs(
    jobs: list[Job],
    *,
    skip_test: bool,
    disable_pred_side_residual: bool,
) -> None:
    missing = [job.base_config_path for job in jobs if not job.base_config_path.exists()]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing base configs:\n{formatted}")
    for job in jobs:
        backbone_cfg = configure_backbone_run(read_yaml(job.base_config_path), job=job)
        write_yaml(backbone_config_path(job), backbone_cfg)
        stage2_cfg = configure_run(
            read_yaml(job.base_config_path),
            job=job,
            skip_test=skip_test,
            disable_pred_side_residual=disable_pred_side_residual,
        )
        write_yaml(job.config_path, stage2_cfg)


def run_job(job: Job, *, python_exe: str, resume: bool, log_dir: Path) -> dict[str, Any]:
    summary_path = job.out_dir / "run_summary.json"
    if resume and completed_summary(summary_path):
        return row_from_summary(job, status="skipped")

    job.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.dataset}_{job.config_path.stem}.log"
    start = time.time()
    env = run_environment_for_job(job)
    cmd = [python_exe, "-m", "src.train", "--config", str(job.config_path)]
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("COMMAND: " + " ".join(cmd) + "\n")
        log_file.write(f"DEVICE={job.device}\n")
        log_file.flush()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        return row_from_summary(
            job,
            status="failed",
            returncode=proc.returncode,
            error=f"see {log_path}",
        )
    if not completed_summary(summary_path):
        return row_from_summary(
            job,
            status="failed",
            returncode=proc.returncode,
            error=f"missing/incomplete {summary_path}",
        )
    row = row_from_summary(job, status="ok", returncode=proc.returncode)
    row["total_sec"] = row.get("total_sec") or round(time.time() - start, 3)
    return row


def run_two_stage_job(job: Job, *, python_exe: str, resume: bool, log_dir: Path) -> dict[str, Any]:
    if resume and completed_summary(job.out_dir / "run_summary.json"):
        return row_from_summary(job, status="skipped")

    backbone_job = replace(
        job,
        config_path=backbone_config_path(job),
        out_dir=backbone_out_dir(job),
    )
    backbone_row = run_job(backbone_job, python_exe=python_exe, resume=resume, log_dir=log_dir)
    if backbone_row.get("status") not in {"ok", "skipped"}:
        return row_from_summary(
            job,
            status="failed",
            returncode=int(backbone_row.get("returncode") or 1),
            error=f"backbone stage failed: {backbone_row.get('error', '')}",
        )

    return run_job(job, python_exe=python_exe, resume=resume, log_dir=log_dir)


def os_environ_utf8() -> dict[str, str]:
    import os

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def run_environment_for_job(job: Job) -> dict[str, str]:
    _ = job
    return os_environ_utf8()


def run_assigned(
    assigned: dict[str, list[Job]],
    *,
    python_exe: str,
    resume: bool,
    summary_path: Path,
    log_dir: Path,
    progress: Callable[[str], None] | None = print_progress,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()
    total_jobs = sum(len(jobs) for jobs in assigned.values())
    completed_jobs = 0
    started_at = time.time()

    def run_worker(worker_key: str, worker_jobs: list[Job]) -> list[dict[str, Any]]:
        nonlocal completed_jobs
        worker_rows: list[dict[str, Any]] = []
        for job in worker_jobs:
            if progress is not None:
                with rows_lock:
                    progress(
                        format_progress_line(
                            completed=completed_jobs,
                            total=total_jobs,
                            job=job,
                            worker_key=worker_key,
                            status="start",
                            elapsed_s=time.time() - started_at,
                        )
                    )
            row = run_two_stage_job(job, python_exe=python_exe, resume=resume, log_dir=log_dir)
            row["worker"] = worker_key
            worker_rows.append(row)
            with rows_lock:
                completed_jobs += 1
                rows.append(row)
                write_rows(summary_path, rows)
                if progress is not None:
                    progress(
                        format_progress_line(
                            completed=completed_jobs,
                            total=total_jobs,
                            job=job,
                            worker_key=worker_key,
                            status=str(row.get("status", "done")),
                            elapsed_s=time.time() - started_at,
                            error=str(row.get("error", "")),
                        )
                    )
        return worker_rows

    with ThreadPoolExecutor(max_workers=len(assigned)) as executor:
        futures = [executor.submit(run_worker, key, jobs) for key, jobs in assigned.items() if jobs]
        for future in as_completed(futures):
            future.result()
    rows.sort(key=lambda row: (str(row["dataset"]), int(row["horizon"])))
    write_rows(summary_path, rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default="outputs/full_learnable_anchor_matrix_20260627")
    parser.add_argument("--devices", default=",".join(DEFAULT_DEVICES))
    parser.add_argument("--workers-per-device", type=int, default=DEFAULT_WORKERS_PER_DEVICE)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset list, or all.")
    parser.add_argument("--horizons", default="all", help="Comma-separated horizon list, or all.")
    parser.add_argument("--dry-run", action="store_true", help="Only generate configs and summary plan.")
    parser.add_argument("--resume", action="store_true", help="Skip jobs with completed run_summary.json.")
    parser.add_argument("--skip-test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-pred-side-residual", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    devices = tuple(device.strip() for device in str(args.devices).split(",") if device.strip())
    datasets = parse_dataset_filter(str(args.datasets))
    horizons = parse_horizon_filter(str(args.horizons))
    jobs = build_matrix(out_root=out_root, devices=devices, datasets=datasets, horizons=horizons)
    prepare_configs(
        jobs,
        skip_test=bool(args.skip_test),
        disable_pred_side_residual=bool(args.disable_pred_side_residual),
    )
    assigned = assign_jobs_to_devices(jobs, devices, int(args.workers_per_device))
    summary_path = out_root / "summary.csv"
    log_dir = out_root / "logs"
    plan_rows: list[dict[str, Any]] = []
    for worker_key, worker_jobs in assigned.items():
        for job in worker_jobs:
            row = row_from_summary(job, status="planned")
            row["worker"] = worker_key
            plan_rows.append(row)
    write_rows(summary_path, plan_rows)
    print(f"Generated {len(jobs)} backbone configs and {len(jobs)} stage2 configs under {out_root / 'configs'}")
    print(f"Summary: {summary_path}")
    print(f"Devices: {', '.join(devices)}; workers/device={args.workers_per_device}")
    print("Training mode: two-stage (backbone checkpoint first, then frozen-backbone PKR-MoE + anchor)")
    if args.dry_run:
        return
    rows = run_assigned(
        assigned,
        python_exe=str(args.python),
        resume=bool(args.resume),
        summary_path=summary_path,
        log_dir=log_dir,
    )
    failed = [row for row in rows if row.get("status") == "failed"]
    if failed:
        raise SystemExit(f"{len(failed)} jobs failed. See {summary_path} and {log_dir}.")


if __name__ == "__main__":
    main()
