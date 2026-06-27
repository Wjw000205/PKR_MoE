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


def iter_dataset_horizons() -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for dataset in STANDARD_DATASETS:
        for horizon in STANDARD_HORIZONS:
            pairs.append((dataset, horizon))
    for dataset in PEMS_DATASETS:
        for horizon in PEMS_HORIZONS:
            pairs.append((dataset, horizon))
    return pairs


def build_matrix(*, out_root: Path, devices: tuple[str, ...] = DEFAULT_DEVICES) -> list[Job]:
    jobs: list[Job] = []
    if not devices:
        raise ValueError("At least one device is required.")
    for idx, (dataset, horizon) in enumerate(iter_dataset_horizons()):
        config_path = out_root / "configs" / dataset / f"H{horizon}.yaml"
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


def configure_run(
    base_cfg: dict[str, Any],
    *,
    job: Job,
    skip_test: bool,
    disable_pred_side_residual: bool,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = f"{job.dataset}_H{job.horizon}_learnable_anchor_full"
    cfg["exp"]["out_dir"] = job.out_dir.as_posix()
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
    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = bool(skip_test)

    cfg.setdefault("moe", {})
    cfg["moe"]["enable"] = True
    if disable_pred_side_residual:
        cfg["moe"].setdefault("pred_side_residual", {})
        cfg["moe"]["pred_side_residual"]["enable"] = False
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
        cfg = configure_run(
            read_yaml(job.base_config_path),
            job=job,
            skip_test=skip_test,
            disable_pred_side_residual=disable_pred_side_residual,
        )
        write_yaml(job.config_path, cfg)


def run_job(job: Job, *, python_exe: str, resume: bool, log_dir: Path) -> dict[str, Any]:
    summary_path = job.out_dir / "run_summary.json"
    if resume and completed_summary(summary_path):
        return row_from_summary(job, status="skipped")

    job.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.dataset}_H{job.horizon}.log"
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
            row = run_job(job, python_exe=python_exe, resume=resume, log_dir=log_dir)
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
    parser.add_argument("--dry-run", action="store_true", help="Only generate configs and summary plan.")
    parser.add_argument("--resume", action="store_true", help="Skip jobs with completed run_summary.json.")
    parser.add_argument("--skip-test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-pred-side-residual", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    devices = tuple(device.strip() for device in str(args.devices).split(",") if device.strip())
    jobs = build_matrix(out_root=out_root, devices=devices)
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
    print(f"Generated {len(jobs)} configs under {out_root / 'configs'}")
    print(f"Summary: {summary_path}")
    print(f"Devices: {', '.join(devices)}; workers/device={args.workers_per_device}")
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
