from __future__ import annotations

import json
from pathlib import Path

import yaml

import scripts.run_full_learnable_anchor_matrix as runner
from scripts.run_full_learnable_anchor_matrix import (
    DEFAULT_DEVICES,
    DEFAULT_WORKERS_PER_DEVICE,
    Job,
    assign_jobs_to_devices,
    as_backbone_job,
    build_matrix,
    backbone_summary_path_for,
    backbone_config_path,
    backbone_out_dir,
    configure_backbone_run,
    configure_run,
    format_duration,
    format_progress_line,
    learnable_anchor_config,
    prepare_configs,
    row_from_summary,
    run_environment_for_job,
    run_assigned,
)


def test_matrix_covers_requested_dataset_horizons() -> None:
    jobs = build_matrix(out_root=Path("outputs/full"), devices=("cuda:0",))

    keys = {(job.dataset, job.horizon) for job in jobs}

    for dataset in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity"):
        for horizon in (96, 192, 336, 720):
            assert (dataset, horizon) in keys
    for dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        for horizon in (12, 24, 48, 96):
            assert (dataset, horizon) in keys
    assert len(jobs) == 40


def test_matrix_can_filter_to_ett_datasets_and_standard_horizons() -> None:
    jobs = build_matrix(
        out_root=Path("outputs/ett"),
        devices=("cuda:0",),
        datasets=("ETTh1", "ETTh2", "ETTm1", "ETTm2"),
        horizons=(96, 192, 336, 720),
    )

    keys = [(job.dataset, job.horizon) for job in jobs]

    assert len(jobs) == 16
    assert keys == [
        (dataset, horizon)
        for dataset in ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
        for horizon in (96, 192, 336, 720)
    ]
    assert {job.device for job in jobs} == {"cuda:0"}


def test_configure_run_enables_pkr_moe_and_learnable_anchor_without_changing_training_schedule() -> None:
    base_cfg = {
        "exp": {"name": "base", "out_dir": "outputs/base", "device": "cuda:7"},
        "window": {"input_len": 96, "pred_len": 96},
        "corr": {"save_path": "outputs/base/corr.npy"},
        "portrait": {"out_dir": "outputs/base/portraits"},
        "memory": {"path": "old/memory.pt", "checkpoint_path": "old/best.pt"},
        "eval": {"skip_test": False},
        "train": {"epochs": 36, "lr": 0.001, "freeze_backbone": False},
        "finetune": {
            "enable": True,
            "checkpoint_path": "outputs/missing_backbone/best_checkpoint.pt",
            "load_model": True,
        },
        "moe": {
            "enable": False,
            "freeze_backbone": True,
            "pred_side_residual": {
                "enable": True,
                "selection_policy": "val_mse_candidate_channel_guarded",
            },
        },
    }
    job = Job(
        dataset="ETTh2",
        horizon=192,
        base_config_path=Path("configs/ETTh2_H192.yaml"),
        config_path=Path("generated/ETTh2_H192.yaml"),
        out_dir=Path("outputs/full/ETTh2/H192"),
        device="cuda:1",
    )

    cfg = configure_run(
        base_cfg,
        job=job,
        skip_test=False,
        disable_pred_side_residual=True,
    )

    assert cfg["exp"]["name"] == "ETTh2_H192_learnable_anchor_full"
    assert cfg["exp"]["device"] == "cuda:1"
    assert cfg["exp"]["out_dir"] == "outputs/full/ETTh2/H192"
    assert cfg["window"]["pred_len"] == 192
    assert cfg["corr"]["save_path"] == "outputs/full/ETTh2/H192/corr.npy"
    assert cfg["memory"]["checkpoint_path"] == "outputs/full/ETTh2/H192/best_checkpoint.pt"
    assert cfg["portrait"]["out_dir"] == "outputs/full/ETTh2/H192/cluster_portraits"
    assert cfg["eval"]["skip_test"] is False
    assert cfg["train"]["epochs"] == 36
    assert cfg["train"]["lr"] == 0.001
    assert cfg["train"]["freeze_backbone"] is True
    assert cfg["finetune"]["enable"] is True
    assert cfg["finetune"]["checkpoint_path"] == "outputs/full/ETTh2/H192_backbone/best_checkpoint.pt"
    assert cfg["finetune"]["load_model"] is True
    assert cfg["finetune"]["strict_window"] is True
    assert cfg["finetune"]["strict_model"] is True
    assert cfg["finetune"]["cluster_map"] == "index"
    assert cfg["moe"]["enable"] is True
    assert cfg["moe"]["freeze_backbone"] is True
    assert cfg["moe"]["pred_side_residual"]["enable"] is False
    assert cfg["moe"]["pred_side_residual"]["selection_policy"] == "none"
    assert cfg["moe"]["learnable_output_anchor_refiner"] == learnable_anchor_config()


def test_configure_run_preserves_pred_side_residual_when_not_explicitly_disabled() -> None:
    base_cfg = {
        "exp": {"name": "base", "out_dir": "outputs/base", "device": "cuda:7"},
        "window": {"input_len": 96, "pred_len": 96},
        "train": {"epochs": 36, "freeze_backbone": False},
        "moe": {
            "enable": True,
            "freeze_backbone": False,
            "pred_side_residual": {
                "enable": True,
                "selection_policy": "val_mse_candidate_channel_guarded",
                "corrector_hidden": 32,
            },
        },
    }
    job = Job(
        dataset="ETTh1",
        horizon=96,
        base_config_path=Path("configs/ETTh1_H96.yaml"),
        config_path=Path("generated/ETTh1_H96.yaml"),
        out_dir=Path("outputs/full/ETTh1/H96"),
        device="cuda:0",
    )

    cfg = configure_run(
        base_cfg,
        job=job,
        skip_test=True,
        disable_pred_side_residual=False,
    )

    assert cfg["moe"]["pred_side_residual"]["enable"] is True
    assert cfg["moe"]["pred_side_residual"]["selection_policy"] == "val_mse_candidate_channel_guarded"
    assert cfg["moe"]["pred_side_residual"]["corrector_hidden"] == 32


def test_configure_backbone_run_trains_and_saves_backbone_checkpoint() -> None:
    base_cfg = {
        "exp": {"name": "base", "out_dir": "outputs/base", "device": "cuda:0"},
        "window": {"input_len": 96, "pred_len": 96},
        "finetune": {
            "enable": True,
            "checkpoint_path": "outputs/missing/best_checkpoint.pt",
            "load_model": True,
        },
        "train": {"epochs": 12, "freeze_backbone": True},
        "moe": {
            "enable": True,
            "freeze_backbone": True,
            "history_anchor_expert": {
                "enable": True,
                "lags": [96],
                "alpha": 0.2,
            },
            "train_stat_anchor_expert": {
                "enable": True,
                "period": 96,
                "alpha": 0.1,
            },
            "train_residual_anchor_expert": {
                "enable": True,
                "period": 96,
                "alpha": 0.2,
            },
            "pred_side_residual": {
                "enable": True,
                "selection_policy": "val_mse_candidate_channel_guarded",
            },
            "learnable_output_anchor_refiner": {"enable": True},
        },
        "memory": {"save_checkpoint": False, "checkpoint_path": "old/best.pt"},
    }
    job = Job(
        dataset="PEMS08",
        horizon=96,
        base_config_path=Path("configs/PEMS08_H96.yaml"),
        config_path=Path("generated/PEMS08_H96.yaml"),
        out_dir=Path("outputs/full/PEMS08/H96"),
        device="cuda:5",
    )

    cfg = configure_backbone_run(
        base_cfg,
        job=job,
    )

    assert cfg["exp"]["name"] == "PEMS08_H96_backbone_full"
    assert cfg["exp"]["out_dir"] == "outputs/full/PEMS08/H96_backbone"
    assert cfg["exp"]["device"] == "cuda:5"
    assert cfg["window"]["pred_len"] == 96
    assert cfg["eval"]["skip_test"] is True
    assert cfg["finetune"] == {"enable": False}
    assert cfg["train"]["epochs"] == 12
    assert cfg["train"]["freeze_backbone"] is False
    assert cfg["moe"]["enable"] is False
    assert cfg["moe"]["freeze_backbone"] is False
    assert cfg["moe"]["history_anchor_expert"] == {"enable": False}
    assert cfg["moe"]["train_stat_anchor_expert"] == {"enable": False}
    assert cfg["moe"]["train_residual_anchor_expert"] == {"enable": False}
    assert cfg["moe"]["pred_side_residual"]["enable"] is False
    assert cfg["moe"]["pred_side_residual"]["selection_policy"] == "none"
    assert cfg["moe"]["learnable_output_anchor_refiner"]["enable"] is False
    assert cfg["memory"]["save_checkpoint"] is True
    assert cfg["memory"]["checkpoint_path"] == "outputs/full/PEMS08/H96_backbone/best_checkpoint.pt"


def test_configure_backbone_run_uses_main_table_epoch_floor_for_stage2_configs() -> None:
    base_cfg = {
        "exp": {"name": "base", "out_dir": "outputs/base", "device": "cuda:0"},
        "window": {"input_len": 96, "pred_len": 96},
        "finetune": {"enable": True, "checkpoint_path": "old/best.pt"},
        "train": {"epochs": 1, "freeze_backbone": True},
        "moe": {"enable": True, "freeze_backbone": True},
        "memory": {"save_checkpoint": False},
    }
    job = Job(
        dataset="ETTh1",
        horizon=96,
        base_config_path=Path("configs/ETTh1_H96.yaml"),
        config_path=Path("generated/ETTh1_H96.yaml"),
        out_dir=Path("outputs/full/ETTh1/H96"),
        device="cuda:0",
    )

    cfg = configure_backbone_run(base_cfg, job=job)
    config_policy_cfg = configure_backbone_run(base_cfg, job=job, backbone_epoch_policy="config")

    assert cfg["train"]["epochs"] == 21
    assert config_policy_cfg["train"]["epochs"] == 1


def test_configure_backbone_run_extends_early_stop_patience_to_reach_main_table_epoch_floor() -> None:
    base_cfg = {
        "exp": {"name": "base", "out_dir": "outputs/base", "device": "cuda:0"},
        "window": {"input_len": 96, "pred_len": 192},
        "finetune": {"enable": True, "checkpoint_path": "old/best.pt"},
        "train": {"epochs": 1, "freeze_backbone": True},
        "early_stop": {"patience": 1, "min_delta": 1.0e-6},
        "moe": {"enable": True, "freeze_backbone": True},
        "memory": {"save_checkpoint": False},
    }
    job = Job(
        dataset="weather",
        horizon=192,
        base_config_path=Path("configs/weather_H192.yaml"),
        config_path=Path("generated/weather_H192.yaml"),
        out_dir=Path("outputs/full/weather/H192"),
        device="cuda:0",
    )

    cfg = configure_backbone_run(base_cfg, job=job)
    config_policy_cfg = configure_backbone_run(base_cfg, job=job, backbone_epoch_policy="config")

    assert cfg["train"]["epochs"] == 55
    assert cfg["early_stop"]["patience"] == 55
    assert config_policy_cfg["early_stop"]["patience"] == 1


def test_configure_backbone_run_restores_main_table_lr_for_frozen_stage2_configs() -> None:
    base_cfg = {
        "exp": {"name": "base", "out_dir": "outputs/base", "device": "cuda:0"},
        "window": {"input_len": 96, "pred_len": 720},
        "finetune": {"enable": True, "checkpoint_path": "old/best.pt"},
        "train": {"epochs": 1, "lr": 0.0, "freeze_backbone": True},
        "early_stop": {"patience": 10, "min_delta": 1.0e-6},
        "moe": {"enable": True, "freeze_backbone": True},
        "memory": {"save_checkpoint": False},
    }
    job = Job(
        dataset="ETTh1",
        horizon=720,
        base_config_path=Path("configs/ETTh1_H720.yaml"),
        config_path=Path("generated/ETTh1_H720.yaml"),
        out_dir=Path("outputs/full/ETTh1/H720"),
        device="cuda:0",
    )

    cfg = configure_backbone_run(base_cfg, job=job)
    config_policy_cfg = configure_backbone_run(base_cfg, job=job, backbone_epoch_policy="config")

    assert cfg["train"]["lr"] == 0.001
    assert config_policy_cfg["train"]["lr"] == 0.0


def test_two_stage_paths_are_derived_from_stage2_job() -> None:
    job = Job(
        dataset="weather",
        horizon=720,
        base_config_path=Path("configs/weather_H720.yaml"),
        config_path=Path("outputs/full/configs/weather/H720_stage2.yaml"),
        out_dir=Path("outputs/full/runs/weather/H720"),
        device="cuda:2",
    )

    assert backbone_config_path(job) == Path("outputs/full/configs/weather/H720_backbone.yaml")
    assert backbone_out_dir(job) == Path("outputs/full/runs/weather/H720_backbone")
    assert as_backbone_job(job).config_path == backbone_config_path(job)
    assert as_backbone_job(job).out_dir == backbone_out_dir(job)


def test_prepare_configs_writes_backbone_and_stage2_configs(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "exp": {"device": "cuda:7", "out_dir": "old"},
                "window": {"input_len": 96, "pred_len": 96},
                "train": {"epochs": 2, "freeze_backbone": True},
                "finetune": {"enable": True, "checkpoint_path": "old/best.pt"},
                "moe": {"enable": True, "freeze_backbone": True},
                "memory": {"save_checkpoint": False},
            }
        ),
        encoding="utf-8",
    )
    job = Job(
        dataset="ETTh1",
        horizon=96,
        base_config_path=base_path,
        config_path=tmp_path / "configs" / "ETTh1" / "H96_stage2.yaml",
        out_dir=tmp_path / "runs" / "ETTh1" / "H96",
        device="cuda:0",
    )

    prepare_configs([job], skip_test=True, disable_pred_side_residual=True)

    backbone_cfg = yaml.safe_load(backbone_config_path(job).read_text(encoding="utf-8"))
    stage2_cfg = yaml.safe_load(job.config_path.read_text(encoding="utf-8"))
    assert backbone_cfg["moe"]["enable"] is False
    assert backbone_cfg["memory"]["save_checkpoint"] is True
    assert backbone_cfg["memory"]["checkpoint_path"].endswith("H96_backbone/best_checkpoint.pt")
    assert stage2_cfg["moe"]["enable"] is True
    assert stage2_cfg["moe"]["freeze_backbone"] is True
    assert stage2_cfg["finetune"]["checkpoint_path"].endswith("H96_backbone/best_checkpoint.pt")


def test_prepare_configs_can_write_backbone_only_repro_configs(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "exp": {"device": "cuda:7", "out_dir": "old"},
                "window": {"input_len": 96, "pred_len": 96},
                "train": {"epochs": 1, "freeze_backbone": True},
                "finetune": {"enable": True, "checkpoint_path": "old/best.pt"},
                "moe": {"enable": True, "freeze_backbone": True},
                "memory": {"save_checkpoint": False},
            }
        ),
        encoding="utf-8",
    )
    job = Job(
        dataset="ETTh1",
        horizon=96,
        base_config_path=base_path,
        config_path=tmp_path / "configs" / "ETTh1" / "H96_stage2.yaml",
        out_dir=tmp_path / "runs" / "ETTh1" / "H96",
        device="cuda:0",
    )

    prepare_configs(
        [job],
        skip_test=True,
        disable_pred_side_residual=False,
        include_stage2=False,
    )

    backbone_cfg = yaml.safe_load(backbone_config_path(job).read_text(encoding="utf-8"))
    assert backbone_cfg["train"]["epochs"] == 21
    assert backbone_cfg["moe"]["enable"] is False
    assert backbone_cfg["finetune"] == {"enable": False}
    assert not job.config_path.exists()


def test_row_from_summary_exports_learnable_test_generalization_metrics(tmp_path: Path) -> None:
    out_dir = tmp_path / "runs" / "ETTh1" / "H96"
    out_dir.mkdir(parents=True)
    (out_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "val": {"avg_mse": 1.0, "avg_mae": 2.0},
                "test": {"avg_mse": 3.0, "avg_mae": 4.0},
                "learnable_output_anchor_refiner": {
                    "val_static_mse": 1.1,
                    "val_refined_mse": 1.0,
                    "val_static_mae": 2.2,
                    "val_refined_mae": 2.0,
                    "test_static_mse": 3.3,
                    "test_refined_mse": 3.0,
                    "test_static_mae": 4.4,
                    "test_refined_mae": 4.0,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    job = Job(
        dataset="ETTh1",
        horizon=96,
        base_config_path=Path("base.yaml"),
        config_path=tmp_path / "cfg.yaml",
        out_dir=out_dir,
        device="cuda:0",
    )

    row = row_from_summary(job, status="ok")

    assert row["learnable_test_static_mse"] == 3.3
    assert row["learnable_test_refined_mse"] == 3.0
    assert row["learnable_test_static_mae"] == 4.4
    assert row["learnable_test_refined_mae"] == 4.0


def test_assign_jobs_to_devices_round_robins_over_workers() -> None:
    jobs = [
        Job(
            dataset="weather",
            horizon=96 + idx,
            base_config_path=Path("base.yaml"),
            config_path=Path(f"cfg{idx}.yaml"),
            out_dir=Path(f"out{idx}"),
            device="",
        )
        for idx in range(5)
    ]

    assigned = assign_jobs_to_devices(jobs, devices=("cuda:0", "cuda:2"), workers_per_device=2)

    assert list(assigned) == ["cuda:0#1", "cuda:2#1", "cuda:0#2", "cuda:2#2"]
    assert [job.device for job in assigned["cuda:0#1"]] == ["cuda:0", "cuda:0"]
    assert [job.horizon for job in assigned["cuda:0#1"]] == [96, 100]
    assert [job.device for job in assigned["cuda:2#1"]] == ["cuda:2"]
    assert [job.horizon for job in assigned["cuda:2#1"]] == [97]


def test_defaults_target_server_parallelism() -> None:
    assert DEFAULT_DEVICES == ("cuda:0", "cuda:2", "cuda:5")
    assert DEFAULT_WORKERS_PER_DEVICE == 2


def test_run_environment_does_not_remap_physical_cuda_device() -> None:
    job = Job(
        dataset="PEMS08",
        horizon=96,
        base_config_path=Path("base.yaml"),
        config_path=Path("cfg.yaml"),
        out_dir=Path("out"),
        device="cuda:5",
    )

    env = run_environment_for_job(job)

    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["PYTHONUTF8"] == "1"


def test_full_stage_runs_all_backbones_before_any_stage2(monkeypatch, tmp_path: Path) -> None:
    jobs = [
        Job(
            dataset="ETTm1",
            horizon=96,
            base_config_path=Path("base.yaml"),
            config_path=tmp_path / "configs" / "ETTm1" / "H96_stage2.yaml",
            out_dir=tmp_path / "runs" / "ETTm1" / "H96",
            device="cuda:0",
        ),
        Job(
            dataset="ETTm2",
            horizon=192,
            base_config_path=Path("base.yaml"),
            config_path=tmp_path / "configs" / "ETTm2" / "H192_stage2.yaml",
            out_dir=tmp_path / "runs" / "ETTm2" / "H192",
            device="cuda:0",
        ),
    ]
    assigned = {"cuda:0#1": jobs}
    calls: list[str] = []

    def fake_run_job(job: Job, *, python_exe: str, resume: bool, log_dir: Path) -> dict:
        _ = python_exe, resume, log_dir
        calls.append(job.out_dir.as_posix())
        job.out_dir.mkdir(parents=True, exist_ok=True)
        (job.out_dir / "run_summary.json").write_text(
            json.dumps({"val": {"avg_mse": 1.0, "avg_mae": 2.0}}),
            encoding="utf-8",
        )
        return runner.row_from_summary(job, status="ok")

    monkeypatch.setattr(runner, "run_job", fake_run_job)

    rows = runner.run_assigned(
        assigned,
        python_exe="python",
        resume=True,
        summary_path=tmp_path / "summary.csv",
        log_dir=tmp_path / "logs",
        stage="full",
        progress=None,
    )

    assert calls == [
        as_backbone_job(jobs[0]).out_dir.as_posix(),
        as_backbone_job(jobs[1]).out_dir.as_posix(),
        jobs[0].out_dir.as_posix(),
        jobs[1].out_dir.as_posix(),
    ]
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"ok"}
    assert backbone_summary_path_for(tmp_path / "summary.csv").exists()


def test_full_stage_skips_stage2_when_any_backbone_fails(monkeypatch, tmp_path: Path) -> None:
    jobs = [
        Job(
            dataset="ETTm1",
            horizon=96,
            base_config_path=Path("base.yaml"),
            config_path=tmp_path / "configs" / "ETTm1" / "H96_stage2.yaml",
            out_dir=tmp_path / "runs" / "ETTm1" / "H96",
            device="cuda:0",
        )
    ]
    calls: list[str] = []

    def fake_run_job(job: Job, *, python_exe: str, resume: bool, log_dir: Path) -> dict:
        _ = python_exe, resume, log_dir
        calls.append(job.out_dir.as_posix())
        return runner.row_from_summary(job, status="failed", returncode=1, error="boom")

    monkeypatch.setattr(runner, "run_job", fake_run_job)

    rows = runner.run_assigned(
        {"cuda:0#1": jobs},
        python_exe="python",
        resume=True,
        summary_path=tmp_path / "summary.csv",
        log_dir=tmp_path / "logs",
        stage="full",
        progress=None,
    )

    assert calls == [as_backbone_job(jobs[0]).out_dir.as_posix()]
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "boom"


def test_format_duration_is_stable_for_progress_output() -> None:
    assert format_duration(0) == "00:00:00"
    assert format_duration(65.9) == "00:01:05"
    assert format_duration(3661) == "01:01:01"


def test_format_progress_line_shows_count_percent_job_worker_and_error() -> None:
    job = Job(
        dataset="ETTh1",
        horizon=96,
        base_config_path=Path("base.yaml"),
        config_path=Path("cfg.yaml"),
        out_dir=Path("out"),
        device="cuda:0",
    )

    line = format_progress_line(
        completed=3,
        total=40,
        job=job,
        worker_key="cuda:0#1",
        status="failed",
        elapsed_s=65,
        error="see outputs/logs/ETTh1_H96.log",
    )

    assert line == (
        "[3/40 7.5%] FAILED ETTh1_H96 device=cuda:0 "
        "worker=cuda:0#1 elapsed=00:01:05 error=see outputs/logs/ETTh1_H96.log"
    )


def test_run_assigned_emits_start_and_finish_progress(tmp_path: Path, monkeypatch) -> None:
    job = Job(
        dataset="weather",
        horizon=96,
        base_config_path=Path("base.yaml"),
        config_path=Path("cfg.yaml"),
        out_dir=tmp_path / "weather_H96",
        device="cuda:0",
    )

    def fake_run_job(job: Job, *, python_exe: str, resume: bool, log_dir: Path):
        assert python_exe
        assert resume is False
        assert log_dir == tmp_path / "logs"
        return {
            "status": "ok",
            "dataset": job.dataset,
            "horizon": job.horizon,
            "device": job.device,
            "config_path": str(job.config_path),
            "out_dir": str(job.out_dir),
        }

    monkeypatch.setattr(runner, "run_job", fake_run_job)
    progress_lines: list[str] = []

    run_assigned(
        {"cuda:0#1": [job]},
        python_exe="python",
        resume=False,
        summary_path=tmp_path / "summary.csv",
        log_dir=tmp_path / "logs",
        stage="backbone",
        progress=progress_lines.append,
    )

    assert progress_lines[0].startswith("[0/1 0.0%] BACKBONE_START weather_H96")
    assert progress_lines[1].startswith("[1/1 100.0%] BACKBONE_OK weather_H96")
