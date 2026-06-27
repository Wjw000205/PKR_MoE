from __future__ import annotations

from pathlib import Path

from scripts.run_full_learnable_anchor_matrix import (
    DEFAULT_DEVICES,
    DEFAULT_WORKERS_PER_DEVICE,
    Job,
    assign_jobs_to_devices,
    build_matrix,
    configure_run,
    learnable_anchor_config,
    run_environment_for_job,
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


def test_configure_run_enables_pkr_moe_and_learnable_anchor_without_changing_training_schedule() -> None:
    base_cfg = {
        "exp": {"name": "base", "out_dir": "outputs/base", "device": "cuda:7"},
        "window": {"input_len": 96, "pred_len": 96},
        "corr": {"save_path": "outputs/base/corr.npy"},
        "portrait": {"out_dir": "outputs/base/portraits"},
        "memory": {"path": "old/memory.pt", "checkpoint_path": "old/best.pt"},
        "eval": {"skip_test": False},
        "train": {"epochs": 36, "lr": 0.001},
        "moe": {
            "enable": False,
            "pred_side_residual": {"enable": True},
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
    assert cfg["train"] == {"epochs": 36, "lr": 0.001}
    assert cfg["moe"]["enable"] is True
    assert cfg["moe"]["pred_side_residual"]["enable"] is False
    assert cfg["moe"]["learnable_output_anchor_refiner"] == learnable_anchor_config()


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
