from __future__ import annotations

from pathlib import Path

from scripts.run_electricity_param_search import (
    DEFAULT_DEVICES,
    DEFAULT_WORKERS_PER_DEVICE,
    SearchJob,
    assign_jobs_to_devices,
    configure,
    seed_candidates,
)


def test_electricity_param_search_defaults_to_cuda0_and_cuda2() -> None:
    assert DEFAULT_DEVICES == ("cuda:0", "cuda:2")
    assert DEFAULT_WORKERS_PER_DEVICE == 2


def test_electricity_param_search_config_is_val_only_by_default(tmp_path: Path) -> None:
    cand = seed_candidates()[0]

    cfg = configure(
        {
            "exp": {"seed": 2026},
            "data": {"csv_path": "data/electricity.csv"},
            "window": {"input_len": 96, "pred_len": 96},
            "cluster": {"random_state": 2026},
            "model": {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.0},
            "moe": {"enable": True},
            "penalties": {"enabled": ["amp_under", "delta"]},
            "train": {"epochs": 100, "batch_size": 64, "lr": 0.001},
            "memory": {"checkpoint_path": "old/best_checkpoint.pt"},
        },
        horizon=192,
        cand=cand,
        phase="search",
        out_dir=tmp_path / "run",
        device="cuda:2",
        epochs=12,
        skip_test=True,
        save_checkpoint=True,
    )

    assert cfg["exp"]["device"] == "cuda:2"
    assert cfg["exp"]["out_dir"] == str(tmp_path / "run")
    assert cfg["data"]["csv_path"] == "data/electricity.csv"
    assert cfg["window"]["input_len"] == 96
    assert cfg["window"]["pred_len"] == 192
    assert cfg["window"]["past_context"] is True
    assert cfg["eval"]["skip_test"] is True
    assert cfg["memory"]["save_checkpoint"] is True
    assert cfg["memory"]["checkpoint_path"] == str(tmp_path / "run" / "best_checkpoint.pt")
    assert cfg["corr"]["save_path"] == str(tmp_path / "run" / "corr.npy")
    assert cfg["train"]["selection_metric"] == "val_mse"
    assert cfg["train"]["epochs"] == 12


def test_electricity_param_search_final_config_can_read_test_once(tmp_path: Path) -> None:
    cand = seed_candidates()[0]

    cfg = configure(
        {},
        horizon=96,
        cand=cand,
        phase="final",
        out_dir=tmp_path / "final",
        device="cuda:0",
        epochs=1,
        skip_test=False,
        save_checkpoint=False,
    )

    assert cfg["eval"]["skip_test"] is False
    assert cfg["memory"]["save_checkpoint"] is False
    assert cfg["exp"]["name"].endswith("_final")


def test_electricity_param_search_uses_train_supported_residual_selection_policy(tmp_path: Path) -> None:
    supported = {
        "none",
        "val_mse_channel",
        "val_mse_scale",
        "val_mse_scale_holdout",
        "val_mse_candidate_channel",
    }

    for cand in seed_candidates():
        cfg = configure(
            {},
            horizon=96,
            cand=cand,
            phase="search",
            out_dir=tmp_path / cand.name,
            device="cuda:0",
            epochs=1,
            skip_test=True,
            save_checkpoint=False,
        )

        assert cfg["moe"]["pred_side_residual"]["selection_policy"] in supported


def test_electricity_param_search_assigns_jobs_round_robin_to_four_workers(tmp_path: Path) -> None:
    jobs = [
        SearchJob(
            horizon=96,
            candidate_name=f"cand{i}",
            config_path=tmp_path / f"cfg{i}.yaml",
            out_dir=tmp_path / f"run{i}",
            device="",
        )
        for i in range(5)
    ]

    assigned = assign_jobs_to_devices(jobs, ("cuda:0", "cuda:2"), workers_per_device=2)

    assert list(assigned) == ["cuda:0#1", "cuda:2#1", "cuda:0#2", "cuda:2#2"]
    assert [job.candidate_name for job in assigned["cuda:0#1"]] == ["cand0", "cand4"]
    assert [job.device for job in assigned["cuda:0#1"]] == ["cuda:0", "cuda:0"]
    assert [job.candidate_name for job in assigned["cuda:2#1"]] == ["cand1"]
    assert [job.device for job in assigned["cuda:2#1"]] == ["cuda:2"]
    assert [job.candidate_name for job in assigned["cuda:0#2"]] == ["cand2"]
    assert [job.device for job in assigned["cuda:0#2"]] == ["cuda:0"]
    assert [job.candidate_name for job in assigned["cuda:2#2"]] == ["cand3"]
    assert [job.device for job in assigned["cuda:2#2"]] == ["cuda:2"]
