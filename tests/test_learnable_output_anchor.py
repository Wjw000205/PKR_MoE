import inspect

import torch

import src.train as train_module
from src.models.learnable_anchor import ClusterwiseLearnableOutputAnchorRefiner
from src.train import eval_loop, train_learnable_output_anchor_refiner


class _ZeroBackbone(torch.nn.Module):
    def eval(self):
        return self

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x_bcl.shape[0], x_bcl.shape[1], 2, device=x_bcl.device, dtype=x_bcl.dtype)


class _UnusedGate(torch.nn.Module):
    def eval(self):
        return self


class _AddOneRefiner(torch.nn.Module):
    def eval(self):
        return self

    def forward(
        self,
        *,
        x_bcl: torch.Tensor,
        base_pred_bch: torch.Tensor,
        static_pred_bch: torch.Tensor,
        cluster_id_c: torch.Tensor,
    ) -> torch.Tensor:
        return static_pred_bch + 1.0


def test_zero_init_preserves_static_anchor_output() -> None:
    torch.manual_seed(7)
    cluster_id_c = torch.tensor([0, 1, 0], dtype=torch.long)
    refiner = ClusterwiseLearnableOutputAnchorRefiner(
        num_clusters=2,
        pred_len=4,
        num_channels=3,
        cluster_id_c=cluster_id_c,
        hidden_dim=8,
        init="zero_delta",
    )
    x = torch.randn(5, 3, 6)
    base = torch.randn(5, 3, 4)
    static = torch.randn(5, 3, 4)

    out = refiner(
        x_bcl=x,
        base_pred_bch=base,
        static_pred_bch=static,
        cluster_id_c=cluster_id_c,
    )

    assert torch.allclose(out, static)


def test_channel_adoption_mask_falls_back_to_static_for_rejected_channels() -> None:
    cluster_id_c = torch.tensor([0, 0, 0], dtype=torch.long)
    refiner = ClusterwiseLearnableOutputAnchorRefiner(
        num_clusters=1,
        pred_len=2,
        num_channels=3,
        cluster_id_c=cluster_id_c,
        hidden_dim=4,
        init="zero_delta",
    )
    with torch.no_grad():
        last = refiner.nets[0][-1]
        assert isinstance(last, torch.nn.Linear)
        last.bias.fill_(0.5)
    refiner.set_channel_adoption_mask(torch.tensor([True, False, True]))

    x = torch.randn(2, 3, 5)
    base = torch.zeros(2, 3, 2)
    static = torch.zeros(2, 3, 2)
    out = refiner(
        x_bcl=x,
        base_pred_bch=base,
        static_pred_bch=static,
        cluster_id_c=cluster_id_c,
    )

    assert torch.allclose(out[:, 1], static[:, 1])
    assert not torch.allclose(out[:, 0], static[:, 0])
    assert not torch.allclose(out[:, 2], static[:, 2])


def test_cluster_params_and_masking_are_cluster_local() -> None:
    cluster_id_c = torch.tensor([0, 1, 0], dtype=torch.long)
    refiner = ClusterwiseLearnableOutputAnchorRefiner(
        num_clusters=2,
        pred_len=2,
        num_channels=3,
        cluster_id_c=cluster_id_c,
        hidden_dim=4,
    )
    params0 = set(refiner.get_cluster_params(0))
    params1 = set(refiner.get_cluster_params(1))

    assert params0
    assert params1
    assert params0.isdisjoint(params1)

    x = torch.randn(2, 3, 5)
    base = torch.zeros(2, 3, 2)
    static = torch.zeros(2, 3, 2)
    out = refiner(
        x_bcl=x,
        base_pred_bch=base,
        static_pred_bch=static,
        cluster_id_c=cluster_id_c,
    )
    out.sum().backward()

    refiner.mask_cluster_grads(torch.tensor([False, True]))

    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in params0)
    assert all(p.grad is None or p.grad.abs().sum().item() == 0 for p in params1)


def test_refiner_can_learn_simple_anchor_residual() -> None:
    torch.manual_seed(11)
    cluster_id_c = torch.zeros(2, dtype=torch.long)
    refiner = ClusterwiseLearnableOutputAnchorRefiner(
        num_clusters=1,
        pred_len=3,
        num_channels=2,
        cluster_id_c=cluster_id_c,
        hidden_dim=16,
        init="zero_delta",
    )
    x = torch.randn(64, 2, 5)
    base = torch.zeros(64, 2, 3)
    static = torch.zeros(64, 2, 3)
    horizon_scale = torch.tensor([0.20, -0.10, 0.35]).view(1, 1, 3)
    target = static + x[..., -1:].expand_as(static) * horizon_scale

    with torch.no_grad():
        initial = (refiner(x_bcl=x, base_pred_bch=base, static_pred_bch=static, cluster_id_c=cluster_id_c) - target).pow(2).mean()

    opt = torch.optim.Adam(refiner.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        pred = refiner(x_bcl=x, base_pred_bch=base, static_pred_bch=static, cluster_id_c=cluster_id_c)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        final = (refiner(x_bcl=x, base_pred_bch=base, static_pred_bch=static, cluster_id_c=cluster_id_c) - target).pow(2).mean()

    assert final.item() < initial.item() * 0.05


def test_eval_loop_applies_learnable_refiner_after_static_anchor_path() -> None:
    x = torch.zeros(1, 1, 3)
    y = torch.ones(1, 1, 2)
    idx = torch.zeros(1, dtype=torch.long)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y, idx), batch_size=1)

    _, mse_k, mae_k, mse_c, mae_c, *_ = eval_loop(
        model=_ZeroBackbone(),
        gate=_UnusedGate(),
        lambda_kp=torch.zeros(1, 0),
        penalty_names=[],
        penalty_fns={},
        loader=loader,
        cluster_id_c=torch.zeros(1, dtype=torch.long),
        K=1,
        moe_cfg={"enable": False, "detach_penalty_grad": True},
        device=torch.device("cpu"),
        select_ranks=None,
        channel_count=1,
        input_len=3,
        learnable_output_anchor_refiner=_AddOneRefiner(),
    )

    assert mse_k.item() == 0.0
    assert mae_k.item() == 0.0
    assert mse_c.item() == 0.0
    assert mae_c.item() == 0.0


def test_posthoc_trainer_adopts_refiner_when_val_improves_static_anchor() -> None:
    torch.manual_seed(21)
    cluster_id_c = torch.zeros(1, dtype=torch.long)
    x_train = torch.randn(48, 1, 5)
    x_val = torch.randn(24, 1, 5)
    horizon_scale = torch.tensor([0.25, -0.15]).view(1, 1, 2)
    y_train = x_train[..., -1:].expand(-1, -1, 2) * horizon_scale
    y_val = x_val[..., -1:].expand(-1, -1, 2) * horizon_scale
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train, torch.arange(x_train.shape[0])),
        batch_size=16,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_val, y_val, torch.arange(x_val.shape[0])),
        batch_size=16,
        shuffle=False,
    )

    refiner, summary = train_learnable_output_anchor_refiner(
        model=_ZeroBackbone(),
        train_loader=train_loader,
        val_loader=val_loader,
        cluster_id_c=cluster_id_c,
        K=1,
        moe_cfg={"enable": False, "detach_penalty_grad": True},
        device=torch.device("cpu"),
        input_len=5,
        channel_count=1,
        cfg={
            "enable": True,
            "hidden_dim": 16,
            "epochs": 120,
            "lr": 0.05,
            "weight_decay": 0.0,
            "selection_metric": "mse",
            "min_rel_improvement": 0.50,
        },
    )

    assert refiner is not None
    assert summary["adopted"] is True
    assert summary["val_refined_mse"] < summary["val_static_mse"] * 0.5


def test_posthoc_trainer_channel_scope_rejects_val_mse_regression_channels() -> None:
    torch.manual_seed(31)
    cluster_id_c = torch.tensor([0, 1], dtype=torch.long)
    x_train = torch.randn(96, 2, 5)
    x_val = torch.randn(48, 2, 5)
    horizon_scale = torch.tensor([0.35, -0.20]).view(1, 1, 2)
    y_train = x_train[..., -1:].expand(-1, -1, 2) * horizon_scale
    y_val = torch.zeros(48, 2, 2)
    y_val[:, 0:1] = x_val[:, 0:1, -1:].expand(-1, -1, 2) * horizon_scale
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train, torch.arange(x_train.shape[0])),
        batch_size=16,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_val, y_val, torch.arange(x_val.shape[0])),
        batch_size=16,
        shuffle=False,
    )

    refiner, summary = train_learnable_output_anchor_refiner(
        model=_ZeroBackbone(),
        train_loader=train_loader,
        val_loader=val_loader,
        cluster_id_c=cluster_id_c,
        K=2,
        moe_cfg={"enable": False, "detach_penalty_grad": True},
        device=torch.device("cpu"),
        input_len=5,
        channel_count=2,
        cfg={
            "enable": True,
            "hidden_dim": 16,
            "epochs": 160,
            "lr": 0.05,
            "weight_decay": 0.0,
            "selection_metric": "mse",
            "adoption_scope": "channel",
        },
    )

    assert refiner is not None
    assert summary["adopted"] is True
    assert summary["adopted_channel_mask"] == [True, False]

    with torch.no_grad():
        static = torch.zeros_like(y_val)
        out = refiner(
            x_bcl=x_val,
            base_pred_bch=static,
            static_pred_bch=static,
            cluster_id_c=cluster_id_c,
        )

    assert torch.allclose(out[:, 1], static[:, 1])
    assert not torch.allclose(out[:, 0], static[:, 0])


def test_main_wires_learnable_output_anchor_into_final_eval_and_summary() -> None:
    source = inspect.getsource(train_module.main)

    assert "learnable_output_anchor_refiner_model" in source
    assert "learnable_output_anchor_refiner=learnable_output_anchor_refiner_model" in source
    assert '"learnable_output_anchor_refiner": learnable_output_anchor_summary' in source
