from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


class ClusterwiseLearnableOutputAnchorRefiner(nn.Module):
    """Per-cluster refiner initialized as an exact no-op over static output anchors."""

    feature_dim = 8

    def __init__(
        self,
        *,
        num_clusters: int,
        pred_len: int,
        num_channels: int,
        cluster_id_c: torch.Tensor,
        hidden_dim: int = 16,
        max_delta_scale: float = 1.0,
        init: str = "zero_delta",
    ) -> None:
        super().__init__()
        self.K = int(num_clusters)
        self.H = int(pred_len)
        self.C = int(num_channels)
        self.hidden_dim = max(int(hidden_dim), 1)
        self.max_delta_scale = float(max_delta_scale)

        cluster_id_c = cluster_id_c.detach().cpu().to(torch.long)
        if int(cluster_id_c.numel()) != self.C:
            raise ValueError("learnable output anchor requires cluster_id_c length to match num_channels.")
        if self.K <= 0:
            raise ValueError("num_clusters must be positive.")
        if self.H <= 0:
            raise ValueError("pred_len must be positive.")
        if bool(((cluster_id_c < 0) | (cluster_id_c >= self.K)).any().item()):
            raise ValueError("cluster_id_c contains ids outside [0, num_clusters).")
        self.register_buffer("cluster_id_c", cluster_id_c, persistent=False)
        self.register_buffer("adopt_channel_c", torch.ones(self.C, dtype=torch.bool))

        self.nets = nn.ModuleList()
        for k in range(self.K):
            idx = (cluster_id_c == k).nonzero(as_tuple=False).view(-1)
            self.register_buffer(f"channel_idx_{k}", idx, persistent=False)
            net = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.H),
            )
            self.nets.append(net)
        self.reset_parameters(init=init)

    def reset_parameters(self, *, init: str = "zero_delta") -> None:
        init = str(init).lower()
        for net in self.nets:
            first = net[0]
            last = net[-1]
            assert isinstance(first, nn.Linear)
            assert isinstance(last, nn.Linear)
            nn.init.xavier_uniform_(first.weight)
            nn.init.zeros_(first.bias)
            if init == "zero_delta":
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            elif init == "xavier":
                nn.init.xavier_uniform_(last.weight)
                nn.init.zeros_(last.bias)
            else:
                raise ValueError(f"Unsupported learnable output anchor init='{init}'.")

    @staticmethod
    def _features(
        *,
        x_bcl: torch.Tensor,
        base_pred_bch: torch.Tensor,
        static_pred_bch: torch.Tensor,
    ) -> torch.Tensor:
        anchor_delta = static_pred_bch - base_pred_bch
        return torch.stack(
            [
                x_bcl[..., -1],
                x_bcl.mean(dim=-1),
                x_bcl.std(dim=-1, unbiased=False),
                x_bcl[..., -1] - x_bcl[..., 0],
                base_pred_bch.mean(dim=-1),
                static_pred_bch.mean(dim=-1),
                anchor_delta.mean(dim=-1),
                anchor_delta.std(dim=-1, unbiased=False),
            ],
            dim=-1,
        )

    def forward(
        self,
        *,
        x_bcl: torch.Tensor,
        base_pred_bch: torch.Tensor,
        static_pred_bch: torch.Tensor,
        cluster_id_c: torch.Tensor,
    ) -> torch.Tensor:
        if base_pred_bch.shape != static_pred_bch.shape:
            raise ValueError("base_pred_bch and static_pred_bch must have the same shape.")
        if x_bcl.ndim != 3 or static_pred_bch.ndim != 3:
            raise ValueError("x_bcl and predictions must be rank-3 tensors.")
        if int(x_bcl.shape[0]) != int(static_pred_bch.shape[0]) or int(x_bcl.shape[1]) != int(static_pred_bch.shape[1]):
            raise ValueError("x_bcl and predictions must share batch and channel dimensions.")
        if int(static_pred_bch.shape[1]) != self.C or int(static_pred_bch.shape[-1]) != self.H:
            raise ValueError("static_pred_bch shape does not match configured channels/pred_len.")
        expected_cluster = self.cluster_id_c.to(device=cluster_id_c.device)
        if int(cluster_id_c.numel()) != self.C or not torch.equal(cluster_id_c.detach().to(torch.long), expected_cluster):
            raise ValueError("cluster_id_c must match the refiner's construction-time channel clusters.")
        if self.max_delta_scale == 0.0:
            return static_pred_bch

        features = self._features(
            x_bcl=x_bcl,
            base_pred_bch=base_pred_bch,
            static_pred_bch=static_pred_bch,
        )
        out = static_pred_bch
        batch_size = int(static_pred_bch.shape[0])
        for k, net in enumerate(self.nets):
            idx = getattr(self, f"channel_idx_{k}").to(device=static_pred_bch.device)
            if idx.numel() == 0:
                continue
            feat_bnf = features.index_select(1, idx)
            delta_bnh = net(feat_bnf.reshape(-1, self.feature_dim)).view(batch_size, int(idx.numel()), self.H)
            refined = static_pred_bch.index_select(1, idx) + self.max_delta_scale * delta_bnh
            adopt_n = self.adopt_channel_c.to(device=static_pred_bch.device).index_select(0, idx).view(1, -1, 1)
            refined = torch.where(adopt_n, refined, static_pred_bch.index_select(1, idx))
            out = out.index_copy(1, idx, refined)
        return out

    def set_channel_adoption_mask(self, adopt_channel_c: torch.Tensor) -> None:
        adopt_channel_c = adopt_channel_c.detach().to(device=self.adopt_channel_c.device, dtype=torch.bool).view(-1)
        if int(adopt_channel_c.numel()) != self.C:
            raise ValueError("learnable output anchor channel adoption mask must match num_channels.")
        self.adopt_channel_c.copy_(adopt_channel_c)

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return list(self.nets[int(k)].parameters())

    def mask_cluster_grads(self, stopped_k: torch.Tensor) -> None:
        for k in range(self.K):
            if not bool(stopped_k[k].item()):
                continue
            for param in self.get_cluster_params(k):
                if param.grad is not None:
                    param.grad.zero_()

    def get_cluster_state(self, k: int) -> Dict[str, object]:
        idx = getattr(self, f"channel_idx_{int(k)}").detach().cpu()
        return {
            "channel_idx": idx,
            "net": {name: tensor.detach().cpu() for name, tensor in self.nets[int(k)].state_dict().items()},
        }

    def load_cluster_state(self, k: int, state: Dict[str, object]) -> None:
        k = int(k)
        idx = getattr(self, f"channel_idx_{k}").detach().cpu()
        saved_idx = state.get("channel_idx", idx)
        if not isinstance(saved_idx, torch.Tensor) or saved_idx.numel() != idx.numel() or not torch.equal(saved_idx.cpu(), idx):
            raise ValueError(f"learnable output anchor cluster {k} channel indices do not match checkpoint state.")
        net_state = state["net"]
        if not isinstance(net_state, dict):
            raise ValueError("learnable output anchor cluster state must contain a net state dict.")
        device = next(self.nets[k].parameters()).device
        self.nets[k].load_state_dict({name: value.to(device) for name, value in net_state.items()}, strict=True)
