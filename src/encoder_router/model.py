"""The encoder-router network and its trainer: one encoder to a latent,
privileged branches (cell, corpus, feature — each optional) whose predictions
feed the route layers, and two acceptability heads (sparse, dense) — the most
probable head clearing its threshold serves; RRF is the hedge when neither
fires, never a prediction. Branch losses are masked, annealed toward zero,
and gradient-gated so they can only help the route loss."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from encoder_router.table import HEAD_ROUTES, serve_from_probabilities


class RouterNet(nn.Module):
    """Forward pass shared by training and serving — branches always run;
    only their losses are training-time."""

    def __init__(
        self,
        in_dim: int,
        cell_dim: int,
        corpus_dim: int,
        feature_dim: int = 0,
        *,
        latent: int = 128,
        hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
        )
        self.cell_branch = nn.Linear(latent, cell_dim) if cell_dim else None
        self.corpus_branch = (
            nn.Linear(latent, corpus_dim) if corpus_dim else None
        )
        self.feature_branch = (
            nn.Linear(latent, feature_dim) if feature_dim else None
        )
        route_in = latent + cell_dim + corpus_dim + feature_dim
        self.route_layers = nn.Sequential(
            nn.Linear(route_in, 64), nn.ReLU(),
            nn.Linear(64, len(HEAD_ROUTES)),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        parts = [z]
        out: dict[str, torch.Tensor] = {"latent": z}
        if self.cell_branch is not None:
            out["cell_logits"] = self.cell_branch(z)
            parts.append(torch.sigmoid(out["cell_logits"]))
        if self.corpus_branch is not None:
            out["corpus_pred"] = self.corpus_branch(z)
            parts.append(out["corpus_pred"])
        if self.feature_branch is not None:
            out["feature_pred"] = self.feature_branch(z)
            parts.append(out["feature_pred"])
        out["route_logits"] = self.route_layers(torch.cat(parts, dim=1))
        return out


def _masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = ~torch.isnan(targets)
    if not mask.any():
        return logits.sum() * 0.0
    raw = nn.functional.binary_cross_entropy_with_logits(
        logits, torch.nan_to_num(targets), reduction="none",
        pos_weight=pos_weight,
    )
    weight = mask.float()
    if row_weights is not None:
        weight = weight * row_weights[:, None]
    return (raw * weight).sum() / weight.sum().clamp(min=1e-8)


def _head_pos_weight(targets: np.ndarray | None) -> torch.Tensor | None:
    """Per-column neg/pos, NaN-aware — below 1 for majority-positive heads
    (the route heads), so rare negatives are not drowned."""
    if targets is None:
        return None
    valid = ~np.isnan(targets)
    positives = np.nansum(targets, axis=0)
    negatives = valid.sum(axis=0) - positives
    return torch.as_tensor(
        np.clip(negatives / np.maximum(positives, 1.0), 0.01, 100.0),
        dtype=torch.float32,
    )


def _masked_mse(pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(targets)
    if not mask.any():
        return pred.sum() * 0.0
    diff = (pred - torch.nan_to_num(targets)) * mask
    return diff.pow(2).sum() / mask.sum()


@dataclass
class EncoderRouter:
    """Trainer and serving surface around `RouterNet`."""

    latent: int = 128
    hidden: int = 256
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    patience: int = 20
    batch_size: int = 1024
    lambda_cell: float = 0.5
    lambda_corpus: float = 0.5
    lambda_feature: float = 0.5
    anneal_share: float = 0.7
    threshold: float = 0.5
    seed: int = 0
    net: RouterNet | None = field(default=None, repr=False)
    history: list[dict[str, float]] = field(default_factory=list, repr=False)

    def fit(
        self,
        x: np.ndarray,
        route_targets: np.ndarray,
        cell_targets: np.ndarray | None,
        corpus_targets: np.ndarray | None,
        feature_targets: np.ndarray | None = None,
        *,
        route_weights: np.ndarray | None = None,
        x_val: np.ndarray | None = None,
        val_route_targets: np.ndarray | None = None,
        val_route_weights: np.ndarray | None = None,
    ) -> "EncoderRouter":
        torch.manual_seed(self.seed)
        cell_dim = 0 if cell_targets is None else cell_targets.shape[1]
        corpus_dim = 0 if corpus_targets is None else corpus_targets.shape[1]
        feature_dim = (
            0 if feature_targets is None else feature_targets.shape[1]
        )
        self.net = RouterNet(
            x.shape[1], cell_dim, corpus_dim, feature_dim,
            latent=self.latent, hidden=self.hidden, dropout=self.dropout,
        )
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        tensors = self._tensors(
            x, route_targets, cell_targets, corpus_targets, feature_targets,
            route_weights,
        )
        pos_weight = _head_pos_weight(cell_targets)
        self._route_pos_weight = _head_pos_weight(route_targets)
        best_state, best_val, best_epoch, stale = None, float("inf"), -1, 0
        self.history = []
        progress = tqdm(range(self.epochs), desc="fit", leave=False)
        for epoch in progress:
            anneal = max(
                0.0, 1.0 - epoch / max(self.anneal_share * self.epochs, 1)
            )
            train_loss = self._run_epoch(
                tensors, optimizer, pos_weight, anneal, epoch
            )
            entry = {"epoch": float(epoch), "train": train_loss}
            self.history.append(entry)
            if x_val is None:
                # No validation signal: early stopping would freeze epoch-0
                # weights (constant 0.0 "improves" once, then only stales).
                progress.set_postfix(train=f"{train_loss:.4f}")
                continue
            score = self._validation_loss(
                x_val, val_route_targets, val_route_weights
            )
            entry["val"] = score
            progress.set_postfix(
                train=f"{train_loss:.4f}", val=f"{score:.4f}",
                best=f"{min(best_val, score):.4f}", stale=stale,
            )
            if score < best_val - 1e-5:
                best_state = {
                    k: v.detach().clone()
                    for k, v in self.net.state_dict().items()
                }
                best_val, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        progress.close()
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_loss = best_val if best_val < float("inf") else float("nan")
        self.best_epoch = best_epoch
        return self

    def _tensors(
        self, x, route, cell, corpus, feature, weights
    ) -> list[torch.Tensor]:
        out = [torch.as_tensor(x, dtype=torch.float32),
               torch.as_tensor(route, dtype=torch.float32)]
        for block in (cell, corpus, feature):
            if block is not None:
                out.append(torch.as_tensor(block, dtype=torch.float32))
            else:
                out.append(None)
        out.append(
            torch.ones(len(x)) if weights is None
            else torch.as_tensor(weights, dtype=torch.float32)
        )
        return out

    def _run_epoch(self, tensors, optimizer, pos_weight, anneal, epoch) -> float:
        x, route, cell, corpus, feature, weights = tensors
        generator = torch.Generator().manual_seed(self.seed * 10_000 + epoch)
        order = torch.randperm(len(x), generator=generator)
        self.net.train()
        losses = []
        for start in range(0, len(order), self.batch_size):
            index = order[start:start + self.batch_size]
            out = self.net(x[index])
            route_loss = _masked_bce(
                out["route_logits"], route[index],
                pos_weight=self._route_pos_weight,
                row_weights=weights[index],
            )
            branch_loss = x.sum() * 0.0
            if cell is not None:
                branch_loss = branch_loss + self.lambda_cell * _masked_bce(
                    out["cell_logits"], cell[index], pos_weight
                )
            if corpus is not None:
                branch_loss = branch_loss + self.lambda_corpus * _masked_mse(
                    out["corpus_pred"], corpus[index]
                )
            if feature is not None:
                branch_loss = branch_loss + self.lambda_feature * _masked_mse(
                    out["feature_pred"], feature[index]
                )
            losses.append(float(route_loss.detach()))
            self._gated_step(optimizer, route_loss, branch_loss * anneal)
        return float(np.mean(losses))

    def _gated_step(self, optimizer, route_loss, branch_loss) -> None:
        """Apply branch gradients only when they do not oppose the route
        gradient on the shared encoder (cosine >= 0)."""
        optimizer.zero_grad()
        if branch_loss.requires_grad:
            branch_loss.backward(retain_graph=True)
            branch_grads = {
                name: p.grad.detach().clone()
                for name, p in self.net.named_parameters()
                if p.grad is not None
            }
            optimizer.zero_grad()
        else:
            branch_grads = {}
        route_loss.backward()
        if branch_grads:
            shared = [
                (p, branch_grads[name])
                for name, p in self.net.named_parameters()
                if name.startswith("encoder") and name in branch_grads
                and p.grad is not None
            ]
            dot = sum((p.grad * g).sum() for p, g in shared)
            if dot >= 0:
                for name, p in self.net.named_parameters():
                    if name in branch_grads:
                        grad = branch_grads[name]
                        p.grad = grad if p.grad is None else p.grad + grad
        optimizer.step()

    def _validation_loss(self, x_val, val_route, val_weights=None) -> float:
        """Same objective as training (pos_weight, tie down-weighting) —
        unweighted BCE RISES as the deflated-probability optimum is
        approached, which read as divergence and stopped every run at
        epoch 0."""
        if x_val is None or val_route is None:
            return 0.0
        self.net.eval()
        with torch.no_grad():
            logits = self.net(
                torch.as_tensor(x_val, dtype=torch.float32)
            )["route_logits"]
            return float(_masked_bce(
                logits, torch.as_tensor(val_route, dtype=torch.float32),
                pos_weight=self._route_pos_weight,
                row_weights=None if val_weights is None else torch.as_tensor(
                    val_weights, dtype=torch.float32
                ),
            ))

    def probabilities(self, x: np.ndarray) -> pd.DataFrame:
        self.net.eval()
        with torch.no_grad():
            logits = self.net(
                torch.as_tensor(x, dtype=torch.float32)
            )["route_logits"]
        return pd.DataFrame(
            torch.sigmoid(logits).numpy(), columns=HEAD_ROUTES
        )

    def predict_routes(
        self, x: np.ndarray, threshold: float | None = None
    ) -> list[str]:
        return serve_from_probabilities(
            self.probabilities(x),
            self.threshold if threshold is None else threshold,
        )

    def save(self, path: str | Path) -> Path:
        """One file: constructor params, network dims, trained weights —
        everything `load` needs to serve again."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            f.name: getattr(self, f.name)
            for f in fields(self) if f.name not in ("net", "history")
        }
        net = self.net
        torch.save({
            "params": params,
            "dims": {
                "in_dim": net.encoder[0].in_features,
                "cell_dim": 0 if net.cell_branch is None
                else net.cell_branch.out_features,
                "corpus_dim": 0 if net.corpus_branch is None
                else net.corpus_branch.out_features,
                "feature_dim": 0 if net.feature_branch is None
                else net.feature_branch.out_features,
            },
            "state_dict": net.state_dict(),
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EncoderRouter":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        router = cls(**blob["params"])
        dims = blob["dims"]
        router.net = RouterNet(
            dims["in_dim"], dims["cell_dim"], dims["corpus_dim"],
            dims["feature_dim"], latent=router.latent,
            hidden=router.hidden, dropout=router.dropout,
        )
        router.net.load_state_dict(blob["state_dict"])
        router.net.eval()
        return router
