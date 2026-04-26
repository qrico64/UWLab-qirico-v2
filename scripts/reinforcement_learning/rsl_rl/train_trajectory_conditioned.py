#!/usr/bin/env python3

# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a trajectory-conditioned transformer policy from collected transitions."""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrajectoryTensors:
    observations: torch.Tensor
    trajectory_observations: torch.Tensor
    targets: torch.Tensor
    noise_index: int
    original_index: int

    @property
    def length(self) -> int:
        return int(self.observations.shape[0])


@dataclass
class DatasetBundle:
    trajectories: list[TrajectoryTensors]
    info: dict | None
    dataset_path: str
    num_total_trajectories: int
    num_kept_trajectories: int
    num_action_filtered_trajectories: int
    num_scenario_filtered_trajectories: int
    num_used_transitions: int
    applied_num_train_scenarios: int | None


class TrajectoryConditionedDataset(Dataset):
    """Trajectory-level dataset that pairs each episode with a same-noise reference episode."""

    def __init__(
        self,
        trajectories: list[TrajectoryTensors],
        context_length: int,
        sample_random_reference: bool,
    ):
        if not trajectories:
            raise ValueError("TrajectoryConditionedDataset requires at least one trajectory.")
        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")

        self.trajectories = trajectories
        self.context_length = context_length
        self.sample_random_reference = sample_random_reference
        self.state_dim = trajectories[0].observations.shape[1]
        self.trajectory_state_dim = trajectories[0].trajectory_observations.shape[1]
        self.action_dim = trajectories[0].targets.shape[1]

        self.indices_by_noise_index: dict[int, list[int]] = {}
        for dataset_index, trajectory in enumerate(trajectories):
            self.indices_by_noise_index.setdefault(trajectory.noise_index, []).append(dataset_index)

        self.reference_candidates: dict[int, list[int]] = {}
        self.fixed_reference_index: dict[int, int] = {}
        for dataset_index, trajectory in enumerate(trajectories):
            same_noise_indices = self.indices_by_noise_index[trajectory.noise_index]
            candidates = [candidate for candidate in same_noise_indices if candidate != dataset_index]
            if not candidates:
                candidates = [dataset_index]
            self.reference_candidates[dataset_index] = candidates
            self.fixed_reference_index[dataset_index] = candidates[0]

    def __len__(self) -> int:
        return len(self.trajectories)

    def _select_reference_index(self, dataset_index: int) -> int:
        candidates = self.reference_candidates[dataset_index]
        if not self.sample_random_reference or len(candidates) == 1:
            return self.fixed_reference_index[dataset_index]
        return candidates[torch.randint(len(candidates), size=(1,)).item()]

    def _select_context_window(self, trajectory: TrajectoryTensors) -> tuple[torch.Tensor, torch.Tensor]:
        if trajectory.length <= self.context_length:
            return trajectory.trajectory_observations, trajectory.targets

        if self.sample_random_reference:
            start = torch.randint(trajectory.length - self.context_length + 1, size=(1,)).item()
            indices = slice(start, start + self.context_length)
        else:
            window_positions = torch.linspace(0, trajectory.length - 1, steps=self.context_length)
            indices = window_positions.round().long()
        return trajectory.trajectory_observations[indices], trajectory.targets[indices]

    def __getitem__(self, dataset_index: int) -> dict[str, torch.Tensor | int]:
        anchor_trajectory = self.trajectories[dataset_index]
        reference_trajectory = self.trajectories[self._select_reference_index(dataset_index)]
        context_states, context_actions = self._select_context_window(reference_trajectory)
        context_tokens = torch.cat([context_states, context_actions], dim=-1)

        return {
            "current_states": anchor_trajectory.observations,
            "targets": anchor_trajectory.targets,
            "context_tokens": context_tokens,
            "query_length": anchor_trajectory.length,
            "context_length": context_tokens.shape[0],
        }


class TrajectoryConditionedTransformer(nn.Module):
    """Transformer encoder over query-state tokens and context transition tokens."""

    def __init__(
        self,
        state_dim: int,
        trajectory_state_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_query_length: int,
        max_context_length: int,
        action_head_hidden_dim: int,
        ffn_dim: int | None = None,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        if max_query_length <= 0 or max_context_length <= 0:
            raise ValueError("Transformer sequence lengths must be positive.")

        self.state_dim = state_dim
        self.trajectory_state_dim = trajectory_state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_query_length = max_query_length
        self.max_context_length = max_context_length

        self.query_projection = nn.Linear(state_dim, hidden_dim)
        self.context_projection = nn.Linear(trajectory_state_dim + action_dim, hidden_dim)
        self.type_embeddings = nn.Embedding(2, hidden_dim)
        self.position_embeddings = nn.Embedding(max_query_length + max_context_length, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim or (4 * hidden_dim),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, action_head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(action_head_hidden_dim, action_dim),
        )

    def _build_attention_mask(
        self,
        query_length: int,
        context_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_length = query_length + context_length
        attention_mask = torch.zeros((total_length, total_length), dtype=torch.float32, device=device)

        if query_length > 1:
            attention_mask[:query_length, :query_length] = float("-inf")
            diagonal_indices = torch.arange(query_length, device=device)
            attention_mask[diagonal_indices, diagonal_indices] = 0.0

        if context_length > 0:
            attention_mask[query_length:, :query_length] = float("-inf")

        return attention_mask

    def forward(
        self,
        current_states: torch.Tensor,
        query_mask: torch.Tensor,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_length, _ = current_states.shape
        _, context_length, _ = context_tokens.shape

        if query_length > self.max_query_length:
            raise ValueError(f"Query length {query_length} exceeds configured max_query_length={self.max_query_length}")
        if context_length > self.max_context_length:
            raise ValueError(
                f"Context length {context_length} exceeds configured max_context_length={self.max_context_length}"
            )

        query_token_ids = torch.zeros((batch_size, query_length), dtype=torch.long, device=current_states.device)
        context_token_ids = torch.ones((batch_size, context_length), dtype=torch.long, device=current_states.device)

        query_positions = torch.arange(query_length, device=current_states.device)
        context_positions = torch.arange(context_length, device=current_states.device) + self.max_query_length

        query_embeddings = self.query_projection(current_states)
        query_embeddings = query_embeddings + self.type_embeddings(query_token_ids)
        query_embeddings = query_embeddings + self.position_embeddings(query_positions).unsqueeze(0)

        if context_length > 0:
            context_embeddings = self.context_projection(context_tokens)
            context_embeddings = context_embeddings + self.type_embeddings(context_token_ids)
            context_embeddings = context_embeddings + self.position_embeddings(context_positions).unsqueeze(0)
            tokens = torch.cat([query_embeddings, context_embeddings], dim=1)
            padding_mask = torch.cat([~query_mask, ~context_mask], dim=1)
        else:
            tokens = query_embeddings
            padding_mask = ~query_mask

        attention_mask = self._build_attention_mask(query_length, context_length, current_states.device)
        encoded_tokens = self.encoder(tokens, mask=attention_mask, src_key_padding_mask=padding_mask)
        query_outputs = self.output_norm(encoded_tokens[:, :query_length])
        return self.action_head(query_outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a trajectory-conditioned transformer policy from collected trajectories."
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="collected_data/data_a2r2o0015n100_1/trajectories.pkl",
        help="Path to the source dataset used for the train/val split.",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="collected_data/data_a2r2o0015n100_2/trajectories.pkl",
        help="Path to the held-out test dataset.",
    )
    parser.add_argument(
        "--state-obs-key",
        type=str,
        default="policy",
        choices=("policy", "policy2"),
        help="Observation group used for current/query state tokens.",
    )
    parser.add_argument(
        "--traj-obs-key",
        "--traj-obe-key",
        dest="traj_obs_key",
        type=str,
        default="policy",
        choices=("policy", "policy2"),
        help="Observation group used for conditioned trajectory context tokens.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/rsl_rl/trajectory_conditioned",
        help="Directory for checkpoints.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="trajectory_conditioned_policy",
        help="wandb project name.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="wandb run name override.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
        help="wandb logging mode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Trajectory batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Transformer embedding dimension.")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer encoder layers.")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument(
        "--ffn-dim",
        type=parse_optional_int,
        default=None,
        help="Transformer feed-forward width. Defaults to 4 * hidden_dim.",
    )
    parser.add_argument(
        "--action-head-hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension used by the final action head.",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of trajectories used for training.")
    parser.add_argument(
        "--with_noise_value",
        "--with-noise-value",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, append per-trajectory obs_noise and act_noise values to every state token.",
    )
    parser.add_argument(
        "--num_train_scenarios",
        "--num-train-scenarios",
        type=parse_optional_int,
        default=None,
        help="If set, keep only train/val trajectories with noise_index in [0, N-1] before splitting.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=60,
        help="Maximum number of same-noise reference transitions used for conditioning.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Clip gradient norm to this value. Set <= 0 to disable clipping.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--device", type=str, default="cuda", help="Training device, e.g. cuda or cpu.")
    parser.add_argument("--save-every", type=int, default=10, help="Save the latest checkpoint every N epochs.")
    return parser.parse_args()


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return int(value)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset_info(dataset_path: str) -> dict | None:
    info_path = Path(dataset_path).with_name("info.pkl")
    if not info_path.exists():
        return None

    with open(info_path, "rb") as f:
        return pickle.load(f)


def trajectory_has_large_action(trajectory: dict, threshold: float = 100.0) -> bool:
    actions = np.asarray(trajectory["actions"], dtype=np.float32)
    if actions.ndim == 1:
        action_magnitudes = np.abs(actions)
    else:
        action_magnitudes = np.linalg.norm(actions, axis=-1)
    return bool(np.any(action_magnitudes > threshold))


def trajectory_in_train_scenarios(trajectory: dict, num_train_scenarios: int | None) -> bool:
    if num_train_scenarios is None:
        return True
    noise_index = trajectory.get("noise_index")
    if noise_index is None:
        return False
    return 0 <= int(noise_index) < num_train_scenarios


def load_trajectories(
    path: str,
    state_obs_key: str,
    traj_obs_key: str,
    num_train_scenarios: int | None = None,
    with_noise_value: bool = False,
) -> DatasetBundle:
    with open(path, "rb") as f:
        trajectories = pickle.load(f)

    dataset_info = load_dataset_info(path)
    kept_trajectories: list[TrajectoryTensors] = []
    num_action_filtered_trajectories = 0
    num_scenario_filtered_trajectories = 0

    for original_index, trajectory in enumerate(trajectories):
        if trajectory_has_large_action(trajectory):
            num_action_filtered_trajectories += 1
            continue

        if not trajectory_in_train_scenarios(trajectory, num_train_scenarios):
            num_scenario_filtered_trajectories += 1
            continue

        observations = np.asarray(trajectory["obs"][state_obs_key], dtype=np.float32)
        trajectory_observations = np.asarray(trajectory["obs"][traj_obs_key], dtype=np.float32)
        actions = np.asarray(trajectory["actions"], dtype=np.float32)
        rand_noise = np.asarray(trajectory["rand_noise"], dtype=np.float32)
        targets = actions - rand_noise

        if observations.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Trajectory length mismatch in {path}: state obs steps={observations.shape[0]}, "
                f"target steps={targets.shape[0]}"
            )
        if trajectory_observations.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Trajectory length mismatch in {path}: conditioned trajectory obs steps="
                f"{trajectory_observations.shape[0]}, target steps={targets.shape[0]}"
            )

        if with_noise_value:
            obs_noise = np.asarray(trajectory["obs_noise"], dtype=np.float32).reshape(1, -1)
            act_noise = np.asarray(trajectory["act_noise"], dtype=np.float32).reshape(1, -1)
            noise_features = np.repeat(np.concatenate([obs_noise, act_noise], axis=-1), observations.shape[0], axis=0)
            observations = np.concatenate([observations, noise_features], axis=-1)
            trajectory_observations = np.concatenate([trajectory_observations, noise_features], axis=-1)

        noise_index = trajectory.get("noise_index")
        if noise_index is None:
            raise ValueError(
                f"Trajectory {original_index} from {path} is missing noise_index, but this model requires same-scenario pairing."
            )

        kept_trajectories.append(
            TrajectoryTensors(
                observations=torch.from_numpy(observations),
                trajectory_observations=torch.from_numpy(trajectory_observations),
                targets=torch.from_numpy(targets),
                noise_index=int(noise_index),
                original_index=original_index,
            )
        )

    if not kept_trajectories:
        raise ValueError(f"No valid trajectories remain after filtering dataset {path}")

    num_used_transitions = sum(trajectory.length for trajectory in kept_trajectories)
    return DatasetBundle(
        trajectories=kept_trajectories,
        info=dataset_info,
        dataset_path=path,
        num_total_trajectories=len(trajectories),
        num_kept_trajectories=len(kept_trajectories),
        num_action_filtered_trajectories=num_action_filtered_trajectories,
        num_scenario_filtered_trajectories=num_scenario_filtered_trajectories,
        num_used_transitions=num_used_transitions,
        applied_num_train_scenarios=num_train_scenarios,
    )


def split_trajectories_by_noise_index(
    trajectories: list[TrajectoryTensors],
    train_fraction: float,
    seed: int,
) -> tuple[list[TrajectoryTensors], list[TrajectoryTensors]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")

    grouped_indices: dict[int, list[int]] = {}
    for trajectory_index, trajectory in enumerate(trajectories):
        grouped_indices.setdefault(trajectory.noise_index, []).append(trajectory_index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for group_indices in grouped_indices.values():
        shuffled = list(group_indices)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_indices.extend(shuffled)
            continue

        num_train = math.floor(len(shuffled) * train_fraction)
        num_train = max(1, min(num_train, len(shuffled) - 1))
        train_indices.extend(shuffled[:num_train])
        val_indices.extend(shuffled[num_train:])

    train_indices.sort()
    val_indices.sort()
    if not train_indices or not val_indices:
        raise ValueError(
            f"Train/val split is empty with {len(trajectories)} trajectories and train_fraction={train_fraction}"
        )

    return [trajectories[index] for index in train_indices], [trajectories[index] for index in val_indices]


def compute_normalization(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    concatenated = torch.cat(tensors, dim=0)
    mean = concatenated.mean(dim=0)
    std = concatenated.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def collate_trajectory_batch(batch: list[dict[str, torch.Tensor | int]]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_query_length = max(int(item["query_length"]) for item in batch)
    max_context_length = max(int(item["context_length"]) for item in batch)
    state_dim = batch[0]["current_states"].shape[1]
    action_dim = batch[0]["targets"].shape[1]
    context_dim = batch[0]["context_tokens"].shape[1]

    current_states = torch.zeros((batch_size, max_query_length, state_dim), dtype=torch.float32)
    targets = torch.zeros((batch_size, max_query_length, action_dim), dtype=torch.float32)
    query_mask = torch.zeros((batch_size, max_query_length), dtype=torch.bool)
    context_tokens = torch.zeros((batch_size, max_context_length, context_dim), dtype=torch.float32)
    context_mask = torch.zeros((batch_size, max_context_length), dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        query_length = int(item["query_length"])
        context_length = int(item["context_length"])
        current_states[batch_index, :query_length] = item["current_states"]
        targets[batch_index, :query_length] = item["targets"]
        query_mask[batch_index, :query_length] = True
        context_tokens[batch_index, :context_length] = item["context_tokens"]
        context_mask[batch_index, :context_length] = True

    return {
        "current_states": current_states,
        "targets": targets,
        "query_mask": query_mask,
        "context_tokens": context_tokens,
        "context_mask": context_mask,
    }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_trajectory_batch,
    )


def normalize_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    trajectory_state_mean: torch.Tensor,
    trajectory_state_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    trajectory_state_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_states = batch["current_states"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    query_mask = batch["query_mask"].to(device, non_blocking=True)
    context_tokens = batch["context_tokens"].to(device, non_blocking=True)
    context_mask = batch["context_mask"].to(device, non_blocking=True)

    normalized_current_states = (current_states - state_mean) / state_std
    normalized_targets = (targets - target_mean) / target_std

    context_states = context_tokens[..., :trajectory_state_dim]
    context_actions = context_tokens[..., trajectory_state_dim:]
    normalized_context_states = (context_states - trajectory_state_mean) / trajectory_state_std
    normalized_context_actions = (context_actions - target_mean) / target_std
    normalized_context_tokens = torch.cat([normalized_context_states, normalized_context_actions], dim=-1)

    return normalized_current_states, normalized_targets, query_mask, normalized_context_tokens, context_mask


def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = (predictions - targets).pow(2).mean(dim=-1)
    masked_error = squared_error[mask]
    if masked_error.numel() == 0:
        return squared_error.new_tensor(0.0)
    return masked_error.mean()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    trajectory_state_mean: torch.Tensor,
    trajectory_state_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    trajectory_state_dim: int,
) -> float:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    with torch.no_grad():
        for batch in loader:
            normalized_current_states, normalized_targets, query_mask, normalized_context_tokens, context_mask = (
                normalize_batch(
                    batch=batch,
                    device=device,
                    state_mean=state_mean,
                    state_std=state_std,
                    trajectory_state_mean=trajectory_state_mean,
                    trajectory_state_std=trajectory_state_std,
                    target_mean=target_mean,
                    target_std=target_std,
                    trajectory_state_dim=trajectory_state_dim,
                )
            )
            predictions = model(
                current_states=normalized_current_states,
                query_mask=query_mask,
                context_tokens=normalized_context_tokens,
                context_mask=context_mask,
            )
            batch_token_count = int(query_mask.sum().item())
            loss_sum += masked_mse_loss(predictions, normalized_targets, query_mask).item() * batch_token_count
            token_count += batch_token_count
    return loss_sum / max(token_count, 1)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    trajectory_state_mean: torch.Tensor,
    trajectory_state_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    save_dict: dict,
) -> None:
    checkpoint = dict(save_dict)
    checkpoint.update(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "state_mean": state_mean.cpu(),
            "state_std": state_std.cpu(),
            "trajectory_state_mean": trajectory_state_mean.cpu(),
            "trajectory_state_std": trajectory_state_std.cpu(),
            "target_mean": target_mean.cpu(),
            "target_std": target_std.cpu(),
        }
    )
    torch.save(checkpoint, path)

    save_dict_path = f"{os.path.splitext(path)[0]}_save_dict.pkl"
    with open(save_dict_path, "wb") as f:
        pickle.dump(checkpoint, f)


def main() -> None:
    args = parse_args()
    args.run_name = args.run_name or None
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError(f"--train-fraction must be between 0 and 1, got {args.train_fraction}")
    if args.num_train_scenarios is not None and args.num_train_scenarios <= 0:
        raise ValueError(f"--num-train-scenarios must be positive when set, got {args.num_train_scenarios}")
    if args.context_length <= 0:
        raise ValueError(f"--context-length must be positive, got {args.context_length}")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = requested_device

    full_trainval_bundle = load_trajectories(
        args.train_data,
        args.state_obs_key,
        args.traj_obs_key,
        num_train_scenarios=args.num_train_scenarios,
        with_noise_value=args.with_noise_value,
    )
    test_bundle = load_trajectories(
        args.test_data,
        args.state_obs_key,
        args.traj_obs_key,
        with_noise_value=args.with_noise_value,
    )

    train_trajectories, val_trajectories = split_trajectories_by_noise_index(
        full_trainval_bundle.trajectories, args.train_fraction, args.seed
    )
    test_trajectories = test_bundle.trajectories

    train_dataset = TrajectoryConditionedDataset(
        trajectories=train_trajectories,
        context_length=args.context_length,
        sample_random_reference=True,
    )
    train_eval_dataset = TrajectoryConditionedDataset(
        trajectories=train_trajectories,
        context_length=args.context_length,
        sample_random_reference=False,
    )
    val_dataset = TrajectoryConditionedDataset(
        trajectories=val_trajectories,
        context_length=args.context_length,
        sample_random_reference=False,
    )
    test_dataset = TrajectoryConditionedDataset(
        trajectories=test_trajectories,
        context_length=args.context_length,
        sample_random_reference=False,
    )

    state_mean, state_std = compute_normalization([trajectory.observations for trajectory in train_trajectories])
    trajectory_state_mean, trajectory_state_std = compute_normalization(
        [trajectory.trajectory_observations for trajectory in train_trajectories]
    )
    target_mean, target_std = compute_normalization([trajectory.targets for trajectory in train_trajectories])

    train_loader = build_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_eval_loader = build_dataloader(train_eval_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = build_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = build_dataloader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    state_dim = train_trajectories[0].observations.shape[1]
    trajectory_state_dim = train_trajectories[0].trajectory_observations.shape[1]
    action_dim = train_trajectories[0].targets.shape[1]
    max_query_length = max(trajectory.length for trajectory in full_trainval_bundle.trajectories + test_trajectories)
    model = TrajectoryConditionedTransformer(
        state_dim=state_dim,
        trajectory_state_dim=trajectory_state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_query_length=max_query_length,
        max_context_length=args.context_length,
        action_head_hidden_dim=args.action_head_hidden_dim,
        ffn_dim=args.ffn_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    state_mean = state_mean.to(device)
    state_std = state_std.to(device)
    trajectory_state_mean = trajectory_state_mean.to(device)
    trajectory_state_std = trajectory_state_std.to(device)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    train_transitions = sum(trajectory.length for trajectory in train_trajectories)
    val_transitions = sum(trajectory.length for trajectory in val_trajectories)
    test_transitions = sum(trajectory.length for trajectory in test_trajectories)
    model_info = {
        "model_class": model.__class__.__name__,
        "state_dim": state_dim,
        "trajectory_state_dim": trajectory_state_dim,
        "action_dim": action_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ffn_dim": args.ffn_dim or (4 * args.hidden_dim),
        "action_head_hidden_dim": args.action_head_hidden_dim,
        "dropout": args.dropout,
        "max_query_length": max_query_length,
        "max_context_length": args.context_length,
        "optimizer": optimizer.__class__.__name__,
        "input_normalization": "train_split_state_mean_std",
        "trajectory_input_normalization": "train_split_trajectory_state_mean_std",
        "target_normalization": "train_split_action_mean_std",
        "with_noise_value": args.with_noise_value,
    }
    dataset_info = {
        "trainval": {
            "dataset_path": full_trainval_bundle.dataset_path,
            "info": full_trainval_bundle.info,
            "num_total_trajectories": full_trainval_bundle.num_total_trajectories,
            "num_kept_trajectories": full_trainval_bundle.num_kept_trajectories,
            "num_action_filtered_trajectories": full_trainval_bundle.num_action_filtered_trajectories,
            "num_scenario_filtered_trajectories": full_trainval_bundle.num_scenario_filtered_trajectories,
            "num_transitions": full_trainval_bundle.num_used_transitions,
            "applied_num_train_scenarios": full_trainval_bundle.applied_num_train_scenarios,
        },
        "test": {
            "dataset_path": test_bundle.dataset_path,
            "info": test_bundle.info,
            "num_total_trajectories": test_bundle.num_total_trajectories,
            "num_kept_trajectories": test_bundle.num_kept_trajectories,
            "num_action_filtered_trajectories": test_bundle.num_action_filtered_trajectories,
            "num_scenario_filtered_trajectories": test_bundle.num_scenario_filtered_trajectories,
            "num_transitions": test_bundle.num_used_transitions,
            "applied_num_train_scenarios": test_bundle.applied_num_train_scenarios,
        },
        "split": {
            "train_fraction": args.train_fraction,
            "train_trajectories": len(train_trajectories),
            "val_trajectories": len(val_trajectories),
            "test_trajectories": len(test_trajectories),
            "train_transitions": train_transitions,
            "val_transitions": val_transitions,
            "test_transitions": test_transitions,
        },
        "filtering": {
            "action_magnitude_threshold": 100.0,
            "num_train_scenarios": args.num_train_scenarios,
            "with_noise_value": args.with_noise_value,
            "context_length": args.context_length,
        },
    }
    base_save_dict = {
        "args": vars(args),
        "model_info": model_info,
        "dataset_info": dataset_info,
    }

    wandb.init(
        project=args.experiment_name,
        name=args.run_name,
        mode=args.wandb_mode,
        config={
            **vars(args),
            "state_dim": state_dim,
            "trajectory_state_dim": trajectory_state_dim,
            "action_dim": action_dim,
            "train_trajectories": len(train_trajectories),
            "val_trajectories": len(val_trajectories),
            "test_trajectories": len(test_trajectories),
            "train_transitions": train_transitions,
            "val_transitions": val_transitions,
            "test_transitions": test_transitions,
            "num_train_scenarios": args.num_train_scenarios,
            "with_noise_value": args.with_noise_value,
            "trainval_action_filtered_trajectories": full_trainval_bundle.num_action_filtered_trajectories,
            "trainval_scenario_filtered_trajectories": full_trainval_bundle.num_scenario_filtered_trajectories,
            "test_action_filtered_trajectories": test_bundle.num_action_filtered_trajectories,
            "test_scenario_filtered_trajectories": test_bundle.num_scenario_filtered_trajectories,
        },
    )

    best_val_loss = float("inf")
    best_path = os.path.join(args.output_dir, "best.pt")
    last_path = os.path.join(args.output_dir, "last.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_token_count = 0

        for batch in train_loader:
            normalized_current_states, normalized_targets, query_mask, normalized_context_tokens, context_mask = (
                normalize_batch(
                    batch=batch,
                    device=device,
                    state_mean=state_mean,
                    state_std=state_std,
                    trajectory_state_mean=trajectory_state_mean,
                    trajectory_state_std=trajectory_state_std,
                    target_mean=target_mean,
                    target_std=target_std,
                    trajectory_state_dim=trajectory_state_dim,
                )
            )

            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                current_states=normalized_current_states,
                query_mask=query_mask,
                context_tokens=normalized_context_tokens,
                context_mask=context_mask,
            )
            loss = masked_mse_loss(predictions, normalized_targets, query_mask)
            loss.backward()
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            batch_token_count = int(query_mask.sum().item())
            train_loss_sum += loss.item() * batch_token_count
            train_token_count += batch_token_count

        train_loss = train_loss_sum / max(train_token_count, 1)
        train_eval_loss = evaluate(
            model=model,
            loader=train_eval_loader,
            device=device,
            state_mean=state_mean,
            state_std=state_std,
            trajectory_state_mean=trajectory_state_mean,
            trajectory_state_std=trajectory_state_std,
            target_mean=target_mean,
            target_std=target_std,
            trajectory_state_dim=trajectory_state_dim,
        )
        val_loss = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            state_mean=state_mean,
            state_std=state_std,
            trajectory_state_mean=trajectory_state_mean,
            trajectory_state_std=trajectory_state_std,
            target_mean=target_mean,
            target_std=target_std,
            trajectory_state_dim=trajectory_state_dim,
        )
        test_loss = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            state_mean=state_mean,
            state_std=state_std,
            trajectory_state_mean=trajectory_state_mean,
            trajectory_state_std=trajectory_state_std,
            target_mean=target_mean,
            target_std=target_std,
            trajectory_state_dim=trajectory_state_dim,
        )

        metrics = {
            "epoch": epoch,
            "loss/train": train_loss,
            "loss/train_eval": train_eval_loss,
            "loss/val": val_loss,
            "loss/test": test_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        wandb.log(metrics)
        print(
            f"Epoch {epoch:04d} | train={train_loss:.6f} | train_eval={train_eval_loss:.6f} "
            f"| val={val_loss:.6f} | test={test_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
                state_mean,
                state_std,
                trajectory_state_mean,
                trajectory_state_std,
                target_mean,
                target_std,
                base_save_dict,
            )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                last_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
                state_mean,
                state_std,
                trajectory_state_mean,
                trajectory_state_std,
                target_mean,
                target_std,
                base_save_dict,
            )

    wandb.finish()


if __name__ == "__main__":
    main()
