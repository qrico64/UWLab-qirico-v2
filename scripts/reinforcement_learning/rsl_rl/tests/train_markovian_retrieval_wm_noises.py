#!/usr/bin/env python3

"""Train Markovian dynamics with prior transition memory by noise id."""

from __future__ import annotations

import argparse
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset


@dataclass
class TransitionData:
    inputs: torch.Tensor
    targets: torch.Tensor
    noise_ids: np.ndarray
    order_in_noise: np.ndarray
    trajectory_in_noise: np.ndarray
    policy_dim: int
    num_trajectories: int

    @property
    def num_transitions(self) -> int:
        return self.inputs.shape[0]


@dataclass
class SplitData:
    inputs: torch.Tensor
    targets: torch.Tensor
    noise_ids: np.ndarray
    trajectory_in_noise: np.ndarray

    @property
    def num_transitions(self) -> int:
        return self.inputs.shape[0]


@dataclass
class FilterStats:
    num_total_trajectories: int
    num_short_trajectories: int
    num_large_action_trajectories: int
    num_removed_trajectories: int
    num_kept_trajectories: int


class TensorDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(f"Mismatched inputs/targets: {inputs.shape[0]} != {targets.shape[0]}")
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="collected_data/data_may4_a2r2o002n500_per100_1/trajectories.pkl")
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Optional held-out dataset. When set, all of --data is split into train/val and this dataset is used as test.",
    )
    parser.add_argument("--train-noise-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--trajectories-per-noise", type=int, default=100)
    parser.add_argument("--expected-horizon", type=int, default=60, help="Task horizon; set <= 0 to skip max-length checks.")
    parser.add_argument("--expected-num-noises", type=int, default=None)
    parser.add_argument("--min-trajectory-length", type=int, default=10)
    parser.add_argument("--max-action-magnitude", type=float, default=100.0)
    parser.add_argument("--hidden-dims", type=str, default="512,512,512")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-backend", type=str, default="auto", choices=("auto", "ckdtree", "torch"))
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="state_action",
        choices=("state_action", "policy", "random", "zero"),
        help=(
            "How to choose prior transitions: nearest by current policy+action input, "
            "nearest by current policy state only, random prior transition, or zeroed memory."
        ),
    )
    parser.add_argument(
        "--retrieval-action-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to normalized action dimensions for state_action retrieval distances.",
    )
    parser.add_argument("--retrieval-batch-size", type=int, default=8192)
    parser.add_argument("--retrieval-initial-k", type=int, default=64)
    parser.add_argument("--retrieval-max-query-elements", type=int, default=5_000_000)
    parser.add_argument("--retrieval-num-workers", type=int, default=-1)
    parser.add_argument("--retrieval-log-every", type=int, default=25)
    parser.add_argument("--normalization-batch-size", type=int, default=200_000)
    parser.add_argument(
        "--save_path",
        "--save-path",
        dest="save_path",
        type=str,
        default="outputs/markovian_retrieval_wm_noises",
        help="Directory for final per-trajectory loss plots and their source data.",
    )
    parser.add_argument("--wandb-project", type=str, default="markovian_retrieval_wm_noises")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=("online", "offline", "disabled"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_normalization(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    return mean, torch.where(std < 1e-6, torch.ones_like(std), std)


def parse_hidden_dims(hidden_dims: str) -> list[int]:
    dims = [int(dim) for dim in hidden_dims.split(",") if dim]
    if not dims:
        raise ValueError("--hidden-dims must contain at least one layer size.")
    return dims


def as_noise_id(value) -> int:
    if value is None:
        raise ValueError("Every trajectory must have a non-None noise_index.")
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def load_trajectories(path: str) -> list[dict]:
    with open(path, "rb") as f:
        trajectories = pickle.load(f)
    if not trajectories:
        raise ValueError(f"No trajectories found in {path}.")
    return trajectories


def group_by_noise(trajectories: list[dict], trajectories_per_noise: int) -> dict[int, list[tuple[int, dict]]]:
    groups: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for trajectory_index, trajectory in enumerate(trajectories):
        groups[as_noise_id(trajectory.get("noise_index"))].append((trajectory_index, trajectory))

    for noise_id, group in groups.items():
        group.sort(key=lambda item: item[0])
    return dict(sorted(groups.items()))


def trajectory_length(trajectory: dict) -> int:
    return int(trajectory["T"])


def has_large_action(trajectory: dict, threshold: float) -> bool:
    actions = np.asarray(trajectory["actions"], dtype=np.float32)
    if actions.size == 0:
        return False
    actions = actions.reshape(actions.shape[0], -1)
    return bool(np.any(np.linalg.norm(actions, axis=-1) > threshold))


def filter_groups(
    groups: dict[int, list[tuple[int, dict]]],
    min_length: int,
    max_action_magnitude: float,
) -> tuple[dict[int, list[tuple[int, dict]]], FilterStats]:
    if min_length < 0:
        raise ValueError("--min-trajectory-length must be non-negative.")
    if max_action_magnitude <= 0.0:
        raise ValueError("--max-action-magnitude must be positive.")

    filtered_groups = {}
    num_short = 0
    num_large_action = 0
    num_removed = 0
    num_total = 0
    for noise_id, group in groups.items():
        kept = []
        for item in group:
            _, trajectory = item
            num_total += 1
            short = trajectory_length(trajectory) < min_length
            large_action = has_large_action(trajectory, max_action_magnitude)
            num_short += int(short)
            num_large_action += int(large_action)
            if short or large_action:
                num_removed += 1
            else:
                kept.append(item)
        filtered_groups[noise_id] = kept

    return filtered_groups, FilterStats(
        num_total_trajectories=num_total,
        num_short_trajectories=num_short,
        num_large_action_trajectories=num_large_action,
        num_removed_trajectories=num_removed,
        num_kept_trajectories=num_total - num_removed,
    )


def split_noise_ids(
    noise_ids: list[int],
    train_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train-noise-fraction must be greater than 0 and less than 1.")
    if len(noise_ids) < 2:
        raise ValueError("Need at least two noise indices to create train/test splits.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.asarray(sorted(noise_ids), dtype=np.int64))
    num_train = int(round(len(shuffled) * train_fraction))
    num_train = max(1, min(num_train, len(shuffled) - 1))
    return sorted(shuffled[:num_train].tolist()), sorted(shuffled[num_train:].tolist())


def trajectory_to_xy(trajectory: dict, expected_horizon: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    if expected_horizon is not None:
        assert trajectory_length(trajectory) <= expected_horizon, (
            f"Trajectory T={trajectory['T']} exceeds expected horizon {expected_horizon}"
        )

    policy = np.asarray(trajectory["obs"]["policy"], dtype=np.float32)
    policy2 = np.asarray(trajectory["obs"]["policy2"], dtype=np.float32)
    actions = np.asarray(trajectory["actions"], dtype=np.float32)
    steps = min(policy.shape[0], actions.shape[0], policy2.shape[0] - 1)
    if steps <= 0:
        raise ValueError("Found a trajectory with no usable next-state transitions.")

    # policy/action are at time t; policy2 is shifted to make the target s_{t+1}.
    states = policy[:steps].reshape(steps, -1)
    actions = actions[:steps].reshape(steps, -1)
    next_states = policy2[1 : steps + 1].reshape(steps, -1)
    return np.concatenate([states, actions], axis=-1), next_states, states.shape[1]


def build_transitions(
    groups: dict[int, list[tuple[int, dict]]],
    selected_noise_ids: list[int],
    expected_horizon: int | None,
) -> TransitionData:
    inputs = []
    targets = []
    noise_ids = []
    order_in_noise = []
    trajectory_in_noise = []
    policy_dim: int | None = None
    num_trajectories = 0

    for noise_id in selected_noise_ids:
        order = 0
        for trajectory_position, (_, trajectory) in enumerate(groups[noise_id], start=1):
            trajectory_inputs, trajectory_targets, trajectory_policy_dim = trajectory_to_xy(trajectory, expected_horizon)
            if policy_dim is None:
                policy_dim = trajectory_policy_dim
            elif policy_dim != trajectory_policy_dim:
                raise ValueError(f"Mismatched policy dims: {policy_dim} != {trajectory_policy_dim}")
            steps = trajectory_inputs.shape[0]
            inputs.append(trajectory_inputs)
            targets.append(trajectory_targets)
            noise_ids.append(np.full(steps, noise_id, dtype=np.int64))
            order_in_noise.append(np.arange(order, order + steps, dtype=np.int64))
            trajectory_in_noise.append(np.full(steps, trajectory_position, dtype=np.int64))
            order += steps
            num_trajectories += 1

    if not inputs:
        raise ValueError("No transitions were built.")
    if policy_dim is None:
        raise ValueError("No policy observations were found.")
    return TransitionData(
        inputs=torch.from_numpy(np.concatenate(inputs, axis=0)),
        targets=torch.from_numpy(np.concatenate(targets, axis=0)),
        noise_ids=np.concatenate(noise_ids, axis=0),
        order_in_noise=np.concatenate(order_in_noise, axis=0),
        trajectory_in_noise=np.concatenate(trajectory_in_noise, axis=0),
        policy_dim=policy_dim,
        num_trajectories=num_trajectories,
    )


def split_train_val_indices(valid_indices: np.ndarray, val_fraction: float, seed: int):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be greater than 0 and less than 1.")
    if valid_indices.shape[0] < 2:
        raise ValueError("Need at least two valid train transitions to create a validation split.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(valid_indices)
    num_val = max(1, min(int(round(shuffled.shape[0] * val_fraction)), shuffled.shape[0] - 1))
    return shuffled[num_val:], shuffled[:num_val]


def resolve_retrieval_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    try:
        import scipy.spatial  # noqa: F401

        return "ckdtree"
    except ImportError:
        return "torch"


def normalized_numpy(
    values: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("--normalization-batch-size must be at least 1.")
    output = np.empty(values.shape, dtype=np.float32)
    for start in range(0, values.shape[0], batch_size):
        end = min(start + batch_size, values.shape[0])
        output[start:end] = ((values[start:end] - mean) / std).numpy()
    return output


def query_ckdtree(tree, points: np.ndarray, k: int, num_workers: int):
    kwargs = {"k": k}
    if num_workers != 1:
        kwargs["workers"] = num_workers
    try:
        return tree.query(points, **kwargs)
    except TypeError:
        kwargs.pop("workers", None)
        return tree.query(points, **kwargs)


def nearest_previous_ckdtree_group(
    keys: np.ndarray,
    batch_size: int,
    initial_k: int,
    max_query_elements: int,
    num_workers: int,
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("SciPy is required for --retrieval-backend ckdtree.") from exc

    num_points = keys.shape[0]
    nearest = np.full(num_points, -1, dtype=np.int64)
    if num_points < 2:
        return nearest

    keys = np.ascontiguousarray(keys, dtype=np.float32)
    tree = cKDTree(keys)
    unresolved = np.arange(1, num_points, dtype=np.int64)
    k = min(num_points, max(2, initial_k))

    while unresolved.size:
        next_unresolved = []
        rows_per_chunk = max(1, min(batch_size, max(1, max_query_elements // max(k, 1))))
        for start in range(0, unresolved.shape[0], rows_per_chunk):
            query_ids = unresolved[start : start + rows_per_chunk]
            _, candidates = query_ckdtree(tree, keys[query_ids], k, num_workers)
            candidates = np.asarray(candidates)
            if candidates.ndim == 1:
                candidates = candidates[:, None]
            candidates = candidates.astype(np.int64, copy=False)

            before = candidates < query_ids[:, None]
            has_previous = before.any(axis=1)
            if has_previous.any():
                rows = np.flatnonzero(has_previous)
                first_previous = before[rows].argmax(axis=1)
                nearest[query_ids[rows]] = candidates[rows, first_previous]
            if not has_previous.all():
                next_unresolved.append(query_ids[~has_previous])

        if not next_unresolved:
            break
        # Increase k until every query has seen its nearest prior transition.
        if k == num_points:
            raise RuntimeError("Failed to find a prior transition even after querying the whole group.")
        unresolved = np.concatenate(next_unresolved, axis=0)
        k = min(num_points, k * 2)

    return nearest


def nearest_previous_torch_group(keys: np.ndarray, batch_size: int) -> np.ndarray:
    num_points = keys.shape[0]
    nearest = np.full(num_points, -1, dtype=np.int64)
    if num_points < 2:
        return nearest
    if batch_size < 1:
        raise ValueError("--retrieval-batch-size must be at least 1.")

    keys_tensor = torch.from_numpy(np.ascontiguousarray(keys, dtype=np.float32))
    norms = keys_tensor.square().sum(dim=1)
    candidate_ids = torch.arange(num_points)
    for start in range(1, num_points, batch_size):
        end = min(start + batch_size, num_points)
        query_ids = torch.arange(start, end)
        query = keys_tensor[start:end]
        distances = query.square().sum(dim=1, keepdim=True) + norms.unsqueeze(0) - 2.0 * query.matmul(keys_tensor.T)
        distances[candidate_ids.unsqueeze(0) >= query_ids.unsqueeze(1)] = torch.inf
        nearest[start:end] = torch.argmin(distances, dim=1).numpy()
    return nearest


def random_previous_group(num_points: int, rng: np.random.Generator) -> np.ndarray:
    previous = np.full(num_points, -1, dtype=np.int64)
    if num_points < 2:
        return previous
    previous[1:] = (rng.random(num_points - 1) * np.arange(1, num_points, dtype=np.float64)).astype(np.int64)
    return previous


def immediate_previous_group(num_points: int) -> np.ndarray:
    previous = np.arange(num_points, dtype=np.int64) - 1
    if num_points > 0:
        previous[0] = -1
    return previous


def previous_retrieval_indices(
    retrieval_keys: np.ndarray | None,
    noise_ids: np.ndarray,
    order_in_noise: np.ndarray,
    retrieval_mode: str,
    backend: str,
    batch_size: int,
    initial_k: int,
    max_query_elements: int,
    num_workers: int,
    log_every: int,
    rng: np.random.Generator | None,
) -> np.ndarray:
    if retrieval_mode not in {"state_action", "policy", "random", "zero"}:
        raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")
    if retrieval_keys is not None and retrieval_keys.shape[0] != noise_ids.shape[0]:
        raise ValueError(f"Mismatched retrieval keys/noise ids: {retrieval_keys.shape[0]} != {noise_ids.shape[0]}")
    if retrieval_mode in {"random", "zero"}:
        if retrieval_mode == "random" and rng is None:
            raise ValueError("Random retrieval requires an RNG.")
        resolved_backend = "none"
        num_transitions = noise_ids.shape[0]
    else:
        if retrieval_keys is None:
            raise ValueError(f"{retrieval_mode} retrieval requires retrieval keys.")
        if batch_size < 1:
            raise ValueError("--retrieval-batch-size must be at least 1.")
        if initial_k < 2:
            raise ValueError("--retrieval-initial-k must be at least 2.")
        if max_query_elements < 1:
            raise ValueError("--retrieval-max-query-elements must be at least 1.")
        resolved_backend = resolve_retrieval_backend(backend)
        num_transitions = retrieval_keys.shape[0]

    nearest = np.full(num_transitions, -1, dtype=np.int64)
    unique_noise_ids = np.unique(noise_ids)

    # Memory is local to each noise id, so test noise ids use only their own past.
    for group_index, noise_id in enumerate(unique_noise_ids, start=1):
        group = np.flatnonzero(noise_ids == noise_id)
        group = group[np.argsort(order_in_noise[group], kind="stable")]
        if not np.array_equal(order_in_noise[group], np.arange(group.shape[0], dtype=np.int64)):
            raise AssertionError(f"Non-contiguous order_in_noise for noise_index={noise_id}.")

        if retrieval_mode == "random":
            local_nearest = random_previous_group(group.shape[0], rng)
        elif retrieval_mode == "zero":
            local_nearest = immediate_previous_group(group.shape[0])
        elif resolved_backend == "ckdtree":
            local_nearest = nearest_previous_ckdtree_group(
                retrieval_keys[group],
                batch_size,
                initial_k,
                max_query_elements,
                num_workers,
            )
        elif resolved_backend == "torch":
            local_nearest = nearest_previous_torch_group(retrieval_keys[group], batch_size)
        else:
            raise ValueError(f"Unsupported retrieval backend: {resolved_backend}")

        valid = local_nearest >= 0
        nearest[group[valid]] = group[local_nearest[valid]]
        if log_every > 0 and (group_index % log_every == 0 or group_index == len(unique_noise_ids)):
            print(
                "retrieval | "
                f"mode={retrieval_mode} "
                f"backend={resolved_backend} "
                f"noise_groups={group_index}/{len(unique_noise_ids)} "
                f"transitions={num_transitions}"
            )

    return nearest


def retrieval_key_values(data: TransitionData, retrieval_mode: str) -> torch.Tensor | None:
    if retrieval_mode == "state_action":
        return data.inputs
    if retrieval_mode == "policy":
        return data.inputs[:, : data.policy_dim]
    if retrieval_mode in {"random", "zero"}:
        return None
    raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")


def apply_retrieval_action_multiplier(keys: np.ndarray, policy_dim: int, multiplier: float) -> None:
    if multiplier < 0.0:
        raise ValueError("--retrieval-action-multiplier must be non-negative.")
    if multiplier == 1.0:
        return
    if not 0 <= policy_dim < keys.shape[1]:
        raise ValueError(f"Invalid policy_dim={policy_dim} for retrieval keys with dim={keys.shape[1]}.")
    keys[:, policy_dim:] *= multiplier


def materialize_split(
    data: TransitionData,
    indices: np.ndarray,
    nearest_indices: np.ndarray,
    split_name: str,
    retrieval_mode: str,
) -> SplitData:
    nearest = nearest_indices[indices]
    if np.any(nearest < 0):
        missing = int(np.sum(nearest < 0))
        raise ValueError(f"{split_name} has {missing} transitions without a prior retrieved transition.")

    query_indices = torch.from_numpy(indices.astype(np.int64, copy=False))
    base_dim = data.inputs.shape[1]
    target_dim = data.targets.shape[1]
    inputs = torch.empty((indices.shape[0], base_dim * 2 + target_dim), dtype=torch.float32)
    inputs[:, :base_dim] = data.inputs[query_indices]
    if retrieval_mode == "zero":
        inputs[:, base_dim:] = 0.0
    else:
        retrieved_indices = torch.from_numpy(nearest.astype(np.int64, copy=False))
        inputs[:, base_dim : 2 * base_dim] = data.inputs[retrieved_indices]
        inputs[:, 2 * base_dim :] = data.targets[retrieved_indices]
    return SplitData(
        inputs=inputs,
        targets=data.targets[query_indices],
        noise_ids=data.noise_ids[indices],
        trajectory_in_noise=data.trajectory_in_noise[indices],
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    count = 0
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = model(normalize_model_inputs(inputs, input_mean, input_std))
            loss = loss_fn(preds, (targets - target_mean) / target_std)
            loss_sum += loss.item() * inputs.shape[0]
            count += inputs.shape[0]
    return loss_sum / max(count, 1)


def normalize_model_inputs(
    inputs: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
) -> torch.Tensor:
    return (inputs - input_mean) / input_std


def split_transition_losses(
    model: nn.Module,
    split: SplitData,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, split.num_transitions, batch_size):
            end = min(start + batch_size, split.num_transitions)
            inputs = split.inputs[start:end].to(device, non_blocking=True)
            targets = split.targets[start:end].to(device, non_blocking=True)
            preds = model(normalize_model_inputs(inputs, input_mean, input_std))
            target_normalized = (targets - target_mean) / target_std
            losses.append((preds - target_normalized).square().mean(dim=1).cpu().numpy())
    return np.concatenate(losses, axis=0)


def trajectory_loss_curve(split: SplitData, losses: np.ndarray) -> dict[str, np.ndarray]:
    if losses.shape[0] != split.num_transitions:
        raise ValueError(f"Mismatched losses/transitions: {losses.shape[0]} != {split.num_transitions}")

    pair_sums: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0])
    for noise_id, trajectory_index, loss in zip(split.noise_ids, split.trajectory_in_noise, losses):
        key = (int(noise_id), int(trajectory_index))
        pair_sums[key][0] += float(loss)
        pair_sums[key][1] += 1

    losses_by_trajectory: dict[int, list[float]] = defaultdict(list)
    transition_counts: dict[int, int] = defaultdict(int)
    for (_, trajectory_index), (loss_sum, count) in pair_sums.items():
        losses_by_trajectory[trajectory_index].append(loss_sum / count)
        transition_counts[trajectory_index] += count

    trajectory_indices = np.asarray(sorted(losses_by_trajectory), dtype=np.int64)
    mean_losses = np.asarray(
        [np.mean(losses_by_trajectory[int(trajectory_index)]) for trajectory_index in trajectory_indices],
        dtype=np.float64,
    )
    noise_counts = np.asarray(
        [len(losses_by_trajectory[int(trajectory_index)]) for trajectory_index in trajectory_indices],
        dtype=np.int64,
    )
    transitions = np.asarray(
        [transition_counts[int(trajectory_index)] for trajectory_index in trajectory_indices],
        dtype=np.int64,
    )
    return {
        "trajectory_in_noise": trajectory_indices,
        "mean_loss": mean_losses,
        "num_noise_indices": noise_counts,
        "num_transitions": transitions,
    }


def save_trajectory_loss_curve(save_dir: Path, split_name: str, curve: dict[str, np.ndarray]) -> tuple[Path, Path]:
    txt_path = save_dir / f"{split_name}_trajectory_loss.txt"
    png_path = save_dir / f"{split_name}_trajectory_loss.png"

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("trajectory_in_noise\tmean_loss_across_noise_indices\tnum_noise_indices\tnum_transitions\n")
        for trajectory_index, mean_loss, num_noise_indices, num_transitions in zip(
            curve["trajectory_in_noise"],
            curve["mean_loss"],
            curve["num_noise_indices"],
            curve["num_transitions"],
        ):
            f.write(f"{trajectory_index}\t{mean_loss:.10g}\t{num_noise_indices}\t{num_transitions}\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve["trajectory_in_noise"], curve["mean_loss"], marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Trajectory number within noise index")
    ax.set_ylabel("Mean normalized MSE")
    ax.set_title(f"{split_name.capitalize()} loss by trajectory number")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return png_path, txt_path


def save_all_trajectory_loss_outputs(
    save_path: str,
    model: nn.Module,
    splits: dict[str, SplitData],
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, tuple[Path, Path]]:
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}
    for split_name, split in splits.items():
        losses = split_transition_losses(
            model,
            split,
            input_mean,
            input_std,
            target_mean,
            target_std,
            device,
            batch_size,
        )
        curve = trajectory_loss_curve(split, losses)
        output_paths[split_name] = save_trajectory_loss_curve(save_dir, split_name, curve)
    return output_paths


def make_loader(split: SplitData, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        TensorDataset(split.inputs, split.targets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    args = parse_args()
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    expected_horizon = args.expected_horizon if args.expected_horizon > 0 else None

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    trajectories = load_trajectories(args.data)
    groups = group_by_noise(trajectories, args.trajectories_per_noise)
    all_noise_ids = sorted(groups)
    if args.expected_num_noises is not None:
        assert len(all_noise_ids) == args.expected_num_noises, (
            f"Found {len(all_noise_ids)} noise indices, expected {args.expected_num_noises}"
        )
    filtered_groups, filter_stats = filter_groups(groups, args.min_trajectory_length, args.max_action_magnitude)
    if filter_stats.num_kept_trajectories == 0:
        raise ValueError("No trajectories remain after filtering.")

    if args.test_data is None:
        train_noise_ids, test_noise_ids = split_noise_ids(all_noise_ids, args.train_noise_fraction, args.seed)
        test_filter_stats = None
        train_full = build_transitions(filtered_groups, train_noise_ids, expected_horizon)
        test_full = build_transitions(filtered_groups, test_noise_ids, expected_horizon)
    else:
        train_noise_ids = all_noise_ids
        test_trajectories = load_trajectories(args.test_data)
        test_groups = group_by_noise(test_trajectories, args.trajectories_per_noise)
        test_noise_ids = sorted(test_groups)
        test_filtered_groups, test_filter_stats = filter_groups(
            test_groups,
            args.min_trajectory_length,
            args.max_action_magnitude,
        )
        if test_filter_stats.num_kept_trajectories == 0:
            raise ValueError("No test trajectories remain after filtering.")
        train_full = build_transitions(filtered_groups, train_noise_ids, expected_horizon)
        test_full = build_transitions(test_filtered_groups, test_noise_ids, expected_horizon)
    if train_full.policy_dim != test_full.policy_dim:
        raise ValueError(f"Mismatched train/test policy dims: {train_full.policy_dim} != {test_full.policy_dim}")

    valid_train_indices = np.flatnonzero(train_full.order_in_noise > 0)
    valid_test_indices = np.flatnonzero(test_full.order_in_noise > 0)
    train_indices, val_indices = split_train_val_indices(
        valid_train_indices,
        args.val_fraction,
        args.seed + 1,
    )
    if valid_test_indices.shape[0] == 0:
        raise ValueError("No valid test transitions remain after dropping transitions with no prior memory.")

    train_key_values = retrieval_key_values(train_full, args.retrieval_mode)
    test_key_values = retrieval_key_values(test_full, args.retrieval_mode)
    if train_key_values is None:
        train_keys = None
        test_keys = None
    else:
        if test_key_values is None:
            raise ValueError(f"{args.retrieval_mode} retrieval requires test retrieval keys.")
        key_mean, key_std = compute_normalization(train_key_values[torch.from_numpy(train_indices)])
        train_keys = normalized_numpy(train_key_values, key_mean, key_std, args.normalization_batch_size)
        test_keys = normalized_numpy(test_key_values, key_mean, key_std, args.normalization_batch_size)
        if args.retrieval_mode == "state_action":
            apply_retrieval_action_multiplier(train_keys, train_full.policy_dim, args.retrieval_action_multiplier)
            apply_retrieval_action_multiplier(test_keys, test_full.policy_dim, args.retrieval_action_multiplier)
    retrieval_rng = np.random.default_rng(args.seed) if args.retrieval_mode == "random" else None

    nearest_train = previous_retrieval_indices(
        train_keys,
        train_full.noise_ids,
        train_full.order_in_noise,
        args.retrieval_mode,
        args.retrieval_backend,
        args.retrieval_batch_size,
        args.retrieval_initial_k,
        args.retrieval_max_query_elements,
        args.retrieval_num_workers,
        args.retrieval_log_every,
        retrieval_rng,
    )
    nearest_test = previous_retrieval_indices(
        test_keys,
        test_full.noise_ids,
        test_full.order_in_noise,
        args.retrieval_mode,
        args.retrieval_backend,
        args.retrieval_batch_size,
        args.retrieval_initial_k,
        args.retrieval_max_query_elements,
        args.retrieval_num_workers,
        args.retrieval_log_every,
        retrieval_rng,
    )
    del train_keys, test_keys, train_key_values, test_key_values

    train = materialize_split(train_full, train_indices, nearest_train, "train", args.retrieval_mode)
    val = materialize_split(train_full, val_indices, nearest_train, "val", args.retrieval_mode)
    test = materialize_split(test_full, valid_test_indices, nearest_test, "test", args.retrieval_mode)

    input_mean, input_std = compute_normalization(train.inputs)
    target_mean, target_std = compute_normalization(train.targets)

    train_loader = make_loader(train, args.batch_size, True, args.num_workers)
    train_eval_loader = make_loader(train, args.batch_size, False, args.num_workers)
    val_loader = make_loader(val, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test, args.batch_size, False, args.num_workers)

    model = MLP(train.inputs.shape[1], train.targets.shape[1], hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    input_mean = input_mean.to(device)
    input_std = input_std.to(device)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    dataset_metrics = {
        "dataset/noise_indices": len(all_noise_ids),
        "dataset/train_noise_indices": len(train_noise_ids),
        "dataset/test_noise_indices": len(test_noise_ids),
        "dataset/trajectories": filter_stats.num_total_trajectories,
        "dataset/kept_trajectories": filter_stats.num_kept_trajectories,
        "dataset/removed_trajectories": filter_stats.num_removed_trajectories,
        "dataset/short_trajectories": filter_stats.num_short_trajectories,
        "dataset/large_action_trajectories": filter_stats.num_large_action_trajectories,
        "dataset/train_trajectories": train_full.num_trajectories,
        "dataset/test_trajectories": test_full.num_trajectories,
        "dataset/train_full_transitions": train_full.num_transitions,
        "dataset/test_full_transitions": test_full.num_transitions,
        "dataset/train_transitions": train.num_transitions,
        "dataset/val_transitions": val.num_transitions,
        "dataset/test_transitions": test.num_transitions,
        "dataset/train_no_prior_transitions": int(np.sum(nearest_train < 0)),
        "dataset/test_no_prior_transitions": int(np.sum(nearest_test < 0)),
    }
    if test_filter_stats is not None:
        dataset_metrics.update(
            {
                "dataset/test_data_noise_indices": len(test_noise_ids),
                "dataset/test_data_trajectories": test_filter_stats.num_total_trajectories,
                "dataset/test_data_kept_trajectories": test_filter_stats.num_kept_trajectories,
                "dataset/test_data_removed_trajectories": test_filter_stats.num_removed_trajectories,
                "dataset/test_data_short_trajectories": test_filter_stats.num_short_trajectories,
                "dataset/test_data_large_action_trajectories": test_filter_stats.num_large_action_trajectories,
            }
        )
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={
            **vars(args),
            "base_input_dim": train_full.inputs.shape[1],
            "retrieved_dim": train_full.inputs.shape[1] + train_full.targets.shape[1],
            "input_dim": train.inputs.shape[1],
            "output_dim": train.targets.shape[1],
            "resolved_retrieval_backend": (
                "none" if args.retrieval_mode in {"random", "zero"} else resolve_retrieval_backend(args.retrieval_backend)
            ),
            **dataset_metrics,
        },
    )
    wandb.summary.update(dataset_metrics)
    wandb.log(dataset_metrics, step=0)
    print(
        "split sizes | "
        f"noise_train={len(train_noise_ids)} noise_test={len(test_noise_ids)} "
        f"kept_trajectories={filter_stats.num_kept_trajectories}/{filter_stats.num_total_trajectories} "
        f"train={train.num_transitions} val={val.num_transitions} test={test.num_transitions}"
    )

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = model(normalize_model_inputs(inputs, input_mean, input_std))
            loss = loss_fn(preds, (targets - target_mean) / target_std)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_loss = evaluate(
            model,
            train_eval_loader,
            input_mean,
            input_std,
            target_mean,
            target_std,
            device,
        )
        val_loss = evaluate(
            model,
            val_loader,
            input_mean,
            input_std,
            target_mean,
            target_std,
            device,
        )
        test_loss = evaluate(
            model,
            test_loader,
            input_mean,
            input_std,
            target_mean,
            target_std,
            device,
        )
        best_val_loss = min(best_val_loss, val_loss)
        metrics = {
            "loss/train": train_loss,
            "loss/val": val_loss,
            "loss/test": test_loss,
            "loss/best_val": best_val_loss,
            "epoch": epoch,
        }
        wandb.log(metrics, step=epoch)
        print(f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} test_loss={test_loss:.6f}")

    trajectory_loss_paths = save_all_trajectory_loss_outputs(
        args.save_path,
        model,
        {"train": train, "val": val, "test": test},
        input_mean,
        input_std,
        target_mean,
        target_std,
        device,
        args.batch_size,
    )
    for split_name, (png_path, txt_path) in trajectory_loss_paths.items():
        wandb.summary[f"trajectory_loss/{split_name}_plot"] = str(png_path)
        wandb.summary[f"trajectory_loss/{split_name}_data"] = str(txt_path)
        print(f"trajectory loss curve | split={split_name} plot={png_path} data={txt_path}")

    wandb.summary["best_val_loss"] = best_val_loss
    wandb.finish()


if __name__ == "__main__":
    main()
