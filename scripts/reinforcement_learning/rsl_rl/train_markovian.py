#!/usr/bin/env python3

# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a Markovian policy from collected transitions."""

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
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class TransitionTensors:
    inputs: torch.Tensor
    targets: torch.Tensor


@dataclass
class DatasetBundle:
    transitions: TransitionTensors
    info: dict | None
    dataset_path: str
    num_total_trajectories: int
    num_kept_trajectories: int
    num_action_filtered_trajectories: int
    num_scenario_filtered_trajectories: int
    num_used_transitions: int
    applied_num_train_scenarios: int | None


class TensorDataset(Dataset):
    """Simple tensor-backed dataset for transition learning."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(f"Mismatched dataset lengths: {inputs.shape[0]} != {targets.shape[0]}")
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


class MarkovianMLP(nn.Module):
    """Deterministic five-layer MLP with dropout regularization."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()

        layers: list[nn.Module] = []
        dims = [input_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, output_dim]
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Markovian policy from collected trajectories.")
    parser.add_argument(
        "--train-data",
        type=str,
        default="collected_data/data_a2r2o0015n100_1/trajectories.pkl",
        help="Path to the source dataset used for the train/val split.",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="",
        help="Optional path to the held-out test dataset.",
    )
    parser.add_argument(
        "--obs-key",
        type=str,
        default="policy",
        choices=("policy", "policy2", "policy_actiononly"),
        help="Observation group to map from.",
    )
    parser.add_argument("--output-dir", type=str, default="logs/rsl_rl/markovian", help="Directory for checkpoints.")
    parser.add_argument("--experiment-name", type=str, default="markovian_policy", help="wandb project name.")
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
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension for each MLP layer.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of transitions used for training.")
    parser.add_argument(
        "--with_noise_value",
        "--with-noise-value",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, append per-trajectory obs_noise and act_noise values to each transition input.",
    )
    parser.add_argument(
        "--num_train_scenarios",
        "--num-train-scenarios",
        type=parse_optional_int,
        default=None,
        help="If set, keep only train/val trajectories with noise_index in [0, N-1] before splitting.",
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


def load_transitions(
    path: str,
    obs_key: str,
    num_train_scenarios: int | None = None,
    with_noise_value: bool = False,
) -> DatasetBundle:
    with open(path, "rb") as f:
        trajectories = pickle.load(f)

    dataset_info = load_dataset_info(path)
    obs_list = []
    target_list = []
    num_action_filtered_trajectories = 0
    num_scenario_filtered_trajectories = 0
    for trajectory in trajectories:
        if trajectory_has_large_action(trajectory):
            num_action_filtered_trajectories += 1
            continue

        if not trajectory_in_train_scenarios(trajectory, num_train_scenarios):
            num_scenario_filtered_trajectories += 1
            continue

        observations = np.asarray(
            trajectory["obs"]["policy" if obs_key == "policy_actiononly" else obs_key], dtype=np.float32
        )
        if obs_key == "policy_actiononly":
            observations = observations[:, 30:65]
        actions = np.asarray(trajectory["actions"], dtype=np.float32)
        rand_noise = np.asarray(trajectory["rand_noise"], dtype=np.float32)
        targets = actions - rand_noise
        if with_noise_value:
            obs_noise = np.asarray(trajectory["obs_noise"], dtype=np.float32).reshape(1, -1)
            act_noise = np.asarray(trajectory["act_noise"], dtype=np.float32).reshape(1, -1)
            noise_features = np.repeat(np.concatenate([obs_noise, act_noise], axis=-1), observations.shape[0], axis=0)
            observations = np.concatenate([observations, noise_features], axis=-1)

        if observations.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Trajectory length mismatch in {path}: obs steps={observations.shape[0]}, target steps={targets.shape[0]}"
            )

        obs_list.append(observations)
        target_list.append(targets)

    if not obs_list:
        raise ValueError(f"No valid trajectories remain after filtering dataset {path}")

    inputs = torch.from_numpy(np.concatenate(obs_list, axis=0))
    targets = torch.from_numpy(np.concatenate(target_list, axis=0))
    return DatasetBundle(
        transitions=TransitionTensors(inputs=inputs, targets=targets),
        info=dataset_info,
        dataset_path=path,
        num_total_trajectories=len(trajectories),
        num_kept_trajectories=len(trajectories) - num_action_filtered_trajectories - num_scenario_filtered_trajectories,
        num_action_filtered_trajectories=num_action_filtered_trajectories,
        num_scenario_filtered_trajectories=num_scenario_filtered_trajectories,
        num_used_transitions=inputs.shape[0],
        applied_num_train_scenarios=num_train_scenarios,
    )


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
    )


def compute_normalization(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = inputs.mean(dim=0)
    std = inputs.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> float:
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            normalized_inputs = (inputs - input_mean) / input_std
            normalized_targets = (targets - target_mean) / target_std
            predictions = model(normalized_inputs)
            batch_size = inputs.shape[0]
            loss_sum += loss_fn(predictions, normalized_targets).item() * batch_size
            count += batch_size
    return loss_sum / max(count, 1)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
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
            "input_mean": input_mean.cpu(),
            "input_std": input_std.cpu(),
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

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = requested_device

    full_trainval_bundle = load_transitions(
        args.train_data,
        args.obs_key,
        num_train_scenarios=args.num_train_scenarios,
        with_noise_value=args.with_noise_value,
    )
    test_bundle = None
    if args.test_data and Path(args.test_data).exists():
        test_bundle = load_transitions(
            args.test_data,
            args.obs_key,
            with_noise_value=args.with_noise_value,
        )
    full_trainval = full_trainval_bundle.transitions

    full_dataset = TensorDataset(full_trainval.inputs, full_trainval.targets)
    num_train = math.floor(len(full_dataset) * args.train_fraction)
    num_val = len(full_dataset) - num_train
    if num_train == 0 or num_val == 0:
        raise ValueError(
            f"Train/val split is empty with {len(full_dataset)} transitions and train_fraction={args.train_fraction}"
        )

    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val], generator=split_generator)
    test_dataset = TensorDataset(test_bundle.transitions.inputs, test_bundle.transitions.targets) if test_bundle else None

    train_indices = train_dataset.indices
    train_inputs = full_trainval.inputs[train_indices]
    train_targets = full_trainval.targets[train_indices]
    input_mean, input_std = compute_normalization(train_inputs)
    target_mean, target_std = compute_normalization(train_targets)

    train_loader = build_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_eval_loader = build_dataloader(train_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = build_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = (
        build_dataloader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
        if test_dataset is not None
        else None
    )

    input_dim = full_trainval.inputs.shape[1]
    output_dim = full_trainval.targets.shape[1]
    model = MarkovianMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    input_mean = input_mean.to(device)
    input_std = input_std.to(device)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)
    model_info = {
        "model_class": model.__class__.__name__,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": args.hidden_dim,
        "num_linear_layers": 5,
        "dropout": args.dropout,
        "loss": loss_fn.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "input_normalization": "train_split_mean_std",
        "target_normalization": "train_split_mean_std",
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
        "test": None
        if test_bundle is None
        else {
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
            "train_transitions": len(train_dataset),
            "val_transitions": len(val_dataset),
        },
        "filtering": {
            "action_magnitude_threshold": 100.0,
            "num_train_scenarios": args.num_train_scenarios,
            "with_noise_value": args.with_noise_value,
        },
        "dataset_sizes_used_for_training": {
            "train_transitions": len(train_dataset),
            "val_transitions": len(val_dataset),
            "test_transitions": len(test_dataset) if test_dataset is not None else 0,
            "trainval_transitions_before_split": len(full_dataset),
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
            "input_dim": input_dim,
            "output_dim": output_dim,
            "train_transitions": len(train_dataset),
            "val_transitions": len(val_dataset),
            "test_transitions": len(test_dataset) if test_dataset is not None else 0,
            "num_train_scenarios": args.num_train_scenarios,
            "with_noise_value": args.with_noise_value,
            "trainval_action_filtered_trajectories": full_trainval_bundle.num_action_filtered_trajectories,
            "trainval_scenario_filtered_trajectories": full_trainval_bundle.num_scenario_filtered_trajectories,
            "test_action_filtered_trajectories": test_bundle.num_action_filtered_trajectories if test_bundle else 0,
            "test_scenario_filtered_trajectories": test_bundle.num_scenario_filtered_trajectories if test_bundle else 0,
        },
    )

    best_val_loss = float("inf")
    best_path = os.path.join(args.output_dir, "best.pt")
    last_path = os.path.join(args.output_dir, "last.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            normalized_inputs = (inputs - input_mean) / input_std
            normalized_targets = (targets - target_mean) / target_std

            optimizer.zero_grad(set_to_none=True)
            predictions = model(normalized_inputs)
            loss = loss_fn(predictions, normalized_targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)
        train_eval_loss = evaluate(
            model, train_eval_loader, loss_fn, device, input_mean, input_std, target_mean, target_std
        )
        val_loss = evaluate(model, val_loader, loss_fn, device, input_mean, input_std, target_mean, target_std)

        metrics = {
            "epoch": epoch,
            "loss/train": train_loss,
            "loss/train_eval": train_eval_loss,
            "loss/val": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        test_loss = None
        if test_loader is not None:
            test_loss = evaluate(model, test_loader, loss_fn, device, input_mean, input_std, target_mean, target_std)
            metrics["loss/test"] = test_loss
        wandb.log(metrics, step=epoch)
        message = (
            f"Epoch {epoch:04d} | "
            f"train_loss={train_loss:.6f} | "
            f"train_eval_loss={train_eval_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )
        if test_loss is not None:
            message += f" | test_loss={test_loss:.6f}"
        print(message)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
                input_mean,
                input_std,
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
                input_mean,
                input_std,
                target_mean,
                target_std,
                base_save_dict,
            )

    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary["best_checkpoint"] = best_path
    wandb.finish()


if __name__ == "__main__":
    main()
