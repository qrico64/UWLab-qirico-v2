#!/usr/bin/env python3

"""Train a Markovian dynamics model from collected trajectories."""

from __future__ import annotations

import argparse
import pickle
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset


FRICTION_SPLIT_THRESHOLDS = {
    "insertive_object.material.shape_0.static_friction": 1.5,
    "insertive_object.material.shape_0.dynamic_friction": 1.4,
    "receptive_object.material.shape_0.static_friction": 0.4,
    "receptive_object.material.shape_0.dynamic_friction": 0.35,
}


@dataclass
class SplitData:
    inputs: torch.Tensor
    targets: torch.Tensor
    num_trajectories: int
    num_transitions: int


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="collected_data/data_may2_r3n1_2/trajectories.pkl")
    parser.add_argument("--obs-key", type=str, default="policy", choices=("policy", "policy2"))
    parser.add_argument("--with-dynamic-parameters", action="store_true")
    parser.add_argument("--hidden-dims", type=str, default="512,512,512")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="markovian_dynamics")
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


def dynamics_vector(trajectory: dict, dynamics_keys: list[str]) -> np.ndarray:
    dynamics = trajectory["dynamics"]
    return np.asarray([dynamics[key] for key in dynamics_keys], dtype=np.float32)


def split_trajectory(trajectory: dict) -> str | None:
    dynamics = trajectory["dynamics"]
    values = [float(dynamics[key]) >= threshold for key, threshold in FRICTION_SPLIT_THRESHOLDS.items()]
    if all(values):
        return "train"
    if not any(values):
        return "test"
    return None


def has_large_action(trajectory: dict, threshold: float = 100.0) -> bool:
    actions = np.asarray(trajectory["actions"], dtype=np.float32)
    magnitudes = np.linalg.norm(actions, axis=-1) if actions.ndim > 1 else np.abs(actions)
    return bool(np.any(magnitudes > threshold))


def build_split(
    trajectories: list[dict],
    split_name: str,
    obs_key: str,
    with_dynamic_parameters: bool,
    dynamics_keys: list[str],
) -> SplitData:
    inputs = []
    targets = []
    num_trajectories = 0

    for trajectory in trajectories:
        observations = np.asarray(trajectory["obs"][obs_key], dtype=np.float32)
        actions = np.asarray(trajectory["actions"], dtype=np.float32)
        steps = min(observations.shape[0], actions.shape[0]) - 1
        if steps <= 0:
            continue

        split_inputs = [observations[:steps], actions[:steps]]
        if with_dynamic_parameters:
            dynamic_params = np.repeat(dynamics_vector(trajectory, dynamics_keys)[None, :], steps, axis=0)
            split_inputs.append(dynamic_params)

        inputs.append(np.concatenate(split_inputs, axis=-1))
        targets.append(observations[1 : steps + 1])
        num_trajectories += 1

    if not inputs:
        raise ValueError(f"No transitions found for {split_name} split.")

    inputs_tensor = torch.from_numpy(np.concatenate(inputs, axis=0))
    targets_tensor = torch.from_numpy(np.concatenate(targets, axis=0))
    return SplitData(inputs_tensor, targets_tensor, num_trajectories, inputs_tensor.shape[0])


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
            preds = model((inputs - input_mean) / input_std)
            loss = loss_fn(preds, (targets - target_mean) / target_std)
            loss_sum += loss.item() * inputs.shape[0]
            count += inputs.shape[0]
    return loss_sum / max(count, 1)


def main() -> None:
    args = parse_args()
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(",") if dim]
    if not hidden_dims:
        raise ValueError("--hidden-dims must contain at least one layer size.")

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    with open(args.data, "rb") as f:
        trajectories = pickle.load(f)
    if not trajectories:
        raise ValueError(f"No trajectories found in {args.data}")
    num_total_trajectories = len(trajectories)
    num_short_trajectories = sum(int(trajectory["T"]) <= 10 for trajectory in trajectories)
    trajectories = [trajectory for trajectory in trajectories if int(trajectory["T"]) > 10]
    num_large_action_trajectories = sum(has_large_action(trajectory) for trajectory in trajectories)
    trajectories = [trajectory for trajectory in trajectories if not has_large_action(trajectory)]
    if not trajectories:
        raise ValueError("No trajectories remain after filtering.")

    dynamics_keys = sorted(trajectories[0]["dynamics"].keys()) if args.with_dynamic_parameters else []
    train_trajectories = []
    test_trajectories = []
    num_unused_split_trajectories = 0
    for trajectory in trajectories:
        split = split_trajectory(trajectory)
        if split == "train":
            train_trajectories.append(trajectory)
        elif split == "test":
            test_trajectories.append(trajectory)
        else:
            num_unused_split_trajectories += 1

    train = build_split(train_trajectories, "train", args.obs_key, args.with_dynamic_parameters, dynamics_keys)
    test = build_split(test_trajectories, "test", args.obs_key, args.with_dynamic_parameters, dynamics_keys)
    input_mean, input_std = compute_normalization(train.inputs)
    target_mean, target_std = compute_normalization(train.targets)

    train_loader = DataLoader(
        TensorDataset(train.inputs, train.targets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    train_eval_loader = DataLoader(
        TensorDataset(train.inputs, train.targets),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        TensorDataset(test.inputs, test.targets),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = MLP(train.inputs.shape[1], train.targets.shape[1], hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    input_mean = input_mean.to(device)
    input_std = input_std.to(device)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={
            **vars(args),
            "input_dim": train.inputs.shape[1],
            "output_dim": train.targets.shape[1],
            "num_dynamics_keys": len(dynamics_keys),
            "train_trajectories": train.num_trajectories,
            "test_trajectories": test.num_trajectories,
            "train_transitions": train.num_transitions,
            "test_transitions": test.num_transitions,
            "num_total_trajectories": num_total_trajectories,
            "num_short_trajectories": num_short_trajectories,
            "num_large_action_trajectories": num_large_action_trajectories,
            "num_kept_trajectories": len(trajectories),
            "num_unused_split_trajectories": num_unused_split_trajectories,
            "friction_split_thresholds": FRICTION_SPLIT_THRESHOLDS,
            "dynamics_keys": dynamics_keys,
        },
    )
    split_sizes = {
        "dataset/train_trajectories": train.num_trajectories,
        "dataset/test_trajectories": test.num_trajectories,
        "dataset/train_transitions": train.num_transitions,
        "dataset/test_transitions": test.num_transitions,
        "dataset/unused_split_trajectories": num_unused_split_trajectories,
    }
    wandb.summary.update(split_sizes)
    wandb.log(split_sizes, step=0)
    print(
        "split sizes | "
        f"train_trajectories={train.num_trajectories} "
        f"test_trajectories={test.num_trajectories} "
        f"train_transitions={train.num_transitions} "
        f"test_transitions={test.num_transitions} "
        f"unused_split_trajectories={num_unused_split_trajectories}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = model((inputs - input_mean) / input_std)
            loss = loss_fn(preds, (targets - target_mean) / target_std)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * inputs.shape[0]
            count += inputs.shape[0]

        train_loss = evaluate(model, train_eval_loader, input_mean, input_std, target_mean, target_std, device)
        test_loss = evaluate(model, test_loader, input_mean, input_std, target_mean, target_std, device)
        wandb.log({"loss/train": train_loss, "loss/test": test_loss, "epoch": epoch}, step=epoch)
        print(f"epoch={epoch:04d} train_loss={train_loss:.6f} test_loss={test_loss:.6f}")

    wandb.finish()


if __name__ == "__main__":
    main()
