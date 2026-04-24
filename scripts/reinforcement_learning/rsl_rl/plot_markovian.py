#!/usr/bin/env python3

# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot per-transition losses for a trained Markovian policy."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from train_markovian import MarkovianMLP, TensorDataset, load_transitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-transition losses for a trained Markovian policy.")
    parser.add_argument("experiment_dir", type=str, help="Directory containing a finished Markovian training run.")
    parser.add_argument("--checkpoint", type=str, default="last.pt", help="Checkpoint filename or path to load.")
    parser.add_argument("--output", type=str, default="markovian_loss_histogram.png", help="Output image filename or path.")
    parser.add_argument("--batch-size", type=int, default=8192, help="Evaluation batch size.")
    parser.add_argument("--bins", type=int, default=100, help="Histogram bin count.")
    parser.add_argument("--device", type=str, default=None, help="Evaluation device. Defaults to cuda if available else cpu.")
    return parser.parse_args()


def resolve_path(path: str, experiment_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return experiment_dir / candidate


def load_checkpoint(checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def build_model(checkpoint: dict) -> MarkovianMLP:
    model_info = checkpoint["model_info"]
    model = MarkovianMLP(
        input_dim=int(model_info["input_dim"]),
        output_dim=int(model_info["output_dim"]),
        hidden_dim=int(model_info["hidden_dim"]),
        dropout=float(model_info["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def split_train_val(dataset: Dataset, train_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    num_train = math.floor(len(dataset) * train_fraction)
    num_val = len(dataset) - num_train
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [num_train, num_val], generator=generator)


def per_transition_losses(
    model: MarkovianMLP,
    dataset: Dataset,
    checkpoint: dict,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    input_mean = checkpoint["input_mean"].to(device)
    input_std = checkpoint["input_std"].to(device)
    target_mean = checkpoint["target_mean"].to(device)
    target_std = checkpoint["target_std"].to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses = []

    model.to(device)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model((inputs - input_mean) / input_std)
            normalized_targets = (targets - target_mean) / target_std
            losses.append(torch.mean((predictions - normalized_targets) ** 2, dim=1).cpu().numpy())

    return np.concatenate(losses, axis=0)


def trim_percent(losses: np.ndarray, percent: float = 1.0) -> np.ndarray:
    if losses.size == 0:
        return losses
    lower, upper = np.percentile(losses, [percent, 100.0 - percent])
    return losses[(losses >= lower) & (losses <= upper)]


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    checkpoint_path = resolve_path(args.checkpoint, experiment_dir)
    output_path = resolve_path(args.output, experiment_dir)
    checkpoint = load_checkpoint(checkpoint_path)
    saved_args = checkpoint["args"]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(checkpoint)
    trainval_bundle = load_transitions(
        saved_args["train_data"],
        saved_args["obs_key"],
        num_train_scenarios=saved_args.get("num_train_scenarios"),
        with_noise_value=saved_args.get("with_noise_value", False),
    )
    trainval_dataset = TensorDataset(trainval_bundle.transitions.inputs, trainval_bundle.transitions.targets)
    train_dataset, val_dataset = split_train_val(
        trainval_dataset,
        saved_args.get("train_fraction", 0.8),
        saved_args.get("seed", 42),
    )

    datasets: list[tuple[str, Dataset]] = [("train", train_dataset), ("val", val_dataset)]
    test_data = saved_args.get("test_data")
    if test_data and Path(test_data).exists():
        test_bundle = load_transitions(
            test_data,
            saved_args["obs_key"],
            with_noise_value=saved_args.get("with_noise_value", False),
        )
        datasets.append(("test", TensorDataset(test_bundle.transitions.inputs, test_bundle.transitions.targets)))

    plt.figure(figsize=(10, 6))
    for name, dataset in datasets:
        losses = per_transition_losses(model, dataset, checkpoint, args.batch_size, device)
        trimmed = trim_percent(losses)
        plt.hist(trimmed, bins=args.bins, alpha=0.5, density=True, label=f"{name} (n={trimmed.size})")
        print(f"{name}: mean={losses.mean():.6f}, trimmed_mean={trimmed.mean():.6f}, n={losses.size}")

    plt.xlabel("Per-transition normalized MSE")
    plt.ylabel("Density")
    plt.title("Markovian per-transition loss distribution")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
