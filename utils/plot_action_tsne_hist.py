#!/usr/bin/env python3
"""Plot 2D t-SNE histograms for two robot action datasets."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load two trajectory pickle files, sample valid actions_expert vectors, "
            "embed them in 2D with t-SNE, and save overlaid 2D histograms."
        )
    )
    parser.add_argument("dataset_a", type=Path, help="First trajectories.pkl file.")
    parser.add_argument("dataset_b", type=Path, help="Second trajectories.pkl file.")
    parser.add_argument(
        "--labels",
        nargs=2,
        default=("dataset_a", "dataset_b"),
        metavar=("LABEL_A", "LABEL_B"),
        help="Labels to use in plot titles and filenames.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("action_tsne_hist.png"),
        help="Path for the output image.",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=10_000,
        help="Maximum number of valid actions to sample from each dataset before t-SNE.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=160,
        help="Number of histogram bins along each t-SNE axis.",
    )
    parser.add_argument(
        "--magnitude-threshold",
        type=float,
        default=100.0,
        help="Drop actions whose L2 magnitude is greater than this threshold.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity. Must be less than the combined sample size.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and t-SNE.")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI.")
    return parser.parse_args()


def sample_valid_actions(
    dataset_path: Path,
    *,
    max_samples: int,
    magnitude_threshold: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    """Load a trajectory list and return a random sample of valid actions_expert rows."""
    if max_samples <= 0:
        raise ValueError("--max-samples-per-dataset must be positive.")

    with dataset_path.open("rb") as f:
        trajectories = pickle.load(f)

    sampled_actions: np.ndarray | None = None
    sampled_keys: np.ndarray | None = None
    total_actions = 0
    total_valid_actions = 0

    for traj_idx, trajectory in enumerate(trajectories):
        if "actions_expert" not in trajectory:
            raise KeyError(f"Trajectory {traj_idx} in {dataset_path} does not contain 'actions_expert'.")

        actions = np.asarray(trajectory["actions_expert"])
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(
                f"Expected actions_expert with shape (T, 7) in trajectory {traj_idx} of {dataset_path}, "
                f"got {actions.shape}."
            )

        actions = actions.astype(np.float32, copy=False)
        finite_mask = np.isfinite(actions).all(axis=1)
        magnitude_mask = np.linalg.norm(actions, axis=1) <= magnitude_threshold
        valid_actions = actions[finite_mask & magnitude_mask]

        total_actions += actions.shape[0]
        total_valid_actions += valid_actions.shape[0]

        if not valid_actions.size:
            continue

        random_keys = rng.random(valid_actions.shape[0])
        if valid_actions.shape[0] > max_samples:
            keep = np.argpartition(random_keys, max_samples - 1)[:max_samples]
            valid_actions = valid_actions[keep]
            random_keys = random_keys[keep]

        if sampled_actions is None or sampled_keys is None:
            sampled_actions = valid_actions.copy()
            sampled_keys = random_keys.copy()
            continue

        sampled_actions = np.concatenate([sampled_actions, valid_actions], axis=0)
        sampled_keys = np.concatenate([sampled_keys, random_keys], axis=0)
        if sampled_actions.shape[0] > max_samples:
            keep = np.argpartition(sampled_keys, max_samples - 1)[:max_samples]
            sampled_actions = sampled_actions[keep]
            sampled_keys = sampled_keys[keep]

    if sampled_actions is None:
        raise ValueError(f"No valid actions found in {dataset_path}.")

    return sampled_actions, total_actions, total_valid_actions


def standardize_combined(actions_a: np.ndarray, actions_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize both datasets using combined action statistics."""
    combined = np.concatenate([actions_a, actions_b], axis=0)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (actions_a - mean) / std, (actions_b - mean) / std


def make_hist_plot(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    *,
    labels: tuple[str, str],
    output_path: Path,
    bins: int,
    dpi: int,
) -> None:
    """Save per-dataset, overlaid, and difference 2D t-SNE histograms in one figure."""
    x_min = min(embedding_a[:, 0].min(), embedding_b[:, 0].min())
    x_max = max(embedding_a[:, 0].max(), embedding_b[:, 0].max())
    y_min = min(embedding_a[:, 1].min(), embedding_b[:, 1].min())
    y_max = max(embedding_a[:, 1].max(), embedding_b[:, 1].max())
    hist_range = [[x_min, x_max], [y_min, y_max]]

    hist_a, x_edges, y_edges = np.histogram2d(embedding_a[:, 0], embedding_a[:, 1], bins=bins, range=hist_range)
    hist_b, _, _ = np.histogram2d(embedding_b[:, 0], embedding_b[:, 1], bins=[x_edges, y_edges])

    hist_a = hist_a.T
    hist_b = hist_b.T
    diff = hist_b / max(hist_b.sum(), 1.0) - hist_a / max(hist_a.sum(), 1.0)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    log_a = np.log1p(hist_a)
    log_b = np.log1p(hist_b)
    norm_a = log_a / max(log_a.max(), 1.0)
    norm_b = log_b / max(log_b.max(), 1.0)
    overlay = np.zeros((*hist_a.shape, 3), dtype=np.float32)
    overlay[..., 0] = norm_b
    overlay[..., 1] = 0.35 * np.minimum(norm_a, norm_b)
    overlay[..., 2] = norm_a

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)

    im_a = axes[0].imshow(hist_a, origin="lower", extent=extent, aspect="auto", cmap="Blues")
    axes[0].set_title(f"{labels[0]} t-SNE density\nN={embedding_a.shape[0]:,}")
    fig.colorbar(im_a, ax=axes[0], label="count")

    im_b = axes[1].imshow(hist_b, origin="lower", extent=extent, aspect="auto", cmap="Oranges")
    axes[1].set_title(f"{labels[1]} t-SNE density\nN={embedding_b.shape[0]:,}")
    fig.colorbar(im_b, ax=axes[1], label="count")

    axes[2].imshow(overlay, origin="lower", extent=extent, aspect="auto")
    axes[2].set_title(f"Overlay\nblue={labels[0]}, red={labels[1]}")

    limit = float(np.max(np.abs(diff))) if diff.size else 1.0
    limit = limit if limit > 0 else 1.0
    im_diff = axes[3].imshow(
        diff,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[3].set_title(f"Normalized density difference\n{labels[1]} - {labels[0]}")
    fig.colorbar(im_diff, ax=axes[3], label="probability/bin difference")

    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    actions_a, total_a, valid_a = sample_valid_actions(
        args.dataset_a,
        max_samples=args.max_samples_per_dataset,
        magnitude_threshold=args.magnitude_threshold,
        rng=rng,
    )
    actions_b, total_b, valid_b = sample_valid_actions(
        args.dataset_b,
        max_samples=args.max_samples_per_dataset,
        magnitude_threshold=args.magnitude_threshold,
        rng=rng,
    )

    scaled_a, scaled_b = standardize_combined(actions_a, actions_b)
    combined = np.concatenate([scaled_a, scaled_b], axis=0)
    if combined.shape[0] <= 1:
        raise ValueError("Need at least two sampled valid actions to run t-SNE.")

    perplexity = min(args.perplexity, max(1.0, (combined.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    )
    embedding = tsne.fit_transform(combined)
    embedding_a = embedding[: scaled_a.shape[0]]
    embedding_b = embedding[scaled_a.shape[0] :]

    make_hist_plot(
        embedding_a,
        embedding_b,
        labels=tuple(args.labels),
        output_path=args.output,
        bins=args.bins,
        dpi=args.dpi,
    )

    print(f"{args.labels[0]}: kept {valid_a:,}/{total_a:,} valid actions; t-SNE sample {actions_a.shape[0]:,}")
    print(f"{args.labels[1]}: kept {valid_b:,}/{total_b:,} valid actions; t-SNE sample {actions_b.shape[0]:,}")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
