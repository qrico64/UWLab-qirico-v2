#!/usr/bin/env python3
"""Plot 2D t-SNE action distributions grouped by trajectory noise_index."""

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
            "Load one trajectory pickle file, group actions by trajectory noise_index, "
            "embed sampled actions in 2D with t-SNE, and save a colored plot."
        )
    )
    parser.add_argument("dataset", type=Path, help="Trajectory pickle file.")
    parser.add_argument(
        "--num-noise-indices",
        type=int,
        default=20,
        help="Plot only noise_index values in [0, N).",
    )
    parser.add_argument(
        "--action-key",
        default="actions_expert",
        help="Trajectory key containing action arrays with shape (T, 7).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("action_tsne_noise_index.png"),
        help="Path for the output image.",
    )
    parser.add_argument(
        "--max-samples-per-noise-index",
        type=int,
        default=2_000,
        help="Maximum number of valid actions sampled for each plotted noise_index.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=240,
        help="Number of bins along each t-SNE axis for the colored density image.",
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


def update_group_sample(
    current_actions: np.ndarray | None,
    current_keys: np.ndarray | None,
    new_actions: np.ndarray,
    *,
    max_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep a bounded random sample for one noise_index group."""
    random_keys = rng.random(new_actions.shape[0])
    if new_actions.shape[0] > max_samples:
        keep = np.argpartition(random_keys, max_samples - 1)[:max_samples]
        new_actions = new_actions[keep]
        random_keys = random_keys[keep]

    if current_actions is None or current_keys is None:
        return new_actions.copy(), random_keys.copy()

    actions = np.concatenate([current_actions, new_actions], axis=0)
    keys = np.concatenate([current_keys, random_keys], axis=0)
    if actions.shape[0] > max_samples:
        keep = np.argpartition(keys, max_samples - 1)[:max_samples]
        actions = actions[keep]
        keys = keys[keep]
    return actions, keys


def sample_actions_by_noise_index(
    dataset_path: Path,
    *,
    num_noise_indices: int,
    action_key: str,
    max_samples_per_noise_index: int,
    magnitude_threshold: float,
    rng: np.random.Generator,
) -> tuple[dict[int, np.ndarray], dict[int, int], dict[int, int], dict[int, int]]:
    """Load trajectories and sample valid actions for noise_index values in [0, N)."""
    if num_noise_indices <= 0:
        raise ValueError("--num-noise-indices must be positive.")
    if max_samples_per_noise_index <= 0:
        raise ValueError("--max-samples-per-noise-index must be positive.")

    with dataset_path.open("rb") as f:
        trajectories = pickle.load(f)

    samples: dict[int, np.ndarray | None] = {idx: None for idx in range(num_noise_indices)}
    sample_keys: dict[int, np.ndarray | None] = {idx: None for idx in range(num_noise_indices)}
    trajectory_counts: dict[int, int] = {idx: 0 for idx in range(num_noise_indices)}
    total_action_counts: dict[int, int] = {idx: 0 for idx in range(num_noise_indices)}
    valid_action_counts: dict[int, int] = {idx: 0 for idx in range(num_noise_indices)}

    for traj_idx, trajectory in enumerate(trajectories):
        if "noise_index" not in trajectory:
            raise KeyError(f"Trajectory {traj_idx} in {dataset_path} does not contain 'noise_index'.")
        if action_key not in trajectory:
            raise KeyError(f"Trajectory {traj_idx} in {dataset_path} does not contain '{action_key}'.")

        noise_index = int(trajectory["noise_index"])
        if noise_index < 0 or noise_index >= num_noise_indices:
            continue

        actions = np.asarray(trajectory[action_key])
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(
                f"Expected {action_key} with shape (T, 7) in trajectory {traj_idx} of {dataset_path}, "
                f"got {actions.shape}."
            )

        actions = actions.astype(np.float32, copy=False)
        finite_mask = np.isfinite(actions).all(axis=1)
        magnitude_mask = np.linalg.norm(actions, axis=1) <= magnitude_threshold
        valid_actions = actions[finite_mask & magnitude_mask]

        trajectory_counts[noise_index] += 1
        total_action_counts[noise_index] += actions.shape[0]
        valid_action_counts[noise_index] += valid_actions.shape[0]
        if not valid_actions.size:
            continue

        samples[noise_index], sample_keys[noise_index] = update_group_sample(
            samples[noise_index],
            sample_keys[noise_index],
            valid_actions,
            max_samples=max_samples_per_noise_index,
            rng=rng,
        )

    final_samples = {idx: actions for idx, actions in samples.items() if actions is not None}
    if not final_samples:
        raise ValueError(f"No valid actions found in {dataset_path} for noise_index values [0, {num_noise_indices}).")

    return final_samples, trajectory_counts, total_action_counts, valid_action_counts


def standardize(actions_by_noise_index: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Standardize actions using statistics from all plotted groups."""
    combined = np.concatenate(list(actions_by_noise_index.values()), axis=0)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {idx: (actions - mean) / std for idx, actions in actions_by_noise_index.items()}


def make_color_density(
    embeddings_by_noise_index: dict[int, np.ndarray],
    *,
    output_path: Path,
    hist_bins: int,
    dpi: int,
) -> None:
    """Save a colored density map and a sampled scatter plot."""
    all_embeddings = np.concatenate(list(embeddings_by_noise_index.values()), axis=0)
    x_min, y_min = all_embeddings.min(axis=0)
    x_max, y_max = all_embeddings.max(axis=0)
    hist_range = [[x_min, x_max], [y_min, y_max]]
    extent = [x_min, x_max, y_min, y_max]

    noise_indices = sorted(embeddings_by_noise_index)
    colors = plt.get_cmap("tab20", max(len(noise_indices), 1))
    density_rgb = np.zeros((hist_bins, hist_bins, 3), dtype=np.float32)
    density_alpha = np.zeros((hist_bins, hist_bins), dtype=np.float32)

    for color_idx, noise_index in enumerate(noise_indices):
        embedding = embeddings_by_noise_index[noise_index]
        hist, _, _ = np.histogram2d(embedding[:, 0], embedding[:, 1], bins=hist_bins, range=hist_range)
        hist = np.log1p(hist.T)
        if hist.max() > 0:
            hist = hist / hist.max()
        color = np.asarray(colors(color_idx)[:3], dtype=np.float32)
        density_rgb += hist[..., None] * color
        density_alpha = np.maximum(density_alpha, hist)

    density_rgb = density_rgb / np.maximum(density_rgb.max(), 1.0)
    density_image = np.clip(density_rgb * density_alpha[..., None], 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    axes[0].imshow(density_image, origin="lower", extent=extent, aspect="auto")
    axes[0].set_title("t-SNE density by noise_index")

    for color_idx, noise_index in enumerate(noise_indices):
        embedding = embeddings_by_noise_index[noise_index]
        axes[1].scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=2,
            alpha=0.8,
            color=colors(color_idx),
            label=f"{noise_index} ({embedding.shape[0]:,})",
            linewidths=0,
        )

    axes[1].set_title("Sampled t-SNE actions")
    axes[1].legend(title="noise_index", loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=3)

    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    actions_by_noise_index, trajectory_counts, total_action_counts, valid_action_counts = sample_actions_by_noise_index(
        args.dataset,
        num_noise_indices=args.num_noise_indices,
        action_key=args.action_key,
        max_samples_per_noise_index=args.max_samples_per_noise_index,
        magnitude_threshold=args.magnitude_threshold,
        rng=rng,
    )

    scaled_by_noise_index = standardize(actions_by_noise_index)
    ordered_indices = sorted(scaled_by_noise_index)
    combined = np.concatenate([scaled_by_noise_index[idx] for idx in ordered_indices], axis=0)
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
    combined_embedding = tsne.fit_transform(combined)

    embeddings_by_noise_index: dict[int, np.ndarray] = {}
    start = 0
    for noise_index in ordered_indices:
        count = scaled_by_noise_index[noise_index].shape[0]
        embeddings_by_noise_index[noise_index] = combined_embedding[start : start + count]
        start += count

    make_color_density(
        embeddings_by_noise_index,
        output_path=args.output,
        hist_bins=args.hist_bins,
        dpi=args.dpi,
    )

    for noise_index in range(args.num_noise_indices):
        if trajectory_counts[noise_index] == 0:
            print(f"noise_index {noise_index}: no trajectories")
            continue
        sample_count = actions_by_noise_index.get(noise_index, np.empty((0, 7))).shape[0]
        print(
            f"noise_index {noise_index}: trajectories={trajectory_counts[noise_index]:,}, "
            f"kept={valid_action_counts[noise_index]:,}/{total_action_counts[noise_index]:,}, "
            f"t-SNE sample={sample_count:,}"
        )
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
