#!/usr/bin/env python3
"""Plot 2D t-SNE histograms for two robot action and observation datasets."""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class SeriesSpec:
    """Description of one per-timestep trajectory array to sample and plot."""

    name: str
    key_path: tuple[str, ...]
    output_suffix: str
    expected_dim: int | None = None
    magnitude_threshold: float | None = None
    abs_threshold: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load two trajectory pickle files, sample valid per-timestep action and "
            "observation vectors, embed them in 2D with t-SNE, and save overlaid "
            "2D histograms."
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
        help=(
            "Path for the action output image. Observation plots are saved beside it "
            "with obs-key suffixes."
        ),
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=10_000,
        help="Maximum number of valid rows to sample from each dataset before t-SNE.",
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
        "--obs-keys",
        nargs="+",
        default=("policy", "policy2"),
        help="Observation keys under traj['obs'] to plot.",
    )
    parser.add_argument(
        "--observation-abs-threshold",
        type=float,
        default=300.0,
        help=(
            "Drop observations containing an absolute value greater than this threshold. "
            "Use a negative value to disable this filter."
        ),
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


def get_nested(mapping: dict[str, Any], key_path: tuple[str, ...]) -> Any:
    """Read a nested trajectory value with a clear error if a key is absent."""
    value: Any = mapping
    traversed: list[str] = []
    for key in key_path:
        traversed.append(key)
        if not isinstance(value, dict) or key not in value:
            joined = "']['".join(traversed)
            raise KeyError(f"Trajectory does not contain ['{joined}'].")
        value = value[key]
    return value


def output_path_for_series(base_output: Path, spec: SeriesSpec) -> Path:
    """Return the output path for a plot series without changing action output names."""
    if not spec.output_suffix:
        return base_output

    suffix = base_output.suffix or ".png"
    stem = base_output.stem if base_output.suffix else base_output.name
    return base_output.with_name(f"{stem}_{spec.output_suffix}{suffix}")


def sample_valid_rows(
    dataset_path: Path,
    spec: SeriesSpec,
    *,
    max_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    """Load trajectories and return a random sample of valid per-timestep rows."""
    if max_samples <= 0:
        raise ValueError("--max-samples-per-dataset must be positive.")

    with dataset_path.open("rb") as f:
        trajectories = pickle.load(f)

    sampled_rows: np.ndarray | None = None
    sampled_keys: np.ndarray | None = None
    total_rows = 0
    total_valid_rows = 0

    for traj_idx, trajectory in enumerate(trajectories):
        rows = np.asarray(get_nested(trajectory, spec.key_path))
        if rows.ndim != 2:
            raise ValueError(
                f"Expected {spec.name} with shape (T, D) in trajectory {traj_idx} of {dataset_path}, "
                f"got {rows.shape}."
            )
        if spec.expected_dim is not None and rows.shape[1] != spec.expected_dim:
            raise ValueError(
                f"Expected {spec.name} with shape (T, {spec.expected_dim}) in trajectory {traj_idx} "
                f"of {dataset_path}, got {rows.shape}."
            )

        rows = rows.astype(np.float32, copy=False)
        valid_mask = np.isfinite(rows).all(axis=1)
        if spec.magnitude_threshold is not None:
            valid_mask &= np.linalg.norm(rows, axis=1) <= spec.magnitude_threshold
        if spec.abs_threshold is not None:
            valid_mask &= (np.abs(rows) <= spec.abs_threshold).all(axis=1)
        valid_rows = rows[valid_mask]

        total_rows += rows.shape[0]
        total_valid_rows += valid_rows.shape[0]

        if not valid_rows.size:
            continue

        random_keys = rng.random(valid_rows.shape[0])
        if valid_rows.shape[0] > max_samples:
            keep = np.argpartition(random_keys, max_samples - 1)[:max_samples]
            valid_rows = valid_rows[keep]
            random_keys = random_keys[keep]

        if sampled_rows is None or sampled_keys is None:
            sampled_rows = valid_rows.copy()
            sampled_keys = random_keys.copy()
            continue

        sampled_rows = np.concatenate([sampled_rows, valid_rows], axis=0)
        sampled_keys = np.concatenate([sampled_keys, random_keys], axis=0)
        if sampled_rows.shape[0] > max_samples:
            keep = np.argpartition(sampled_keys, max_samples - 1)[:max_samples]
            sampled_rows = sampled_rows[keep]
            sampled_keys = sampled_keys[keep]

    if sampled_rows is None:
        key_name = ".".join(spec.key_path)
        raise ValueError(f"No valid {key_name} rows found in {dataset_path}.")

    return sampled_rows, total_rows, total_valid_rows


def standardize_combined(rows_a: np.ndarray, rows_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize both datasets using combined feature statistics."""
    combined = np.concatenate([rows_a, rows_b], axis=0)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (rows_a - mean) / std, (rows_b - mean) / std


def nearest_neighbor_coverage_diagnostic(
    scaled_a: np.ndarray,
    scaled_b: np.ndarray,
    *,
    labels: tuple[str, str],
    output_path: Path,
    k: int = 10,
    dpi: int = 300,
) -> None:
    # B -> A distances: how far each B point is from the A distribution
    nn_a = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn_a.fit(scaled_a)

    b_to_a_distances, _ = nn_a.kneighbors(scaled_b)
    b_to_a_d1 = b_to_a_distances[:, 0]
    b_to_a_dk_mean = b_to_a_distances.mean(axis=1)

    # A -> A leave-one-out distances: normal within-A nearest-neighbor scale
    # Need k + 1 because each A point's closest neighbor is itself.
    nn_a_self = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn_a_self.fit(scaled_a)

    a_to_a_distances, _ = nn_a_self.kneighbors(scaled_a)
    a_to_a_d1 = a_to_a_distances[:, 1]          # skip self
    a_to_a_dk_mean = a_to_a_distances[:, 1:].mean(axis=1)

    # Quantile-based coverage score.
    # Example: a B point is "covered" if its nearest-A distance is within
    # the 95th percentile of normal A-to-A nearest-neighbor distances.
    threshold_d1 = np.quantile(a_to_a_d1, 0.99)
    threshold_dk = np.quantile(a_to_a_dk_mean, 0.99)

    coverage_d1 = np.mean(b_to_a_d1 <= threshold_d1)
    coverage_dk = np.mean(b_to_a_dk_mean <= threshold_dk)

    print(f"Nearest-neighbor coverage diagnostic: {labels[1]} against {labels[0]}")
    print(f"  A->A d1 95th percentile: {threshold_d1:.4f}")
    print(f"  B->A d1 covered fraction: {coverage_d1:.4f}")
    print(f"  A->A mean-k 95th percentile: {threshold_dk:.4f}")
    print(f"  B->A mean-k covered fraction: {coverage_dk:.4f}")

    # Clip plot x-axis to the lower 95% of distances so rare outliers do not blow up the scale.
    d1_xmax = np.quantile(np.concatenate([a_to_a_d1, b_to_a_d1]), 0.99)
    dk_xmax = np.quantile(np.concatenate([a_to_a_dk_mean, b_to_a_dk_mean]), 0.99)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].hist(
        a_to_a_d1,
        bins=80,
        range=(0, d1_xmax),
        alpha=0.6,
        density=True,
        label=f"{labels[0]} -> {labels[0]} leave-one-out",
    )
    axes[0].hist(
        b_to_a_d1,
        bins=80,
        range=(0, d1_xmax),
        alpha=0.6,
        density=True,
        label=f"{labels[1]} -> {labels[0]}",
    )
    axes[0].axvline(threshold_d1, linestyle="--", linewidth=1)
    axes[0].set_xlim(0, d1_xmax)
    axes[0].set_title("Nearest-neighbor distance")
    axes[0].set_xlabel("distance")
    axes[0].set_ylabel("density")
    axes[0].legend()

    axes[1].hist(
        a_to_a_dk_mean,
        bins=80,
        range=(0, dk_xmax),
        alpha=0.6,
        density=True,
        label=f"{labels[0]} -> {labels[0]} leave-one-out",
    )
    axes[1].hist(
        b_to_a_dk_mean,
        bins=80,
        range=(0, dk_xmax),
        alpha=0.6,
        density=True,
        label=f"{labels[1]} -> {labels[0]}",
    )
    axes[1].axvline(threshold_dk, linestyle="--", linewidth=1)
    axes[1].set_xlim(0, dk_xmax)
    axes[1].set_title(f"Mean distance to {k} nearest neighbors")
    axes[1].set_xlabel("mean distance")
    axes[1].set_ylabel("density")
    axes[1].legend()

    nn_output_path = output_path.with_name(f"{output_path.stem}_nn_coverage{output_path.suffix}")
    fig.savefig(nn_output_path, dpi=dpi)
    plt.close(fig)

    print(f"Saved nearest-neighbor coverage plot to {nn_output_path}")


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


def plot_series(
    dataset_a: Path,
    dataset_b: Path,
    spec: SeriesSpec,
    *,
    labels: tuple[str, str],
    output_path: Path,
    max_samples: int,
    bins: int,
    perplexity: float,
    seed: int,
    dpi: int,
) -> None:
    """Sample, embed, and plot one series from both datasets."""
    rng = np.random.default_rng(seed)
    rows_a, total_a, valid_a = sample_valid_rows(
        dataset_a,
        spec,
        max_samples=max_samples,
        rng=rng,
    )
    rows_b, total_b, valid_b = sample_valid_rows(
        dataset_b,
        spec,
        max_samples=max_samples,
        rng=rng,
    )

    scaled_a, scaled_b = standardize_combined(rows_a, rows_b)
    nearest_neighbor_coverage_diagnostic(
        scaled_a,
        scaled_b,
        labels=labels,
        output_path=output_path,
        k=10,
        dpi=dpi,
    )
    combined = np.concatenate([scaled_a, scaled_b], axis=0)
    if combined.shape[0] <= 1:
        raise ValueError(f"Need at least two sampled valid {spec.name} rows to run t-SNE.")

    effective_perplexity = min(perplexity, max(1.0, (combined.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    embedding = tsne.fit_transform(combined)
    embedding_a = embedding[: scaled_a.shape[0]]
    embedding_b = embedding[scaled_a.shape[0] :]

    make_hist_plot(
        embedding_a,
        embedding_b,
        labels=labels,
        output_path=output_path,
        bins=bins,
        dpi=dpi,
    )

    print(f"{labels[0]}: kept {valid_a:,}/{total_a:,} valid {spec.name} rows; t-SNE sample {rows_a.shape[0]:,}")
    print(f"{labels[1]}: kept {valid_b:,}/{total_b:,} valid {spec.name} rows; t-SNE sample {rows_b.shape[0]:,}")
    print(f"Saved {spec.name} plot to {output_path}")


def build_series_specs(args: argparse.Namespace) -> list[SeriesSpec]:
    """Build action and observation series from CLI options."""
    obs_abs_threshold = args.observation_abs_threshold
    if obs_abs_threshold < 0:
        obs_abs_threshold = None

    specs = [
        SeriesSpec(
            name="actions_expert",
            key_path=("actions_expert",),
            output_suffix="",
            expected_dim=7,
            magnitude_threshold=args.magnitude_threshold,
        )
    ]
    for obs_key in args.obs_keys:
        safe_key = obs_key.replace("/", "_").replace(".", "_")
        specs.append(
            SeriesSpec(
                name=f"obs.{obs_key}",
                key_path=("obs", obs_key),
                output_suffix=f"obs_{safe_key}",
                abs_threshold=obs_abs_threshold,
            )
        )
    return specs


def main() -> None:
    args = parse_args()
    labels = tuple(args.labels)

    for spec in build_series_specs(args):
        plot_series(
            args.dataset_a,
            args.dataset_b,
            spec,
            labels=labels,
            output_path=output_path_for_series(args.output, spec),
            max_samples=args.max_samples_per_dataset,
            bins=args.bins,
            perplexity=args.perplexity,
            seed=args.seed,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
