#!/usr/bin/env python3
"""Plot one x/y column pair from one or more loss table files."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot selected numeric columns from one or more whitespace- or delimiter-separated "
            "loss table files, such as test_trajectory_loss.txt."
        )
    )
    parser.add_argument(
        "loss_files",
        nargs="+",
        type=Path,
        help="Loss table files to plot.",
    )
    parser.add_argument(
        "--x-key",
        default="trajectory_in_noise",
        help="Column name to use for the x-axis.",
    )
    parser.add_argument(
        "--y-key",
        default="mean_loss_across_noise_indices",
        help="Column name to use for the y-axis.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help=(
            "Optional labels for the plotted files. Must provide one label per loss file. "
            "Defaults to the parent directory name for test_trajectory_loss.txt files."
        ),
    )
    parser.add_argument(
        "--difference",
        action="store_true",
        help=(
            "With exactly two input files, plot the first file's curve minus the second file's curve. "
            "Rows are matched by the selected x key."
        ),
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help=(
            "Column delimiter. Defaults to any whitespace. Use '\\t' for tab-delimited files "
            "or ',' for CSV-style files."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("trajectory_loss_plot.png"),
        help="Output plot path.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title. Defaults to '<y-key> vs <x-key>'.",
    )
    parser.add_argument(
        "--sort-x",
        action="store_true",
        help="Sort each curve by x value before plotting.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic y-axis.",
    )
    parser.add_argument(
        "--marker",
        default="o",
        help="Matplotlib marker for each curve. Use 'None' for no marker.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image DPI.",
    )
    return parser.parse_args()


def normalize_delimiter(delimiter: str | None) -> str | None:
    if delimiter is None:
        return None
    if delimiter == r"\t":
        return "\t"
    return delimiter


def split_table_line(line: str, delimiter: str | None) -> list[str]:
    if delimiter is None:
        return line.split()
    return [part.strip() for part in line.rstrip("\n").split(delimiter)]


def read_columns(path: Path, *, delimiter: str | None) -> dict[str, list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")

    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty table: {path}")

    header = split_table_line(lines[0], delimiter)
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate column names in {path}: {header}")

    columns: dict[str, list[float]] = {name: [] for name in header}
    for line_number, line in enumerate(lines[1:], start=2):
        values = split_table_line(line, delimiter)
        if len(values) != len(header):
            raise ValueError(
                f"Expected {len(header)} columns in {path}:{line_number}, got {len(values)}: {line!r}"
            )

        for name, value in zip(header, values):
            try:
                columns[name].append(float(value))
            except ValueError as e:
                raise ValueError(f"Could not parse numeric value for '{name}' in {path}:{line_number}: {value}") from e

    return columns


def require_column(columns: dict[str, list[float]], key: str, path: Path) -> list[float]:
    if key not in columns:
        available = ", ".join(columns)
        raise KeyError(f"{path} does not contain column '{key}'. Available columns: {available}")
    return columns[key]


def default_label(path: Path) -> str:
    if path.name == "test_trajectory_loss.txt" and path.parent.name:
        return path.parent.name
    return path.stem


def validate_labels(labels: list[str] | None, num_files: int) -> list[str] | None:
    if labels is None:
        return None
    if len(labels) != num_files:
        raise ValueError(f"--labels needs {num_files} labels, got {len(labels)}.")
    return labels


def make_xy_map(x_values: list[float], y_values: list[float], path: Path) -> dict[float, float]:
    xy_map: dict[float, float] = {}
    for x_value, y_value in zip(x_values, y_values):
        if x_value in xy_map:
            raise ValueError(f"Duplicate x value {x_value} in {path}; cannot compute a unique difference curve.")
        xy_map[x_value] = y_value
    return xy_map


def difference_curve(
    x_values_a: list[float],
    y_values_a: list[float],
    path_a: Path,
    x_values_b: list[float],
    y_values_b: list[float],
    path_b: Path,
) -> tuple[list[float], list[float]]:
    xy_a = make_xy_map(x_values_a, y_values_a, path_a)
    xy_b = make_xy_map(x_values_b, y_values_b, path_b)

    missing_from_b = [x_value for x_value in xy_a if x_value not in xy_b]
    missing_from_a = [x_value for x_value in xy_b if x_value not in xy_a]
    if missing_from_b or missing_from_a:
        details = []
        if missing_from_b:
            details.append(f"{path_b} is missing x values from {path_a}: {missing_from_b[:10]}")
        if missing_from_a:
            details.append(f"{path_a} is missing x values from {path_b}: {missing_from_a[:10]}")
        raise ValueError("; ".join(details))

    return x_values_a, [xy_a[x_value] - xy_b[x_value] for x_value in x_values_a]


def main() -> None:
    args = parse_args()
    if args.difference and len(args.loss_files) != 2:
        raise ValueError("--difference requires exactly two input files.")

    delimiter = normalize_delimiter(args.delimiter)
    labels = validate_labels(args.labels, len(args.loss_files))
    marker = None if args.marker.lower() == "none" else args.marker

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    xy_values: list[tuple[Path, list[float], list[float]]] = []
    plot_labels: list[str] = []
    for index, loss_file in enumerate(args.loss_files):
        columns = read_columns(loss_file, delimiter=delimiter)
        xy_values.append(
            (
                loss_file,
                require_column(columns, args.x_key, loss_file),
                require_column(columns, args.y_key, loss_file),
            )
        )
        plot_labels.append(labels[index] if labels is not None else default_label(loss_file))

    if args.difference:
        path_a, x_values_a, y_values_a = xy_values[0]
        path_b, x_values_b, y_values_b = xy_values[1]
        x_values, y_values = difference_curve(x_values_a, y_values_a, path_a, x_values_b, y_values_b, path_b)

        if args.sort_x:
            sorted_points = sorted(zip(x_values, y_values), key=lambda point: point[0])
            x_values = [point[0] for point in sorted_points]
            y_values = [point[1] for point in sorted_points]

        label = f"{plot_labels[0]} - {plot_labels[1]}"
        cutoff = len(x_values) // 10 * 10
        ax.plot(x_values[:cutoff], y_values[:cutoff], marker=marker, linewidth=1.8, markersize=3, label=label)
    else:
        for index, (_loss_file, x_values, y_values) in enumerate(xy_values):
            if args.sort_x:
                sorted_points = sorted(zip(x_values, y_values), key=lambda point: point[0])
                x_values = [point[0] for point in sorted_points]
                y_values = [point[1] for point in sorted_points]

            label = plot_labels[index]
            cutoff = len(x_values) // 10 * 10
            ax.plot(x_values[:cutoff], y_values[:cutoff], marker=marker, linewidth=1.8, markersize=3, label=label)

    ax.set_xlabel(args.x_key)
    ax.set_ylabel(f"{args.y_key} difference" if args.difference else args.y_key)
    if args.title is not None:
        ax.set_title(args.title)
    elif args.difference:
        ax.set_title(f"{plot_labels[0]} - {plot_labels[1]}: {args.y_key} vs {args.x_key}")
    else:
        ax.set_title(f"{args.y_key} vs {args.x_key}")
    if args.log_y:
        ax.set_yscale("log")
    if len(args.loss_files) > 1 or args.difference:
        ax.legend()
    ax.grid(True, alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
