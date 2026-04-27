import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def extract_success_rate(eval_dir: Path, subdir: str) -> float:
    """
    Read success rate from:
      N-ckpt-eval_viz/<subdir>/final_success_rate.txt

    Expected format example:
      /path/to/402-ckpt.pt
      None
      Eval mode: default
      [1156, 0, 846]
      0.4225774109363556

    We take the 5th non-empty line if present; otherwise fall back to the
    last non-empty line.
    """
    rate_file = eval_dir / subdir / "final_success_rate.txt"
    if not rate_file.is_file():
        return None

    lines = [line.strip() for line in rate_file.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty file: {rate_file}")

    if len(lines) >= 5:
        candidate = lines[4]
    else:
        candidate = lines[-1]

    try:
        return float(candidate)
    except ValueError as e:
        raise ValueError(f"Could not parse success rate from {rate_file}: {candidate}") from e


def find_checkpoint_success_rates(root: Path, subdir: str) -> list[tuple[int, float]]:
    """
    For each checkpoint file N-ckpt.pt in `root`, read success rate from
    N-ckpt-eval_viz/<subdir>/final_success_rate.txt.
    """
    results = []

    for ckpt_path in root.glob("*-ckpt-eval_viz"):
        stem = ckpt_path.name
        prefix = stem[:-len("-ckpt-eval_viz")]

        try:
            ckpt_num = int(prefix)
        except ValueError:
            continue

        eval_dir = root / f"{prefix}-ckpt-eval_viz"
        success_rate = extract_success_rate(eval_dir, subdir)
        if success_rate is None:
            continue
        results.append((ckpt_num, success_rate))

    results.sort(key=lambda x: x[0])
    return results


def write_txt(results: list[tuple[int, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ckpt, rate in results:
            f.write(f"{ckpt} {rate}\n")


def write_plot(results: list[tuple[int, float]], out_path: Path) -> None:
    if not results:
        raise ValueError("No checkpoint success rates found to plot.")

    checkpoints = [x[0] for x in results]
    success_rates = [x[1] for x in results]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(checkpoints, success_rates, marker="o")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Independent Success Rate")
    fig.savefig(str(out_path))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing N-ckpt.pt and N-ckpt-eval_viz/",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default="xleq035",
        help="Subdirectory inside N-ckpt-eval_viz containing final_success_rate.txt",
    )
    args = parser.parse_args()

    root = Path(args.directory).expanduser().resolve()
    assert root.is_dir(), f"Not a directory: {root}"

    results = find_checkpoint_success_rates(root, args.subdir)

    viz_dir = root / "viz"
    txt_path = viz_dir / "success_rate_over_checkpoints.txt"
    png_path = viz_dir / "success_rate_over_checkpoints.png"

    write_txt(results, txt_path)
    write_plot(results, png_path)

    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
