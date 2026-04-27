import os
import re
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
REFERENCE_FILE = [
    "experiments/apr25/residual_a2r2o0015n20_n20_actualkl/finetune-id_expert_noisei0_filtering/viz/success_rate_over_checkpoints.txt",
    "experiments/apr25/residual_a2r2o0015n20_n20_actualkl/finetune-id_expert_noisei1_filtering/viz/success_rate_over_checkpoints.txt",
    "experiments/apr25/residual_a2r2o0015n20_n20_actualkl/finetune-id_expert_noisei2/viz/success_rate_over_checkpoints.txt",
    "experiments/apr25/residual_a2r2o0015n20_n20_actualkl/finetune-id_expert_noisei3/viz/success_rate_over_checkpoints.txt",
    "experiments/apr25/residual_a2r2o0015n20_n20_actualkl/finetune-id_expert_noisei4/viz/success_rate_over_checkpoints.txt",
]

NUM_SEEDS = 3

PLOT_MEAN_STD = False  # True → mean+std, False → individual curves
OUTPUT_NAME = "success_rate_across_seeds.png" if PLOT_MEAN_STD else "success_rate_across_seeds_individual.png"
DATA_OUTPUT_NAME = "success_rate_across_seeds.txt"


def load_success_rates(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            step, rate = line.strip().split()
            step = int(step)
            data[step - step % 10] = float(rate)
    return data


def get_seed_paths(reference_file, num_seeds):
    if re.search(r"seed\d+", reference_file) is None:
        raise ValueError(f"Could not find 'seedN' pattern in: {reference_file}")
    return [
        re.sub(r"seed\d+", f"seed{i}", reference_file)
        for i in range(1, num_seeds + 1)
    ]


def resolve_paths():
    if isinstance(REFERENCE_FILE, list):
        return REFERENCE_FILE
    else:
        return get_seed_paths(REFERENCE_FILE, NUM_SEEDS)


def main():
    seed_files = resolve_paths()
    runs = []

    for path in seed_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        runs.append(load_success_rates(path))

    all_steps = sorted(set().union(*[run.keys() for run in runs]))
    steps = np.array(all_steps)

    # Calculate statistics for both plotting and file saving
    means, stds = [], []
    for step in all_steps:
        vals = [run[step] for run in runs if step in run]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    means, stds = np.array(means), np.array(stds)

    plt.figure(figsize=(8, 5))

    if PLOT_MEAN_STD:
        plt.plot(steps, means, marker="o", linewidth=2, label="Mean")
        plt.fill_between(steps, means - stds, means + stds, alpha=0.2, label="Std")
    else:
        for i, run in enumerate(runs):
            run_steps = sorted(run.keys())
            run_vals = [run[s] for s in run_steps]
            plt.plot(run_steps, run_vals, marker="o", linewidth=1.5, label=f"seed {i+1}")

    plt.xlabel("Checkpoint")
    plt.ylabel("Success Rate")

    title_n = len(seed_files)
    mode = "Mean ± Std" if PLOT_MEAN_STD else "Individual Runs"
    plt.title(f"Success Rate ({mode}, n={title_n})")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    for path in seed_files:
        dir_path = os.path.dirname(path)
        
        # Save Plot
        out_path = os.path.join(dir_path, OUTPUT_NAME)
        plt.savefig(out_path, dpi=200)
        
        # Save Data File
        txt_path = os.path.join(dir_path, DATA_OUTPUT_NAME)
        with open(txt_path, "w") as f:
            for s, m, std in zip(steps, means, stds):
                f.write(f"{s} {m} {std}\n")
                
        print(f"Saved: {out_path} and {txt_path}")

    plt.close()


if __name__ == "__main__":
    main()