import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def format_array_3dec(x: np.ndarray) -> str:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating):
        # Format float arrays to 3 decimal places
        return np.array2string(
            x,
            precision=3,
            floatmode='fixed',
            suppress_small=False
        )
    return str(x)


def format_array_exact(name, x, dtype):
    body = np.array2string(
        np.asarray(x),
        separator=", ",
        max_line_width=160,
        formatter={"float_kind": lambda value: repr(float(value))},
    )
    return f"{name} = np.array({body}, dtype={dtype})"


def print_and_save_noise_values(trajs, filename: Path):
    act_noise = np.asarray([traj["act_noise"] for traj in trajs], dtype=np.float32)
    obs_noise = np.asarray([traj["obs_noise"] for traj in trajs], dtype=np.float32)
    noise_index = np.asarray([traj["noise_index"] for traj in trajs], dtype=np.int64)
    scenario_ids = np.unique(noise_index)

    act_noise = np.stack([act_noise[noise_index == i][0] for i in scenario_ids])
    obs_noise = np.stack([obs_noise[noise_index == i][0] for i in scenario_ids])

    lines = [
        format_array_exact("noise_index", scenario_ids, "np.int64"),
        format_array_exact("act_noise", act_noise, "np.float32"),
        format_array_exact("obs_noise", obs_noise, "np.float32"),
    ]
    print("\n".join(lines))

    output_path = filename.parent / "noise_values.txt"
    with open(output_path, "w") as fo:
        fo.write("\n\n".join(lines) + "\n")
    print(f"Saved noise values to {output_path}")


def save_point_distribution_image(x, out_path="dist.png", bins=400, dpi=200, fixed_bounds=False):
    """
    x: torch.Tensor, shape (N, 2) on CPU or GPU
    Saves a 2D histogram (density map) visualization to out_path.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.is_cuda:
            x = x.cpu()
        x = x.float().numpy()

    fig, ax = plt.subplots()
    if not fixed_bounds:
        ax.hist2d(x[:, 0], x[:, 1], bins=bins)
    else:
        ax.hist2d(x[:, 0], x[:, 1], bins=bins, range=[[0.25, 0.6], [-0.15, 0.55]])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("2D point distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved to: {out_path}")


def save_histogram(x, filename, bins=100):
    # convert to numpy
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    x = x.reshape(-1)

    plt.figure()
    plt.hist(x, bins=bins)
    plt.tight_layout()
    plt.savefig(filename)
    print(filename)
    plt.close()


def plot_actions_tsne(actions, n_components=2, filename="tsne_plot.png"):
    """
    Reduces 7D actions to 2D or 3D and saves a visualization.
    
    Args:
        actions (np.ndarray): Array of shape (N, 7).
        n_components (int): Dimensionality of the embedding (2 or 3).
        filename (str): The filename for the output image.
    """
    # t-SNE can be computationally expensive (O(N log N)). 
    # If your dataset is huge (e.g., >20k points), consider sub-sampling.
    if len(actions) > 10000:
        indices = np.random.choice(len(actions), 10000, replace=False)
        actions_to_fit = actions[indices]
    else:
        actions_to_fit = actions

    # Initialize t-SNE
    # Use init='pca' and learning_rate='auto' for better convergence
    tsne = TSNE(
        n_components=n_components, 
        init='pca', 
        learning_rate='auto', 
        random_state=42
    )
    actions_reduced = tsne.fit_transform(actions_to_fit)
    
    plt.clf() # Ensure a clean canvas
    
    if n_components == 2:
        plt.scatter(
            actions_reduced[:, 0], 
            actions_reduced[:, 1], 
            alpha=0.5, 
            s=1, 
            cmap='viridis'
        )
        plt.xlabel('$z_1$ (t-SNE)')
        plt.ylabel('$z_2$ (t-SNE)')
        plt.title(f'2D t-SNE Action Distribution (N={len(actions_to_fit)})')
        
    elif n_components == 3:
        ax = plt.subplot(111, projection='3d')
        # We use the 3rd component as a color map to help with depth perception
        scatter = ax.scatter(
            actions_reduced[:, 0], 
            actions_reduced[:, 1], 
            actions_reduced[:, 2], 
            c=actions_reduced[:, 2],
            alpha=0.5, 
            s=1, 
            cmap='viridis'
        )
        ax.set_xlabel('$z_1$')
        ax.set_ylabel('$z_2$')
        ax.set_zlabel('$z_3$')
        plt.title('3D t-SNE Action Distribution')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300) # High DPI for publication quality
    print(filename)


def main():
    FILENAME = "collected_data/data_apr25_a2r2o0015n20_1/trajectories.pkl"
    FILENAME = Path(FILENAME)
    VIZ_DIR = FILENAME.parent / "viz"
    VIZ_DIR.mkdir(exist_ok=True, parents=True)
    with open(FILENAME, "rb") as fi:
        trajs = pickle.load(fi)
    print_and_save_noise_values(trajs, FILENAME)
    lengths = [traj['actions'].shape[0] for traj in trajs]
    save_histogram(lengths, VIZ_DIR / "lengths.png", bins=40)
    # receptive_starting_positions = np.stack([traj['starting_position']['receptive_position'][:2] for traj in trajs], axis=0)
    # save_point_distribution_image(receptive_starting_positions, VIZ_DIR / "receptive_starting_positions.png", fixed_bounds=True)
    # insertive_starting_positions = np.stack([traj['starting_position']['insertive_position'][:2] for traj in trajs], axis=0)
    # save_point_distribution_image(insertive_starting_positions, VIZ_DIR / "insertive_starting_positions.png", fixed_bounds=True)
    rewards = np.concatenate([traj['rewards'] for traj in trajs], axis=0)
    rewards = np.maximum(rewards, np.quantile(rewards, 0.01))
    save_histogram(rewards, VIZ_DIR / "rewards.png", bins=100)
    actions = np.concatenate([traj['actions'] for traj in trajs], axis=0)
    action_magnitudes = np.linalg.norm(actions, axis=1)
    save_histogram(action_magnitudes, VIZ_DIR / "action_magnitudes.png", bins=100)

    action_low = []
    action_high = []
    expert_actions = np.concatenate([traj['actions_expert'] for traj in trajs], axis=0)
    residual_actions = expert_actions - actions
    
    sys_noise = np.array([traj['act_noise'] for traj in trajs])
    obs_receptive_noise = np.array([traj['obs_noise'] for traj in trajs])

    for i in range(7):
        actions_1dim = actions[:, i]
        action_low.append(round(np.quantile(actions_1dim, 0.1), 2))
        action_high.append(round(np.quantile(actions_1dim, 0.9), 2))
        print(f"Action dim {i}: 10% = {np.quantile(actions_1dim, 0.1)}, 90% = {np.quantile(actions_1dim, 0.9)}")
        actions_1dim = np.clip(actions_1dim, np.quantile(actions_1dim, 0.002), np.quantile(actions_1dim, 0.998))
        save_histogram(actions_1dim, VIZ_DIR / f"action_dim_{i}.png", bins=100)

        sys_noise_1dim = sys_noise[:, i]
        save_histogram(sys_noise_1dim, VIZ_DIR / f"sys_noise_{i}.png", bins=100)

        residual_actions_1dim = residual_actions[:, i]
        save_histogram(residual_actions_1dim, VIZ_DIR / f"residual_action_{i}.png", bins=100)
    
    for i in range(2):
        obs_receptive_noise_1dim = obs_receptive_noise[:, i]
        save_histogram(obs_receptive_noise_1dim, VIZ_DIR / f"obs_receptive_noise_{i}.png", bins=100)
    
    print(f"action low = {action_low}")
    print(f"action high = {action_high}")
    print(f"action mean = {actions.mean(axis=0).tolist()}")
    print(f"action std = {actions.std(axis=0).tolist()}")
    success = np.array([np.any(traj['rewards'] > 0.11) for traj in trajs])
    success_rate = np.mean(success)
    print(f"Success rate = {success_rate}")
    pass

if __name__ == "__main__":
    main()
