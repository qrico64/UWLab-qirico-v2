import torch
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pickle

GENERAL_NOISE_SCALES = np.array([2.9608822, 4.3582673, 2.5497098, 8.63183, 8.950732, 2.6481836, 5.6350408], dtype=np.float32) / 5

def axis_angle_to_matrix(axis_angle):
    # axis_angle shape: (N, 3)
    angle = torch.norm(axis_angle, dim=-1, keepdim=True) # (N, 1)
    axis = axis_angle / (angle + 1e-8) # (N, 3)
    
    cos = torch.cos(angle).unsqueeze(-1) # (N, 1, 1)
    sin = torch.sin(angle).unsqueeze(-1) # (N, 1, 1)
    vers = 1 - cos
    
    x = axis[:, 0:1].unsqueeze(-1) # (N, 1, 1)
    y = axis[:, 1:2].unsqueeze(-1) # (N, 1, 1)
    z = axis[:, 2:3].unsqueeze(-1) # (N, 1, 1)
    
    # Building the rotation matrix row by row for the batch
    row1 = torch.cat([cos + x*x*vers, x*y*vers - z*sin, x*z*vers + y*sin], dim=-1)
    row2 = torch.cat([y*x*vers + z*sin, cos + y*y*vers, y*z*vers - x*sin], dim=-1)
    row3 = torch.cat([z*x*vers - y*sin, z*y*vers + x*sin, cos + z*z*vers], dim=-1)
    
    return torch.cat([row1, row2, row3], dim=-2) # (N, 3, 3)

def matrix_to_axis_angle(matrix):
    # matrix shape: (N, 3, 3)
    # Trace-based angle calculation
    dims = list(range(len(matrix.shape)))
    trace = matrix.diagonal(offset=0, dim1=dims[-2], dim2=dims[-1]).sum(-1)
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    angle = torch.acos(cos_theta).unsqueeze(-1) # (N, 1)
    
    # Extracting the axis from skew-symmetric part
    axis = torch.stack([
        matrix[:, 2, 1] - matrix[:, 1, 2],
        matrix[:, 0, 2] - matrix[:, 2, 0],
        matrix[:, 1, 0] - matrix[:, 0, 1]
    ], dim=-1)
    
    # Normalize and scale by angle
    return axis * (angle / (2 * torch.sin(angle) + 1e-8))

def predict_relative_pose(insertive_pose, receptive_pose):
    """
    Computes T_ins_in_rec = T_rec_in_gripper^-1 * T_ins_in_gripper
    Input shapes: (N, 6), (N, 6)
    Output shape: (N, 6)
    """
    # 1. Decomposition
    p_ins = insertive_pose[:, :3].unsqueeze(-1) # (N, 3, 1)
    p_rec = receptive_pose[:, :3].unsqueeze(-1) # (N, 3, 1)
    
    R_ins = axis_angle_to_matrix(insertive_pose[:, 3:]) # (N, 3, 3)
    R_rec = axis_angle_to_matrix(receptive_pose[:, 3:]) # (N, 3, 3)
    
    # 2. Relative Rotation: R_rel = R_rec.T @ R_ins
    # We use transpose on the last two dims to invert the rotation
    R_rec_T = R_rec.transpose(-1, -2)
    R_rel = torch.bmm(R_rec_T, R_ins) # (N, 3, 3)
    
    # 3. Relative Position: p_rel = R_rec.T @ (p_ins - p_rec)
    p_rel = torch.bmm(R_rec_T, (p_ins - p_rec)).squeeze(-1) # (N, 3)
    
    # 4. Convert Rotation back to Axis-Angle
    r_rel_aa = matrix_to_axis_angle(R_rel) # (N, 3)
    
    return torch.cat([p_rel, r_rel_aa], dim=-1) # (N, 6)



def apply_obs_noise_policy2(obs: torch.Tensor, obsnoise: torch.Tensor):
    obs = obs.clone()
    receptive_asset_pose = obs[:,37:43]
    receptive_asset_pose[:, :2] += obsnoise
    insertive_asset_pose = obs[:,31:37]
    insertive_asset_in_receptive_asset_frame = predict_relative_pose(insertive_asset_pose, receptive_asset_pose)
    obs[:, 37:43] = receptive_asset_pose
    obs[:, :6] = insertive_asset_in_receptive_asset_frame
    return obs

def apply_obs_noise_policy(obs: torch.Tensor, obsnoise: torch.Tensor):
    obs = obs.clone()
    receptive_asset_pose = obs[:,37*5:43*5].reshape(-1, 5, 6)
    receptive_asset_pose[:, :, :2] += obsnoise.reshape(-1, 1, 2)
    receptive_asset_pose = receptive_asset_pose.reshape(-1, 6)
    insertive_asset_pose = obs[:,31*5:37*5].reshape(-1, 6)
    insertive_asset_in_receptive_asset_frame = predict_relative_pose(insertive_asset_pose, receptive_asset_pose)
    obs[:, 37*5:43*5] = receptive_asset_pose.reshape(-1, 30)
    obs[:, :6*5] = insertive_asset_in_receptive_asset_frame.reshape(-1, 30)
    return obs

def save_info_dict(info: dict, filename: str | pathlib.Path):
    filepath = pathlib.Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as fi:
        pickle.dump(info, fi)
    with open(filepath.with_suffix(".txt"), "w") as fi:
        for k, v in info.items():
            fi.write(f"{k}: {v}\n")
    print(f"Saved info to {filepath}")

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


def plot_success_grid(coords, success, bins=10, save_path=None, fixed_bounds=False):
    coords = np.asarray(coords)
    success = np.asarray(success).astype(float)

    x, y = coords[:, 0], coords[:, 1]

    xrange = (x.min(), x.max())
    yrange = (y.min(), y.max())
    if fixed_bounds:
        xrange = (-0.15, 0.6)
        yrange = (-0.15, 0.6)
        bins = (15, 15)

    counts, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[xrange, yrange]
    )
    sums, _, _ = np.histogram2d(
        x, y, bins=bins, range=[xrange, yrange], weights=success
    )

    rate = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

    fig, ax = plt.subplots()

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=1)

    im = ax.imshow(
        rate.T,
        origin="lower",
        extent=[*xrange, *yrange],
        aspect="auto",
        vmin=0, vmax=1,
        cmap=cmap,
    )
    fig.colorbar(im, ax=ax, label="Success rate")

    # --- overlay per-cell text (skip empty / NaN cells) ---
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    # rate has shape (nx, ny) corresponding to x-bins then y-bins
    nx, ny = rate.shape
    for i in range(nx):
        for j in range(ny):
            val = rate[i, j]
            if not np.isfinite(val):  # empty cell -> don't annotate
                continue

            # Determine text color based on background luminance
            r, g, b, _ = cmap(norm(val))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            txt_color = "black" if luminance > 0.6 else "white"

            ax.text(
                xcenters[i],
                ycenters[j],
                f"{val:.2f}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=7,
            )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Grid Success Rate / {len(success)}")

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(save_path)

    plt.close(fig)
