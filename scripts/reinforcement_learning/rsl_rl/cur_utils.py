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



def apply_obs_noise(obs, obsnoise): # policy2 only
    obs = obs.clone()
    receptive_asset_pose = obs['policy2'][:,37:43]
    receptive_asset_pose[:, :2] += obsnoise
    insertive_asset_in_receptive_asset_frame = predict_relative_pose(obs['policy2'][:,31:37], receptive_asset_pose)
    obs['policy2'][:, 37:43] = receptive_asset_pose
    obs['policy2'][:, :6] = insertive_asset_in_receptive_asset_frame
    return obs
