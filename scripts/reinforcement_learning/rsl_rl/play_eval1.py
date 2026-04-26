# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--horizon", type=int, default=60, help="Horizon, max steps, duration, whatever you call it.")
parser.add_argument("--correction_model", type=str, default=None, help="Residual model .pt file.")
parser.add_argument("--plot_residual", action="store_true", default=False, help="Open second screen & plot residual.")
parser.add_argument("--video_path", type=str, default=None, help="Save location for videos.")
parser.add_argument("--num_evals", type=int, default=2000, help="Number of trajectories we eval for.")
parser.add_argument("--base_policy", type=str, default=None, help="Base model .pt file.")
parser.add_argument("--reset_mode", type=str, default='none', help="Options: none, xleq035.")
parser.add_argument("--no_viz", action="store_true", default=False, help="Whether to disable all visualization (including videos).")
parser.add_argument("--eval_mode", type=str, default='default', help="Options: default, sysnoise3, obsnoise001.")
parser.add_argument("--our_task", choices=["drawer", "leg", "peg"], default=None)
parser.add_argument("--sim_device", type=str, default="cuda:0", help="Device to run the simulation on.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# our task specific arguments
if args_cli.our_task is not None:
    insertive, receptive, checkpoint = {
        "drawer": ("fbdrawerbottom", "fbdrawerbox", "expert_policies/fbdrawerbottom_state_rl_expert.pt"),
        "leg": ("fbleg", "fbtabletop", "expert_policies/fbleg_state_rl_expert.pt"),
        "peg": ("peg", "peghole", "expert_policies/peg_state_rl_expert_seed42.pt"),
    }[args_cli.our_task]
    args_cli.checkpoint = checkpoint
    hydra_args += [
        f"env.scene.insertive_object={insertive}",
        f"env.scene.receptive_object={receptive}",
    ]

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import pathlib
import matplotlib.pyplot as plt
import re

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
    ManagerBasedEnv,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
import isaaclab.sim as sim_utils

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config
from train_lib import RobotTransformerPolicy, load_robot_policy
import cur_utils

# PLACEHOLDER: Extension template (do not remove this comment)


def save_video(frames, path, fps=30):
    """Saves a list of frames (numpy arrays) to a video file."""
    print(f"[INFO] Saving video to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"[INFO] Video saved successfully.")

def get_positions(env: ManagerBasedEnv):
    asset = env.scene["receptive_object"]
    positions = asset.data.root_pos_w
    orientations = asset.data.root_quat_w
    asset2 = env.scene["insertive_object"]
    positions2 = asset2.data.root_pos_w
    orientations2 = asset2.data.root_quat_w
    return torch.cat([positions, orientations, positions2, orientations2], dim=-1)

def set_positions_completely(env: ManagerBasedEnv, position: torch.Tensor, env_id: int):
    asset = env.scene["receptive_object"]
    asset.write_root_pose_to_sim(position[:7].unsqueeze(0).clone(), env_ids=torch.tensor([env_id], device=env.device))
    asset2 = env.scene["insertive_object"]
    asset2.write_root_pose_to_sim(position[7:].unsqueeze(0).clone(), env_ids=torch.tensor([env_id], device=env.device))

def get_starting_position(env: ManagerBasedEnv):
    origins = env.unwrapped.scene.env_origins
    receptive_position = env.unwrapped.scene["receptive_object"].data.root_pos_w - origins
    insertive_position = env.unwrapped.scene["insertive_object"].data.root_pos_w - origins
    return torch.concatenate([receptive_position[:, :2], insertive_position[:, :2]], dim=1)

def render_frame(frame: np.ndarray, caption: str, display_action=None, display_action2=None):
    captions = caption.splitlines()
    IMAGE_SIZE = frame.shape[:2]
    BOUNDARY_X = 8
    BOUNDARY_Y = 8

    if display_action is not None:
        # DRAW SECOND SCREEN #
        SECOND_SCREEN_TOP_MARGIN = 40
        SECOND_SCREEN_SIZE = (IMAGE_SIZE[0] - SECOND_SCREEN_TOP_MARGIN, IMAGE_SIZE[1])
        second_screen = np.zeros((*SECOND_SCREEN_SIZE, 3), dtype=np.uint8)
        h, w = SECOND_SCREEN_SIZE
        center = (w // 2, h // 2)

        # Draw Axes (White = 255)
        cv2.line(second_screen, (0, center[1]), (w, center[1]), (255, 255, 255), 1) # X-axis
        cv2.line(second_screen, (center[0], 0), (center[0], h), (255, 255, 255), 1) # Y-axis

        # Map the -5 to +5 range to pixel coordinates
        # Scale factor: pixels per unit
        scale_x = w / (BOUNDARY_X + BOUNDARY_X)
        scale_y = h / (BOUNDARY_Y + BOUNDARY_Y)
        
        # Draw Ticks at multiples of 2
        tick_size = 5
        for i in range(-BOUNDARY_X, BOUNDARY_X + 1, 1):
            tx = int(center[0] + i * scale_x)
            cv2.line(second_screen, (tx, center[1] - tick_size), (tx, center[1] + tick_size), (255, 255, 255), 1)
        for i in range(-BOUNDARY_Y, BOUNDARY_Y + 1, 1):
            ty = int(center[1] - i * scale_y)
            cv2.line(second_screen, (center[0] - tick_size, ty), (center[0] + tick_size, ty), (255, 255, 255), 1)

        # Calculate pixel position (Note: Y is inverted in screen space)
        coord = display_action[:2]
        px = int(center[0] + coord[0] * scale_x)
        py = int(center[1] - coord[1] * scale_y)
        if abs(coord[0]) < BOUNDARY_X and abs(coord[1]) < BOUNDARY_Y:
            cv2.circle(second_screen, (px, py), 5, (0, 255, 0), -1)
        else:
            cv2.line(second_screen, center, (px, py), (0, 255, 0), 2)
        
        if display_action2 is not None:
            coord = display_action2[:2]
            px = int(center[0] + coord[0] * scale_x)
            py = int(center[1] - coord[1] * scale_y)
            if abs(coord[0]) < BOUNDARY_X and abs(coord[1]) < BOUNDARY_Y:
                cv2.circle(second_screen, (px, py), 5, (255, 0, 0), -1)
            else:
                cv2.line(second_screen, center, (px, py), (255, 0, 0), 2)
        # END DRAW SECOND SCREEN #

        top_margin = np.full((SECOND_SCREEN_TOP_MARGIN, IMAGE_SIZE[1], 3), 255, dtype=np.uint8)
        frame = np.concatenate([frame, np.concatenate([top_margin, second_screen], axis=0)], axis=1)

    cv2.putText(
        frame,
        captions[0],
        org=(5, 25),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 0, 0),  # BGR: black
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        ' '.join(captions[1:]),
        org=(5, 60),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 0, 0),  # BGR: black
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return frame


def evaluate_model_raw(
    env,
    expert_policy,
    policy_nn,
    base_policy_file,
    correction_model_file,
    num_evals,
    reset_mode='none',
    need_reset_envs=True,
    enable_cameras=False,
    IMAGE_SIZE=(400, 400),
    plot_residual=False,
    video_path=None,
    horizon=60,
    device='cuda',
    no_viz: bool = False,
    eval_mode: str = 'default',
):
    # Correction model, base policy -> Correction model on base policy;
    # Correction model only -> Correction model on noised PPO expert;
    # Base policy only -> Base policy;
    # None -> noised PPO expert;
    T_DIM = horizon
    N = env.num_envs
    S_DIM = env.observation_space['policy2'].shape[-1]
    A_DIM = env.action_space.shape[-1]

    # Rico: Instantiate base policy!!
    BASE_POLICY_FILE = pathlib.Path(base_policy_file) if base_policy_file is not None else None
    if BASE_POLICY_FILE is not None:
        base_policy_raw, base_policy_info = load_robot_policy(BASE_POLICY_FILE, device=device)
        print(f"Using obs_receptive_noise of {base_policy_info['obs_receptive_noise']}")
        print(f"Using obs_insertive_noise of {base_policy_info['obs_insertive_noise']}")
        print(f"Using sys_noise of {base_policy_info['sys_noise']}")
        assert base_policy_info['infer_mode'] == "expert" or base_policy_info['infer_mode'] == "expert_new"
        def base_policy(obs_input_dict_original):
            with torch.no_grad():
                obs_receptive_noise = base_policy_info['obs_receptive_noise']
                obs_insertive_noise = base_policy_info['obs_insertive_noise']
                sys_noise = base_policy_info['sys_noise']
                sys_noise = torch.tensor(sys_noise, dtype=torch.float32, device=device)
                obs_input_dict = obs_input_dict_original.clone()
                obs_input_dict['policy'] = cur_utils.apply_obs_noise_policy(obs_input_dict_original['policy'], obs_receptive_noise)
                obs_input_dict['policy2'] = cur_utils.apply_obs_noise_policy2(obs_input_dict_original['policy2'], obs_receptive_noise)
                base_actions_raw = base_policy_raw.get_action(
                    torch.zeros(len(obs_input_dict), T_DIM, base_policy_info['context_dim'], device=device),
                    torch.cat([obs_input_dict['policy'], torch.zeros(len(obs_input_dict), 7, device=device)], dim=-1),
                    torch.zeros(len(obs_input_dict), T_DIM, base_policy_info['label_dim'], device=device),
                    padding_mask=torch.zeros(len(obs_input_dict), T_DIM, dtype=torch.bool, device=device),
                )
                return base_actions_raw + sys_noise
    else:
        base_policy = expert_policy
    
    # Correction model stuff
    CORRECTION_MODEL_FILE = pathlib.Path(correction_model_file) if correction_model_file is not None else None
    if CORRECTION_MODEL_FILE is not None:
        print(f"Loading model at {CORRECTION_MODEL_FILE}")
        assert CORRECTION_MODEL_FILE.is_file()
        correction_model, correction_model_info = load_robot_policy(CORRECTION_MODEL_FILE, device=device)
        assert correction_model_info['infer_mode'] not in ["expert", "expert_new"]
        BASE_POLICY_ONLY = False
    else:
        correction_model = None
        correction_model_info = {}
        BASE_POLICY_ONLY = True

    # Eval mode stuff
    SYS_NOISE_SCALE = 0.0
    RAND_NOISE_SCALE = 0.0
    OBS_RECEPTIVE_NOISE_SCALE = 0.0
    OBS_INSERTIVE_NOISE_SCALE = 0.0
    if BASE_POLICY_FILE is not None:
        pass
    elif eval_mode == 'sysnoise4':
        SYS_NOISE_SCALE = 4.0
    elif eval_mode == 'obsnoise001':
        OBS_RECEPTIVE_NOISE_SCALE = 0.01
    elif eval_mode == 'o1s2':
        OBS_RECEPTIVE_NOISE_SCALE = 0.01
        SYS_NOISE_SCALE = 2.0
    elif eval_mode == 'default' and CORRECTION_MODEL_FILE is None:
        pass
    elif eval_mode == 'default' and CORRECTION_MODEL_FILE is not None:
        SYS_NOISE_SCALE = correction_model_info['sys_noise_scale']
        OBS_RECEPTIVE_NOISE_SCALE = correction_model_info['obs_receptive_noise_scale']
        OBS_INSERTIVE_NOISE_SCALE = correction_model_info['obs_insertive_noise_scale']
    elif re.match(r"o\d+s\d+r\d+", eval_mode):
        m = re.match(r"o(\d+)s(\d+)r(\d+)", eval_mode)
        OBS_RECEPTIVE_NOISE_SCALE = int(m.group(1)) / 1000.0
        SYS_NOISE_SCALE = float(m.group(2))
        RAND_NOISE_SCALE = float(m.group(3))
        print(f"Dynamic eval mode: obs={OBS_RECEPTIVE_NOISE_SCALE}, sys={SYS_NOISE_SCALE}, rand={RAND_NOISE_SCALE}")
    elif re.match(r"o\d+s\d+", eval_mode):
        m = re.match(r"o(\d+)s(\d+)", eval_mode)
        OBS_RECEPTIVE_NOISE_SCALE = int(m.group(1)) / 1000.0
        SYS_NOISE_SCALE = float(m.group(2))
        print(f"Dynamic eval mode: obs={OBS_RECEPTIVE_NOISE_SCALE}, sys={SYS_NOISE_SCALE}")
    else:
        raise NotImplementedError(f"Eval mode {eval_mode} not implemented.")

    # Viz directory
    if no_viz:
        VIZ_DIRECTORY = pathlib.Path("/tmp/qirico/eval_viz")
    elif CORRECTION_MODEL_FILE is not None:
        temp_viz_directory_end = reset_mode + ("-" + BASE_POLICY_FILE.parent.name if BASE_POLICY_FILE is not None else "")
        temp_viz_directory_end += ("-" + eval_mode) if eval_mode != 'default' else ""
        VIZ_DIRECTORY = CORRECTION_MODEL_FILE.parent / CORRECTION_MODEL_FILE.name.replace(".pt", "-eval_viz") / temp_viz_directory_end
    elif BASE_POLICY_FILE is not None:
        VIZ_DIRECTORY = BASE_POLICY_FILE.parent / BASE_POLICY_FILE.name.replace(".pt", "-eval_viz") / reset_mode
    else:
        VIZ_DIRECTORY = "viz/expert_peg/"
        assert VIZ_DIRECTORY != ""
        VIZ_DIRECTORY = pathlib.Path(VIZ_DIRECTORY)
    VIZ_DIRECTORY.mkdir(parents=True, exist_ok=True)

    N_DIM = 2
    completed_reset = torch.full((N,), not need_reset_envs, dtype=torch.bool, device=device)
    timesteps = torch.zeros(N, N_DIM, dtype=torch.int64, device=device)
    successes = torch.zeros(N, N_DIM, dtype=torch.bool, device=device)
    rec_observations = torch.zeros(N, N_DIM, T_DIM, S_DIM, dtype=torch.float32, device=device)
    rec_observations_policy = torch.zeros(N, N_DIM, T_DIM, 215, dtype=torch.float32, device=device)
    rec_actions = torch.zeros(N, N_DIM, T_DIM, A_DIM, dtype=torch.float32, device=device)
    rec_rewards = torch.zeros(N, N_DIM, T_DIM, dtype=torch.float32, device=device)
    rec_expert_actions = torch.zeros(N, N_DIM, T_DIM, A_DIM, dtype=torch.float32, device=device)
    curstates = torch.zeros(N, dtype=torch.int64, device=device)
    trajectory_start_position = get_starting_position(env)

    result_success_distribution = []
    first_traj_mse_sum = 0.0
    first_traj_mse_count = 0
    mse_list = []
    
    obs = env.get_observations().to(device)

    if CORRECTION_MODEL_FILE is not None and (correction_model_info['obs_receptive_noise_scale'] != 0 or correction_model_info['obs_insertive_noise_scale'] != 0):
        assert 'policy_aaaaaa' in obs.keys()
    if CORRECTION_MODEL_FILE is not None and correction_model_info['obs_insertive_noise_scale'] != 0:
        raise Exception("Right now doesn't support insertive noise!!")

    GENERAL_NOISE_SCALES = torch.tensor(cur_utils.GENERAL_NOISE_SCALES, device=device)

    sys_noises = torch.randn(N, A_DIM, device=device) * SYS_NOISE_SCALE * GENERAL_NOISE_SCALES
    print(f"Using systematic noise of {SYS_NOISE_SCALE}")
    obs_receptive_noise = torch.randn(N, 2, device=device) * OBS_RECEPTIVE_NOISE_SCALE
    print(f"Using obs_receptive_noise of {OBS_RECEPTIVE_NOISE_SCALE}")
    obs_insertive_noise = torch.cat([torch.randn(N, 2, device=device) * OBS_INSERTIVE_NOISE_SCALE, torch.zeros(N, 4, device=device)], dim=-1)
    print(f"Using obs_insertive_noise of {OBS_INSERTIVE_NOISE_SCALE}")

    ENABLE_CAMERAS = enable_cameras
    if ENABLE_CAMERAS:
        PLOT_RESIDUAL = plot_residual
        rec_video = np.zeros((N, 2, T_DIM, IMAGE_SIZE[0], IMAGE_SIZE[1] * 2 if PLOT_RESIDUAL else IMAGE_SIZE[1], 3), dtype=np.uint8)
        VIDEO_PATH = video_path or str(VIZ_DIRECTORY / "video" / "videos.mp4")
        os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
        videopath_generator = lambda x, y: VIDEO_PATH[:VIDEO_PATH.rfind('.')] + f"_{x}_{y}" + VIDEO_PATH[VIDEO_PATH.rfind('.'):]
        NUM_VIDEOS = 10
        VIDEO_FPS = 6
    
    starting_positions = get_positions(env.env.env)

    # reset environment

    global_timestep = 0
    count_success = torch.zeros(N_DIM + 1, dtype=torch.int64, device='cpu')
    while count_success.sum() < num_evals:
        global_timestep += 1
        with torch.inference_mode():
            expert_actions = expert_policy(obs)
            noised_obs = obs.clone()
            noised_obs['policy'] = cur_utils.apply_obs_noise_policy(obs['policy'], obs_receptive_noise)
            noised_obs['policy2'] = cur_utils.apply_obs_noise_policy2(obs['policy2'], obs_receptive_noise)

            base_actions_raw = base_policy(noised_obs)

            rand_noise = torch.randn(N, A_DIM, device=device) * RAND_NOISE_SCALE * GENERAL_NOISE_SCALES
            base_actions_raw += sys_noises + rand_noise
            base_actions = base_actions_raw.clone()

            need_residuals = curstates > 0
            need_residuals_count = need_residuals.sum()
            if need_residuals_count > 0:
                assert not BASE_POLICY_ONLY, "Shouldn't get here."
                contexts = torch.cat([rec_observations[need_residuals, 0, :, :], rec_actions[need_residuals, 0, :, :]], dim=2)
                currents = obs['policy2'][need_residuals].clone()
                padding_mask = torch.arange(T_DIM, device=device).repeat(need_residuals_count, 1) >= timesteps[need_residuals, 0].unsqueeze(1)
                cur_base_actions = base_actions[need_residuals].clone()
                temp_timesteps = timesteps[need_residuals, 0:1].to(dtype=torch.float32)
                currents_policy = obs['policy'][need_residuals].clone()
                currents = torch.cat([currents, cur_base_actions, temp_timesteps, currents_policy], dim=-1)
                
                if correction_model_info["force_mu_conditioning"] == "none":
                    mu_conditioning = None
                elif correction_model_info["force_mu_conditioning"] == "obsnoise":
                    mu_conditioning = obs_receptive_noise[need_residuals, :2]

                residual_actions = correction_model.get_action(contexts, currents, cur_base_actions, padding_mask, mu_conditioning=mu_conditioning)
                base_actions[need_residuals, :] = residual_actions
            
            # step
            next_obs, reward, dones, info = env.step(base_actions.to(env.unwrapped.device))
            next_obs, reward, dones = next_obs.to(device), reward.to(device), dones.to(device)

            # handle non-dones
            indices = torch.arange(N, device=device)
            cur_timesteps = timesteps[indices, curstates]
            rec_observations[indices, curstates, cur_timesteps, :] = obs['policy2']
            rec_observations_policy[indices, curstates, cur_timesteps, :] = obs['policy']
            rec_actions[indices, curstates, cur_timesteps, :] = base_actions
            rec_expert_actions[indices, curstates, cur_timesteps, :] = expert_actions
            rec_rewards[indices, curstates, cur_timesteps] = reward
            successes[indices, curstates] |= env.unwrapped.termination_manager.get_term("success")

            if ENABLE_CAMERAS:
                frames = obs['rgb'].cpu().detach().numpy().transpose(0, 2, 3, 1)
                if frames.max() <= 1.0 + 1e-4:
                    frames = (frames * 255).astype(np.uint8)
                
                for i in range(N):
                    assert frames[i].shape == (*IMAGE_SIZE, 3), f"{frames[i].shape} != {IMAGE_SIZE}"
                    if PLOT_RESIDUAL:
                        display_action = expert_actions[i] - base_actions_raw[i]
                        display_action2 = base_actions[i] - base_actions_raw[i]
                        display_action_str = '[' + ', '.join([f"{x:.3f}" for x in display_action.tolist()]) + ']'
                        display_action2_str = '[' + ', '.join([f"{x:.3f}" for x in display_action2.tolist()]) + ']'
                    else:
                        display_action = display_action2 = None
                        display_action_str = display_action2_str = "None"
                    caption = f"t={timesteps[i].tolist()} r={reward[i]:.5f} done={dones[i]}"
                    if PLOT_RESIDUAL:
                        caption += f" residual-action={display_action_str}"
                    caption += f"\npred-action={display_action2_str}"
                    final_screen = render_frame(frames[i], caption, display_action=display_action, display_action2=display_action2)
                    rec_video[i, curstates[i], timesteps[i, curstates[i]]] = final_screen
            
            timesteps[indices, curstates] += 1
            obs = next_obs

            # handle dones
            policy_nn.reset(dones)
            for i in torch.nonzero(dones).squeeze(-1):
                if not completed_reset[i]:
                    completed_reset[i] = True
                    rec_observations[i] *= 0
                    rec_observations_policy[i] *= 0
                    rec_actions[i] *= 0
                    rec_expert_actions[i] *= 0
                    rec_rewards[i] *= 0
                    timesteps[i] *= 0
                    successes[i] = False
                    curstates[i] *= 0
                    sys_noises[i] = torch.randn(A_DIM, device=device) * SYS_NOISE_SCALE * GENERAL_NOISE_SCALES
                    obs_receptive_noise[i] = torch.randn(2, device=device) * OBS_RECEPTIVE_NOISE_SCALE
                    obs_insertive_noise[i] = torch.cat([torch.randn(2, device=device) * OBS_INSERTIVE_NOISE_SCALE, torch.zeros(4, device=device)], dim=-1)
                    starting_positions[i] = get_positions(env.env.env)[i]
                    trajectory_start_position[i] = get_starting_position(env)[i]
                    continue
                
                if curstates[i] > 0 or successes[i, curstates[i]] or BASE_POLICY_ONLY:
                    curi = curstates[i].cpu() + 1 if successes[i, curstates[i]] else 0
                    # curi = 0 -> Both failed; curi = 1 -> First try success; curi = 2 -> Second try success;
                    count_success[curi] += 1
                    if not BASE_POLICY_ONLY and (curi == 2 or curi == 0):
                        result_success_distribution.append([trajectory_start_position[i].detach().cpu().numpy().copy(), curi == 2])
                    if BASE_POLICY_ONLY:
                        result_success_distribution.append([trajectory_start_position[i].detach().cpu().numpy().copy(), curi == 1])

                    if ENABLE_CAMERAS and count_success[curi] <= NUM_VIDEOS:
                        videopath = videopath_generator(curi, count_success[curi].item())
                        maxt = (rec_rewards[i, curstates[i]] < SUCCESS_THRESHOLD).sum()
                        maxt = min(maxt + 1, timesteps[i, curstates[i]])
                        if BASE_POLICY_ONLY:
                            # visualize expert trajectory
                            frames = rec_video[i, curstates[i], :maxt]
                        elif curi == 1:
                            # first try success
                            frames = rec_video[i, curstates[i], :maxt]
                        elif curi == 2:
                            # second try success
                            frames = np.concatenate([rec_video[i, 0, :timesteps[i, 0]], rec_video[i, 1, :maxt]], axis=0)
                        else:
                            frames = np.concatenate([rec_video[i, 0, :timesteps[i, 0]], rec_video[i, 1, :timesteps[i, 1]]], axis=0)
                        save_video(frames, videopath, fps=VIDEO_FPS)

                    if count_success.sum() % 20 == 0:
                        print(f"First try success rate: {count_success[1] / count_success.sum()}; Second try success rate: {count_success[2] / (count_success.sum() - count_success[1])}")
                        print(f"Current average MSE: {first_traj_mse_sum / (first_traj_mse_count + 1e-8)}")
                        print(f"{count_success.sum()} {count_success[1]} {count_success[2]}")
                    
                    if count_success.sum() % 100 == 0 and len(result_success_distribution) > 0:
                        print(f"Success distribution size {len(result_success_distribution)}")
                        success_dist_locations = np.stack([instance[0] for instance in result_success_distribution], axis=0)
                        success_dist_successes = np.array([instance[1] for instance in result_success_distribution])
                        cur_utils.plot_success_grid(success_dist_locations[:, :2], success_dist_successes, save_path = VIZ_DIRECTORY / "eval_success_by_receptive_location2.png", fixed_bounds=True)
                        cur_utils.plot_success_grid(success_dist_locations[:, 2:], success_dist_successes, save_path = VIZ_DIRECTORY / "eval_success_by_insertive_location2.png", fixed_bounds=True)
                        cur_utils.plot_success_grid(success_dist_locations[:, [0, 2]], success_dist_successes, save_path = VIZ_DIRECTORY / "eval_success_by_x.png", fixed_bounds=True)
                        cur_utils.plot_success_grid(success_dist_locations[:, [1, 3]], success_dist_successes, save_path = VIZ_DIRECTORY / "eval_success_by_y.png", fixed_bounds=True)
                        print()

                    rec_observations[i] *= 0
                    rec_observations_policy[i] *= 0
                    rec_actions[i] *= 0
                    rec_expert_actions[i] *= 0
                    rec_rewards[i] *= 0
                    timesteps[i] *= 0
                    successes[i] = False
                    curstates[i] *= 0
                    sys_noises[i] = torch.randn(A_DIM, device=device) * SYS_NOISE_SCALE * GENERAL_NOISE_SCALES
                    obs_receptive_noise[i] = torch.randn(2, device=device) * OBS_RECEPTIVE_NOISE_SCALE
                    obs_insertive_noise[i] = torch.cat([torch.randn(2, device=device) * OBS_INSERTIVE_NOISE_SCALE, torch.zeros(4, device=device)], dim=-1)
                    starting_positions[i] = get_positions(env.env.env)[i]
                    trajectory_start_position[i] = get_starting_position(env)[i]
                else:
                    T = timesteps[i, 0].item()
                    if T > 0:
                        context = torch.cat([rec_observations[i, 0], rec_actions[i, 0]], dim=1).repeat(T, 1, 1)
                        current = rec_observations[i, 0, :T]
                        current_policy = rec_observations_policy[i, 0, :T]
                        cur_base_actions = rec_actions[i, 0, :T]
                        temp_timesteps = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
                        current = torch.cat([current, cur_base_actions, temp_timesteps, current_policy], dim=-1)
                        padding_mask = torch.arange(T_DIM, device=device).repeat(T, 1) >= T
                
                        if correction_model_info["force_mu_conditioning"] == "none":
                            mu_conditioning = None
                        elif correction_model_info["force_mu_conditioning"] == "obsnoise":
                            mu_conditioning = obs_receptive_noise[i, :2].repeat(T, 1)

                        pred_actions = correction_model.get_action(context, current, cur_base_actions, mu_conditioning=mu_conditioning)
                        actual_expert_actions = rec_expert_actions[i, 0, :T]
                        residual_mse = torch.linalg.norm(pred_actions - actual_expert_actions, dim=-1).mean()
                        if (torch.linalg.norm(cur_base_actions, dim=-1) < 100).all():
                            first_traj_mse_sum += residual_mse.item()
                            first_traj_mse_count += 1
                        mse_list.append(residual_mse.item())

                    curstates[i] += 1
                    set_positions_completely(env.env.env, starting_positions[i], i)
        
        if ENABLE_CAMERAS and torch.all(count_success >= NUM_VIDEOS):
            break
    
    print(f"Correction model at {CORRECTION_MODEL_FILE}")
    print(f"Base policy at {BASE_POLICY_FILE}")
    print(f"Eval mode: {eval_mode}")
    if BASE_POLICY_ONLY:
        final_success_rate = (count_success[1] / count_success.sum()).detach().cpu().item()
    else:
        final_success_rate = (count_success[2] / (count_success.sum() - count_success[1])).detach().cpu().item()

    avg_first_traj_mse = first_traj_mse_sum / first_traj_mse_count if first_traj_mse_count > 0 else float("nan")
    with open(VIZ_DIRECTORY / "first_traj_mse.txt", 'w') as f:
        f.write(f"{mse_list}\n")
        f.write(f"threshold 100 on base_actions\n")
        f.write(f"average_mse {avg_first_traj_mse}\n")
        f.write(f"num_trajectories {first_traj_mse_count}\n")

    with open(VIZ_DIRECTORY / "final_success_rate.txt", 'w') as f:
        f.write(f"{CORRECTION_MODEL_FILE}\n")
        f.write(f"{BASE_POLICY_FILE}\n")
        f.write(f"Eval mode: {eval_mode}\n")
        f.write(f"{count_success.tolist()}\n")
        f.write(f"{final_success_rate}")
    return final_success_rate

def evaluate_model(*args, **kwargs):
    try:
        return evaluate_model_raw(*args, **kwargs)
    except Exception as e:
        raise





@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # make config compatible with installed rsl-rl version
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.sim_device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # set horizon based on the configured environment step time
    env_step_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.episode_length_s = args_cli.horizon * env_step_dt
    print(f"Horizon: {args_cli.horizon}, Dt: {env_step_dt}")

    # set camera & video
    IMAGE_SIZE = (800, 800)
    if args_cli.enable_cameras:
        assert IMAGE_SIZE[0] == IMAGE_SIZE[1]
        env_cfg.scene.side_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/rgb_side_camera",
            update_period=0,
            height=IMAGE_SIZE[0],
            width=IMAGE_SIZE[1],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.65, 0, 0.15),
                rot=(0.5, 0.5, 0.5, 0.5), # (w, x, y, z), -z direction.
                convention="opengl",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=21.9
            )
        )
        env_cfg.observations.rgb = env_cfg.observations.RGBCfg()
        env_cfg.observations.rgb.side_rgb.params['output_size'] = IMAGE_SIZE
        print(f"Video generation on at size/resolution {IMAGE_SIZE}")

    env_cfg.events.reset_from_reset_states.params['reset_mode'] = args_cli.reset_mode

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # Expert policy as in Omnireset.
    policy = runner.get_inference_policy(device=args_cli.device)
    policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
    
    base_policy_file = args_cli.base_policy
    base_policy_file = None if base_policy_file in (None, "", "none") else pathlib.Path(base_policy_file)
    correction_model_file = args_cli.correction_model
    correction_model_file = None if correction_model_file in (None, "", "none") else pathlib.Path(correction_model_file)
    
    if correction_model_file is not None and correction_model_file.is_dir():
        # Eval list of correction models
        checkpoints = list(correction_model_file.glob("*.pt"))
        checkpoints = [int(ckpt.name.replace("-ckpt.pt", "")) for ckpt in checkpoints if ckpt.name.endswith("-ckpt.pt")]
        checkpoints = sorted(checkpoints)
        checkpoints = [checkpoints[-1]] + checkpoints[:-1]
        correction_model_files = [correction_model_file / f"{ckpt}-ckpt.pt" for ckpt in checkpoints]
        success_rates = []
        for correction_model_path in correction_model_files:
            print(f"Evaluating correction model at {correction_model_path}...")
            success_rate = evaluate_model(
                env,
                policy,
                policy_nn,
                base_policy_file=base_policy_file,
                correction_model_file=correction_model_path,
                num_evals=args_cli.num_evals * 4 if correction_model_path == correction_model_files[-1] else args_cli.num_evals,
                reset_mode=args_cli.reset_mode,
                enable_cameras=args_cli.enable_cameras,
                IMAGE_SIZE=IMAGE_SIZE,
                plot_residual=args_cli.plot_residual,
                video_path=str(args_cli.video_path) if args_cli.video_path else None,
                horizon=args_cli.horizon,
                device=agent_cfg.device,
                need_reset_envs=True,
                no_viz=args_cli.no_viz,
                eval_mode=args_cli.eval_mode,
            )
            success_rates.append(success_rate)
        
        print(checkpoints)
        print(success_rates)
        (correction_model_file / "viz").mkdir(parents=True, exist_ok=True)
        with open(correction_model_file / "viz" / "success_rate_over_checkpoints.txt", 'w') as f:
            for ckpt, rate in zip(checkpoints, success_rates):
                f.write(f"{ckpt} {rate}\n")

        fig, ax = plt.subplots()
        ax.plot(checkpoints, success_rates, marker='o')
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Independent Success Rate")
        (correction_model_file / "viz").mkdir(parents=True, exist_ok=True)
        fig.savefig(str(correction_model_file / "viz" / "success_rate_over_checkpoints.png"))
        print(correction_model_file / "viz" / "success_rate_over_checkpoints.png")
        plt.close(fig)
    elif base_policy_file is not None and base_policy_file.is_dir():
        # Eval list of base policies
        checkpoints = list(base_policy_file.glob("*.pt"))
        checkpoints = [int(ckpt.name.replace("-ckpt.pt", "")) for ckpt in checkpoints if ckpt.name.endswith("-ckpt.pt")]
        checkpoints = sorted(checkpoints)
        checkpoints = [checkpoints[-1]] + checkpoints[:-1]
        base_policy_files = [base_policy_file / f"{ckpt}-ckpt.pt" for ckpt in checkpoints]
        success_rates = []
        for base_policy_path in base_policy_files:
            print(f"Evaluating base policy at {base_policy_path}...")
            success_rate = evaluate_model(
                env,
                policy,
                policy_nn,
                base_policy_file=base_policy_path,
                correction_model_file=correction_model_file,
                num_evals=args_cli.num_evals * 4 if base_policy_path == base_policy_files[-1] else args_cli.num_evals,
                reset_mode=args_cli.reset_mode,
                enable_cameras=args_cli.enable_cameras,
                IMAGE_SIZE=IMAGE_SIZE,
                plot_residual=args_cli.plot_residual,
                video_path=str(args_cli.video_path) if args_cli.video_path else None,
                horizon=args_cli.horizon,
                device=agent_cfg.device,
                need_reset_envs=True,
                no_viz=args_cli.no_viz,
                eval_mode=args_cli.eval_mode,
            )
            if not np.isnan(success_rate):
                success_rates.append(success_rate)
        
        print(checkpoints)
        print(success_rates)
        (base_policy_file / "viz").mkdir(parents=True, exist_ok=True)
        with open(base_policy_file / "viz" / "success_rate_over_checkpoints.txt", 'w') as f:
            for ckpt, rate in zip(checkpoints, success_rates):
                f.write(f"{ckpt} {rate}\n")

        fig, ax = plt.subplots()
        ax.plot(checkpoints, success_rates, marker='o')
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Independent Success Rate")
        (base_policy_file / "viz").mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base_policy_file / "viz" / "success_rate_over_checkpoints.png"))
        print(base_policy_file / "viz" / "success_rate_over_checkpoints.png")
        plt.close(fig)
    else:
        evaluate_model(
            env,
            policy,
            policy_nn,
            base_policy_file=base_policy_file,
            correction_model_file=correction_model_file,
            num_evals=args_cli.num_evals,
            reset_mode=args_cli.reset_mode,
            enable_cameras=args_cli.enable_cameras,
            IMAGE_SIZE=IMAGE_SIZE,
            plot_residual=args_cli.plot_residual,
            video_path=args_cli.video_path,
            horizon=args_cli.horizon,
            device=agent_cfg.device,
            need_reset_envs=False,
            no_viz=args_cli.no_viz,
            eval_mode=args_cli.eval_mode,
        )
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
