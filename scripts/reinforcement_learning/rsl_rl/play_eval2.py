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
parser.add_argument("--correction_model", type=str, default="N/A", help="Residual model .pt file.")
parser.add_argument("--plot_residual", action="store_true", default=False, help="Open second screen & plot residual.")
parser.add_argument("--video_path", type=str, default=None, help="Save location for videos.")
parser.add_argument("--num_evals", type=int, default=2000, help="Number of trajectories we eval for.")
parser.add_argument("--base_policy", type=str, default=None, help="Base model .pt file.")
parser.add_argument("--finetune_mode", type=str, default="residual", help="Options: residual, expert.")
parser.add_argument("--save_path", type=str, default=None, help="Save location for checkpoints.")
parser.add_argument("--utd_ratio", type=float, default=1.0, help="Utd ratio for finetuning.")
parser.add_argument("--finetune_arch", type=str, default="lora", help="Options: lora, full.")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for finetuning.")
parser.add_argument("--reset_mode", type=str, default='none', help="Options: none, xleq035, recxgeq05.")
parser.add_argument("--our_task", choices=["drawer", "leg", "peg"], default=None)
parser.add_argument("--sim_device", type=str, default="cuda:0", help="Device to run the simulation on.")
parser.add_argument("--noise_scale", type=float, default=1, help="Noise scale on both obs and action noises.")
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
import wandb
import math
import pickle

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
import train_lib
import cur_utils
import train_lora_lib
import expert_utils

ENABLE_WANDB = True

# PLACEHOLDER: Extension template (do not remove this comment)


def save_video(frames, path, fps=30):
    """Saves a list of frames (numpy arrays) to a video file."""
    print(f"[INFO] Saving video to {path}...")
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
    BOUNDARY_X = 5
    BOUNDARY_Y = 5

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
        org=(5, 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=(0, 0, 0),  # BGR: black
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        ' '.join(captions[1:]),
        org=(5, 28),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=(0, 0, 0),  # BGR: black
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return frame

def save_model_at_checkpoint(model: train_lib.RobotTransformerPolicy, save_path: str, epoch: int, finetuning_arch: str):
    if finetuning_arch == "lora":
        model = train_lora_lib.convert_lora_model_to_plain_robot_policy(model)
    csp = os.path.join(save_path, f"{epoch}-ckpt.pt")
    torch.save(model.state_dict(), csp)
    print(f"Model at epoch {epoch} saved to {csp}")



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
    expert_policy = runner.get_inference_policy(device=env.unwrapped.device)
    expert_policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
    
    # Rico: Instantiate base policy!!
    assert args_cli.base_policy is not None
    base_policy, base_policy_info = train_lib.load_robot_policy(args_cli.base_policy, device=args_cli.device, our_task=args_cli.our_task)
    assert base_policy_info['infer_mode'] in ["expert", "expert_new"], f"infer_mode = {base_policy_info['infer_mode']}"
    for k in ["current_means", "current_stds", "context_means", "context_stds", "label_means", "label_stds"]:
        base_policy_info[k + "_tensor"] = torch.tensor(base_policy_info[k], dtype=torch.float32, device=args_cli.device)
    CURRENT_DIM = 215 + 7
    OBS_RECEPTIVE_NOISE_SCALE = 0.0
    OBS_INSERTIVE_NOISE_SCALE = 0.0
    SYS_NOISE_SCALE = 0.0
    
    if args_cli.finetune_arch == "lora":
        report = train_lora_lib.verify_lora_conversion_from_model(base_policy)
        print(report)
        base_policy.head = train_lora_lib.apply_lora(base_policy.head, r=8, alpha=16, lora_dropout=0, init_std=0.01)
        base_policy = train_lora_lib.freeze_all_but_lora(base_policy)
        base_policy.eval()
    elif args_cli.finetune_arch == "full":
        base_policy.eval()
    else:
        raise NotImplementedError(f"Finetune architecture {args_cli.finetune_arch} not supported.")

    print("****** Base Policy Architecture ******")
    print(base_policy)
    print("****** End Base Policy Architecture ******")

    # Correction model stuff

    RESIDUAL_S_DIM = env.observation_space['policy2'].shape[-1]
    A_DIM = env.action_space.shape[-1]
    RESIDUAL_CONTEXT_DIM = RESIDUAL_S_DIM + A_DIM
    T_DIM = args_cli.horizon
    CORRECTION_MODEL_FILE = os.path.abspath(args_cli.correction_model)
    print(f"Loading model at {CORRECTION_MODEL_FILE}")
    VIZ_DIRECTORY = pathlib.Path(CORRECTION_MODEL_FILE).parent / "viz"
    VIZ_DIRECTORY.mkdir(parents=True, exist_ok=True)
    correction_model, correction_model_info = train_lib.load_robot_policy(CORRECTION_MODEL_FILE, device=args_cli.device)

    N_DIM = 1
    timesteps = torch.zeros(env.num_envs, N_DIM, dtype=torch.int64, device=args_cli.device)
    successes = torch.zeros(env.num_envs, N_DIM, dtype=torch.bool, device=args_cli.device)
    rec_observations = torch.zeros(env.num_envs, N_DIM, T_DIM, RESIDUAL_S_DIM, dtype=torch.float32, device=args_cli.device)
    rec_observations_policy = torch.zeros(env.num_envs, N_DIM, T_DIM, 215, dtype=torch.float32, device=args_cli.device)
    rec_currents = torch.zeros(env.num_envs, N_DIM, T_DIM, CURRENT_DIM, dtype=torch.float32, device=args_cli.device)
    rec_actions = torch.zeros(env.num_envs, N_DIM, T_DIM, A_DIM, dtype=torch.float32, device=args_cli.device)
    rec_rewards = torch.zeros(env.num_envs, N_DIM, T_DIM, dtype=torch.float32, device=args_cli.device)
    curstates = torch.zeros(env.num_envs, dtype=torch.int64, device=args_cli.device)
    trajectory_start_position = get_starting_position(env)
    rec_expert_actions = torch.zeros(env.num_envs, N_DIM, T_DIM, A_DIM, dtype=torch.float32, device=args_cli.device)

    obs = env.get_observations()

    if OBS_RECEPTIVE_NOISE_SCALE != 0 or OBS_INSERTIVE_NOISE_SCALE != 0:
        assert 'policy_aaaaaa' in obs.keys()
    if OBS_INSERTIVE_NOISE_SCALE != 0:
        raise Exception("Right now doesn't support insertive noise!!")

    GENERAL_NOISE_SCALES = torch.tensor(cur_utils.GENERAL_NOISE_SCALES, device=args_cli.device)
    sys_noises = torch.randn(A_DIM, device=args_cli.device) * SYS_NOISE_SCALE * GENERAL_NOISE_SCALES
    obs_receptive_noise = torch.randn(2, device=args_cli.device) * OBS_RECEPTIVE_NOISE_SCALE
    obs_insertive_noise = torch.cat([torch.randn(2, device=args_cli.device) * OBS_INSERTIVE_NOISE_SCALE, torch.zeros(4, device=args_cli.device)], dim=-1)
    USING_FIXED_NOISE = False
    if args_cli.base_policy == "expert":
        USING_FIXED_NOISE = True
        sys_noises = torch.tensor([0.7432, 0.5431, -0.6655, 0.2321, 0.1166, 0.2186, 0.8714], dtype=torch.float32, device=args_cli.device) * correction_model_info['act_noise_scale'] * GENERAL_NOISE_SCALES
        obs_receptive_noise = torch.tensor([0.3047, -1.0399], dtype=torch.float32, device=args_cli.device) * correction_model_info['obs_receptive_noise_scale']
        sys_noises *= args_cli.noise_scale
        obs_receptive_noise *= args_cli.noise_scale
    base_policy_info['obs_receptive_noise'] = obs_receptive_noise
    base_policy_info['obs_insertive_noise'] = obs_insertive_noise
    base_policy_info['sys_noise'] = sys_noises
    print("Using sysnoise of ", sys_noises)
    print("Using obs_receptive_noise of ", obs_receptive_noise)
    print("Using obs_insertive_noise of ", obs_insertive_noise)

    if args_cli.enable_cameras:
        PLOT_RESIDUAL = args_cli.plot_residual
        rec_video = np.zeros((env.num_envs, 2, T_DIM, IMAGE_SIZE[0], IMAGE_SIZE[1] * 2 if PLOT_RESIDUAL else IMAGE_SIZE[1], 3), dtype=np.uint8)
        VIDEO_PATH = args_cli.video_path or str(VIZ_DIRECTORY / "videos")
        os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
        videopath_generator = lambda x, y: VIDEO_PATH[:VIDEO_PATH.rfind('.')] + f"_{x}_{y}" + VIDEO_PATH[VIDEO_PATH.rfind('.'):]
        NUM_VIDEOS = 6
        VIDEO_FPS = 5
    
    starting_positions = get_positions(env.env.env)

    # TRAINING CONFIG
    finetune_args = {
        "base_policy": args_cli.base_policy,
        "correction_model": args_cli.correction_model,
        "finetune_arch": args_cli.finetune_arch,
        "finetune_mode": args_cli.finetune_mode,
        "utd_ratio": args_cli.utd_ratio,
        "lr": args_cli.lr,
        "sys_noise_scale": SYS_NOISE_SCALE,
        "obs_receptive_noise_scale": OBS_RECEPTIVE_NOISE_SCALE,
        "obs_insertive_noise_scale": OBS_INSERTIVE_NOISE_SCALE,
        "sys_noise": sys_noises.cpu().numpy(),
        "obs_receptive_noise": obs_receptive_noise.cpu().numpy(),
        "obs_insertive_noise": obs_insertive_noise.cpu().numpy(),
        "reset_mode": args_cli.reset_mode,
        "using_fixed_noise": USING_FIXED_NOISE,
        "noise_scale": args_cli.noise_scale,
    }
    base_policy_info['finetune_args'] = finetune_args

    replay_buffer = {
        'index': 0,
        'current': torch.zeros(args_cli.num_evals * T_DIM, CURRENT_DIM, dtype=torch.float32, device='cpu'),
        'label': torch.zeros(args_cli.num_evals * T_DIM, A_DIM, dtype=torch.float32, device='cpu'),
    }
    BATCH_SIZE = 64
    LABEL_MAGNITUDE_LIMIT = 50.0
    optimizer = torch.optim.AdamW(base_policy.parameters(), lr=args_cli.lr, weight_decay=1e-5)

    SAVE_DIRECTORY = args_cli.save_path
    if SAVE_DIRECTORY is not None:
        SAVE_DIRECTORY = pathlib.Path(SAVE_DIRECTORY)
        SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        # Save the base model as ckpt_0.pt
        save_model_at_checkpoint(base_policy.model, str(SAVE_DIRECTORY), 0, finetuning_arch=args_cli.finetune_arch)
        cur_utils.save_info_dict(base_policy_info, SAVE_DIRECTORY / f"info.pkl")

    num_epochs_so_far = 0
    num_trajs_so_far = 0

    UTD_RATIO = args_cli.utd_ratio
    print(f"Using UTD ratio of {UTD_RATIO}")

    LOG_INTERVAL = 20
    SAVE_INTERVAL = lambda x: 0 if x < 20 else (5 * int(math.log10(x) - 1) + int(x / 10 ** int(math.log10(x)) / 2))

    # reset environment

    global_timestep = 0

    count_success = torch.zeros(N_DIM + 1, dtype=torch.int64, device='cpu')

    # Initialize wandb
    if ENABLE_WANDB:
        WANDB_PROJECT = "robot-transformer-bc-finetuning-eval"
        WANDB_NAME = os.path.basename(os.path.dirname(SAVE_DIRECTORY))
        wandb.init(project=WANDB_PROJECT, config=vars(args_cli), name=WANDB_NAME)
        wandb.watch(base_policy)

    try:
        while count_success.sum() < args_cli.num_evals:
            global_timestep += 1
            with torch.inference_mode():
                expert_action = expert_policy(obs)
                noised_obs = obs.clone()
                noised_obs['policy'] = cur_utils.apply_obs_noise_policy(obs['policy'], obs_receptive_noise)
                noised_obs['policy2'] = cur_utils.apply_obs_noise_policy2(obs['policy2'], obs_receptive_noise)

                fake_context = torch.zeros(env.num_envs, args_cli.horizon, base_policy_info['context_dim'], dtype=torch.float32, device=args_cli.device)
                fake_padding_mask = torch.zeros(env.num_envs, args_cli.horizon, dtype=torch.bool, device=args_cli.device)
                currents = torch.cat([noised_obs['policy'], torch.zeros(env.num_envs, 7, dtype=torch.float32, device=args_cli.device)], dim=-1)
                fake_base_actions = torch.zeros(env.num_envs, base_policy_info['label_dim'], dtype=torch.float32, device=args_cli.device)
                base_actions = base_policy.get_action(fake_context, currents, fake_base_actions, padding_mask=fake_padding_mask)
                base_actions += sys_noises
                
                # step
                next_obs, reward, dones, info = env.step(base_actions)

                # handle non-dones
                indices = torch.arange(env.num_envs, device=args_cli.device)
                cur_timesteps = timesteps[indices, curstates]
                rec_observations[indices, curstates, cur_timesteps, :] = obs['policy2']
                rec_observations_policy[indices, curstates, cur_timesteps, :] = obs['policy']
                rec_currents[indices, curstates, cur_timesteps, :] = currents
                rec_actions[indices, curstates, cur_timesteps, :] = base_actions
                rec_rewards[indices, curstates, cur_timesteps] = reward
                rec_expert_actions[indices, curstates, cur_timesteps] = expert_action
                successes[indices, curstates] |= env.unwrapped.termination_manager.get_term("success")
                
                timesteps[indices, curstates] += 1
                obs = next_obs

                # handle dones
                expert_policy_nn.reset(dones)
                for i in torch.nonzero(dones).squeeze(-1):
                    count_success[successes[i, 0].long()] += 1

                    T = timesteps[i, 0]
                    context = torch.cat([rec_observations[i, 0, :T], rec_actions[i, 0, :T]], axis=1)
                    current_policy = rec_observations_policy[i, 0, :T]
                    cur_base_actions = rec_actions[i, 0, :T]
                    padded_current = torch.cat([current_policy, cur_base_actions], dim=-1)
                    
                    residual_actions = correction_model.get_action(
                        context.repeat(T, 1, 1),
                        padded_current,
                        cur_base_actions,
                    )

                    stored_currents = rec_currents[i, 0, :T]
                    stored_labels = rec_expert_actions[i, 0, :T] if args_cli.finetune_mode == "expert" else residual_actions
                    stored_labels = stored_labels - sys_noises # for both expert and residual supervision
                    valid_label_mask = (
                        torch.isfinite(stored_labels).all(dim=-1)
                        & (stored_labels.abs() <= LABEL_MAGNITUDE_LIMIT).all(dim=-1)
                    )
                    num_valid = valid_label_mask.sum().item()
                    if num_valid > 0:
                        replay_start = replay_buffer['index']
                        replay_end = replay_start + num_valid
                        replay_buffer['current'][replay_start:replay_end] = stored_currents[valid_label_mask].cpu()
                        replay_buffer['label'][replay_start:replay_end] = stored_labels[valid_label_mask].cpu()
                        replay_buffer['index'] = replay_end

                    rec_observations[i] *= 0
                    rec_observations_policy[i] *= 0
                    rec_currents[i] *= 0
                    rec_actions[i] *= 0
                    rec_rewards[i] *= 0
                    timesteps[i] *= 0
                    successes[i] = False
                    curstates[i] *= 0
                    starting_positions[i] = get_positions(env.env.env)[i]
                    trajectory_start_position[i] = get_starting_position(env)[i]
                
            if dones.sum() > 0:
                current_step = replay_buffer['index']
                current_traj = count_success.sum().item()

                if current_traj // LOG_INTERVAL > num_trajs_so_far // LOG_INTERVAL:
                    print(f"Success count: {count_success.tolist()} Success rate: {count_success[1].item() / count_success.sum().item():.3f} at {current_traj} evaluations")

                if ENABLE_WANDB:
                    wandb.log({
                        "success_rate": count_success[1].item() / count_success.sum().item(),
                        "num_success": count_success[1].item(),
                        "num_total": count_success.sum().item(),
                        "eval_step": current_step,
                    }, step=count_success.sum().item())

                batch_size = min(BATCH_SIZE, replay_buffer['index'])
                num_epochs = int(current_step * UTD_RATIO) - num_epochs_so_far

                if num_epochs > 0:
                    base_policy.train()
                    train_loss = 0
                    total_info = {}

                    for _ in range(num_epochs):
                        idxs = torch.randint(0, replay_buffer['index'], size=(batch_size,), device=replay_buffer['current'].device)
                        batch_current = replay_buffer['current'][idxs].to(args_cli.device)
                        batch_label = replay_buffer['label'][idxs].to(args_cli.device)
                        fake_context = torch.zeros(batch_size, args_cli.horizon, RESIDUAL_CONTEXT_DIM, dtype=torch.float32, device=args_cli.device)
                        fake_padding_mask = torch.zeros(batch_size, args_cli.horizon, dtype=torch.bool, device=args_cli.device)
                        fake_base_actions = torch.zeros(batch_size, A_DIM, dtype=torch.float32, device=args_cli.device)

                        optimizer.zero_grad()
                        loss, info = base_policy.loss(fake_context, batch_current, fake_base_actions, batch_label, fake_padding_mask)
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()

                        for k, v in info.items():
                            if k not in total_info:
                                total_info[k] = 0
                            total_info[k] += v
                    
                    avg_train_loss = train_loss / num_epochs
                    if current_traj // LOG_INTERVAL > num_trajs_so_far // LOG_INTERVAL:
                        print(f"Trained for {num_epochs} epochs with batch size {batch_size} on {replay_buffer['index']} samples in replay buffer.")
                        print(f"Step {replay_buffer['index']} - Train Loss: {avg_train_loss:.6f}")
                        print()

                    if ENABLE_WANDB:
                        wandb.log({
                            "train_loss": avg_train_loss,
                            "step": replay_buffer['index'],
                            **{f"train/{k}": v / num_epochs for k, v in total_info.items()},
                        }, step=count_success.sum().item())

                    base_policy.eval()

                    if SAVE_DIRECTORY is not None and SAVE_INTERVAL(current_traj) > SAVE_INTERVAL(num_trajs_so_far):
                        save_model_at_checkpoint(base_policy.model, SAVE_DIRECTORY, current_traj, finetuning_arch=args_cli.finetune_arch)
                        cur_utils.save_info_dict(base_policy_info, SAVE_DIRECTORY / f"info.pkl")
                    
                    num_epochs_so_far += num_epochs
                
                num_trajs_so_far = current_traj
    finally:
        # close the simulator
        env.close()
        print(f"Loading model at {CORRECTION_MODEL_FILE}")

        if ENABLE_WANDB:
            wandb.finish()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
