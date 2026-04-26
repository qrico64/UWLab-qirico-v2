import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import wandb
import random
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import cur_utils
from train_lib import RobotTransformerPolicy
import expert_utils
import signal
import tempfile


STOP_REQUESTED = False
STOP_SIGNAL = None

def _request_stop(signum, frame):
    global STOP_REQUESTED, STOP_SIGNAL
    STOP_REQUESTED = True
    STOP_SIGNAL = signum
    print(f"[signal] Received signal {signum}; will checkpoint and exit safely.")

ENABLE_WANDB = True

# --- Model Definition ---

def save_latest_checkpoint_atomic(
    save_path,
    epoch_to_resume,
    model,
    optimizer,
    scheduler,
    best_loss,
    best_loss_epoch,
):
    latest_dict = {
        "epoch": epoch_to_resume,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "wandb_run_id": wandb.run.id if ENABLE_WANDB and wandb.run is not None else None,
        "best_loss": best_loss,
        "best_loss_epoch": best_loss_epoch,
    }

    final_path = os.path.join(save_path, "latest.pt")
    fd, tmp_path = tempfile.mkstemp(dir=save_path, prefix="latest.pt.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(latest_dict, tmp_path)
        os.replace(tmp_path, final_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

class IndependentTrajectoryDataset(Dataset):
    def __init__(
            self,
            data,
            train_mode,
            closest_neighbors_radius: float = 0,
        ):
        """
        data: List of dicts containing 'context', 'current', 'label', and 'choosable'
        """
        self.data = data

        self.train_mode = train_mode
        if train_mode == "closest-neighbors":
            assert closest_neighbors_radius > 0
            self.choosable_trajs = []
            self.closest_neighbors_radius = closest_neighbors_radius
            self.all_receptive_noises = np.stack([traj['obs_receptive_noise'] for traj in data], axis=0)
            self.valid_seconds = []
            for i, traj in tqdm(enumerate(data)):
                if not traj.get('choosable', False):
                    if i < 20: print("(skipped due to unchoosable)")
                    continue
                cur_distances = np.linalg.norm(self.all_receptive_noises - traj['obs_receptive_noise'], axis=-1)
                if (((cur_distances <= closest_neighbors_radius) & (cur_distances > 0)).sum() == 0):
                    if i < 20: print("(skipped due to no neighbors)")
                    continue
                cur_seconds = np.where((cur_distances <= closest_neighbors_radius) & (cur_distances > 0))[0]
                self.choosable_trajs.append(traj)
                self.valid_seconds.append(cur_seconds)
                if i < 20:
                    print(self.valid_seconds[-1].shape)
        elif train_mode == "single-traj":
            self.choosable_trajs = [traj for traj in data if traj.get('choosable', False)]
        elif train_mode == "autoregressive":
            self.choosable_trajs = [traj for traj in data if traj.get('choosable', False)]
        elif train_mode == "full-traj":
            self.choosable_trajs = [traj for traj in data if traj.get('choosable', False)]
        elif train_mode == "perfect-coverage":
            self.choosable_trajs = [traj for traj in data if traj.get('choosable', False)]
        elif train_mode == "expert":
            self.choosable_trajs = []
            self.context_dim = data[0]['context'].shape[-1]
            self.current_dim = data[0]['current'].shape[-1]
            self.action_dim = data[0]['expert_actions'].shape[-1]
            self.choosable_currents = np.concatenate([traj['current'] for traj in data if traj.get('choosable', False)], axis=0)
            self.corresponding_expert_actions = np.concatenate([traj['expert_actions'] for traj in data if traj.get('choosable', False)], axis=0)
        else:
            raise NotImplementedError(train_mode)

    def __len__(self):
        if self.train_mode == "expert":
            return self.choosable_currents.shape[0]
        else:
            return len(self.choosable_trajs)

    def __getitem__(self, idx):
        if self.train_mode == "expert":
            fake_context = torch.zeros(1, self.context_dim, dtype=torch.float32)
            current = torch.tensor(self.choosable_currents[idx], dtype=torch.float32)
            fake_base_action = torch.zeros(self.action_dim, dtype=torch.float32)
            expert_action = torch.tensor(self.corresponding_expert_actions[idx], dtype=torch.float32)
            fake_data_source = "ds"
            fake_sys_noise = torch.zeros(self.action_dim, dtype=torch.float32)
            fake_obs_noise = torch.zeros(self.action_dim, dtype=torch.float32)
            fake__ref_traj = None
            return fake_context, current, fake_base_action, expert_action, fake_data_source, fake_sys_noise, fake_obs_noise, fake__ref_traj

        # Get the context and label from a "choosable" trajectory
        traj = self.choosable_trajs[idx]

        sys_noise = torch.tensor(traj['act_noise'], dtype=torch.float32)
        obs_noise = torch.tensor(traj['obs_noise'], dtype=torch.float32)

        if self.train_mode == "closest-neighbors":
            context = torch.tensor(traj['context'], dtype=torch.float32)
            second_traj = self.data[np.random.choice(self.valid_seconds[idx])]
            st = random.randint(0, second_traj['current'].shape[0] - 1)
            current = torch.tensor(second_traj['current'][st], dtype=torch.float32)
            base_action = torch.tensor(second_traj['base_actions'][st], dtype=torch.float32)
            expert_action = torch.tensor(second_traj['expert_actions'][st], dtype=torch.float32)
        elif self.train_mode == "single-traj":
            T = traj['context'].shape[0]
            assert T > 6, f"{T}"
            zt = random.randint(6, T - 1)
            st = random.randint(zt, T - 1)
            context = torch.tensor(traj['context'][:zt], dtype=torch.float32)
            current = torch.tensor(traj['current'][st], dtype=torch.float32)
            base_action = torch.tensor(traj['base_actions'][st], dtype=torch.float32)
            expert_action = torch.tensor(traj['expert_actions'][st], dtype=torch.float32)
        elif self.train_mode == "autoregressive":
            T = traj['context'].shape[0]
            assert T > 6, f"{T}"
            zt = random.randint(1, T - 1)
            context = torch.tensor(traj['context'][:zt], dtype=torch.float32)
            current = torch.tensor(traj['current'][zt], dtype=torch.float32)
            base_action = torch.tensor(traj['base_actions'][zt], dtype=torch.float32)
            expert_action = torch.tensor(traj['expert_actions'][zt], dtype=torch.float32)
        elif self.train_mode == "full-traj":
            T = traj['current'].shape[0]
            assert T > 6, f"{T}"
            zt = random.randint(0, T - 1)
            context = torch.tensor(traj['context'], dtype=torch.float32)
            current = torch.tensor(traj['current'][zt], dtype=torch.float32)
            base_action = torch.tensor(traj['base_actions'][zt], dtype=torch.float32)
            expert_action = torch.tensor(traj['expert_actions'][zt], dtype=torch.float32)
            # _ref_traj = (
            #     traj['__log']['obs']['policy'][zt],
            #     traj['__log']['obs']['policy_aaaaaa']['receptive_asset_pose'][zt],
            #     traj['__log']['obs']['policy_aaaaaa']['insertive_asset_pose'][zt],
            # )
        elif self.train_mode == "perfect-coverage":
            context = torch.tensor(traj['context'], dtype=torch.float32)
            si = random.randint(0, len(self.choosable_trajs) - 1)
            second_traj = self.choosable_trajs[si]
            st = random.randint(0, second_traj['current'].shape[0] - 1)
            current = torch.tensor(second_traj['current'][st], dtype=torch.float32)
            base_action = torch.tensor(second_traj['base_actions'][st], dtype=torch.float32)
            expert_action = torch.tensor(second_traj['expert_actions'][st], dtype=torch.float32)
        else:
            raise NotImplementedError(self.train_mode)
        
        data_source = traj['data_source']
        
        return context, current, base_action, expert_action, data_source, sys_noise, obs_noise

def collate_fn(batch):
    """
    Custom collator to pad trajectories of different lengths.
    """
    contexts, currents, base_actions, expert_actions, data_sources, sys_noises, obs_noises = zip(*batch)
    
    # Pad sequences to the max length in this specific batch
    # padded_contexts shape: (Batch, Max_T, Context_Dim)
    padded_contexts = torch.nn.utils.rnn.pad_sequence(contexts, batch_first=True)
    
    # Create a mask: True for padded positions, False for real data
    # This is for PyTorch's src_key_padding_mask
    padding_mask = torch.zeros(padded_contexts.shape[0], padded_contexts.shape[1], dtype=torch.bool)
    for i, ctx in enumerate(contexts):
        padding_mask[i, len(ctx):] = True
        
    currents = torch.stack(currents)
    base_actions = torch.stack(base_actions)
    expert_actions = torch.stack(expert_actions)
    sys_noises = torch.stack(sys_noises)
    obs_noises = torch.stack(obs_noises)

    return padded_contexts, currents, base_actions, expert_actions, padding_mask, data_sources, sys_noises, obs_noises

def train_behavior_cloning(
        model,
        train_data,
        val_data,
        epochs=100,
        lr=1e-4,
        batch_size=64,
        device="cuda",
        save_path: str = None,
        train_mode: str = "single-traj",
        closest_neighbors_radius: float = 0.001,
        warm_start: int = 0,
        force_mu_conditioning: str = "none",
        ref_label_means = None,
        ref_label_stds = None,
        ref_current_means = None,
        ref_current_stds = None,
        our_task: str = "peg",
    ):
    unique_data_sources_train = {}
    for traj in train_data:
        unique_data_sources_train[traj['data_source']] = unique_data_sources_train.get(traj['data_source'], 0) + 1
    unique_data_sources_val = {}
    for traj in val_data:
        unique_data_sources_val[traj['data_source']] = unique_data_sources_val.get(traj['data_source'], 0) + 1
    
    ref_label_means = torch.tensor(ref_label_means, dtype=torch.float32, device=device)
    ref_label_stds = torch.tensor(ref_label_stds, dtype=torch.float32, device=device)
    ref_current_means = torch.tensor(ref_current_means, dtype=torch.float32, device=device)
    ref_current_stds = torch.tensor(ref_current_stds, dtype=torch.float32, device=device)
    GENERAL_NOISE_SCALES = torch.tensor(cur_utils.GENERAL_NOISE_SCALES, device=device)

    expert_model = expert_utils.load_expert_by_task(our_task, device='cuda')[0]
    expert_pred = (expert_model(torch.tensor(train_data[0]['__log']['obs']['policy'], device=device)) - ref_label_means) / ref_label_stds
    assert np.allclose(train_data[0]['expert_actions'], expert_pred.detach().cpu().numpy(), atol=1e-5)
    noised_obss = cur_utils.apply_obs_noise_policy(torch.tensor(train_data[0]['current'], device=device) * ref_current_stds + ref_current_means, torch.tensor(train_data[0]['obs_noise'], device=device))
    expert_pred = (expert_model(noised_obss[:, :215]) + torch.tensor(train_data[0]['act_noise'], device=device) * GENERAL_NOISE_SCALES - ref_label_means) / ref_label_stds
    assert np.allclose(train_data[0]['base_actions'], expert_pred.detach().cpu().numpy(), atol=1e-4)
    # assert np.allclose(
    #     train_data[0]['current'][0][:6],
    #     (train_data[0]['__log']['obs']['policy'][0][:6] - ref_current_means[:6]) / ref_current_stds[:6],
    #     rtol=1e-4,
    # )
    # assert np.allclose(
    #     train_data[0]['current'][0][39:45],
    #     (train_data[0]['__log']['obs']['policy'][0][195:201] - ref_current_means[39:45]) / ref_current_stds[39:45],
    #     rtol=1e-4,
    # )

    train_loader = DataLoader(
        IndependentTrajectoryDataset(
            train_data,
            train_mode=train_mode,
            closest_neighbors_radius=closest_neighbors_radius,
        ),
        batch_size=batch_size, shuffle=True, num_workers=4, 
        collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        IndependentTrajectoryDataset(
            val_data,
            train_mode=train_mode,
            closest_neighbors_radius=closest_neighbors_radius,
        ),
        batch_size=batch_size, shuffle=False, num_workers=4, 
        collate_fn=collate_fn, pin_memory=True
    )

    assert epochs % 5 == 0, f"epochs={epochs} must be divisible by 5"
    SAVE_INTERVAL = epochs // 5

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # Learning rate scheduler for better convergence
    if warm_start <= 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warm_start),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs - warm_start),
            ],
            milestones=[warm_start],
        )
    
    epoch = 0
    best_loss = 100000
    best_loss_epoch = -1
    if os.path.exists(os.path.join(save_path, f"latest.pt")):
        resume_dict = torch.load(os.path.join(save_path, f"latest.pt"), map_location=device)
        model.load_state_dict(resume_dict["model_state_dict"])
        optimizer.load_state_dict(resume_dict["optimizer_state_dict"])
        scheduler.load_state_dict(resume_dict["scheduler_state_dict"])
        epoch = resume_dict["epoch"]
        best_loss = resume_dict["best_loss"]
        best_loss_epoch = resume_dict["best_loss_epoch"]

    while epoch < epochs:
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_info = {}
        
        for context, current, base_actions, expert_actions, padding_mask, data_sources, sys_noises, obs_noises in pbar:
            context, current, base_actions, expert_actions = context.to(device), current.to(device), base_actions.to(device), expert_actions.to(device)
            padding_mask = padding_mask.to(device)
            sys_noises = sys_noises.to(device)
            obs_noises = obs_noises.to(device)
            loss_mask = torch.ones(current.shape[0], dtype=torch.float32, device=device)

            if train_mode == "perfect-coverage":
                opolicy = (current * ref_current_stds + ref_current_means)[:, :215]
                noised_obss = cur_utils.apply_obs_noise_policy(opolicy, obs_noises)
                base_actions = expert_model(noised_obss) + sys_noises * GENERAL_NOISE_SCALES
                base_actions = (base_actions - ref_label_means) / ref_label_stds
                loss_mask[(torch.linalg.norm(base_actions, dim=-1) > 50).any(dim=-1)] = 0

            if force_mu_conditioning == "none":
                mu_conditioning = None
            elif force_mu_conditioning == "obsnoise":
                mu_conditioning = obs_noises[:, :2]
            else:
                raise NotImplementedError(f"{force_mu_conditioning}")

            optimizer.zero_grad()
            loss, info = model.loss(context, current, base_actions, expert_actions, padding_mask=padding_mask, mu_conditioning=mu_conditioning, loss_mask=loss_mask)
            info["loss_mask"] = loss_mask.cpu().numpy()

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            for data_source in unique_data_sources_train.keys():
                correspondent = np.array([ds == data_source for ds in data_sources])
                if correspondent.sum() <= 0:
                    continue
                for k, v in info.items():
                    if isinstance(v, np.ndarray):
                        total_info[f"{data_source}/{k}"] = total_info.get(f"{data_source}/{k}", 0) + v[correspondent].sum()
                    else:
                        total_info[f"{data_source}/{k}"] = total_info.get(f"{data_source}/{k}", 0) + v * correspondent.sum()

        # Validation phase
        model.eval()
        val_loss = 0
        total_vinfo = {}
        with torch.no_grad():
            for context, current, base_actions, expert_actions, padding_mask, data_sources, sys_noises, obs_noises in val_loader:
                context, current, base_actions, expert_actions = context.to(device), current.to(device), base_actions.to(device), expert_actions.to(device)
                padding_mask = padding_mask.to(device)
                sys_noises = sys_noises.to(device)
                obs_noises = obs_noises.to(device)
                loss_mask = torch.ones(current.shape[0], dtype=torch.float32, device=device)

                if train_mode == "perfect-coverage":
                    opolicy = (current * ref_current_stds + ref_current_means)[:, :215]
                    noised_obss = cur_utils.apply_obs_noise_policy(opolicy, obs_noises)
                    base_actions = expert_model(noised_obss) + sys_noises * GENERAL_NOISE_SCALES
                    base_actions = (base_actions - ref_label_means) / ref_label_stds
                    loss_mask[(torch.linalg.norm(base_actions, dim=-1) > 50).any(dim=-1)] = 0

                if force_mu_conditioning == "none":
                    mu_conditioning = None
                elif force_mu_conditioning == "obsnoise":
                    mu_conditioning = obs_noises[:, :2]
                else:
                    raise NotImplementedError(f"{force_mu_conditioning}")

                vloss, vinfo = model.loss(context, current, base_actions, expert_actions, padding_mask=padding_mask, mu_conditioning=mu_conditioning, loss_mask=loss_mask)
                vinfo["loss_mask"] = loss_mask.cpu().numpy()
                val_loss += vloss.item()

                for data_source in unique_data_sources_val.keys():
                    correspondent = np.array([ds == data_source for ds in data_sources])
                    if correspondent.sum() <= 0:
                        continue
                    for k, v in vinfo.items():
                        if isinstance(v, np.ndarray):
                            total_vinfo[f"{data_source}/{k}"] = total_vinfo.get(f"{data_source}/{k}", 0) + v[correspondent].sum()
                        else:
                            total_vinfo[f"{data_source}/{k}"] = total_vinfo.get(f"{data_source}/{k}", 0) + v * correspondent.sum()
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(f"Summary - Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        if ENABLE_WANDB:
            wandblog = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": optimizer.param_groups[0]['lr'],
            }
            for k in total_info.keys():
                data_source, metric_name = k[:k.find('/')], k[k.find('/')+1:]
                wandblog[f"train_{data_source}/{metric_name}"] = total_info[k] / unique_data_sources_train[data_source]
            for k in total_vinfo.keys():
                data_source, metric_name = k[:k.find('/')], k[k.find('/')+1:]
                wandblog[f"val_{data_source}/{metric_name}"] = total_vinfo[k] / unique_data_sources_val[data_source]
            wandb.log(wandblog)
        
        if (epoch + 1) % SAVE_INTERVAL == 0 and save_path is not None:
            csp = os.path.join(save_path, f"{epoch}-ckpt.pt")
            torch.save(model.state_dict(), csp)
            print(f"Model at epoch {epoch} saved to {csp}")
        elif epoch > SAVE_INTERVAL - 10 and avg_val_loss < best_loss and save_path is not None:
            best_loss = avg_val_loss
            csp = os.path.join(save_path, f"{best_loss_epoch}-ckpt.pt")
            if best_loss_epoch >= 0 and os.path.exists(csp):
                os.unlink(csp)
                print(f"Model at epoch {best_loss_epoch} removed.")
            best_loss_epoch = epoch
            csp = os.path.join(save_path, f"{epoch}-ckpt.pt")
            torch.save(model.state_dict(), csp)
            print(f"Best model at epoch {epoch} saved to {csp}")
        
        save_latest_checkpoint_atomic(
            save_path=save_path,
            epoch_to_resume=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_loss=best_loss,
            best_loss_epoch=best_loss_epoch,
        )

        epoch += 1

    
    if save_path is not None:
        csp = os.path.join(save_path, f"{epochs}-ckpt.pt")
        torch.save(model.state_dict(), csp)
        print(f"Model at epoch {epochs} saved to {csp}")

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Train Robot Transformer Policy")
    
    # Adding parameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--d_model", type=int, default=256, help="Transformer & MLP hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--save_path", type=str, default="policy_checkpoint.pt", help="Path to save the model")
    parser.add_argument("--dataset_path", type=str, nargs='+', default=["N/A"], help="Path(s) to load the dataset (multiple paths will be combined)")
    parser.add_argument("--train_mode", type=str, default="single-traj", help="Options: single-traj, closest-neighbors, autoregressive, full-traj.")
    parser.add_argument("--closest_neighbors_radius", type=float, default=0.001, help="If train_mode is closest-neighbors.")
    parser.add_argument("--warm_start", type=int, default=0, help="Number of warm start epochs.")
    parser.add_argument("--train_percent", type=float, default=0.8, help="Percentage of data used for train.")
    parser.add_argument("--val_percent", type=float, default=None, help="Percentage of data used for val.")
    parser.add_argument("--infer_mode", type=str, default="residual", help="Options: residual, expert, res_scale_shift.")
    parser.add_argument("--state_type", type=str, default="state", help="Which states to condition on.")
    parser.add_argument("--current_dim", type=int, default=215, help="Dimension of current / state, depends on state_type.")
    parser.add_argument("--our_task", type=str, default="peg", help="Options: peg, drawer, leg.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Mu stuff
    parser.add_argument("--mu_head_arch", type=str, default="none", help="Options: none, identity, linear, 2layer.")
    parser.add_argument("--mu_size", type=int, default=512, help="Dimension of mu.")
    parser.add_argument("--mu_kl_factor", type=float, default=0.0, help="KL factor for mu.")
    parser.add_argument("--force_mu_conditioning", type=str, default="none", help="Options: none, obsnoise.")
    parser.add_argument("--force_mu_conditioning_size", type=int, default=2, help="Dimension of forced mu.")

    # Curr KL stuff
    parser.add_argument("--current_norm", action="store_true", help="Whether to apply layer normalization to current embeddings.")
    parser.add_argument("--current_head_arch", type=str, default="none", help="Options: none, linear.")
    parser.add_argument("--current_emb_size", type=int, default=512, help="Dimension of current.")
    parser.add_argument("--current_kl_factor", type=float, default=0.0, help="KL factor for current.")

    # Combined head stuff
    parser.add_argument("--combined_head_arch", type=str, default="none", help="Options: none, linear, 2layer.")
    parser.add_argument("--combined_emb_size", type=int, default=512, help="Dimension of combined.")
    parser.add_argument("--combined_kl_factor", type=float, default=0.0, help="KL factor for combined.")

    # Head architecture
    parser.add_argument("--head_arch_version", type=str, default="ancient", help="Options: ancient, blocked, mlpblock_v1.")
    parser.add_argument("--num_head_layers", type=int, default=3, help="Number of Linear layers in the head.")
    parser.add_argument("--d_model_head", type=int, default=1024, help="Size of each Linear layer in the head.")
    parser.add_argument("--dropout_head", type=float, default=0.0, help="Dropout rate for head layers.")
    parser.add_argument("--act_head", type=str, default="relu", help="Activation function for head layers: relu, elu.")

    args = parser.parse_args()

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGUSR1, _request_stop)

    # Accessing the parameters
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    D_MODEL = args.d_model
    NUM_LAYERS = args.num_layers
    DROPOUT = args.dropout

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    CONTEXT_DIM = 43 + 7
    CURRENT_DIM = args.current_dim
    LABEL_DIM = 7

    TRAIN_PERCENT = args.train_percent
    VAL_PERCENT = args.val_percent
    if VAL_PERCENT is None: 
        VAL_PERCENT = 1 - TRAIN_PERCENT
    assert TRAIN_PERCENT + VAL_PERCENT <= 1

    TRAIN_EXPERT = args.infer_mode in ["expert", "expert_new"]
    if TRAIN_EXPERT:
        assert args.infer_mode == "expert_new", "expert is deprecated."
        assert args.train_mode in ["expert"]

    if ENABLE_WANDB:
        wandb_id_path = os.path.join(save_path, "wandb_run_id.txt")
        if os.path.exists(wandb_id_path):
            with open(wandb_id_path, "r") as f:
                wandb_run_id = f.read().strip()
        else:
            wandb_run_id = wandb.util.generate_id()
            with open(wandb_id_path, "w") as f:
                f.write(wandb_run_id)
        
        WANDB_PROJECT = "robot-transformer-bc-deterministic-normalized-labels" if not TRAIN_EXPERT else "robot-mlp-bc"
        WANDB_NAME = os.path.basename(save_path)
        wandb.init(project=WANDB_PROJECT, config=vars(args), name=WANDB_NAME, id=wandb_run_id, resume="allow")
    
    DATASET_PATHS = args.dataset_path

    DATASET_NAMES = {
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb26/fourthtry_receptive_0_sys3_rand2_recxgeq05/job-True-3.0-2.0-100000-60--0.0-0.0/cut-trajectories.pkl": "sysnoise_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb17/fourthtry_receptive_0.01_with_randnoise_2.0_recxgeq05/job-True-0.0-2.0-100000-60--0.01-0.0/cut-trajectories.pkl": "obsnoise_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb19/fourthtry_receptive_0006_sys4_rand2_recxgeq05/job-True-4.0-2.0-100000-60--0.006-0.0/cut-trajectories.pkl": "obs0006_sys4_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar5/obs001r2_dataset_recxgeq05/job-True-0.0-2.0-100000-60--0.01-0.0/cut-trajectories.pkl": "obsnoise_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar9/y4_id_rand2_expert/cut-trajectories.pkl": "y4_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar9/y2_id_rand2_expert/cut-trajectories.pkl": "y2_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar9/y3_id_rand2_expert/cut-trajectories.pkl": "y3_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar10/recxgeq05_id_obsnoise005_rand2/cut-trajectories.pkl": "obsnoise_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar10/peg_recxgeq05_id_obs003_sys4_r2/cut-trajectories.pkl": "obsnoise_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar10/drawer_y4_id_r2/cut-trajectories.pkl": "drawer_y4_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar10/drawer_y4_id_r05/cut-trajectories.pkl": "drawer_y4_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar10/drawer_y4_id_obs001_r2/cut-trajectories.pkl": "drawer_o001_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar9/y4_id_rand2_/cut-trajectories.pkl": "peg_y4_o001_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar11/peg_r4_id/cut-trajectories.pkl": "peg_r4_exp_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar5/obs001r2_dataset_recxgeq05/job-True-0.0-2.0-100000-60--0.01-0.0/cut-trajectories.pkl": "o001r2_ds",
        "/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar27/peg_recxgeq05_s4r2/cut-trajectories.pkl": "s4r2_ds",
    }

    datasets = []
    total_trajs = 0
    for DATASET_PATH in DATASET_PATHS:
        try:
            with open(DATASET_PATH, "rb") as fi:
                loaded = pickle.load(fi)
                datasets.append((loaded, DATASET_PATH))
                total_trajs += len(loaded)
                print(f"Loaded {len(loaded)} trajectories from {DATASET_PATH}.")
        except FileNotFoundError:
            print(f"Data file not found: {DATASET_PATH}")
            return
    # Use the first dataset path for info.pkl lookup (noise scale metadata)
    first_path = DATASET_PATHS[0]
    with open(os.path.join(os.path.dirname(first_path), "info.pkl"), "rb") as fi:
        load_dict = pickle.load(fi)

    processed_data = []
    for dataset in datasets:
        trajs, data_source = dataset
        dataset_name = DATASET_NAMES.get(data_source, "unknown_ds")
        for traj in trajs:
            if traj['actions'].shape[0] <= 6 or np.any(np.linalg.norm(traj['actions_expert'], axis=-1) > 100) or np.any(np.abs(traj['obs']['policy']) > 300):
                continue

            if traj['rewards'].ndim == 1:
                traj['rewards'] = traj['rewards'][:, None]
            
            unnoised_actions = traj['actions'] - traj['rand_noise']
            processed_traj = {
                'context': np.concatenate([traj['obs']['policy2'], traj['actions']], axis=1),
                'current': np.concatenate([traj['obs']['policy'], unnoised_actions], axis=1),
                'base_actions': unnoised_actions,
                'expert_actions': traj['actions_expert'],
                'choosable': True,
                'obs_noise': traj['obs_noise'] * load_dict['obs_receptive_noise_scale'],
                'act_noise': traj['act_noise'] * load_dict['act_noise_scale'],
                'data_source': dataset_name,
                '__log': traj,
            }
            
            processed_data.append(processed_traj)
    assert processed_data[0]['context'].shape[-1] == CONTEXT_DIM
    print(f"Kept {len(processed_data)}/{total_trajs} ({len(processed_data)/total_trajs}) trajectories.")

    # Current normalization
    all_currents = np.concatenate([traj['current'] for traj in processed_data], axis=0)
    current_means = all_currents.mean(axis=0)
    current_stds = all_currents.std(axis=0)
    all_contexts = np.concatenate([traj['context'] for traj in processed_data], axis=0)
    context_means = all_contexts.mean(axis=0)
    context_stds = all_contexts.std(axis=0) + 1e-9
    if TRAIN_EXPERT:
        all_labels = np.concatenate([traj['expert_actions'] for traj in processed_data], axis=0)
    elif args.infer_mode == "res_scale_shift":
        all_labels = np.concatenate([traj['expert_actions'] for traj in processed_data] + [traj['base_actions'] for traj in processed_data], axis=0)
    elif args.infer_mode == "residual":
        all_labels = np.concatenate([traj['expert_actions'] - traj['base_actions'] for traj in processed_data], axis=0)
    label_means = all_labels.mean(axis=0)
    label_stds = all_labels.std(axis=0)
    for traj in processed_data:
        traj['current'] = (traj['current'] - current_means) / current_stds
        traj['context'] = (traj['context'] - context_means) / context_stds
        if args.infer_mode == "res_scale_shift":
            traj['base_actions'] = (traj['base_actions'] - label_means) / label_stds
        else:
            traj['base_actions'] = traj['base_actions'] / label_stds
        traj['expert_actions'] = (traj['expert_actions'] - label_means) / label_stds

    save_dict = {
        'dataset_origin': [os.path.abspath(p) for p in DATASET_PATHS],
        'dataset_size': len(processed_data),
        'save_path': save_path,
        'current_means': current_means,
        'current_stds': current_stds,
        'context_means': context_means,
        'context_stds': context_stds,
        'label_means': label_means,
        'label_stds': label_stds,
        'context_dim': CONTEXT_DIM,
        'current_dim': CURRENT_DIM,
        'label_dim': LABEL_DIM,
        'd_model': D_MODEL,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'train_mode': args.train_mode,
        'closest_neighbors_radius': args.closest_neighbors_radius,
        'warm_start': args.warm_start,
        'train_percent': TRAIN_PERCENT,
        'val_percent': VAL_PERCENT,
        'train_expert': TRAIN_EXPERT,
        'infer_mode': args.infer_mode,

        'mu_head_arch': args.mu_head_arch,
        'mu_size': args.mu_size,
        'mu_kl_factor': args.mu_kl_factor,
        'force_mu_conditioning': args.force_mu_conditioning,
        'force_mu_conditioning_size': args.force_mu_conditioning_size,

        'head_arch_version': args.head_arch_version,
        'num_head_layers': args.num_head_layers,
        'd_model_head': args.d_model_head,
        'dropout_head': args.dropout_head,

        'current_norm': args.current_norm,
        'current_head_arch': args.current_head_arch,
        'current_emb_size': args.current_emb_size,
        'current_kl_factor': args.current_kl_factor,

        'combined_head_arch': args.combined_head_arch,
        'combined_emb_size': args.combined_emb_size,
        'combined_kl_factor': args.combined_kl_factor,

        'state_type': args.state_type,

        'act_noise_scale': load_dict['act_noise_scale'],
        'rand_noise_scale': load_dict['rand_noise_scale'],
        'obs_receptive_noise_scale': load_dict['obs_receptive_noise_scale'],
    }
    cur_utils.save_info_dict(save_dict, os.path.join(save_path, "info.pkl"))

    # Visualization
    viz_path = os.path.join(save_path, "viz")
    os.makedirs(viz_path, exist_ok=True)
    all_base_actions_viz = np.concatenate([traj['base_actions'] for traj in processed_data], axis=0)
    for i in range(LABEL_DIM):
        cur_utils.save_histogram(all_base_actions_viz[:, i], os.path.join(viz_path, f"base_action_{i}.png"))
    all_expert_actions_viz = np.concatenate([traj['expert_actions'] for traj in processed_data], axis=0)
    for i in range(LABEL_DIM):
        cur_utils.save_histogram(all_expert_actions_viz[:, i], os.path.join(viz_path, f"expert_action_{i}.png"))
    all_residual_actions_viz = np.concatenate([traj['expert_actions'] - traj['base_actions'] for traj in processed_data], axis=0)
    for i in range(LABEL_DIM):
        cur_utils.save_histogram(all_residual_actions_viz[:, i], os.path.join(viz_path, f"residual_action_{i}.png"))

    num_choosable = sum(1 for d in processed_data if d['choosable'])
    print(f"Total Trajectories: {len(processed_data)}")
    print(f"Choosable Trajectories: {num_choosable}")
    
    if num_choosable == 0:
        print("Error: No choosable trajectories found. Check reward thresholds.")
        return

    random.shuffle(processed_data)
    split = int(len(processed_data) * TRAIN_PERCENT)
    split2 = split + int(len(processed_data) * VAL_PERCENT)
    train_data = processed_data[:split]
    val_data = processed_data[split:split2]
    print(f"Train percent: {TRAIN_PERCENT} !")
    print(f"Val percent: {VAL_PERCENT} !")

    # Final safeguard: ensure both splits have at least one choosable traj
    if not any(d['choosable'] for d in val_data):
        print("Warning: Validation set has no choosable trajectories. Re-shuffling...")
        # In a real scenario, you might want a Stratified Split here

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RobotTransformerPolicy(
        CONTEXT_DIM, CURRENT_DIM, LABEL_DIM, num_layers=NUM_LAYERS, d_model=D_MODEL, dropout=DROPOUT,
        head_arch_version=args.head_arch_version,
        num_head_layers=args.num_head_layers,
        d_model_head=args.d_model_head,
        dropout_head=args.dropout_head,
        act_head=args.act_head,
        infer_mode=args.infer_mode,
        mu_head_arch=args.mu_head_arch,
        mu_size=args.mu_size,
        mu_kl_factor=args.mu_kl_factor,
        force_mu_conditioning=args.force_mu_conditioning,
        force_mu_conditioning_size=args.force_mu_conditioning_size,
        current_norm=args.current_norm,
        current_head_arch=args.current_head_arch,
        current_emb_size=args.current_emb_size,
        current_kl_factor=args.current_kl_factor,
        combined_head_arch=args.combined_head_arch,
        combined_emb_size=args.combined_emb_size,
        combined_kl_factor=args.combined_kl_factor,
        state_type=args.state_type,
    )
    model.to(device)

    try:
        train_behavior_cloning(
            model,
            train_data,
            val_data,
            epochs=EPOCHS,
            lr=LR,
            batch_size=BATCH_SIZE,
            device=device,
            save_path=save_path,
            train_mode=args.train_mode,
            closest_neighbors_radius=args.closest_neighbors_radius,
            warm_start=args.warm_start,
            force_mu_conditioning=args.force_mu_conditioning,
            ref_label_means=label_means,
            ref_label_stds=label_stds,
            ref_current_means=current_means,
            ref_current_stds=current_stds,
            our_task=args.our_task,
        )
    finally:
        if ENABLE_WANDB:
            wandb.finish()

if __name__ == '__main__':
    main()
