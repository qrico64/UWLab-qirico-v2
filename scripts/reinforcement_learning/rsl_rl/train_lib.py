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
import pathlib
import expert_utils

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, act="relu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU(), "elu": nn.ELU()}[act]
        listofmodules = [
            nn.Linear(in_dim, out_dim),
            act_fn,
        ]
        if dropout > 1e-6:
            listofmodules.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*listofmodules)

    def forward(self, x):
        return self.net(x)

def block(in_dim: int, out_dim: int, dropout: float) -> list[nn.Module]:
    return [
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
    ]

class RobotTransformerPolicy(nn.Module):
    def __init__(
            self, context_dim, current_dim, label_dim, nhead=8, num_layers=4, d_model=512, dropout=0.1,
            head_arch_version="blocked", # Options: "ancient", "blocked", "mlpblock_v1"
            num_head_layers=3,
            d_model_head=1024,
            dropout_head=0,
            act_head="relu",
            infer_mode: str = "residual", # Options: "residual", "expert", "res_scale_shift".
            mu_head_arch: str = "none", # Options: "none", "identity", "linear", "2layer".
            mu_size: int = 512,
            mu_kl_factor: float = 0.0,
            current_norm: bool = True,
            current_head_arch: str = "none", # Options: "none", "linear".
            current_emb_size: int = 512,
            current_kl_factor: float = 0.0,
            combined_head_arch: str = "none", # Options: "none", "linear", "2layer".
            combined_emb_size: int = 512,
            combined_kl_factor: float = 0.0,
            state_type: str = "standard",
            force_mu_conditioning: str = "obsnoise", # Options: "none", "obsnoise".
            force_mu_conditioning_size: int = 2,
        ):
        super().__init__()

        self.policy_cfg = dict(
            context_dim=context_dim,
            current_dim=current_dim,
            label_dim=label_dim,
            nhead=nhead,
            num_layers=num_layers,
            d_model=d_model,
            dropout=dropout,
            head_arch_version=head_arch_version,
            num_head_layers=num_head_layers,
            d_model_head=d_model_head,
            dropout_head=dropout_head,
            act_head=act_head,
            infer_mode=infer_mode,
            mu_head_arch=mu_head_arch,
            mu_size=mu_size,
            mu_kl_factor=mu_kl_factor,
            current_norm=current_norm,
            current_head_arch=current_head_arch,
            current_emb_size=current_emb_size,
            current_kl_factor=current_kl_factor,
            combined_head_arch=combined_head_arch,
            combined_emb_size=combined_emb_size,
            combined_kl_factor=combined_kl_factor,
            state_type=state_type,
            force_mu_conditioning=force_mu_conditioning,
            force_mu_conditioning_size=force_mu_conditioning_size,
        )

        self.context_proj = nn.Linear(context_dim, d_model)
        self.ctx_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # mu head
        assert force_mu_conditioning in ["none", "obsnoise"], f"{force_mu_conditioning}"
        mu_head_in_dim = force_mu_conditioning_size if force_mu_conditioning != "none" else d_model
        if mu_head_arch == "none":
            assert mu_size == mu_head_in_dim, f"{mu_size} != {mu_head_in_dim}"
        elif mu_head_arch == "identity":
            assert mu_size * 2 == mu_head_in_dim, f"{mu_size} * 2 != {d_model}"
            self.mu_head = nn.Identity()
        elif mu_head_arch == "linear":
            self.mu_head = nn.Linear(mu_head_in_dim, mu_size * 2)
        elif mu_head_arch == "2layer":
            self.mu_head = nn.Sequential(
                nn.Linear(mu_head_in_dim, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, mu_size * 2),
            )
        elif mu_head_arch == "3layer":
            self.mu_head = nn.Sequential(
                nn.Linear(mu_head_in_dim, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, mu_size * 2),
            )
        else:
            raise NotImplementedError(f"Unknown mu_head_arch: {mu_head_arch}")
        
        # current head
        current_head_in_dim = current_dim
        if current_head_arch == "none":
            assert current_emb_size == d_model, f"{current_emb_size} != {d_model}"
            self.current_proj = nn.Linear(current_head_in_dim, current_emb_size)
        elif current_head_arch == "linear":
            assert not current_norm
            self.current_proj = nn.Linear(current_head_in_dim, current_emb_size * 2)
        elif current_head_arch == "2layer":
            assert not current_norm
            self.current_proj = nn.Sequential(
                nn.Linear(current_head_in_dim, current_emb_size * 4),
                nn.ReLU(),
                nn.Linear(current_emb_size * 4, current_emb_size * 2),
            )
        elif current_head_arch == "3layer":
            assert not current_norm
            self.current_proj = nn.Sequential(
                nn.Linear(current_head_in_dim, current_emb_size * 4),
                nn.ReLU(),
                nn.Linear(current_emb_size * 4, current_emb_size * 4),
                nn.ReLU(),
                nn.Linear(current_emb_size * 4, current_emb_size * 2),
            )
        else:
            raise NotImplementedError(f"Unknown current_head_arch: {current_head_arch}")
        if current_norm:
            self.curr_norm = nn.LayerNorm(current_emb_size)
        
        # combined head
        combined_head_in_dim = current_emb_size + mu_size
        if combined_head_arch == "none":
            combined_head_out_dim = mu_size + current_emb_size
        elif combined_head_arch == "linear":
            self.combined_proj = nn.Linear(combined_head_in_dim, combined_emb_size * 2)
            combined_head_out_dim = combined_emb_size
        elif combined_head_arch == "2layer":
            self.combined_proj = nn.Sequential(
                nn.Linear(combined_head_in_dim, combined_emb_size * 4),
                nn.ReLU(),
                nn.Linear(combined_emb_size * 4, combined_emb_size * 2),
            )
            combined_head_out_dim = combined_emb_size
        else:
            raise NotImplementedError(f"Unknown combined_head_arch: {combined_head_arch}")

        # head
        head_in_dim = current_dim if infer_mode == "expert_new" else combined_head_out_dim
        head_out_dim = label_dim * 2 if infer_mode == "res_scale_shift" else label_dim
        if head_arch_version == "ancient":
            self.head = nn.Sequential(
                nn.Linear(head_in_dim, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, head_out_dim)
            )
        elif head_arch_version == "blocked":
            assert num_head_layers >= 3
            head_layers = block(head_in_dim, d_model_head, dropout)
            for _ in range(num_head_layers - 3):
                head_layers += block(d_model_head, d_model_head, dropout)
            head_layers += block(d_model_head, d_model, dropout)
            head_layers += [nn.Linear(d_model, head_out_dim)]
            self.head = nn.Sequential(*head_layers)
        elif head_arch_version == "mlpblock_v1":
            assert num_head_layers >= 3
            head_layers = [MLPBlock(head_in_dim, d_model_head, dropout_head, act_head)]
            for _ in range(num_head_layers - 3):
                head_layers += [MLPBlock(d_model_head, d_model_head, dropout_head, act_head)]
            head_layers += [MLPBlock(d_model_head, d_model, dropout_head, act_head)]
            head_layers += [nn.Linear(d_model, head_out_dim)]
            self.head = nn.Sequential(*head_layers)
        elif head_arch_version == "ppo_expert_arch":
            self.head = nn.Sequential(
                nn.Linear(head_in_dim, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, head_out_dim),
            )
        else:
            raise NotImplementedError(f"Unknown head_arch_version: {head_arch_version}")
        print()
        print("****** Creating Transformer Policy ******")
        print("Policy Config:")
        print(self.policy_cfg)
        print("Model Architecture:")
        print(self)
        print("****** End Transformer Policy ******")
        print()

    def forward(self, context, current, base_actions, padding_mask=None):
        return torch.zeros_like(base_actions)

    def forward_transformer(self, context, padding_mask=None):
        ctx_emb = self.ctx_norm(self.context_proj(context))
        ctx_emb = self.pos_encoder(ctx_emb)
        
        # padding_mask: (Batch, Seq_Len)
        ctx_out = self.transformer(ctx_emb, src_key_padding_mask=padding_mask)
        
        if padding_mask is not None:
            ctx_out = ctx_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            lengths = (~padding_mask).sum(dim=1, keepdim=True)
            ctx_agg = ctx_out.sum(dim=1) / lengths
        else:
            ctx_agg = torch.mean(ctx_out, dim=1)
        
        return ctx_agg
    
    def process_current(self, current):
        if self.policy_cfg["state_type"] == "state":
            return current[:, :215]
        elif self.policy_cfg["state_type"] == "state_baseaction":
            return current[:, :215 + 7]
        else:
            raise NotImplementedError(f"Unknown state_type: {self.policy_cfg['state_type']}")

    def loss(self, context, current, base_actions, expert_actions, padding_mask=None,
            target_mu=None,
            mu_conditioning=None,
            loss_mask=None,
        ):
        loss = 0
        info = {}

        current = self.process_current(current)

        if self.policy_cfg["infer_mode"] == "expert_new":
            new_actions = self.head(current)
        else:
            if self.policy_cfg["force_mu_conditioning"] != "none":
                assert mu_conditioning is not None
                ctx_agg = mu_conditioning
            else:
                ctx_agg = self.forward_transformer(context, padding_mask=padding_mask)

            # mu head
            MU_DIM = self.policy_cfg["mu_size"]
            if self.policy_cfg["mu_head_arch"] == "none":
                mu_emb = ctx_agg
            else:
                mu = self.mu_head(ctx_agg)
                mu_mean = mu[:, :MU_DIM]
                mu_logvar = mu[:, MU_DIM:]
                mu_logvar = torch.clamp(mu_logvar, -20, 20)
                kl_loss = -0.5 * torch.sum(1 + mu_logvar - mu_mean.pow(2) - mu_logvar.exp(), dim=-1).mean(dim=-1)
                loss += self.policy_cfg["mu_kl_factor"] * kl_loss.mean()
                info = info | {
                    "kl_loss_mu": self.policy_cfg["mu_kl_factor"] * kl_loss.detach().cpu().numpy().mean(),
                    "mu_kl_factor": self.policy_cfg["mu_kl_factor"],
                    "mu_logvar_mean": torch.linalg.norm(mu_logvar, dim=-1).detach().cpu().numpy(),
                    "mu_logvar_truemean": torch.mean(mu_logvar, dim=-1).detach().cpu().numpy(),
                    "mu_logvar_std": torch.std(mu_logvar, dim=-1).detach().cpu().numpy(),
                    "mu_mean_mean": torch.linalg.norm(mu_mean, dim=-1).detach().cpu().numpy(),
                    "mu_mean_std": torch.std(mu_mean, dim=-1).detach().cpu().numpy(),
                }
                mu_std = torch.exp(0.5 * mu_logvar)
                eps = torch.randn_like(mu_mean)
                mu_emb = mu_mean + eps * mu_std

            # target mu
            if target_mu is not None:
                N, T_MU = target_mu.shape
                if T_MU < MU_DIM:
                    target_mu = F.pad(target_mu, (0, MU_DIM - T_MU, 0, 0), value=0.0)
                elif T_MU > MU_DIM:
                    raise Exception(f"{T_MU} > {MU_DIM}")
                target_mu_loss = F.mse_loss(mu_emb, target_mu, reduction="none").mean(dim=-1)
                TARGET_MU_MULTIPLIER = 1
                loss += TARGET_MU_MULTIPLIER * target_mu_loss.mean()
                info = info | {
                    "target_mu_loss": TARGET_MU_MULTIPLIER * target_mu_loss.detach().cpu().numpy().mean(),
                    "target_mu_multiplier": TARGET_MU_MULTIPLIER,
                }

            # current head
            curr_emb = self.current_proj(current)
            if self.policy_cfg["current_head_arch"] == "none":
                pass
            elif self.policy_cfg["current_head_arch"] == "linear" or self.policy_cfg["current_head_arch"] == "2layer" or self.policy_cfg["current_head_arch"] == "3layer":
                curr_emb_mean = curr_emb[:, :self.policy_cfg["current_emb_size"]]
                curr_emb_logvar = curr_emb[:, self.policy_cfg["current_emb_size"]:]
                curr_emb_logvar = torch.clamp(curr_emb_logvar, -20, 20)
                kl_loss_curr = -0.5 * torch.sum(1 + curr_emb_logvar - curr_emb_mean.pow(2) - curr_emb_logvar.exp(), dim=-1).mean(dim=-1)
                loss += self.policy_cfg["current_kl_factor"] * kl_loss_curr.mean()
                info = info | {
                    "kl_loss_current": self.policy_cfg["current_kl_factor"] * kl_loss_curr.detach().cpu().numpy().mean(),
                    "current_kl_factor": self.policy_cfg["current_kl_factor"],
                    "curr_emb_logvar_mean": torch.linalg.norm(curr_emb_logvar, dim=-1).detach().cpu().numpy(),
                    "curr_emb_logvar_truemean": torch.mean(curr_emb_logvar, dim=-1).detach().cpu().numpy(),
                    "curr_emb_logvar_std": torch.std(curr_emb_logvar, dim=-1).detach().cpu().numpy(),
                    "curr_emb_mean_mean": torch.linalg.norm(curr_emb_mean, dim=-1).detach().cpu().numpy(),
                    "curr_emb_mean_std": torch.std(curr_emb_mean, dim=-1).detach().cpu().numpy(),
                }
                curr_emb_std = torch.exp(0.5 * curr_emb_logvar)
                curr_eps = torch.randn_like(curr_emb_mean)
                curr_emb = curr_emb_mean + curr_eps * curr_emb_std
            if self.policy_cfg["current_norm"]:
                curr_emb = self.curr_norm(curr_emb)
            
            # combined head
            combined = torch.cat([mu_emb, curr_emb], dim=-1)
            if self.policy_cfg["combined_head_arch"] == "none":
                combined_head_out = combined
            else:
                combined_head_output = self.combined_proj(combined)
                combined_head_mean = combined_head_output[:, :self.policy_cfg["combined_emb_size"]]
                combined_head_logvar = combined_head_output[:, self.policy_cfg["combined_emb_size"]:]
                combined_head_logvar = torch.clamp(combined_head_logvar, -20, 20)
                kl_loss_combined = -0.5 * torch.sum(1 + combined_head_logvar - combined_head_mean.pow(2) - combined_head_logvar.exp(), dim=-1).mean(dim=-1)
                loss += self.policy_cfg["combined_kl_factor"] * kl_loss_combined.mean()
                info = info | {
                    "kl_loss_combined": self.policy_cfg["combined_kl_factor"] * kl_loss_combined.detach().cpu().numpy().mean(),
                    "combined_kl_factor": self.policy_cfg["combined_kl_factor"],
                    "combined_head_logvar_mean": torch.linalg.norm(combined_head_logvar, dim=-1).detach().cpu().numpy(),
                    "combined_head_logvar_truemean": torch.mean(combined_head_logvar, dim=-1).detach().cpu().numpy(),
                    "combined_head_logvar_std": torch.std(combined_head_logvar, dim=-1).detach().cpu().numpy(),
                    "combined_head_mean_mean": torch.linalg.norm(combined_head_mean, dim=-1).detach().cpu().numpy(),
                    "combined_head_mean_std": torch.std(combined_head_mean, dim=-1).detach().cpu().numpy(),
                }
                combined_head_std = torch.exp(0.5 * combined_head_logvar)
                combined_head_eps = torch.randn_like(combined_head_mean)
                combined_head_out = combined_head_mean + combined_head_eps * combined_head_std

            output = self.head(combined_head_out)
            if self.policy_cfg["infer_mode"] == "expert":
                new_actions = output
            elif self.policy_cfg["infer_mode"] == "residual":
                new_actions = base_actions + output
                info = info | {
                    "shift/mean": torch.mean(output, dim=-1).detach().cpu().numpy(),
                    "shift/max": torch.max(output, dim=-1).values.detach().cpu().numpy(),
                    "shift/min": torch.min(output, dim=-1).values.detach().cpu().numpy(),
                    "shift/std": torch.std(output, dim=-1).detach().cpu().numpy(),
                }
            elif self.policy_cfg["infer_mode"] == "res_scale_shift":
                scale = output[:, :self.policy_cfg["label_dim"]]
                shift = output[:, self.policy_cfg["label_dim"]:]
                scale = torch.clamp(scale, -2, 2)
                info = info | {
                    "scale/mean": torch.mean(scale, dim=-1).detach().cpu().numpy(),
                    "scale/max": torch.max(scale, dim=-1).values.detach().cpu().numpy(),
                    "scale/min": torch.min(scale, dim=-1).values.detach().cpu().numpy(),
                    "scale/std": torch.std(scale, dim=-1).detach().cpu().numpy(),
                    "shift/mean": torch.mean(shift, dim=-1).detach().cpu().numpy(),
                    "shift/std": torch.std(shift, dim=-1).detach().cpu().numpy(),
                }
                new_actions = base_actions * torch.exp(scale) + shift
            else:
                raise NotImplementedError(f"Unknown infer_mode: {self.policy_cfg['infer_mode']}")
        
        loss_mse = F.mse_loss(new_actions, expert_actions, reduction="none").mean(dim=-1)
        loss_mse *= loss_mask
        loss += loss_mse.mean()
        info = info | {
            "loss_mse": loss_mse.detach().cpu().numpy(),
            "loss": loss.item(),
        }
        
        return loss, info

    def get_action(self, context, current, base_actions, padding_mask=None,
                mu_conditioning=None,
            ):
        with torch.no_grad():
            current = self.process_current(current)

            if self.policy_cfg["infer_mode"] == "expert_new":
                new_actions = self.head(current)
            else:
                if self.policy_cfg["force_mu_conditioning"] != "none":
                    assert mu_conditioning is not None
                    ctx_agg = mu_conditioning
                else:
                    ctx_agg = self.forward_transformer(context, padding_mask=padding_mask)
                
                # Mu head
                if self.policy_cfg["mu_head_arch"] == "none":
                    mu_emb = ctx_agg
                else:
                    mu = self.mu_head(ctx_agg)
                    mu_mean = mu[:, :self.policy_cfg["mu_size"]]
                    mu_emb = mu_mean
                    
                # head stuff
                curr_emb = self.current_proj(current)
                if self.policy_cfg["current_head_arch"] == "none":
                    pass
                elif self.policy_cfg["current_head_arch"] == "linear" or self.policy_cfg["current_head_arch"] == "2layer" or self.policy_cfg["current_head_arch"] == "3layer":
                    curr_emb_mean = curr_emb[:, :self.policy_cfg["current_emb_size"]]
                    curr_emb = curr_emb_mean
                if self.policy_cfg["current_norm"]:
                    curr_emb = self.curr_norm(curr_emb)

                combined = torch.cat([mu_emb, curr_emb], dim=-1)
                # combined head
                if self.policy_cfg["combined_head_arch"] == "none":
                    combined_head_out = combined
                else:
                    combined_head_output = self.combined_proj(combined)
                    combined_head_mean = combined_head_output[:, :self.policy_cfg["combined_emb_size"]]
                    combined_head_out = combined_head_mean

                output = self.head(combined_head_out)
                if self.policy_cfg["infer_mode"] == "expert":
                    new_actions = output
                elif self.policy_cfg["infer_mode"] == "residual":
                    new_actions = base_actions + output
                elif self.policy_cfg["infer_mode"] == "res_scale_shift":
                    scale = output[:, :self.policy_cfg["label_dim"]]
                    shift = output[:, self.policy_cfg["label_dim"]:]
                    new_actions = base_actions * torch.exp(scale) + shift
                else:
                    raise NotImplementedError(f"Unknown infer_mode: {self.policy_cfg['infer_mode']}")
        return new_actions


class ProcessedRobotTransformerPolicy(nn.Module):
    def __init__(self, save_path: str | pathlib.Path, device: str = "cpu", our_task: str = "peg"):
        super().__init__()
        print(f"Loading {save_path}")
        self.device = torch.device(device)
        self.save_path = str(save_path)

        if self.save_path == "expert":
            # Load the RL Expert
            self.model, _ = expert_utils.load_expert_by_task(our_task, device=device)
            
            # Alias head to actor so LoRA code works
            self.head = self.model.actor

            # Compatibility save_dict with identity means/stds
            self.save_dict = {
                "overall_arch": "expert",
                "context_dim": 1,
                "current_dim": 215,
                "label_dim": 7,
                "infer_mode": "expert",
                "state_type": "expert_raw",
                "context_means": np.zeros((43 + 7,)),
                "context_stds":  np.ones((43 + 7,)),
                "current_means": np.zeros((215 + 7,)),
                "current_stds":  np.ones((215 + 7,)),
                "label_means":   np.zeros((7,)),
                "label_stds":    np.ones((7,)),
            }
            
            # Register buffers normally from save_dict
            self.register_buffer("context_means", torch.as_tensor(self.save_dict["context_means"], device=self.device, dtype=torch.float32))
            self.register_buffer("context_stds",  torch.as_tensor(self.save_dict["context_stds"],  device=self.device, dtype=torch.float32))
            self.register_buffer("current_means", torch.as_tensor(self.save_dict["current_means"], device=self.device, dtype=torch.float32))
            self.register_buffer("current_stds",  torch.as_tensor(self.save_dict["current_stds"],  device=self.device, dtype=torch.float32))
            self.register_buffer("label_means",   torch.as_tensor(self.save_dict["label_means"],   device=self.device, dtype=torch.float32))
            self.register_buffer("label_stds",    torch.as_tensor(self.save_dict["label_stds"],    device=self.device, dtype=torch.float32))
            
            self.to(self.device).eval()
            return # Exit early for expert policies

        # --- Standard Transformer Loading Logic ---
        save_path = self.save_path
        info_path = os.path.join(os.path.dirname(save_path), "info.pkl")
        with open(info_path, "rb") as fi:
            save_dict = pickle.load(fi)

        save_dict = {
            "overall_arch": "robot_transformer",
            "current_means": np.zeros((save_dict["current_dim"],)),
            "current_stds": np.ones((save_dict["current_dim"],)),
            "context_means": np.zeros((save_dict["context_dim"],)),
            "context_stds": np.ones((save_dict["context_dim"],)),
            "label_means": np.zeros((save_dict["label_dim"],)),
            "label_stds": np.ones((save_dict["label_dim"],)),
            "train_expert": False,
            "use_new_head_arch": False,
            "num_head_layers": 2,
            "d_model_head": 1024,
            "dropout_head": 0.0,
            "mu_head_arch": "none",
            "mu_size": 512,
            "mu_kl_factor": 0.0,
            "current_norm": True,
            "current_head_arch": "none",
            "current_emb_size": 512,
            "current_kl_factor": 0.0,
            "combined_head_arch": "none",
            "combined_emb_size": 512,
            "combined_kl_factor": 0.0,
            "force_mu_conditioning": "none",
            "force_mu_conditioning_size": 2,
            "act_head": "relu",
            "sys_noise": np.zeros((7,)),
            "obs_receptive_noise": np.zeros((6,)),
            "obs_insertive_noise": np.zeros((6,)),
        } | save_dict

        REQUIRED_LENGTH = 215 + 7 if save_dict["overall_arch"] == "robot_transformer" else 215
        if save_dict['current_means'].shape[0] < REQUIRED_LENGTH:
            save_dict["current_means"] = np.concatenate([save_dict["current_means"], np.zeros((REQUIRED_LENGTH - save_dict['current_means'].shape[0],))], axis=0)
            save_dict["current_stds"] = np.concatenate([save_dict["current_stds"], np.ones((REQUIRED_LENGTH - save_dict['current_stds'].shape[0],))], axis=0)
        
        save_dict = {
            "state_type": "state",
            "infer_mode": "res_scale_shift" if "scale" in save_path else ("expert" if save_dict["train_expert"] else "residual"),
            "head_arch_version": "blocked" if save_dict["use_new_head_arch"] else "ancient",
        } | save_dict
        self.save_dict = save_dict

        if save_dict["overall_arch"] == "robot_transformer":
            self.model = RobotTransformerPolicy(
                save_dict["context_dim"], save_dict["current_dim"], save_dict["label_dim"],
                num_layers=save_dict["num_layers"], d_model=save_dict["d_model"],
                dropout=save_dict["dropout"], head_arch_version=save_dict["head_arch_version"],
                num_head_layers=save_dict["num_head_layers"], d_model_head=save_dict["d_model_head"],
                infer_mode=save_dict["infer_mode"], dropout_head=save_dict["dropout_head"],
                act_head=save_dict["act_head"], mu_head_arch=save_dict["mu_head_arch"],
                mu_size=save_dict["mu_size"], mu_kl_factor=save_dict["mu_kl_factor"],
                current_norm=save_dict["current_norm"], current_head_arch=save_dict["current_head_arch"],
                current_emb_size=save_dict["current_emb_size"], current_kl_factor=save_dict["current_kl_factor"],
                combined_head_arch=save_dict["combined_head_arch"], combined_emb_size=save_dict["combined_emb_size"],
                combined_kl_factor=save_dict["combined_kl_factor"], state_type=save_dict["state_type"],
                force_mu_conditioning=save_dict["force_mu_conditioning"],
                force_mu_conditioning_size=save_dict["force_mu_conditioning_size"],
            )
            self.head = self.model.head
        elif save_dict["overall_arch"] == "expert":
            self.model, _ = expert_utils.load_expert_by_task(our_task, device=device)
            self.head = self.model.actor

        state = torch.load(save_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)

        def reg(name, arr, shape):
            t = torch.as_tensor(arr, device=self.device, dtype=torch.float32).view(*shape)
            self.register_buffer(name, t)

        reg("context_means", save_dict["context_means"], (1, 1, -1))
        reg("context_stds",  save_dict["context_stds"],  (1, 1, -1))
        reg("current_means", save_dict["current_means"], (1, -1))
        reg("current_stds",  save_dict["current_stds"],  (1, -1))
        reg("label_means",   save_dict["label_means"],   (1, -1))
        reg("label_stds",    save_dict["label_stds"],    (1, -1))

        print()
        print(f"Loaded {save_path} !!!")
        print(f"infer_mode: {self.save_dict['infer_mode']}")
        print(f"context_means: {self.context_means}")
        print(f"context_stds: {self.context_stds}")
        print(f"current_means: {self.current_means}")
        print(f"current_stds: {self.current_stds}")
        print(f"label_means: {self.label_means}")
        print(f"label_stds: {self.label_stds}")
        print()

        self.to(self.device).eval()

    @torch.no_grad()
    def forward(self, context, current, base_actions, padding_mask=None):
        return self.get_action(context, current, base_actions, padding_mask=padding_mask)
    
    def get_action(self, context, current, base_actions, padding_mask=None, mu_conditioning=None):
        context = torch.as_tensor(context, device=self.device, dtype=torch.float32)
        current = torch.as_tensor(current, device=self.device, dtype=torch.float32)
        base_actions = torch.as_tensor(base_actions, device=self.device, dtype=torch.float32)

        eps = 1e-8
        context_n = (context - self.context_means) / self.context_stds.clamp_min(eps)
        current_n = (current - self.current_means) / self.current_stds.clamp_min(eps)
        if self.save_dict["infer_mode"] == "res_scale_shift":
            base_actions_n = (base_actions - self.label_means) / self.label_stds.clamp_min(eps)
        else:
            base_actions_n = base_actions / self.label_stds.clamp_min(eps)

        if self.save_dict["overall_arch"] == "expert":
            # Extract RL obs from the end of the state vector
            expert_obs = current_n[:, :215]
            out_n = self.model(expert_obs)
        else:
            out_n = self.model.get_action(context_n, current_n, base_actions_n, padding_mask=padding_mask, mu_conditioning=mu_conditioning)
        
        out = out_n * self.label_stds + self.label_means
        return out

    def loss(self, context, current, base_actions, expert_actions, padding_mask=None, target_mu=None, mu_conditioning=None, loss_mask=None):
        context = torch.as_tensor(context, device=self.device, dtype=torch.float32)
        current = torch.as_tensor(current, device=self.device, dtype=torch.float32)
        base_actions = torch.as_tensor(base_actions, device=self.device, dtype=torch.float32)
        expert_actions = torch.as_tensor(expert_actions, device=self.device, dtype=torch.float32)

        eps = 1e-8
        context_n = (context - self.context_means) / self.context_stds.clamp_min(eps)
        current_n = (current - self.current_means) / self.current_stds.clamp_min(eps)
        if self.save_dict["infer_mode"] == "res_scale_shift":
            base_actions_n = (base_actions - self.label_means) / self.label_stds.clamp_min(eps)
        else:
            base_actions_n = base_actions / self.label_stds.clamp_min(eps)
        
        expert_actions_n = (expert_actions - self.label_means) / self.label_stds.clamp_min(eps)

        if self.save_dict["overall_arch"] == "expert":
            expert_obs = current_n[:, :215]
            new_actions = self.model(expert_obs)
            loss_mse = F.mse_loss(new_actions, expert_actions_n)
            return loss_mse, {"loss_mse": loss_mse.detach().cpu().numpy(), "loss": loss_mse.item()}

        return self.model.loss(context_n, current_n, base_actions_n, expert_actions_n, padding_mask=padding_mask, target_mu=target_mu, mu_conditioning=mu_conditioning, loss_mask=loss_mask)


def load_robot_policy(save_path: str | pathlib.Path, device="cpu", our_task="peg"):
    model = ProcessedRobotTransformerPolicy(save_path, device=device, our_task=our_task)
    model.eval()
    return model, model.save_dict

