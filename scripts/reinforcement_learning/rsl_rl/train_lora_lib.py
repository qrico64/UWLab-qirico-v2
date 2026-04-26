import os
import copy
from typing import Optional, Dict, Any
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
import cur_utils
from train_lib import RobotTransformerPolicy, ProcessedRobotTransformerPolicy

class LoRALinear(nn.Module):
    """
    Standard LoRA wrapper for nn.Linear.

    Common conventions:
      - delta(x) = (dropout(x) @ A^T @ B^T) * (alpha/r)
      - A: (r, in_features) ~ N(0, 0.01)
      - B: (out_features, r) = 0  => starts as no-op
      - base weights frozen
      - supports merge/unmerge into base weight
    """
    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_std: float = 0.01,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = (self.alpha / self.r) if self.r > 0 else 0.0
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.merged = False

        # Freeze base params
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        if self.r > 0:
            in_dim = base.in_features
            out_dim = base.out_features
            # Get device and dtype from base layer
            device = base.weight.device
            dtype = base.weight.dtype
            # A: (r, in), B: (out, r)
            self.A = nn.Parameter(torch.empty(self.r, in_dim, device=device, dtype=dtype))
            self.B = nn.Parameter(torch.zeros(out_dim, self.r, device=device, dtype=dtype))
            # Common init: A small normal, B zeros -> initial delta = 0
            nn.init.normal_(self.A, mean=0.0, std=init_std)
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    @torch.no_grad()
    def _delta_weight(self) -> torch.Tensor:
        """
        Returns deltaW in the same shape as base.weight: (out_features, in_features)
        deltaW = (B @ A) * scaling
        """
        if self.r <= 0:
            return torch.zeros_like(self.base.weight)
        return (self.B @ self.A) * self.scaling

    @torch.no_grad()
    def merge(self):
        """
        Merge LoRA weights into base.weight and mark merged.
        After merge(), forward is identical but LoRA branch is not applied separately.
        """
        if self.merged or self.r <= 0:
            self.merged = True
            return
        self.base.weight.add_(self._delta_weight())
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        """
        Revert merge() by subtracting deltaW from base.weight.
        """
        if (not self.merged) or self.r <= 0:
            self.merged = False
            return
        self.base.weight.sub_(self._delta_weight())
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0 and not self.merged:
            x_d = self.lora_dropout(x)
            # (batch, in) @ (in, r) -> (batch, r), then @ (r, out) -> (batch, out)
            delta = (x_d @ self.A.t()) @ self.B.t()
            y = y + delta * self.scaling
        return y


def apply_lora(model: nn.Module, r=8, alpha=16.0, lora_dropout=0.0, init_std=0.01):
    """
    In-place: replace nn.Linear layers inside model with LoRALinear.
    """
    def _replace(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=lora_dropout, init_std=init_std))
            else:
                _replace(child)

    _replace(model)
    return model


def freeze_all_but_lora(model: nn.Module):
    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)
    # Unfreeze only LoRA A/B
    for m in model.modules():
        if isinstance(m, LoRALinear) and m.r > 0:
            m.A.requires_grad_(True)
            m.B.requires_grad_(True)
    return model





def _infer_policy_cfg_from_model(m: nn.Module) -> dict:
    """
    Best-effort fallback when m.policy_cfg doesn't exist.
    Works for RobotTransformerPolicy where linears may be LoRALinear-wrapped.
    """
    # context_dim / current_dim / d_model from projections
    def _in_features(layer):
        return layer.base.in_features if isinstance(layer, LoRALinear) else layer.in_features

    def _out_features(layer):
        return layer.base.out_features if isinstance(layer, LoRALinear) else layer.out_features

    context_dim = _in_features(m.context_proj)
    current_dim = _in_features(m.current_proj)
    d_model = _out_features(m.context_proj)

    # label_dim from final Linear in head
    last_linear = None
    for mod in reversed(list(m.head.modules())):
        if isinstance(mod, (nn.Linear, LoRALinear)):
            last_linear = mod
            break
    if last_linear is None:
        raise ValueError("Could not infer label_dim: no Linear found in m.head.")
    label_dim = _out_features(last_linear)

    # nhead / num_layers / dropout from transformer
    num_layers = len(getattr(m.transformer, "layers", []))
    if num_layers <= 0:
        raise ValueError("Could not infer num_layers from m.transformer.layers.")

    first_layer = m.transformer.layers[0]
    nhead = int(first_layer.self_attn.num_heads)
    dropout = float(getattr(first_layer, "dropout", 0.0))

    # Try to infer head style
    use_new_head_arch = getattr(m, "policy_cfg", {}).get("use_new_head_arch", False)
    num_head_layers = getattr(m, "policy_cfg", {}).get("num_head_layers", 3)
    d_model_head = getattr(m, "policy_cfg", {}).get("d_model_head", 1024)

    return dict(
        context_dim=int(context_dim),
        current_dim=int(current_dim),
        label_dim=int(label_dim),
        nhead=int(nhead),
        num_layers=int(num_layers),
        d_model=int(d_model),
        dropout=float(dropout),
        use_new_head_arch=bool(use_new_head_arch),
        num_head_layers=int(num_head_layers),
        d_model_head=int(d_model_head),
    )

def convert_lora_model_to_plain_robot_policy(
    lora_model: RobotTransformerPolicy,
    *,
    device=None,
) -> RobotTransformerPolicy:
    """
    Create a NEW plain RobotTransformerPolicy (no LoRA modules) and load weights
    equivalent to the LoRA model with merged deltas, WITHOUT mutating lora_model.

    Requirements:
      - lora_model is a RobotTransformerPolicy whose nn.Linear may be replaced by LoRALinear.
      - lora_model should ideally have lora_model.policy_cfg (recommended).
    """
    # if not isinstance(lora_model, RobotTransformerPolicy):
    #     raise TypeError(f"Expected lora_model to be RobotTransformerPolicy, got {type(lora_model)}")

    # 1) Get architecture from model (prefer saved config)
    save_dict = dict(lora_model.policy_cfg)

    # 2) Construct fresh plain policy
    plain = RobotTransformerPolicy(
        context_dim=save_dict["context_dim"],
        current_dim=save_dict["current_dim"],
        label_dim=save_dict["label_dim"],
        nhead=save_dict["nhead"],
        num_layers=save_dict["num_layers"],
        d_model=save_dict["d_model"],
        dropout=save_dict["dropout"],
        head_arch_version=save_dict["head_arch_version"],
        num_head_layers=save_dict["num_head_layers"],
        d_model_head=save_dict["d_model_head"],
        infer_mode=save_dict["infer_mode"],
        dropout_head=save_dict["dropout_head"],
        act_head=save_dict["act_head"],
        mu_head_arch=save_dict["mu_head_arch"],
        mu_size=save_dict["mu_size"],
        mu_kl_factor=save_dict["mu_kl_factor"],
        current_head_arch=save_dict["current_head_arch"],
        current_emb_size=save_dict["current_emb_size"],
        current_kl_factor=save_dict["current_kl_factor"],
        combined_head_arch=save_dict["combined_head_arch"],
        combined_emb_size=save_dict["combined_emb_size"],
        combined_kl_factor=save_dict["combined_kl_factor"],
        state_type=save_dict["state_type"],
        force_mu_conditioning=save_dict["force_mu_conditioning"],
        force_mu_conditioning_size=save_dict["force_mu_conditioning_size"],
    )

    if device is not None:
        plain = plain.to(device)

    target_device = next(plain.parameters()).device

    # 3) Load non-LoRA weights/buffers where possible (handles base.* keys)
    lora_sd = lora_model.state_dict()
    plain_sd = plain.state_dict()
    new_sd = {}

    for k, v in plain_sd.items():
        if k in lora_sd and lora_sd[k].shape == v.shape:
            new_sd[k] = lora_sd[k].to(device=target_device, dtype=v.dtype)
            continue

        k_base = k.replace(".weight", ".base.weight").replace(".bias", ".base.bias")
        if k_base in lora_sd and lora_sd[k_base].shape == v.shape:
            new_sd[k] = lora_sd[k_base].to(device=target_device, dtype=v.dtype)
            continue

        # keep init
        new_sd[k] = v

    plain.load_state_dict(new_sd, strict=True)

    # 4) Patch merged weights for LoRALinear modules (no mutation to lora_model)
    lora_modules = dict(lora_model.named_modules())
    plain_modules = dict(plain.named_modules())

    with torch.no_grad():
        for name, mod in lora_modules.items():
            if not isinstance(mod, LoRALinear):
                continue

            if name not in plain_modules:
                raise KeyError(
                    f"LoRA module '{name}' exists in lora_model but not in plain model. "
                    f"Architectures likely differ."
                )

            dest = plain_modules[name]
            if not isinstance(dest, nn.Linear):
                raise TypeError(f"Expected plain module '{name}' to be nn.Linear, got {type(dest)}")

            base_w = mod.base.weight.detach()
            if mod.r > 0:
                delta_w = (mod.B.detach() @ mod.A.detach()) * mod.scaling
                merged_w = base_w + delta_w
            else:
                merged_w = base_w

            dest.weight.copy_(merged_w.to(device=target_device, dtype=dest.weight.dtype))

            if dest.bias is not None and mod.base.bias is not None:
                dest.bias.copy_(mod.base.bias.detach().to(device=target_device, dtype=dest.bias.dtype))

    return plain








@torch.no_grad()
def verify_lora_conversion_from_model(lora_model: RobotTransformerPolicy):
    """
    Verifies LoRA conversion consistency using an existing model instance.
    Uses inherent attributes to avoid manual parameter passing.
    """
    print("\n--- Verifying LoRA Conversion (No Mutation) ---")
    if isinstance(lora_model, ProcessedRobotTransformerPolicy):
        lora_model = lora_model.model
    
    # 1. Determine device from the existing model
    device = next(lora_model.parameters()).device
    
    # 2. Extract dimensions from inherent policy_cfg
    cfg = lora_model.policy_cfg
    batch_size = 2
    seq_len = 4
    
    # 3. Generate dummy input based on model attributes
    ctx = torch.randn(batch_size, seq_len, cfg["context_dim"]).to(device)
    curr = torch.randn(batch_size, cfg["current_dim"]).to(device)
    basea = torch.zeros(batch_size, cfg["label_dim"]).to(device)
    
    # 4. Get output from LoRA model (Ensure eval mode for consistency)
    was_training = lora_model.training
    lora_model.eval()
    with torch.no_grad():
        output_lora = lora_model(ctx, curr, basea)
    
    # 5. Convert to a NEW plain model (this function handles the logic)
    # Your toolkit's function creates a 'plain' copy and performs the math
    plain_model = convert_lora_model_to_plain_robot_policy(lora_model, device=device)
    plain_model.eval()
    
    # 6. Get output from the reconstructed plain model
    with torch.no_grad():
        output_plain = plain_model(ctx, curr, basea)
    
    # 7. Compare outputs
    # We use a small epsilon for float32 precision limits
    max_diff = torch.abs(output_lora - output_plain).max().item()
    mean_diff = torch.abs(output_lora - output_plain).mean().item()
    
    print(f"Results for model architecture: {cfg['d_model']} d_model, {cfg['num_layers']} layers")
    print(f"  Max Diff:  {max_diff:.2e}")
    print(f"  Mean Diff: {mean_diff:.2e}")
    
    # Restore original training state
    if was_training:
        lora_model.train()
        
    if max_diff < 1e-5:
        print("✅ SUCCESS: Converted model output matches LoRA model output.")
        return True
    else:
        print("❌ FAILURE: Discrepancy found. Check scaling (alpha/r) or weight merging logic.")
        return False
