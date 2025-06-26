from __future__ import annotations

from pathlib import Path
import logging
import torch
from torch import nn
import os
from omegaconf import OmegaConf
import urllib.request
from huggingface_hub import hf_hub_download
from hydra.utils import instantiate

class SAM2(nn.Module):
    def __init__(
        self,
        arch="hiera-base-plus",
        output="dense",
        layer=-1,
        return_multilayer=False,
        mode="original",
        use_trunk_features=False,
    ):
        super().__init__()

        assert output in ["gap", "dense"], "Options: [gap, dense]"
        self.output = output
        
        # Set HuggingFace token for authentication
        #TODO: add hf
        
        # Map architecture to HuggingFace model IDs and config files (using absolute paths)
        sam2_package_path = "/home/stud/nguyenti/anaconda3/envs/probe3d_2/lib/python3.10/site-packages/sam2"
        
        model_info = {
            "hiera-tiny": ("facebook/sam2.1-hiera-tiny", f"{sam2_package_path}/configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
            "hiera-small": ("facebook/sam2.1-hiera-small", f"{sam2_package_path}/configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"), 
            "hiera-base-plus": ("facebook/sam2.1-hiera-base-plus", f"{sam2_package_path}/configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
            "hiera-large": ("facebook/sam2.1-hiera-large", f"{sam2_package_path}/configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
        }

        model_id, config_path, checkpoint_name = model_info[arch]
        
        # Download checkpoint from HuggingFace
        ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
        
        # Verify config file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        # Load and resolve config
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        
        # Build model directly using Hydra instantiate
        model = instantiate(cfg.model, _recursive_=True)
        
        # Load checkpoint
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logging.error(f"Unexpected keys: {unexpected_keys}")
        
        model = model.to(device)
        model.eval()
        print(model)
        
        self.encoder = model.image_encoder
        
        # Add model name for logging
        self.name = f"SAM2-{arch}"
        
        # Add checkpoint name for logging
        self.checkpoint_name = f"sam2.1_{arch}"

        # Get model dimensions from the neck's d_model (output feature dimension)
        feat_dim = self.encoder.neck.d_model
        
        # Store the trunk feature option
        self.use_trunk_features = use_trunk_features
        
        # Add projection layer for trunk features if needed
        if use_trunk_features:
            trunk_dim = 896  # Last trunk layer has 896 channels
            self.trunk_proj = nn.Conv2d(trunk_dim, feat_dim, kernel_size=1, stride=1, padding=0)
            # Initialize the projection layer properly
            nn.init.xavier_uniform_(self.trunk_proj.weight)
            nn.init.zeros_(self.trunk_proj.bias)
        
        self.patch_size = 16  # SAM2 uses 16x16 patches
        self.image_size = (1024, 1024)  # SAM2 uses 1024x1024 images

        # Configure multilayers based on feature type
        if use_trunk_features:
            # Use last 4 trunk layers
            trunk_multilayers = [-4, -3, -2, -1]  # Last 4 layers of trunk
            self.trunk_multilayers = trunk_multilayers
            total_multilayers = len(trunk_multilayers)
        else:
            # Use FPN features (current default behavior)
            fpn_multilayers = list(range(3))  # [0, 1, 2] - use all 3 FPN features
            self.fpn_multilayers = fpn_multilayers
            total_multilayers = len(fpn_multilayers)

        if return_multilayer:
            self.feat_dim = [feat_dim] * total_multilayers
            self.multilayers = list(range(total_multilayers))  # Just indices for logging
        else:
            self.feat_dim = feat_dim
            self.multilayers = [total_multilayers - 1]  # Use the last feature

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)
        assert mode in ["original", "resize"], "Options: [original, resize]"
        self.mode = mode


    def forward(self, x):
        _, _, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0, f"{h}, {w}"

        if h != self.image_size[0] or w != self.image_size[1]:
            if self.mode == "resize":
                x = torch.nn.functional.interpolate(
                    x, self.image_size, mode="bicubic"
                )
                h, w = self.image_size
            # For "original" mode, SAM2 handles dynamic resolution internally

        if self.use_trunk_features:
            # Extract features from trunk (last 4 layers)
            trunk_features = self.encoder.trunk(x)  # List of features from different trunk layers
            
            # Extract features from specified trunk layers
            embeds = []
            for layer_idx in self.trunk_multilayers:
                if layer_idx < 0:
                    # Negative indexing for last layers
                    trunk_feat = trunk_features[layer_idx]
                    embeds.append(trunk_feat)
                else:
                    trunk_feat = trunk_features[layer_idx]
                    embeds.append(trunk_feat)
        else:
            # Forward through SAM2 ImageEncoder to get FPN features (default behavior)
            encoder_output = self.encoder(x)
            
            # Extract multi-scale features from backbone_fpn (3 features)
            fpn_features = encoder_output["backbone_fpn"]  # List of features at different scales
            
            # Select features from specified FPN layers
            embeds = []
            for layer_idx in self.fpn_multilayers:
                if layer_idx < len(fpn_features):
                    embeds.append(fpn_features[layer_idx])
                else:
                    # Fallback to the highest resolution feature if layer_idx is out of range
                    embeds.append(fpn_features[-1])
        
        # Select only the requested multilayers
        if len(self.multilayers) < len(embeds):
            embeds = [embeds[i] for i in self.multilayers if i < len(embeds)]

        if self.output == "gap":
            embeds = [feat.mean(dim=(-2, -1)) for feat in embeds]
        return embeds[0] if len(embeds) == 1 else embeds 