from functools import partial
import math
import os
from pathlib import Path
import torch
from torch import nn
from .utils import center_padding, fit_to_patch, tokens_to_output
import logging
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as tf

VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
    'vit_gigantic': 1664,
}

logger = logging.getLogger()


class VJEPA2(nn.Module):
    def __init__(
        self,
        arch="vit_large",
        output="dense",
        layer=-1,
        return_multilayer=False,
        mode="img_resize",
        num_frames=1,
    ):
        """
        VJEPA2 model wrapper for depth estimation experiments.
        
        Args:
            arch: Model architecture size (vit_large, vit_huge, vit_giant)
            output: Output format ("dense", "dense-temporal")
            layer: Which layer to extract features from (-1 for last layer)
            return_multilayer: Whether to return features from multiple layers
            mode: Processing mode (same as VJEPA: "video_resize", "video", "img", "img_resize", "video_cutpos", "video_resize_cutpos")
            num_frames: Number of frames (1 for single image)
        """
        super().__init__()
        assert output in ["dense", "dense-temporal"], "Options: [dense, dense-temporal]"
        self.output = output
        self.checkpoint_name = f"vjepa2_{arch}"
        
        # Map architectures to PyTorch Hub model names
        hub_model_mapping = {
            "vit_large": "vjepa2_vit_large",
            "vit_huge": "vjepa2_vit_huge", 
            "vit_giant": "vjepa2_vit_giant",
            "vit_giant_384": "vjepa2_vit_giant_384",
        }
        
        assert arch in hub_model_mapping, f"Architecture {arch} not supported. Available: {list(hub_model_mapping.keys())}"
        
        hub_model_name = hub_model_mapping[arch]
        
        # Load model from PyTorch Hub
        hub_result = torch.hub.load('facebookresearch/vjepa2', hub_model_name, trust_repo=True)
        
        # Handle different return formats from PyTorch Hub
        if isinstance(hub_result, tuple):
            # VJEPA2 returns (encoder, predictor) tuple
            self.vit = hub_result[0]  # Use encoder
            self.predictor = hub_result[1] if len(hub_result) > 1 else None
        else:
            self.vit = hub_result
            
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad = False

        # Set header parameters based on architecture
        self.patch_size = 16
        self.image_size = [256, 256]
        self.num_frames = num_frames
        self.tubelet_size = 2

        if arch == "vit_giant_384":
            self.image_size = [384, 384]
            

        self.name = f"{self.checkpoint_name}_{self.num_frames}_{self.tubelet_size}"
        feat_dim = self.vit.embed_dim

        
        if num_frames == 1:
            #This is single image case, which means we have to adapt embedding projection
            original_projection = self.vit.patch_embed.proj
            original_weights = original_projection.weight
            new_weights = original_weights.sum(dim=2, keepdim=True) #sum temporal kernel dimension to get 2d kernel
            new_projection = torch.nn.Conv3d(
                in_channels=3,
                out_channels=original_projection.out_channels,
                kernel_size=(1, original_projection.kernel_size[1], original_projection.kernel_size[2]),
                stride=(1, original_projection.stride[1], original_projection.stride[2])
            )

            new_projection.weight.data = new_weights
            if original_projection.bias is not None:
                new_projection.bias.data = original_projection.bias.data.clone()

            self.vit.patch_embed.proj = new_projection
            self.tubelet_size=1

        elif num_frames > 1:
            assert num_frames % self.tubelet_size == 0

        # Setup multilayer extraction
        num_layers = len(self.vit.blocks)
            
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]
        
        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
            self.vit.out_layers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        self.layer = "-".join(str(_x) for _x in self.multilayers)
        assert mode in ["video_resize", "video", "img", "img_resize",
                        "video_cutpos", "video_resize_cutpos"], "Invalid mode"
        self.mode = mode
        
    def preprocess(self, x):
        if x.ndim == 4:
            B, C, H, W = x.shape
            # Preprocess to [B,C,T,H,W]
            if self.mode in ["img_resize", "video_resize", "video_resize_cutpos"]:
                x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)
            x, hw = fit_to_patch(x, self.patch_size)
            x = x.unsqueeze(2)  # inserts new dimension at index 2
            x = x.expand(-1, -1, self.num_frames, -1, -1)  # [B,C,T,H,W] example [B,3,16,224,224]
        else:
            B, T, C, H, W = x.shape
            # Swap T and C dimensions to match expected format [B,C,T,H,W]
            x = x.permute(0, 2, 1, 3, 4)
            if self.mode in ["video_resize", "video_resize_cutpos"]:
                # Reshape for interpolation: [B*T, C, H, W]
                BT, C, H, W = x.view(-1, C, H, W).shape
                x = F.interpolate(x.view(BT, C, H, W), size=self.image_size, mode="bilinear", align_corners=False)
                x = x.view(B, T, C, self.image_size[0], self.image_size[1]).permute(0, 2, 1, 3, 4)
            x, hw = fit_to_patch(x, self.patch_size)
        return x


    def forward(self, x):
        # Preprocess input - this already converts to [B, C, T, H, W] format
        x = self.preprocess(x)
        # After preprocessing, videos should be in [B, C, T, H, W] format
        assert x.dim() == 5, f"Expected 5D input after preprocessing, got {x.dim()}D"
        B, C, T, H, W = x.shape
        T = T // self.tubelet_size
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        if not self.vit.handle_nonsquare_inputs:
            T = H_patches = W_patches = None

        if not self.vit.use_rope:
            pos_embed = self.vit.interpolate_pos_encoding(x, self.vit.pos_embed)
            x = self.vit.patch_embed(x)
            x += pos_embed
        else:
            x = self.vit.patch_embed(x)

        # Fwd prop
        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            if self.vit.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, None, None, T=T, H_patches=H_patches, W_patches=W_patches, use_reentrant=False
                )
            else:
                x = blk(x, mask=None, attn_mask=None, T=T, H_patches=H_patches, W_patches=W_patches)
            if self.multilayers is not None and i in self.multilayers:
                embeds.append(self.vit.norm(x))
            if len(embeds) == len(self.multilayers):
                break

        outputs = []
        for i, x_i in enumerate(embeds):
            if self.output == "dense-temporal":
                tokens_for_output = x_i
            else:
                # For dense output, only take spatial tokens (ignore temporal dimension)
                tokens_for_output = x_i[:, : H_patches * W_patches]
            
            x_i = tokens_to_output(
                self.output,
                tokens_for_output,
                None,
                (H_patches, W_patches),
            )
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs


def load_pretrained(encoder, pretrained, checkpoint_key='target_encoder'):
    """Load pretrained weights - kept for compatibility"""
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    
    pretrained_dict = checkpoint.get(checkpoint_key, checkpoint.get('encoder'))
    assert pretrained_dict is not None, f"Could not find weights under key '{checkpoint_key}' or 'encoder'"

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder 