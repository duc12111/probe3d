from __future__ import annotations
import math
import os

import numpy as np
import torch
from torch import nn
from transformers import VideoMAEForPreTraining
import torch.nn.functional as F
from transformers.models.videomae.modeling_videomae import get_sinusoid_encoding_table



from .utils import tokens_to_output, fit_to_patch


class VideoMAE(nn.Module):
    def __init__(
        self,
        checkpoint="MCG-NJU/videomae-base",
        output="dense-temperal",
        layer=-1,
        return_multilayer=False,
        num_frames=1,
        mode="img_resize"
    ):
        """Code based on transformer database"""
        super().__init__()

        assert output in [ "dense", "dense-temporal"], "Options: [dense, dense-temporal]"
        self.output = output

        self.checkpoint_name = checkpoint.split("/")[1]

        self.video_mae = VideoMAEForPreTraining.from_pretrained(checkpoint).videomae
        self.video_mae = self.video_mae.eval()


        # Set header parameters
        patch_size = self.video_mae.config.patch_size
        self.patch_size = patch_size
        self.layer = layer

        self.tubelet_size = self.video_mae.embeddings.patch_embeddings.tubelet_size
        self.image_size = self.video_mae.embeddings.patch_embeddings.image_size
        # By default, we interpolate the pos_embed to the correct size.
        assert mode in ["video_resize", "video", "img", "img_resize",
                        "video_cutpos","video_resize_cutpos"], "Invalid mode"
        self.mode = mode
        self.num_frames = num_frames

        if num_frames == 1:
            #This is single image case, which means we have to adapt embedding projection
            original_projection = self.video_mae.embeddings.patch_embeddings.projection
            original_weights = original_projection.weight
            new_weights = original_weights.sum(dim=2, keepdim=True) #sum temporal kernel dimension to get 2d kernel
            new_projection = nn.Conv3d(
                in_channels=3,
                out_channels=original_projection.out_channels,
                kernel_size=(1, original_projection.kernel_size[1], original_projection.kernel_size[2]),
                stride=(1, original_projection.stride[1], original_projection.stride[2])
            )

            new_projection.weight.data = new_weights
            if original_projection.bias is not None:
                new_projection.bias.data = original_projection.bias.data.clone()

            self.video_mae.embeddings.patch_embeddings.projection = new_projection
            self.tubelet_size=1

        elif num_frames > 1:
            assert num_frames % self.tubelet_size == 0

        self.pretrained_num_frames = 16
        self.pretrained_num_patches = self.video_mae.embeddings.patch_embeddings.num_patches
        self.pretrained_spatial_patch_dim = (self.video_mae.embeddings.patch_embeddings.image_size[0] // self.patch_size, self.video_mae.embeddings.patch_embeddings.image_size[1] // self.patch_size)

        self.name = f"{self.checkpoint_name}_{self.num_frames}_{self.tubelet_size}"

        feat_dim = self.video_mae.config.hidden_size 

        num_layers = len(self.video_mae.encoder.layer)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            if self.num_frames > 1 and self.output == "dense-temporal":
                feat_dim = feat_dim*(self.num_frames//self.tubelet_size)
                self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            else:
                self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)
    
    def _interpolate_pos_embed_3d(self, pos_embed, x, N_t):
        # Convert depth, height, width of input to be measured in patches
        # instead of pixels/frames
        _, N, dim = pos_embed.shape
        _, T, C, H, W = x.shape
        T = T // self.tubelet_size
        H = H // self.patch_size
        W = W // self.patch_size

        # Compute the initialized shape of the positional embedding measured
        # in patches
        N_h = self.pretrained_spatial_patch_dim[0]
        N_w = self.pretrained_spatial_patch_dim[1]
        # assert N_h * N_w * N_t == N, 'Positional embedding initialized incorrectly'

        # Compute scale factor for spatio-temporal interpolation
        scale_factor = (T/N_t, H/N_h, W/N_w)

        pos_embed = torch.nn.functional.interpolate(
            pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
            scale_factor=scale_factor,
            mode='trilinear')
        pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return pos_embed
    
    def _get_pos_encoding(self, x):
        pos_embed = self.video_mae.embeddings.position_embeddings
        _, N, dim = pos_embed.shape
        _, T, C, H, W = x.shape

        if self.num_frames > 1:
            if "cutpos" in self.mode:
                pos_embed = pos_embed[:, : self.pretrained_spatial_patch_dim[0]*self.pretrained_spatial_patch_dim[1]*(self.num_frames//self.tubelet_size)]

            if H == self.image_size[0] and W == self.image_size[1] and T == self.pretrained_num_frames:
                return pos_embed
            if "cutpos" in self.mode:
                pos_embed = self._interpolate_pos_embed_3d(pos_embed, x,self.num_frames // self.tubelet_size)
            else:
                pos_embed = self._interpolate_pos_embed_3d(pos_embed, x,self.pretrained_num_frames // self.tubelet_size)
            return pos_embed
        else: #Single image case
            #Only take pos encoding from first frame since no temporal component?
            pos_embed_first_frame = pos_embed[:, : self.pretrained_spatial_patch_dim[0]*self.pretrained_spatial_patch_dim[1]]
            if H == self.image_size[0] and W == self.image_size[1]:
                return pos_embed_first_frame

            # Compute scale factor for spatial interpolation
            npatch = (H // self.patch_size , W // self.patch_size) #New patch N
            scale_factor = (npatch[0] / self.pretrained_spatial_patch_dim[0], npatch[1] / self.pretrained_spatial_patch_dim[1]) #Scale factor

            pos_embed = torch.nn.functional.interpolate(
                pos_embed_first_frame.reshape(1, self.pretrained_spatial_patch_dim[0], self.pretrained_spatial_patch_dim[1], dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode='bicubic')
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

            return pos_embed
    
    def _patch_embeddings(self, pixel_values):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        if num_channels != self.video_mae.embeddings.patch_embeddings.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # permute to (batch_size, num_channels, num_frames, height, width)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        embeddings = self.video_mae.embeddings.patch_embeddings.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings
        
    def _get_embeddings(self, x, bool_masked_pos=None):
        B, T, C, H, W = x.shape
        embeddings = self._patch_embeddings(x)
        pos_embed = self._get_pos_encoding(x)
        # add position embeddings
        embeddings = embeddings + pos_embed.type_as(embeddings).to(embeddings.device).clone().detach()
        # only keep visible patches
        # ~bool_masked_pos means visible
        if bool_masked_pos is not None:
            batch_size, _, num_channels = embeddings.shape
            embeddings = embeddings[~bool_masked_pos]
            embeddings = embeddings.reshape(batch_size, -1, num_channels)
        
        return embeddings
    
    def preprocess(self, x):
        if x.ndim == 4:
            if "resize" in self.mode:
                x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)

            if "img" in self.mode:
                x,hw = fit_to_patch(x, self.patch_size)
                x = x.unsqueeze(1)  # inserts new dimension at index
            elif "video" in self.mode:
                x,hw = fit_to_patch(x, self.patch_size)
                x = x.unsqueeze(1)  # inserts new dimension at index
                x = x.expand(-1, self.num_frames, -1, -1, -1)
            else:
                raise NotImplementedError(f"Mode {self.mode} is not yet supported")

        elif x.ndim == 5:
            if self.num_frames != x.shape[1]:
                raise NotImplementedError(f"Video Model expects {self.num_frames} frames as input but got {x.shape}")
            b, t, c, h, w = x.shape
            x_reshaped = x.reshape(b * t, c, h, w)
            x_patched, hw = fit_to_patch(x_reshaped, self.patch_size)
            x = x_patched.reshape(b, t, c, hw[0]*self.patch_size , hw[1]*self.patch_size)
        else:
            raise ValueError(f"Different input shape {x.shape} is not yet supported")
        return x

    def forward(self, images):
        images = self.preprocess(images)
        B, T, C, H, W = images.shape
        feat_h = H // self.patch_size
        feat_w = W // self.patch_size

        # ---- hidden ----
        embedding_output =  self._get_embeddings(images,bool_masked_pos=None)
        encoder_outputs = self.video_mae.encoder(
            embedding_output,
            head_mask=None,
            output_attentions=self.video_mae.config.output_attentions,
            output_hidden_states=True,
            return_dict=self.video_mae.config.return_dict,
        )

        outputs = []
        for layer_i in self.multilayers:
            x_i = encoder_outputs.hidden_states[layer_i]
            x_i = tokens_to_output(
                self.output, x_i if self.output == "dense-temporal" else x_i[:, :feat_h*feat_w],
                None, (feat_h, feat_w)
            )
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
