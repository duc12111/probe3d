from __future__ import annotations
import math

import numpy as np
import torch
from torch import nn
from transformers import VideoMAEForPreTraining
import torch.nn.functional as F
from transformers.models.videomae.modeling_videomae import get_sinusoid_encoding_table



from .utils import tokens_to_output


class VideoMAE(nn.Module):
    def __init__(
        self,
        checkpoint="MCG-NJU/videomae-base",
        output="dense-temperal",
        layer=-1,
        return_multilayer=False,
    ):
        """Code based on transformer database"""
        super().__init__()

        assert output in [ "dense", "dense-temperal"], "Options: [dense, dense-temperal]"
        self.output = output

        self.checkpoint_name = checkpoint.split("/")[1]

        self.video_mae = VideoMAEForPreTraining.from_pretrained(checkpoint).videomae
        self.video_mae = self.video_mae.eval()


        # resize pos embedding
        # resize embedding for new size
        patch_size = self.video_mae.config.patch_size
        self.patch_size = patch_size
        self.layer = layer

        self.image_size = self.video_mae.embeddings.patch_embeddings.image_size
        self.tubelet_size = self.video_mae.embeddings.patch_embeddings.tubelet_size

        self.num_frames =  self.tubelet_size
        self.pretrained_num_frames = 16

        feat_dim = self.video_mae.config.hidden_size 
        if self.output == "dense-temperal":
            feat_dim = feat_dim * self.num_frames // self.tubelet_size
        num_layers = len(self.video_mae.encoder.layer)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def _interpolate_pos_encoding(self, x, pos_embed, is_video=True):

        _, N, dim = pos_embed.shape

        if is_video:

            # If pos_embed already corret size, just return
            _, T, C, H, W = x.shape
            if H == self.image_size[0] and W == self.image_size[1] and T == self.num_frames:
                return pos_embed

            # Convert depth, height, width of input to be measured in patches
            # instead of pixels/frames
            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size

            # Compute the initialized shape of the positional embedding measured
            # in patches
            N_t = self.pretrained_num_frames // self.tubelet_size # (16 is the pretraining number of frames for VideoMAE)
            N_h = self.image_size[0] // self.patch_size
            N_w = self.image_size[1] // self.patch_size
            assert N_h * N_w * N_t == N, 'Positional embedding initialized incorrectly'

            # Compute scale factor for spatio-temporal interpolation
            scale_factor = (T/N_t, H/N_h, W/N_w)

            pos_embed = torch.nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode='trilinear')
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed

        else:

            # If pos_embed already corret size, just return
            _, _, H, W = x.shape
            if H == self.image_size[0] and W == self.image_size[1]:
                return pos_embed

            # Compute scale factor for spatial interpolation
            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)

            pos_embed = torch.nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode='bicubic')
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed
    
    def _patch_embeddings(self, pixel_values):
        B, T, C, H, W = pixel_values.shape
        if C != self.video_mae.embeddings.patch_embeddings.num_channels:
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

        pos_embed = self.video_mae.embeddings.position_embeddings
        pos_embed = self._interpolate_pos_encoding(x, pos_embed, is_video=True)
        # add position embeddings
        embeddings = embeddings + pos_embed.type_as(embeddings).to(embeddings.device).clone().detach()
        # only keep visible patches
        # ~bool_masked_pos means visible
        if bool_masked_pos is not None:
            batch_size, _, num_channels = embeddings.shape
            embeddings = embeddings[~bool_masked_pos]
            embeddings = embeddings.reshape(batch_size, -1, num_channels)

        return embeddings

    def forward(self, images):
        images = images.unsqueeze(1)
        images = images.expand(-1, self.num_frames, -1, -1, -1)  # [B,N,C,H,W] expand to [B,16,3,224,224]
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
                self.output, x_i if self.output == "dense-temperal" else x_i[:, :feat_h*feat_w],
                None, (feat_h, feat_w)
            )
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
