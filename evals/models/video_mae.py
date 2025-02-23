from __future__ import annotations

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
        self.feat_h = self.image_size[0] // self.patch_size
        self.feat_w = self.image_size[1] // self.patch_size

        self.num_frames =  self.tubelet_size

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

    def forward(self, images):
        images = images.unsqueeze(1)  # inserts new dimension at index 2
        images = images.expand(-1, self.num_frames, -1, -1, -1)  # [B,N,C,H,W] expand to [B,16,3,224,224]

        # ---- hidden ----
        embedding_output =  self.video_mae.embeddings(images,bool_masked_pos=None)
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
                self.output, x_i if self.output == "dense-temperal" else x_i[:, :self.feat_h*self.feat_w],
                None, (self.feat_h, self.feat_w)
            )
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
