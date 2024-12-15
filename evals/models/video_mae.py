from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import VideoMAEForPreTraining
from transformers import ViTMAEForPreTraining
import torch.nn.functional as F



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
        self.feat_h = self.image_size[0] // self.patch_size
        self.feat_w = self.image_size[1] // self.patch_size

        feat_dim = self.video_mae.config.hidden_size
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

    # def resize_pos_embed(self, image_size):
    #     assert image_size[0] % self.patch_size == 0
    #     assert image_size[1] % self.patch_size == 0
    #     self.feat_h = image_size[0] // self.patch_size
    #     self.feat_w = image_size[1] // self.patch_size
    #     self.video_mae.embeddings.patch_embeddings.image_size = image_size
    #     pos_embed = get_sinusoid_encoding_table(
    #         self.video_mae.embeddings.num_patches, self.video_mae.config.hidden_size
    #     )
    #     # there should be an easier way ... TODO
    #     device = self.video_mae.embeddings.patch_embeddings.projection.weight.device
    #     self.video_mae.embeddings.position_embeddings = nn.Parameter(
    #         torch.from_numpy(pos_embed).float().unsqueeze(0).to(device=device),
    #         requires_grad=False,
    #     )
    
    def get_sinusoid_encoding_table(n_position, d_hid):
        """Sinusoid position encoding table"""

        # TODO: make it with torch instead of numpy
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)


    def forward(self, images):
        resized_image = F.interpolate(images, 
                            size=(self.image_size[0], self.image_size[1]),  # tuple of (H, W)
                            mode='bilinear',  # or 'bicubic', 'nearest', etc.
                            align_corners=False)
        images = resized_image.unsqueeze(1)  # inserts new dimension at index 2
        images = images.expand(-1, 16, -1, -1, -1)  # [1,16,3,224,224]

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
                self.output, x_i[:, :self.feat_h*self.feat_w], None, (self.feat_h, self.feat_w)
            )
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
