from __future__ import annotations

import einops as E
import open_clip
import torch
from torch import nn

from .utils import center_padding, resize_pos_embed, tokens_to_output


class CLIP(nn.Module):
    def __init__(
        self,
        arch="ViT-B-16",
        checkpoint="openai",
        output="dense",
        layer=-1,
        return_multilayer=False,
        mode="original",
    ):
        super().__init__()
        assert output in ["dense-cls", "cls", "gap", "dense"]
        self.output = output
        self.checkpoint_name = "clip_" + arch.replace("-", "").lower() + checkpoint

        # Initialize a pre-trained CLIP image encoder and freeze it.
        _clip_model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=checkpoint
        )
        _clip_model = _clip_model.eval().to(torch.float32)
        self.visual = _clip_model.visual
        self.image_size = self.visual.image_size
        del _clip_model

        # Extract some attributes from CLIP module for easy access.
        self.patch_size = self.visual.conv1.stride[0]

        # get feature dimension
        feat_dim = self.visual.transformer.width
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim
        feat_dims = [feat_dim, feat_dim, feat_dim, feat_dim]

        # get extraction targets
        n_layers = len(self.visual.transformer.resblocks)
        multilayers = [
            n_layers // 4 - 1,
            n_layers // 2 - 1,
            n_layers // 4 * 3 - 1,
            n_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dims
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        self.layer = "-".join(str(_x) for _x in self.multilayers)
        assert mode in ["original", "resize"], "Options: [original, resize]"
        self.mode = mode

    def forward(self, images):
        images = center_padding(images, self.patch_size)
        img_h, img_w = images.shape[-2:]
        out_hw = (img_h // self.patch_size, img_w // self.patch_size)

        if (img_h, img_w) != self.image_size:
            if self.mode == "original":
                pos_embed = resize_pos_embed(self.visual.positional_embedding, out_hw)
            elif self.mode == "resize":
                images = torch.nn.functional.interpolate(images, self.image_size, mode="bicubic")
                img_h, img_w = self.image_size
                pos_embed = self.visual.positional_embedding
                out_hw = (self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        else:
            pos_embed = self.visual.positional_embedding

        # clip stuff
        x = self.visual.conv1(images)
        x_hw = x.shape[-2:]
        x = E.rearrange(x, "b c h w -> b (h w) c")

        # concat cls token
        _cls_embed = E.repeat(self.visual.class_embedding, "c -> b 1 c", b=x.shape[0])
        x = torch.cat([_cls_embed.to(x.dtype), x], dim=1)

        # add pos embed
        x = self.visual.ln_pre(x + pos_embed.to(x.dtype))

        embeds = []
        for i, blk in enumerate(self.visual.transformer.resblocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        for i, _x in enumerate(embeds):
            _x = tokens_to_output(self.output, _x[:, 1:], _x[:, 0], out_hw)
            outputs.append(_x)

        return outputs[0] if len(outputs) == 1 else outputs
