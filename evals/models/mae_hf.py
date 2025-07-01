import torch
from torch import nn
from transformers import ViTMAEForPreTraining

from .utils import get_2d_sincos_pos_embed, tokens_to_output
import torchvision.transforms as tf
import torch.nn.functional as F


class MAE(nn.Module):
    def __init__(
        self,
        checkpoint="facebook/vit-mae-base",
        output="dense",
        layer=-1,
        return_multilayer=False,
        mode="original",
    ):
        """MAE Model with HuggingFace token support and enhanced preprocessing"""
        super().__init__()

        assert output in ["cls", "gap", "dense"], "Options: [cls, gap, dense]"

        self.output = output

        self.checkpoint_name = checkpoint.split("/")[1]

        self.vit = ViTMAEForPreTraining.from_pretrained(checkpoint,token="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").vit
        self.vit = self.vit.eval()

        # resize pos embedding
        # resize embedding for new size
        patch_size = self.vit.config.patch_size
        self.patch_size = patch_size
        self.layer = layer

        self.image_size = self.vit.embeddings.patch_embeddings.image_size
        self.feat_h = self.image_size[0] // self.patch_size
        self.feat_w = self.image_size[1] // self.patch_size

        feat_dim = self.vit.config.hidden_size
        num_layers = len(self.vit.encoder.layer)
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
        assert mode in ["original", "resize"], f"Options: [original, resize] {mode}"
        self.mode = mode
        
        # define name for logging/evaluation
        self.name = f"mae_hf_{self.checkpoint_name}_{self.layer}"

    def resize_pos_embed(self, image_size):
        assert image_size[0] % self.patch_size == 0
        assert image_size[1] % self.patch_size == 0
        self.feat_h = image_size[0] // self.patch_size
        self.feat_w = image_size[1] // self.patch_size
        embed_dim = self.vit.config.hidden_size
        self.vit.embeddings.patch_embeddings.image_size = image_size
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.feat_h, self.feat_w), add_cls_token=True
        )
        # there should be an easier way ... TODO
        device = self.vit.embeddings.patch_embeddings.projection.weight.device
        self.vit.embeddings.position_embeddings = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0).to(device=device),
            requires_grad=False,
        )

    def embed_forward(self, embedder, pixel_values, interpolate_pos_encoding=False):
        # No masking here ...
        batch_size, num_channels, height, width = pixel_values.shape
        
        embeddings = embedder.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if interpolate_pos_encoding:
            position_embeddings = embedder.interpolate_pos_encoding(embeddings, height, width)
        else:
            position_embeddings = embedder.position_embeddings

        # add position embeddings w/o cls token
        embeddings = embeddings + position_embeddings[:, 1:, :]

        # append cls token
        cls_token = embedder.cls_token + position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings


    def forward(self, images):
        # Apply ImageNet normalization
        if images.max() > 2:  # Check if images are in [0, 255] range
            images = images / 255.0
        norm = tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images = norm(images)
        
        # check if positional embeddings are correct
        if self.image_size != images.shape[-2:]:
            if self.mode == "resize":
                images = torch.nn.functional.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)
                self.feat_h, self.feat_w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size

            elif self.mode == "interpolate_pos_embed":
                images = images
                self.feat_h, self.feat_w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
            else:
                self.resize_pos_embed(images.shape[-2:])
        # from MAE implementation
        head_mask = self.vit.get_head_mask(None, self.vit.config.num_hidden_layers)
        
        # ---- hidden ----
        if self.mode == "interpolate_pos_embed":
            embedding_output = self.embed_forward(self.vit.embeddings, images, interpolate_pos_encoding=True)
        else:
            embedding_output = self.embed_forward(self.vit.embeddings, images)
        encoder_outputs = self.vit.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=self.vit.config.output_attentions,
            output_hidden_states=True,
            return_dict=self.vit.config.return_dict,
        )

        outputs = []
        for layer_i in self.multilayers:
            x_i = encoder_outputs.hidden_states[layer_i]
            x_i = tokens_to_output(
                self.output, x_i[:, 1:], x_i[:, 0], (self.feat_h, self.feat_w)
            )
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs 