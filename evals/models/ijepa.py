from transformers import IJepaModel
import torch
from .utils import center_padding, tokens_to_output
import torch.nn.functional as F
class IJEPA(torch.nn.Module):
    def __init__(self, 
                checkpoint="facebook/ijepa_vith16_1k",
                output="dense",
                layer=-1,
                return_multilayer=False,
                mode="original"):
        super().__init__()
        #TODO: check if support dense-cls
        assert output in ["gap", "dense"], "Options: [gap, dense]"
        self.output = output
        self.checkpoint_name = f"{checkpoint}"
        self.vit = IJepaModel.from_pretrained(checkpoint)
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad = False

        feat_dim = self.vit.config.hidden_size
        self.patch_size = self.vit.config.patch_size
        self.image_size = [self.vit.config.image_size, self.vit.config.image_size]

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
        
        # Add name attribute for logging (similar to other models)
        self.name = f"ijepa_{self.checkpoint_name.replace('/', '_')}_{self.layer}"

    def forward(self, images):
        if self.mode == "resize":
            images = F.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        if self.mode == "resize":
            x = self.vit.embeddings(pixel_values=images)
        else:
            x = self.vit.embeddings(pixel_values=images, interpolate_pos_encoding=True)

        embeds = []
        for i, blk in enumerate(self.vit.encoder.layer):
            outs = blk(x)
            x = outs[0]
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            # ignoring register tokens
            spatial = x_i[:, -1 * num_spatial :]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs    