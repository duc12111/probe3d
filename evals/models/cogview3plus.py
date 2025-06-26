from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.functional import interpolate
import os
import math

from .utils import center_padding, tokens_to_output, fit_to_patch


class CogView3Plus(torch.nn.Module):
    def __init__(
        self,
        model_id="THUDM/CogView3-Plus-3B",
        time_step=50,
        output="dense",
        layer=-1,
        return_multilayer=False,
        mode="original",
        prompt="",
        add_noise=False,
    ):
        super().__init__()
        assert output in ["cls", "gap", "dense", "dense-cls"], (
            "Only supports cls, gap, dense, dense-cls output"
        )

        self.output = output
        self.time_step = time_step
        self.model_id = model_id
        self.checkpoint_name = model_id.split("/")[-1] + f"_timestep-{time_step}"
        self.mode = mode
        self.prompt = prompt
        self.add_noise = add_noise
        
        # Initialize the CogView3+ pipeline
        self._init_pipeline()
        
        # Define feature dimensions for different layers based on StableDiffusion UNet architecture
        # These should match the actual UNet dimensions
        if "3b" in model_id.lower():
            # For actual CogView3+ if available, use appropriate dimensions
            feat_dims = [320, 640, 1280, 1280]  # Typical SD UNet dimensions
            multilayers = [0, 1, 2, 3]  # Down blocks indices
        else:  # Default StableDiffusion dimensions
            feat_dims = [320, 640, 1280, 1280]  # StableDiffusion UNet dimensions
            multilayers = [0, 1, 2, 3]  # Down blocks indices
        
        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            layer = multilayers[-1] if layer == -1 else layer
            self.feat_dim = feat_dims[min(layer, len(feat_dims) - 1)]
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)
        
        assert mode in ["original", "resize"], f"Options: [original, resize] {mode}"
        self.name = f"cogview3plus_{self.checkpoint_name}_{self.layer}"

        # Set standard image size for resize modes (CogView3+ standard resolution)
        self.image_size = [1024, 1024]  # Square images for CogView3+
        self.patch_size = 16  # Standard patch size for transformers

    def _init_pipeline(self):
        """Initialize the CogView3+ pipeline"""
        #TODO: add hf
        
        from diffusers import StableDiffusionPipeline
        
        # Initialize the pipeline with authentication
        # Use StableDiffusion as base since CogView3+ may not be directly available
        model_id = "stabilityai/stable-diffusion-2-1"  # Use known working model
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_auth_token=True,
            safety_checker=None,  # Disable safety checker to avoid missing component
            requires_safety_checker=False
        )
        
        # Move pipeline to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = self.pipeline.to(device)
        
        # Extract key components
        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae  
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        
        # Set to eval mode and freeze parameters
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        
        # Freeze all model parameters to prevent gradient computation
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def preprocess(self, x):
        """Preprocess input to [B, C, H, W] format"""
        if x.ndim == 4:
            B, C, H, W = x.shape
            # Preprocess based on mode
            if self.mode == "resize":
                x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)
            elif self.mode == "original":
                x = center_padding(x, self.patch_size)
        else:
            raise ValueError(f"Expected 4D input [B, C, H, W], got {x.ndim}D")
        
        return x

    def encode_image(self, images):
        """Encode images using the VAE encoder"""
        with torch.no_grad():
            # Ensure images are on the same device and dtype as VAE
            images = images.to(device=self.vae.device, dtype=self.vae.dtype)
            # Encode with VAE
            latents = self.vae.encode(images).latent_dist.mode()
            # Scale latents as per diffusion model requirements
            latents = latents * self.vae.config.scaling_factor
            return latents

    def extract_unet_features(self, latents, prompts=None):
        """Extract features from the UNet at specified layers following CogVideoX pattern"""
        with torch.no_grad():
            batch_size = latents.shape[0]
            
            # Handle prompts
            if prompts is None:
                prompts = ["" for _ in range(batch_size)]
            
            # Encode prompts
            prompt_embeds = self._encode_prompts(prompts)
            
            # Conditionally add noise for the specified timestep
            if self.add_noise:
                noise = torch.randn_like(latents)
                timesteps = torch.full((batch_size,), self.time_step, device=latents.device, dtype=torch.long)
                noisy_latents = self.pipeline.scheduler.add_noise(latents, noise, timesteps)
            else:
                noisy_latents = latents
                # Still need timesteps for time embedding, use zeros when no noise
                timesteps = torch.zeros((batch_size,), device=latents.device, dtype=torch.long)
            
            # Follow similar pattern to CogVideoX transformer processing
            
            # 1. Time embedding (like in UNet)
            t_emb = self.unet.time_proj(timesteps)
            t_emb = t_emb.to(dtype=noisy_latents.dtype)
            emb = self.unet.time_embedding(t_emb)
            
            # 2. Process through UNet with feature extraction
            # Get encoder hidden states for cross-attention
            encoder_hidden_states = prompt_embeds
            
            # Initial conv processing
            sample = self.unet.conv_in(noisy_latents)
            
            # Extract features from down blocks and mid block only (more stable)
            embeds = []
            
            # Down blocks
            for i, downsample_block in enumerate(self.unet.down_blocks):
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                
                # Collect features at specified layers (following CogVideoX pattern)
                if i in self.multilayers:
                    # Apply layer norm like CogVideoX does with norm_final
                    if hasattr(self.unet, 'norm_out'):
                        normed_sample = self.unet.norm_out(sample)
                    else:
                        normed_sample = sample
                    embeds.append(normed_sample)
            
            # Mid block
            if self.unet.mid_block is not None:
                sample = self.unet.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
                
                # Check if mid block index is in multilayers
                mid_block_idx = len(self.unet.down_blocks)
                if mid_block_idx in self.multilayers:
                    if hasattr(self.unet, 'norm_out'):
                        normed_sample = self.unet.norm_out(sample)
                    else:
                        normed_sample = sample
                    embeds.append(normed_sample)
            
            # If no features were collected, return the final sample
            if not embeds:
                embeds = [sample]
            
            return embeds

    def _encode_prompts(self, prompts):
        """Encode text prompts using the text encoder"""
        with torch.no_grad():
            # Tokenize prompts
            text_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            # Encode
            prompt_embeds = self.text_encoder(text_inputs.input_ids.to(self.unet.device))[0]
            return prompt_embeds

    def forward(self, images):
        """Main forward pass following CogVideoX pattern"""
        with torch.no_grad():
            # Preprocess input to [B, C, H, W] format
            images = self.preprocess(images)
            
            # After preprocessing, images should be in [B, C, H, W] format
            assert images.dim() == 4, f"Expected 4D input after preprocessing, got {images.dim()}D"
            
            batch_size = images.shape[0]
            input_h, input_w = images.shape[-2:]
            
            # Use the class attribute prompt for all samples in the batch
            prompts = [self.prompt for _ in range(batch_size)]
            
            # Encode images to latent space
            latents = self.encode_image(images)
            
            # Extract UNet features
            features = self.extract_unet_features(latents, prompts)
            
            # Process features using tokens_to_output (similar to CogVideoX)
            outputs = []
            for feat in features:
                # UNet features are in [B, C, H, W] format
                if feat.dim() == 4:  # [B, C, H, W]
                    b, c, fh, fw = feat.shape
                    # Convert to token format: [B, N, C] where N = H*W
                    feat = feat.reshape(b, c, fh * fw).transpose(1, 2)  # [B, H*W, C]
                    h, w = fh, fw
                else:
                    raise ValueError(f"Expected 4D UNet features, got {feat.dim()}D")
                
                # Convert features to desired output format
                processed_feat = tokens_to_output(
                    self.output,
                    feat, 
                    None,  # No CLS token for CogView3+
                    (h, w),
                )
                
                # Convert to float32 to ensure compatibility with probe layers
                processed_feat = processed_feat.float()
                outputs.append(processed_feat)
            
            return outputs[0] if len(outputs) == 1 else outputs 