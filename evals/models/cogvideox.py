from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.functional import interpolate
import os

from .utils import center_padding, tokens_to_output, fit_to_patch


class CogVideoX(torch.nn.Module):
    def __init__(
        self,
        model_id="THUDM/CogVideoX-5b",
        time_step=50,
        output="dense",
        layer=-1,
        return_multilayer=False,
        mode="img",
        num_frames=1,
        temporal_compression_ratio=4,
        prompt="",
        add_noise=False,
        # Memory optimization parameters
        vae_chunk_size=2,  # Process frames in smaller chunks
        clear_cache_freq=1,  # Clear cache every N forward passes
        low_memory_mode=False,  # Enable aggressive memory saving
    ):
        super().__init__()
        assert output in ["cls", "gap", "dense", "dense-cls", "dense-temporal"], (
            "Only supports cls, gap, dense, dense-cls, dense-temporal output"
        )

        self.output = output
        self.time_step = time_step
        self.model_id = model_id
        self.checkpoint_name = model_id.split("/")[-1] + f"_timestep-{time_step}"
        self.mode = mode
        self.num_frames = num_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.prompt = prompt
        self.add_noise = add_noise
        
        # Memory optimization settings
        self.vae_chunk_size = vae_chunk_size
        self.clear_cache_freq = clear_cache_freq
        self.low_memory_mode = low_memory_mode
        self.forward_count = 0
        
        # Initialize the CogVideoX pipeline
        self._init_pipeline()
        
        # Define feature dimensions for different layers based on CogVideoX architecture
        # These should match the actual transformer dimensions
        if "2b" in model_id.lower():
            feat_dims = [1920, 1920, 1920, 1920]  # CogVideoX-2B actual dimensions
            multilayers = [20, 21, 22, 23]  # Last 4 layers for 2B model (0-indexed, assuming 24 total layers)
        elif "5b" in model_id.lower():  # 5B model
            feat_dims = [3072, 3072, 3072, 3072]  # CogVideoX-5B actual dimensions (not 1024!)
            multilayers = [28, 29, 30, 31]  # Last 4 layers for 5B model (0-indexed, assuming 32 total layers)
        else:
            raise ValueError(f"Model {model_id} not supported")
        
        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            layer = multilayers[-1] if layer == -1 else layer
            self.feat_dim = feat_dims[min(layer, len(feat_dims) - 1)]
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)
        
        assert mode in ["video", "video_resize", "img", "img_resize"], f"Options: [video, video_resize, img, img_resize] {mode}"
        self.name = f"cogvideox_{self.checkpoint_name}_{self.layer}"

        # Set standard image size for resize modes (CogVideoX standard resolution)
        self.image_size = [480, 720]  # Height, Width
        self.patch_size = 16  # Standard patch size for transformers

    def _init_pipeline(self):
        """Initialize the CogVideoX pipeline"""
        #TODO: add hf
        
        from diffusers import CogVideoXPipeline
        
        # Initialize the pipeline with authentication
        self.pipeline = CogVideoXPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            use_auth_token=True
        )
        
        # Move pipeline to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = self.pipeline.to(device)
        
        # Extract key components
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        
        # Set to eval mode and freeze parameters
        self.transformer.eval()
        self.vae.eval()
        self.text_encoder.eval()
        
        # Freeze all model parameters to prevent gradient computation
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def _pipeline_style_preprocess(self, images):
        """Pipeline-inspired preprocessing with better memory handling"""
        with torch.no_grad():
            if images.ndim == 4:
                # Convert [B, C, H, W] to [B, C, T, H, W] 
                B, C, H, W = images.shape
                if "resize" in self.mode:
                    # Use pipeline-style resizing
                    images = F.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)
                
                images, hw = fit_to_patch(images, self.patch_size)
                # Add temporal dimension and expand
                images = images.unsqueeze(2)  # [B, C, 1, H, W]
                images = images.expand(-1, -1, self.num_frames, -1, -1)  # [B, C, T, H, W]
                
            elif images.ndim == 5:
                B, T, C, H, W = images.shape
                # Permute to [B, C, T, H, W] format expected by VAE
                images = images.permute(0, 2, 1, 3, 4)
                
                if "resize" in self.mode:
                    # Reshape for efficient resizing: [B*T, C, H, W]
                    B, C, T, H, W = images.shape
                    images_reshaped = images.reshape(B * T, C, H, W)
                    
                    # Resize all frames at once (more efficient than per-frame)
                    images_resized = F.interpolate(
                        images_reshaped, 
                        size=self.image_size, 
                        mode="bilinear", 
                        align_corners=False
                    )
                    
                    # Reshape back to [B, C, T, H, W]
                    images = images_resized.reshape(B, C, T, self.image_size[0], self.image_size[1])
                
                images, hw = fit_to_patch(images, self.patch_size)
            
            return images

    def encode_image(self, images):
        """Encode images using the VAE encoder with memory optimization inspired by CogVideoXImageToVideoPipeline"""
        with torch.no_grad():
            # Ensure images are on the same device and dtype as VAE
            images = images.to(device=self.vae.device, dtype=self.vae.dtype)
            
            B, C, T, H, W = images.shape
            
            # Process in chunks if temporal dimension is large
            if T > self.vae_chunk_size and self.low_memory_mode:
                latent_chunks = []
                for i in range(0, T, self.vae_chunk_size):
                    end_idx = min(i + self.vae_chunk_size, T)
                    chunk = images[:, :, i:end_idx]
                    
                    # Encode chunk directly
                    chunk_latents = self.vae.encode(chunk).latent_dist.mode()
                    
                    # Apply scaling factor
                    if not self.vae.config.invert_scale_latents:
                        chunk_latents = self.vae.config.scaling_factor * chunk_latents
                    else:
                        chunk_latents = chunk_latents / self.vae.config.scaling_factor
                        
                    latent_chunks.append(chunk_latents.detach())
                    
                    # Clear cache after each chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Concatenate along temporal dimension
                latents = torch.cat(latent_chunks, dim=2)
            else:
                # Process all at once (original behavior) with pipeline-style encoding
                encoded = self.vae.encode(images)
                if hasattr(encoded, 'latent_dist'):
                    latents = encoded.latent_dist.mode()
                else:
                    latents = encoded
                    
                # Apply scaling factor like in the pipeline
                if not self.vae.config.invert_scale_latents:
                    latents = self.vae.config.scaling_factor * latents
                else:
                    latents = latents / self.vae.config.scaling_factor
            
            # Detach to prevent gradient accumulation
            return latents.detach()

    def extract_transformer_features(self, latents, prompts=None):
        """Extract features from the transformer at specified layers following CogVideoX flow"""
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
            
            # Follow CogVideoX transformer forward flow exactly
            
            # 1. Time embedding (like in CogVideoX transformer)
            t_emb = self.transformer.time_proj(timesteps)
            t_emb = t_emb.to(dtype=noisy_latents.dtype)
            emb = self.transformer.time_embedding(t_emb)
            
            # 2. Patch embedding (like in CogVideoX transformer)
            hidden_states = self.transformer.patch_embed(prompt_embeds, noisy_latents)
            hidden_states = self.transformer.embedding_dropout(hidden_states)
            
            # 3. Separate text and image embeddings (like in CogVideoX transformer)
            text_seq_length = prompt_embeds.shape[1]
            encoder_hidden_states = hidden_states[:, :text_seq_length]
            hidden_states = hidden_states[:, text_seq_length:]
            
            # 4. Extract features from transformer blocks (like VJEPA)
            embeds = []
            for i, block in enumerate(self.transformer.transformer_blocks):
                # Forward through block (like CogVideoX transformer)
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                )
                
                # Collect features at specified layers (like VJEPA)
                if i in self.multilayers:
                    embeds.append(self.transformer.norm_final(hidden_states).detach())
                    if len(embeds) == len(self.multilayers):
                        break
            
            return embeds

    def _encode_prompts(self, prompts):
        """Encode self.prompt using the text encoder"""
        with torch.no_grad():
            # Use self.prompt for all samples in the batch
            actual_prompts = [self.prompt for _ in range(len(prompts))]
            
            # Tokenize prompts
            text_inputs = self.tokenizer(
                actual_prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            # Encode
            prompt_embeds = self.text_encoder(text_inputs.input_ids.to(self.transformer.device))[0]
            # Detach to prevent gradient accumulation
            return prompt_embeds.detach()

    def forward(self, images):
        """Main forward pass following CogVideoX pattern with memory optimization"""
        with torch.no_grad():
            # Clear cache periodically
            self.forward_count += 1
            if self.forward_count % self.clear_cache_freq == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Use pipeline-inspired preprocessing for better memory efficiency
            images = self._pipeline_style_preprocess(images)
            
            # After preprocessing, images should be in [B, C, T, H, W] format
            assert images.dim() == 5, f"Expected 5D input after preprocessing, got {images.dim()}D"
            
            batch_size = images.shape[0]
            # Calculate patch dimensions from original image size
            h, w = images.shape[-2:] 
            h, w = h // self.patch_size, w // self.patch_size
            
            # Use the class attribute prompt for all samples in the batch
            prompts = [self.prompt for _ in range(batch_size)]
            
            # Encode images to latent space
            latents = self.encode_image(images)
            latents = latents.permute(0, 2, 1, 3, 4)
            
            # Clear intermediate memory
            del images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Extract transformer features
            features = self.extract_transformer_features(latents, prompts)
            
            # Clear latents memory
            del latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            outputs = []
            for feat in features:
                # Convert features to token format and then to desired output
                processed_feat = tokens_to_output(
                    self.output,
                    feat if self.output == "dense-temporal" else feat[:, : h * w], 
                    None,  # No CLS token for CogVideoX
                    (h, w),
                )
                
                # Convert to float32 to ensure compatibility with probe layers
                processed_feat = processed_feat.float().detach()  # Detach to prevent gradient leaks
                outputs.append(processed_feat)
            
            return outputs[0] if len(outputs) == 1 else outputs 