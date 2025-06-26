from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers import CogVideoXImageToVideoPipeline
from torchvision.transforms.functional import to_pil_image
import os

from .utils import tokens_to_output


class CogVideoXPipelineBased(torch.nn.Module):
    def __init__(
        self,
        model_id="THUDM/CogVideoX-5b-I2V",  # Use the correct image-to-video variant
        output="dense",
        layer=-1,
        return_multilayer=False,
        mode="img_resize",
        num_frames=1,
        prompt="",
        low_memory_mode=True,
    ):
        super().__init__()
        assert output in ["cls", "gap", "dense", "dense-cls", "dense-temporal"], (
            "Only supports cls, gap, dense, dense-cls, dense-temporal output"
        )

        self.output = output
        self.model_id = model_id
        self.checkpoint_name = model_id.split("/")[-1]
        self.mode = mode
        self.num_frames = num_frames
        self.prompt = prompt
        self.low_memory_mode = low_memory_mode
        
        # Initialize the CogVideoX Image-to-Video pipeline
        self._init_pipeline()
        
        # Define feature dimensions for different layers based on CogVideoX architecture
        if "2b" in model_id.lower():
            feat_dims = [1920, 1920, 1920, 1920]
            multilayers = [20, 21, 22, 23]
        elif "5b" in model_id.lower():
            feat_dims = [3072, 3072, 3072, 3072]
            multilayers = [28, 29, 30, 31]
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
        self.name = f"cogvideox_pipeline_{self.checkpoint_name}_{self.layer}"

        # Set standard image size
        self.image_size = [480, 720]  # Height, Width
        self.patch_size = 16

    def _init_pipeline(self):
        """Initialize the CogVideoX Image-to-Video pipeline"""
        #TODO: add hf
        
        # Initialize the image-to-video pipeline
        self.pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            use_auth_token=True
        )
        
        # Move pipeline to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = self.pipeline.to(device)
        
        # # Enable memory efficient attention if available
        # if hasattr(self.pipeline, 'enable_model_cpu_offload'):
        #     if self.low_memory_mode:
        #         self.pipeline.enable_model_cpu_offload()
        
        if hasattr(self.pipeline, 'enable_vae_slicing'):
            if self.low_memory_mode:
                self.pipeline.enable_vae_slicing()
        
        # Extract key components for feature extraction
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.video_processor = self.pipeline.video_processor
        
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

    def preprocess_image(self, images):
        """Simple and robust image preprocessing"""
        with torch.no_grad():
            if images.ndim == 4:
                # Convert [B, C, H, W] to [B, T, C, H, W] format
                B, C, H, W = images.shape
                
                # Simple resize if needed
                if "resize" in self.mode:
                    images = F.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)
                
                # Ensure correct device and dtype
                images = images.to(device=self.vae.device, dtype=self.vae.dtype)
                
                # Add temporal dimension: [B, C, H, W] -> [B, C, 1, H, W]
                images = images.unsqueeze(2)
                
                # Expand to desired number of frames: [B, C, 1, H, W] -> [B, C, T, H, W]
                if self.num_frames > 1:
                    images = images.expand(-1, -1, self.num_frames, -1, -1)
            
            elif images.ndim == 5:
                # Already in [B, T, C, H, W] format, just ensure correct device/dtype
                images = images.to(device=self.vae.device, dtype=self.vae.dtype)
                if "resize" in self.mode:
                    B, T, C, H, W = images.shape
                    # Reshape to [B*T, C, H, W] for resizing
                    images_reshaped = images.reshape(B * T, C, H, W)
                    images_resized = F.interpolate(images_reshaped, size=self.image_size, mode="bilinear", align_corners=False)
                    # Reshape back to [B, T, C, H, W]
                    images = images_resized.reshape(B, T, C, self.image_size[0], self.image_size[1])
            
            return images

    def encode_with_pipeline_vae(self, images):
        """Use pipeline's VAE encoding with proper scaling"""
        with torch.no_grad():
            # Images should already be in the correct format from preprocessing
            encoded_output = self.vae.encode(images)
            latents = encoded_output.latent_dist.mode()
            
            # Apply scaling factor like in cogvideox.py
            if not self.vae.config.invert_scale_latents:
                latents = self.vae.config.scaling_factor * latents
            else:
                latents = latents / self.vae.config.scaling_factor
            
            return latents

    def extract_transformer_features(self, latents, prompts=None):
        """Extract features from transformer blocks without full generation"""
        with torch.no_grad():
            batch_size = latents.shape[0]
            
            # Handle prompts
            if prompts is None:
                prompts = [self.prompt for _ in range(batch_size)]
            
            # Encode prompts using pipeline's method
            prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompts,
                negative_prompt=None,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                device=latents.device,
                dtype=latents.dtype,
            )
            
            # Prepare rotary embeddings using pipeline's method
            latent_height, latent_width = latents.shape[-2:]
            height = latent_height * self.pipeline.vae_scale_factor_spatial
            width = latent_width * self.pipeline.vae_scale_factor_spatial
            num_frames = latents.shape[2]
            
            image_rotary_emb = None
            if self.transformer.config.use_rotary_positional_embeddings:
                image_rotary_emb = self.pipeline._prepare_rotary_positional_embeddings(
                    height, width, num_frames, latents.device
                )
            
            timestep = torch.zeros((batch_size,), device=latents.device, dtype=torch.long)
            
            # Create proper time embeddings like in cogvideox.py
            t_emb = self.transformer.time_proj(timestep)
            t_emb = t_emb.to(dtype=latents.dtype)
            temb = self.transformer.time_embedding(t_emb)
            
            # Convert from [B, C, T, H, W] to [B, T, C, H, W]
            B, C, T, H, W = latents.shape
            latents_for_patch_embed = latents.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]
            
            # Apply patch embedding like in pipeline
            hidden_states = self.transformer.patch_embed(prompt_embeds, latents_for_patch_embed)
            hidden_states = self.transformer.embedding_dropout(hidden_states)
            
            # Separate text and image embeddings
            text_seq_length = prompt_embeds.shape[1]
            encoder_hidden_states = hidden_states[:, :text_seq_length]
            hidden_states = hidden_states[:, text_seq_length:]
            
            # Extract features from specified layers
            embeds = []
            for i, block in enumerate(self.transformer.transformer_blocks):
                # Forward through block
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )
                
                # Collect features at specified layers
                if i in self.multilayers:
                    embeds.append(self.transformer.norm_final(hidden_states).detach())
                    if len(embeds) == len(self.multilayers):
                        break
            
            return embeds

    def forward(self, images):
        """Main forward pass using pipeline components"""
        with torch.no_grad():
            # Preprocess using pipeline methods
            images = self.preprocess_image(images)
            
            batch_size = images.shape[0]
            # Calculate patch dimensions
            h, w = images.shape[-2:] 
            h, w = h // self.patch_size, w // self.patch_size
            
            # Encode using pipeline VAE
            latents = self.encode_with_pipeline_vae(images)
            
            # Clear intermediate memory
            del images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Extract transformer features
            prompts = [self.prompt for _ in range(batch_size)]
            features = self.extract_transformer_features(latents, prompts)
            
            # Clear latents memory
            del latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            outputs = []
            for feat in features:
                # Features from transformer blocks are already in [B, H*W, C] format
                # Convert features to token format and then to desired output
                processed_feat = tokens_to_output(
                    self.output,
                    feat if self.output == "dense-temporal" else feat[:, : h * w], 
                    None,  # No CLS token for CogVideoX
                    (h, w),
                )
                
                # Convert to float32 to ensure compatibility with probe layers
                processed_feat = processed_feat.float().detach()
                outputs.append(processed_feat)
            
            return outputs[0] if len(outputs) == 1 else outputs 