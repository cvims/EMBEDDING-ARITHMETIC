import os
import gc
import time
import json
import torch
import inspect
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Union

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKL,FluxTransformer2DModel
from diffusers.pipelines.flux import FluxPipeline
from diffusers.utils.torch_utils import randn_tensor


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FluxCustomPipeline(FluxPipeline):
    """
    A custom FLUX pipeline that modularizes generation and separates the
    reverse-forward diffusion process.
    """

    def _setup_common_components(self,
                                 prompt,
                                 num_inference_steps,
                                 num_images_per_prompt,
                                 max_sequence_length,
                                 guidance_scale,
                                 height,
                                 width,
                                 generator,
                                 latents,
                                 sigmas):
        device = self._execution_device
        batch_size = 1

        # 1. Encode prompt to get pooled embeddings
        (
            prompt_embeds, # Will be replaced by zeros
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        
        # use zeros for text embeddings and rely only on pooled embeddings
        prompt_zeros = torch.zeros_like(prompt_embeds)

        # 2. Prepare latents
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 3. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
            
        return latents, timesteps, guidance, pooled_prompt_embeds, prompt_zeros, text_ids, latent_image_ids
    

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None,
        return_all_latents: bool = False,
        **kwargs,
    ):
        """
        Standard reverse diffusion process.
        """
        latents, timesteps, guidance, pooled_prompt_embeds, prompt_zeros, text_ids, latent_image_ids = self._setup_common_components(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            max_sequence_length=512,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            latents=None,
            sigmas=None
        )

        all_latents_decoded = []
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress:
            for i, t in enumerate(timesteps):
                self._current_timestep = t
                timesteps_expanded = t.expand(latents.shape[0]).to(self._execution_device)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timesteps_expanded / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_zeros,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if return_all_latents:
                    all_latents_decoded.append(self.decode_latents(latents, height, width))
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress.update()

        image = self.decode_latents(latents, height, width)
        
        if return_all_latents:
            return image, all_latents_decoded
        return image, pooled_prompt_embeds
    
    @torch.no_grad()
    def generate_with_inversion(
        self,
        initial_prompt: str,
        new_prompt: str,
        inversion_step: int,
        forward_steps: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        new_guidance_scale: Optional[float] = None,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        """
        Performs reverse diffusion, then forward diffusion (inversion), and then reverse diffusion again with a new prompt.
        """
        if new_guidance_scale is None:
            new_guidance_scale = guidance_scale

        latents, timesteps, guidance, pooled_prompt_embeds, prompt_zeros, text_ids, latent_image_ids = self._setup_common_components(
            prompt=initial_prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            max_sequence_length=512,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            latents=None,
            sigmas=None
        )

        i = 0
        with self.progress_bar(total=num_inference_steps) as progress:
            while i < len(timesteps):
                t = timesteps[i]
                self._current_timestep = t
                timesteps_expanded = t.expand(latents.shape[0]).to(self._execution_device)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timesteps_expanded / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_zeros,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                progress.update()

                # --- Inversion Logic ---
                if i == inversion_step:
                    t_start_idx = i
                    t_end_idx = i - forward_steps
                    
                    print(f"Performing forward diffusion from step {t_start_idx} to {t_end_idx}")
                    latents = self.forward_diffuse(latents, t_start_idx, t_end_idx, self.scheduler)
                    
                    print(f"Switching to new prompt: '{new_prompt}'")
                    (
                        new_prompt_embeds,
                        pooled_prompt_embeds, # Update pooled embeds
                        text_ids,
                    ) = self.encode_prompt(
                        prompt=new_prompt, prompt_2=new_prompt, device=self._execution_device,
                        num_images_per_prompt=1, max_sequence_length=512
                    )
                    prompt_zeros = torch.zeros_like(new_prompt_embeds) # Update zeros shape
                    
                    guidance = torch.full([1], new_guidance_scale, device=self._execution_device, dtype=torch.float32)
                    
                    # Jump back in the timeline
                    i = t_end_idx
                    self.scheduler._step_index = i
                    continue
                
                i += 1

        image = self.decode_latents(latents, height, width)
        return image
    
    @torch.no_grad()
    def generate_with_custom_embedding(
        self,
        pooled_prompt_embedding:torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None,
        return_all_latents: bool = False,
        **kwargs,
    ):
        """
        Standard reverse diffusion process.
        """
        latents, timesteps, guidance, pooled_prompt_embeds, prompt_zeros, text_ids, latent_image_ids = self._setup_common_components(
            prompt="",
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            max_sequence_length=512,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            latents=None,
            sigmas=None
        )
        pooled_prompt_embeds = pooled_prompt_embedding
        all_latents_decoded = []
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        with self.progress_bar(total=num_inference_steps) as progress:
            for i, t in enumerate(timesteps):
                self._current_timestep = t
                timesteps_expanded = t.expand(latents.shape[0]).to(self._execution_device)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timesteps_expanded / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_zeros,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if return_all_latents:
                    all_latents_decoded.append(self.decode_latents(latents, height, width))
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress.update()

        image = self.decode_latents(latents, height, width)
        
        if return_all_latents:
            return image, all_latents_decoded
        return image, pooled_prompt_embeds
    
    @torch.no_grad()
    def perform_concept_algebra(self, main_prompt, concept_prompt, prompt_plus, prompt_minus,
                                num_inference_steps: int = 50,
                                guidance_scale: float = 7.5,
                                height: int = 512,
                                width: int = 512,
                                generator: Optional[torch.Generator] = None,
                                return_all_latents: bool = False,
                                **kwargs,):
        latents, timesteps, guidance, pooled_prompt_embeds, prompt_zeros, text_ids, latent_image_ids = self._setup_common_components(
            prompt=main_prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            max_sequence_length=512,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            latents=None,
            sigmas=None
        )

        (
            _, # Will be replaced by zeros
            pooled_prompt_embeds_concept,
            _,
        ) = self.encode_prompt(
            prompt=concept_prompt,
            prompt_2=concept_prompt,
            device=self._execution_device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

        (
            _, # Will be replaced by zeros
            pooled_prompt_embeds_plus,
            _,
        ) = self.encode_prompt(
            prompt=prompt_plus,
            prompt_2=prompt_plus,
            device=self._execution_device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

        (
            _, # Will be replaced by zeros
            pooled_prompt_embeds_minus,
            _,
        ) = self.encode_prompt(
            prompt=prompt_minus,
            prompt_2=prompt_minus,
            device=self._execution_device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        pooled_embeddings = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds_plus, pooled_prompt_embeds_minus, pooled_prompt_embeds_concept])
        zero_embeddings = torch.cat([prompt_zeros,prompt_zeros,prompt_zeros,prompt_zeros,])
        all_latents_decoded = []
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        with self.progress_bar(total=num_inference_steps) as progress:
            for i, t in enumerate(timesteps):
                self._current_timestep = t
                latents_model_input = torch.cat([latents]*4)
                timesteps_expanded = t.expand(latents.shape[0]).to(self._execution_device)

                noise_pred = self.transformer(
                    hidden_states=latents_model_input,
                    timestep=timesteps_expanded / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_embeddings,
                    encoder_hidden_states=zero_embeddings,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                noise_main, noise_plus, noise_minus, noise_concept = noise_pred.chunk(5)
                noise_tmp = noise_main - noise_concept

                u = noise_plus - noise_minus
                u /= torch.sqrt((u**2).sum())

                noise_main -= (noise_tmp*u).sum()*u

                latents = self.scheduler.step(noise_main, t, latents, return_dict=False)[0]

                if return_all_latents:
                    all_latents_decoded.append(self.decode_latents(latents, height, width))
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress.update()

        image = self.decode_latents(latents, height, width)
        
        if return_all_latents:
            return image, all_latents_decoded
        return image, pooled_prompt_embeds






    def forward_diffuse(self, latents, t_start_idx, t_end_idx, scheduler):
        """
        Simulates the forward diffusion process by adding noise to latents.
        """
        noise = randn_tensor(shape=latents.shape, device=self._execution_device, dtype=latents.dtype)
        
        # Get the sigma for the target noise level
        sigma_start = scheduler.sigmas[t_start_idx]
        sigma_end = scheduler.sigmas[t_end_idx]
        
        # Calculate the amount of noise to add to get from sigma_start to sigma_end
        # sigma_t^2 = alpha_t^2 * sigma_s^2 + (alpha_t * sigma_s)^2
        # Simplified: we want to reach the noise level of sigma_end from a less noisy latent
        # A simple way is to use the DDIM forward equation: sqrt(alpha_end) * x0 + sqrt(1-alpha_end) * noise
        # Since we have x_t, not x0, we can use a simpler formulation based on sigmas.
        # x_t = sqrt(1 - sigma_t^2) * x_0 + sigma_t * noise
        # A direct jump:
        alpha_t_sq = 1 - sigma_start**2
        alpha_s_sq = 1 - sigma_end**2
        
        beta = (1 - alpha_s_sq / alpha_t_sq).sqrt()
        
        latents = (1 - beta**2).sqrt() * latents + beta * noise
        
        return latents

    def decode_latents(self, latents, height, width):
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        images = self.vae.decode(latents, return_dict=False)[0]
        images = self.image_processor.postprocess(images.detach(), output_type='pil')
        return images[0]

