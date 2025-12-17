import torch
import inspect
import numpy as np
from typing import List, Optional, Union

from diffusers.pipelines.flux import FluxPipeline


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
    
    def decode_latents(self, latents, height, width):
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        images = self.vae.decode(latents, return_dict=False)[0]
        images = self.image_processor.postprocess(images.detach(), output_type='pil')
        return images[0]