import argparse
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch
from einops import rearrange

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import ConditioningItem
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.video_vae import TilingConfig, VideoEncoder, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, AudioLatentShape, VideoLatentShape, VideoPixelShape
from ltx_pipelines.iclora_utils import (
    append_ic_lora_reference_video_conditionings,
    read_lora_reference_downscale_factor,
    read_lora_reference_temporal_scale_factor,
)
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    VideoConditioningAction,
    VideoMaskConditioningAction,
    default_2_stage_distilled_arg_parser,
    detect_checkpoint_path,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMAS,
    STAGE_2_DISTILLED_SIGMAS,
    detect_params,
)
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings, get_device
from ltx_pipelines.utils.media_io import decode_video_by_frame, encode_video, resize_and_center_crop, video_preprocess
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode
import ltx_core.model.transformer.global_args as global_args 


@dataclass(frozen=True)
class TrackModality(Modality):
    """Keep custom fields local; stock ModalitySpec/SimpleDenoiser need no patch."""

    track_xy: torch.Tensor | None = None
    track_valid: torch.Tensor | None = None
    target_token_count: int | None = None
    reference_token_count: int | None = None

    def split(self, sizes: list[int]) -> list["TrackModality"]:
        # Modality.split() in upstream versions reconstructs Modality, losing
        # subclass fields. Preserve them when BatchSplitAdapter splits a batch.
        parts = {
            field.name: (
                list(value.split(sizes, dim=0))
                if isinstance(value := getattr(self, field.name), torch.Tensor)
                else [value] * len(sizes)
            )
            for field in fields(self)
        }
        return [type(self)(**{key: values[i] for key, values in parts.items()}) for i in range(len(sizes))]


class TrackDenoiser(SimpleDenoiser):
    def __init__(self, v_context, a_context, *, target_token_count, track_xy=None, track_valid=None):
        super().__init__(v_context, a_context)
        self.target_token_count = target_token_count
        self.track_xy = track_xy
        self.track_valid = track_valid

    def __call__(self, transformer, video_state, audio_state, sigmas, step_index):
        def with_track(*, video, audio, perturbations):
            if video is not None:
                reference_count = video.latent.shape[1] - self.target_token_count
                if self.target_token_count <= 0 or reference_count < 0:
                    raise ValueError("Target/reference video token counts do not partition the latent sequence")
                video = TrackModality(
                    **{field.name: getattr(video, field.name) for field in fields(Modality)}
                )
                video = replace(
                    video, target_token_count=self.target_token_count,
                    reference_token_count=reference_count,
                )
            if audio is not None:
                if self.track_xy is not None:
                    if self.track_xy.shape != (*audio.latent.shape[:2], 2):
                        raise ValueError("track_xy must match the audio latent shape [B, T, 2]")
                    if self.track_valid is None or self.track_valid.shape != audio.latent.shape[:2]:
                        raise ValueError("track_valid must match the audio latent shape [B, T]")
                audio = TrackModality(
                    **{field.name: getattr(audio, field.name) for field in fields(Modality)}
                )
                audio = replace(audio, track_xy=self.track_xy, track_valid=self.track_valid)
            return transformer(video=video, audio=audio, perturbations=perturbations)

        # Delegate sigma handling and result wrapping to the installed pipeline.
        return super().__call__(with_track, video_state, audio_state, sigmas, step_index)


def _track_model_configurator(transformer_args: dict[str, Any]):
    """Build the project class before checkpoint loading, not by changing flags afterwards."""
    from ltx_core.model.model_protocol import ModelConfigurator
    from ltx_core.model.transformer.model_configurator import _build_caption_projections
    from ltx_core.model.transformer.model_track import TrackLTXModel
    from ltx_core.model.transformer.rope import LTXRopeType
    from ltx_core.utils import check_config_value

    class TrackModelConfigurator(ModelConfigurator):
        @classmethod
        def from_config(cls, config: dict, ops=None):
            caption_projection, audio_caption_projection = _build_caption_projections(config, is_av=True)
            cfg = config.get("transformer", {})
            # Same architecture constraints as the project's base LTX configurator.
            required = {
                "dropout": 0.0, "attention_bias": True, "num_vector_embeds": None,
                "activation_fn": "gelu-approximate", "num_embeds_ada_norm": 1000,
                "use_linear_projection": False, "only_cross_attention": False,
                "cross_attention_norm": True, "double_self_attention": False,
                "upcast_attention": False, "standardization_norm": "rms_norm",
                "norm_elementwise_affine": False, "qk_norm": "rms_norm",
                "positional_embedding_type": "rope", "use_audio_video_cross_attention": True,
                "share_ff": False, "av_cross_ada_norm": True, "use_middle_indices_grid": True,
                # This project's model predates architectures without FFN bias
                # or prompt AdaLN; never silently interpret their checkpoints.
                "ff_bias": True, "audio_ff_bias": True, "use_prompt_adaln_single": True,
                "use_keyframes_abs_pos_embedding": False,
            }
            for key, value in required.items():
                if key in cfg:
                    check_config_value(cfg, key, value)
            defaults = {
                "num_attention_heads": 32, "attention_head_dim": 128,
                "in_channels": 128, "out_channels": 128, "num_layers": 48,
                "cross_attention_dim": 4096, "norm_eps": 1e-6,
                "positional_embedding_theta": 10000.0,
                "positional_embedding_max_pos": [20, 2048, 2048],
                "timestep_scale_multiplier": 1000, "use_middle_indices_grid": True,
                "audio_num_attention_heads": 32, "audio_attention_head_dim": 64,
                "audio_in_channels": 128, "audio_out_channels": 128,
                "audio_cross_attention_dim": 2048, "audio_positional_embedding_max_pos": [20],
                "av_ca_timestep_scale_multiplier": 1, "apply_gated_attention": False,
                "cross_attention_adaln": False,
            }
            kwargs = {key: cfg.get(key, value) for key, value in defaults.items()}
            if ops is not None:
                kwargs["ops"] = ops
            return TrackLTXModel(
                **kwargs,
                rope_type=LTXRopeType(cfg.get("rope_type", "split")),
                double_precision_rope=cfg.get("frequencies_precision") == "float64",
                caption_projection=caption_projection,
                audio_caption_projection=audio_caption_projection,
                transformer_args=dict(transformer_args),
            )

        @classmethod
        def from_metadata(cls, metadata: dict, ops=None):
            return cls.from_config(metadata.get("config", {}), ops=ops)

    return TrackModelConfigurator


class ICLoraPipeline:
    """
    Two-stage video generation pipeline with In-Context (IC) LoRA support.
    Allows conditioning the generated video on control signals such as depth maps,
    human pose, or image edges via the video_conditioning parameter.
    The specific IC-LoRA model should be provided via the loras parameter.
    Stage 1 generates video at half of the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    Both stages use distilled models for efficiency.
    """

    def __init__(
        self,
        distilled_checkpoint_path: str,
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        track_injection: str | None = None,
        track_conditioning_path: str | Path | None = None,
        simple_summary_path: str | Path | None = None,
        use_track_rope: bool = False,
        apply_track_with_timesteps: bool = False,
        track_ste_kwargs: dict[str, Any] | None = None,
        track_rope_kwargs: dict[str, Any] | None = None,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self._validate_track_options(
            track_injection=track_injection,
            track_conditioning_path=track_conditioning_path,
            simple_summary_path=simple_summary_path,
            use_track_rope=use_track_rope,
            apply_track_with_timesteps=apply_track_with_timesteps,
            track_ste_kwargs=track_ste_kwargs,
            track_rope_kwargs=track_rope_kwargs,
            offload_mode=offload_mode,
            compilation_config=compilation_config,
            quantization=quantization,
        )
        self.track_injection = track_injection
        self.use_track_rope = use_track_rope
        self.requires_track = track_injection is not None or use_track_rope
        self.use_simple_summary = simple_summary_path is not None
        self.track_conditioning_path = self._validate_weights_path(track_conditioning_path)
        self.simple_summary_path = self._validate_weights_path(simple_summary_path)
        stage_1_options = {}
        if self.requires_track or self.use_simple_summary:
            stage_1_options["model_configurator"] = _track_model_configurator({
                "track_injection": track_injection,
                "use_simple_summary": self.use_simple_summary,
                "use_track_rope": use_track_rope,
                "apply_track_with_timesteps": apply_track_with_timesteps,
                "track_ste_kwargs": dict(track_ste_kwargs or {}),
                "track_rope_kwargs": dict(track_rope_kwargs or {}),
            })

        self.prompt_encoder = PromptEncoder(
            distilled_checkpoint_path,
            gemma_root,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
        )

        self.image_conditioner = ImageConditioner(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.stage_1 = DiffusionStage(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            # Late modules must not survive in a cached shell that the base
            # loader cannot rematerialize on the next sample.
            registry=DummyRegistry() if stage_1_options else registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            **stage_1_options,
        )
        self.stage_2 = DiffusionStage(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            loras=(),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
        )
        self.upsampler = VideoUpsampler(
            distilled_checkpoint_path, spatial_upsampler_path, self.dtype, self.device, registry=registry
        )
        self.video_decoder = VideoDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.audio_decoder = AudioDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)

        if self.requires_track or self.use_simple_summary:
            self._wrap_transformer_context(self.stage_1)

        # Read reference scale factors from LoRA metadata.
        # IC-LoRAs trained with scaled reference videos store these factors
        # so inference can resize/subsample reference videos to match training conditions.
        self.reference_downscale_factor = 2
        self.reference_temporal_scale_factor = 1
        for lora in loras:
            scale = read_lora_reference_downscale_factor(lora.path)
            if scale != 1:
                if self.reference_downscale_factor not in (1, scale):
                    raise ValueError(
                        f"Conflicting reference_downscale_factor values in LoRAs: "
                        f"already have {self.reference_downscale_factor}, but {lora.path} "
                        f"specifies {scale}. Cannot combine LoRAs with different reference scales."
                    )
                self.reference_downscale_factor = scale
            temporal = read_lora_reference_temporal_scale_factor(lora.path)
            if temporal != 1:
                if self.reference_temporal_scale_factor not in (1, temporal):
                    raise ValueError(
                        f"Conflicting reference_temporal_scale_factor values in LoRAs: "
                        f"already have {self.reference_temporal_scale_factor}, but {lora.path} "
                        f"specifies {temporal}. Cannot combine LoRAs with different temporal scales."
                    )
                self.reference_temporal_scale_factor = temporal

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
        conditioning_attention_mask: torch.Tensor | None = None,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        track: str | Path | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        """
        Generate video with IC-LoRA conditioning.
        Args:
            prompt: Text prompt for video generation.
            seed: Random seed for reproducibility.
            height: Output video height in pixels (must be divisible by 64).
            width: Output video width in pixels (must be divisible by 64).
            num_frames: Number of frames to generate.
            frame_rate: Output video frame rate.
            images: List of (path, frame_idx, strength) tuples for image conditioning.
            video_conditioning: List of (path, strength) tuples for IC-LoRA video conditioning.
            enhance_prompt: Whether to enhance the prompt using the text encoder.
            tiling_config: Optional tiling configuration for VAE decoding.
            conditioning_attention_strength: Scale factor for IC-LoRA conditioning attention.
                Controls how strongly the conditioning video influences the output.
                0.0 = ignore conditioning, 1.0 = full conditioning influence. Default 1.0.
                When conditioning_attention_mask is provided, the mask is multiplied by
                this strength before being passed to the conditioning items.
            skip_stage_2: If True, skip Stage 2 upsampling and refinement. Output will be
                at half resolution (height//2, width//2). Default is False.
            conditioning_attention_mask: Optional pixel-space attention mask with the same
                spatial-temporal dimensions as the input reference video. Shape should be
                (B, 1, F, H, W) or (1, 1, F, H, W) where F, H, W match the reference
                video's pixel dimensions. Values in [0, 1].
                The mask is downsampled to latent space using VAE scale factors (with
                causal temporal handling for the first frame), then multiplied by
                conditioning_attention_strength.
                When None (default): scalar conditioning_attention_strength is used
                directly.
        Returns:
            Tuple of (video_iterator, audio_tensor).
        """
        assert_resolution(height=height, width=width, is_two_stage=True)
        if not (0.0 <= conditioning_attention_strength <= 1.0):
            raise ValueError(
                f"conditioning_attention_strength must be in [0.0, 1.0], got {conditioning_attention_strength}"
            )

        if self.requires_track and track is None:
            raise ValueError("The selected track module configuration requires a track for every sample")
        if self.use_simple_summary and not video_conditioning:
            raise ValueError("Simple Summary requires IC reference video conditioning in Stage 1")
        track_xy, track_valid = self._prepare_track(
            track if self.requires_track else None, num_frames, frame_rate
        )
        # breakpoint()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        # Stage 1: Initial low resolution video generation.
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )

        # Encode conditionings using the video encoder block
        stage_1_conditionings = self.image_conditioner(
            lambda enc: self._create_conditionings(
                images=images,
                video_conditioning=video_conditioning,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=enc,
                num_frames=num_frames,
                conditioning_attention_strength=conditioning_attention_strength,
                conditioning_attention_mask=conditioning_attention_mask,
            )
        )
        # breakpoint()
        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        video_state, audio_state = self.stage_1(
            denoiser=TrackDenoiser(
                video_context, audio_context, track_xy=track_xy, track_valid=track_valid,
                target_token_count=VideoLatentShape.from_pixel_shape(stage_1_output_shape).token_count(),
            ),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_1_conditionings,
            ),
            audio=ModalitySpec(context=audio_context),
        )

        if skip_stage_2:
            logging.info("[IC-LoRA] Skipping Stage 2 (--skip-stage-2 enabled)")
            decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
            decoded_audio = self.audio_decoder(audio_state.latent)
            return decoded_video, decoded_audio

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = self.upsampler(video_state.latent[:1])

        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=enc,
                dtype=self.dtype,
                device=self.device,
            )
        )

        video_state, audio_state = self.stage_2(
            denoiser=TrackDenoiser(
                video_context, audio_context,
                target_token_count=VideoLatentShape.from_pixel_shape(stage_2_output_shape).token_count(),
            ),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio

    @staticmethod
    def _validate_track_options(
        *, track_injection, track_conditioning_path, simple_summary_path,
        use_track_rope, apply_track_with_timesteps, track_ste_kwargs,
        track_rope_kwargs, offload_mode, compilation_config, quantization,
    ) -> None:
        if track_injection not in (None, "v2"):
            raise ValueError("track_injection must be None or 'v2'")
        requires_track = track_injection is not None or use_track_rope
        if requires_track != (track_conditioning_path is not None):
            raise ValueError(
                "--track-conditioning-weights requires --track-injection v2 "
                "or --use-track-rope, and each enabled Track mode requires its weights"
            )
        if track_injection == "v2" and simple_summary_path is None:
            raise ValueError("track_injection='v2' requires --simple-summary-weights")
        if apply_track_with_timesteps and track_injection != "v2":
            raise ValueError("--apply-track-with-timesteps requires --track-injection v2")
        if track_ste_kwargs and track_injection != "v2":
            raise ValueError("--track-ste-kwargs applies only to the model-level v2 STE")
        if track_rope_kwargs and not use_track_rope:
            raise ValueError("--track-rope-kwargs requires --use-track-rope")
        if requires_track or simple_summary_path is not None:
            # Streaming's weights provider only knows the base checkpoint; late
            # modules are not part of its block storage. Do not silently omit them.
            if offload_mode != OffloadMode.NONE:
                raise ValueError("Custom Track/Summary modules currently require --offload-mode none")
            if compilation_config is not None:
                raise ValueError("Disable compilation when loading custom Track/Summary modules")
            if quantization is not None:
                raise ValueError("Custom Track/Summary inference currently requires unquantized BF16")

    @staticmethod
    def _validate_weights_path(path: str | Path | None) -> Path | None:
        if path is None:
            return None
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Module weights do not exist: {resolved}")
        return resolved

    def _wrap_transformer_context(self, stage: DiffusionStage) -> None:
        """Initialize extras after base loading, before the first denoising step."""
        original_transformer_ctx = stage._transformer_ctx

        @contextmanager
        def track_aware_transformer_ctx(*args: Any, **kwargs: Any):
            # Build dtype must be explicit: the stage otherwise may retain FP32.
            kwargs["dtype"] = self.dtype
            with original_transformer_ctx(*args, **kwargs) as transformer:
                self._configure_transformer_instance(transformer)
                yield transformer

        stage._transformer_ctx = track_aware_transformer_ctx

    def _configure_transformer_instance(self, transformer: torch.nn.Module) -> None:
        core = getattr(transformer, "velocity_model", transformer)
        if getattr(core, "track_injection", None) != self.track_injection:
            raise RuntimeError("The loaded transformer does not match the requested Track mode")
        initializer = getattr(core, "initialize_track_conditioning_modules", None)
        if initializer is None:
            raise RuntimeError("Expected TrackLTXModel; synchronize the project's model_track.py")
        initializer(device=self.device, dtype=self.dtype)
        for block in core.transformer_blocks:
            if self.track_injection is not None:
                block.initialize_track_injection_modules(device=self.device, dtype=self.dtype)
            if self.use_simple_summary:
                block.initialize_simple_summary(device=self.device, dtype=self.dtype)
        # Each context owns a fresh shell; restore every trained extra.
        if self.track_conditioning_path is not None:
            self._load_module_weights(core, self.track_conditioning_path, "track_conditioning")
        if self.simple_summary_path is not None:
            self._load_module_weights(core, self.simple_summary_path, "simple_summary")
        meta_names = [name for name, value in core.state_dict().items() if value.is_meta]
        if meta_names:
            raise RuntimeError(f"Unmaterialized transformer tensors after loading: {meta_names[:5]}")
        # Newly attached STE dropout layers otherwise remain in training mode.
        transformer.eval()

    @staticmethod
    def _module_anchor(key: str, marker: str) -> str | None:
        """Normalize known PEFT/FSDP/X0 wrappers without suffix guessing."""
        wrappers = ("module.", "_fsdp_wrapped_module.", "_orig_mod.",
                    "base_model.model.", "model.", "velocity_model.")
        while True:
            prefix = next((prefix for prefix in wrappers if key.startswith(prefix)), None)
            if prefix is None:
                break
            key = key[len(prefix):]
        key = key.replace("._fsdp_wrapped_module.", ".").replace("._orig_mod.", ".")
        if marker == "simple_summary":
            if key.startswith("transformer_blocks.") and (
                ".simple_summary." in key or key.endswith(".simple_summary_scale")
            ):
                return key
        elif marker == "track_conditioning":
            if key.startswith(("spatial_track_encoder.", "residual_track_rope.", "track_timesteps_adaln.")):
                return key
            if key.startswith("transformer_blocks.") and (
                ".spatial_track_encoder_version1." in key
                or key.endswith(".scale_shift_table_audio_track")
            ):
                return key
        else:
            raise ValueError(f"Unknown module marker: {marker}")
        return None

    @staticmethod
    def _load_module_weights(model: torch.nn.Module, path: str | Path, marker: str) -> None:
        """Require complete, shape-compatible module weights before copying anything."""
        path = Path(path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(path), device="cpu")
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"Expected a state dict in {path}")

        expected = {}
        for key, value in model.state_dict().items():
            anchor = ICLoraPipeline._module_anchor(key, marker)
            if anchor is not None:
                if anchor in expected:
                    raise ValueError(f"Ambiguous model parameter: {anchor}")
                expected[anchor] = (key, value)
        selected = {}
        unexpected = []
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            anchor = ICLoraPipeline._module_anchor(key, marker)
            if anchor is None:
                unexpected.append(key)
                continue
            if anchor in selected:
                raise ValueError(f"Duplicate checkpoint parameter after prefix normalization: {anchor}")
            selected[anchor] = value

        missing = sorted(expected.keys() - selected.keys())
        unexpected.extend(sorted(selected.keys() - expected.keys()))
        mismatched = [
            f"{key}: checkpoint {tuple(selected[key].shape)} != model {tuple(value.shape)}"
            for key, (_, value) in expected.items()
            if key in selected and selected[key].shape != value.shape
        ]
        if not expected or missing or unexpected or mismatched:
            raise ValueError(
                f"Incompatible {marker} checkpoint {path}. "
                f"Missing: {missing[:5]}; unexpected: {unexpected[:5]}; shapes: {mismatched[:5]}. "
                "Use the complete trainer module checkpoint and the same v2, "
                "Track-RoPE, timestep, STE and Summary configuration as training."
            )
        model.load_state_dict({expected[anchor][0]: value for anchor, value in selected.items()}, strict=False)
        logging.info("Loaded all %d %s tensors from %s", len(selected), marker, path)

    def _prepare_track(
        self, path: str | Path | None, num_frames: int, frame_rate: float
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if path is None:
            return None, None
        raw = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
        if not isinstance(raw, (torch.Tensor, dict)):
            raise ValueError(f"Track {path} must be a tensor or a track_xy/track_valid dict")
        xy = raw if isinstance(raw, torch.Tensor) else raw.get("track_xy")
        valid = None if isinstance(raw, torch.Tensor) else raw.get("track_valid")
        if not isinstance(xy, torch.Tensor):
            raise ValueError(f"Track {path} must be a tensor or contain track_xy")
        if xy.ndim == 3 and xy.shape[0] == 1:
            xy = xy[0]
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"Track {path} must have shape [T, 2], got {tuple(xy.shape)}")
        if xy.shape[0] == 0:
            raise ValueError(f"Track {path} must contain at least one point")
        if valid is not None and not isinstance(valid, torch.Tensor):
            raise ValueError(f"Track {path} track_valid must be a tensor")
        if valid is not None and valid.ndim == 2 and valid.shape[0] == 1:
            valid = valid[0]
        valid = torch.ones(xy.shape[0], dtype=torch.bool) if valid is None else valid.bool()
        if valid.shape != (xy.shape[0],):
            raise ValueError(
                f"Track {path} track_valid must have shape {(xy.shape[0],)}, got {tuple(valid.shape)}"
            )
        valid = valid & torch.isfinite(xy).all(-1)
        token_count = AudioLatentShape.from_duration(1, num_frames / frame_rate).token_count()
        xy = torch.nn.functional.interpolate(
            torch.where(valid[:, None], xy.float(), 0.0).T[None],
            size=token_count, mode="linear", align_corners=True
        ).transpose(1, 2)
        valid = torch.nn.functional.interpolate(
            valid.float()[None, None], size=token_count, mode="nearest"
        ).squeeze(1).bool()
        return xy.to(self.device, self.dtype), valid.to(self.device)

    def _create_conditionings(
        self,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        video_encoder: VideoEncoder,
        conditioning_attention_strength: float = 1.0,
        conditioning_attention_mask: torch.Tensor | None = None,
    ) -> list[ConditioningItem]:
        conditionings = combined_image_conditionings(
            images=images,
            height=height,
            width=width,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
        )

        append_ic_lora_reference_video_conditionings(
            conditionings,
            video_conditioning,
            height=height,
            width=width,
            num_frames=num_frames,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
            reference_downscale_factor=self.reference_downscale_factor,
            reference_temporal_scale_factor=self.reference_temporal_scale_factor,
            conditioning_attention_strength=conditioning_attention_strength,
            conditioning_attention_mask=conditioning_attention_mask,
            tiling_config=None,
        )

        if video_conditioning:
            logging.info("[IC-LoRA] Added %d video conditioning(s)", len(video_conditioning))

        return conditionings


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    parser = default_2_stage_distilled_arg_parser(params=params)
    prompt_action = next(
        (action for action in parser._actions if action.dest == "prompt"),
        None,
    )
    if prompt_action is None:
        raise RuntimeError("The LTX argument parser does not define --prompt")
    prompt_action.required = False
    parser.add_argument(
        "--video-conditioning",
        action=VideoConditioningAction,
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=False,
    )
    parser.add_argument(
        "--conditioning-attention-mask",
        action=VideoMaskConditioningAction,
        nargs=2,
        metavar=("MASK_PATH", "STRENGTH"),
        default=None,
        help=(
            "Optional spatial attention mask: path to a grayscale mask video and "
            "attention strength. The mask video pixel values in [0,1] control "
            "per-region conditioning attention strength. The strength scalar is "
            "multiplied with the spatial mask. "
            "0.0 = ignore IC-LoRA conditioning, 1.0 = full conditioning influence. "
            "When not provided, full conditioning strength (1.0) is used. "
            "Example: --conditioning-attention-mask path/to/mask.mp4 0.5"
        ),
    )
    parser.add_argument(
        "--batch-json",
        type=Path,
        help=(
            "JSON array (or an object with a 'samples' array). Each sample must contain "
            "prompt and reference_video; track is required for Track modes. "
            "output_path and per-sample overrides are optional."
        ),
    )
    parser.add_argument(
        "--sample-num",
        type=int,
        default=None,
        help=(
            "Generate SAMPLE_NUM videos for every input sample. The first video uses "
            "the sample/base seed, and later videos use consecutive seeds. When this "
            "argument is supplied, every sample is saved in its own subdirectory."
        ),
    )
    parser.add_argument("--track-injection", choices=("v2",))
    parser.add_argument("--track", type=Path, help="Track .pt file for single-sample inference.")
    parser.add_argument(
        "--track-conditioning-weights",
        type=Path,
        help="Complete track_conditioning_step_*.safetensors saved by the v2 trainer.",
    )
    parser.add_argument(
        "--simple-summary-weights",
        type=Path,
        help="Enable Simple-Summary and load its trained module checkpoint.",
    )
    parser.add_argument("--use-track-rope", action="store_true",
                        help="Enable the current model-level ResidualTrackRoPE, as configured in training.")
    parser.add_argument("--apply-track-with-timesteps", action="store_true")
    parser.add_argument("--track-ste-kwargs", type=_json_object, default=None,
                        help="JSON object matching training's v2 track_ste_kwargs.")
    parser.add_argument("--track-rope-kwargs", type=_json_object, default=None,
                        help="JSON object matching training's track_rope_kwargs.")
    parser.add_argument(
        "--skip-stage-2",
        action="store_true",
        help=(
            "Skip Stage 2 upsampling and refinement. Output will be at half resolution "
            "(height//2, width//2). Useful for faster iteration or when GPU memory is limited."
        ),
    )
    parser.add_argument(
        "--decode-mode",
        action="store_true",
        help=(
            "After saving each generated video, decode its Mid/Side (mono diff) audio "
            "back into Left/Right stereo and save the result under a 'generate_decode' "
            "folder inside --output-path."
        ),
    )
    parser.add_argument(
        "--use-ltxattn",
        action="store_true",
    )
    parser.add_argument(
        "--trackadaln-version",
        default="V1",
        type=str,
    )
    
    parser.add_argument(
        "--track-rope-ratio",
        default=0.25,
        type=float,
    )
    parser.add_argument(
        "--summary-token-num",
        default=32,
        type=int,
    )
    
    args = parser.parse_args()

    if args.sample_num is not None and args.sample_num < 1:
        parser.error("--sample-num must be greater than or equal to 1")

    if args.batch_json is None:
        if not args.prompt:
            parser.error("--prompt is required when --batch-json is not used")
        if not args.video_conditioning:
            parser.error("--video-conditioning is required when --batch-json is not used")
    elif args.track is not None:
        parser.error("With --batch-json, put track paths in each sample instead of --track")

    global_args.USE_LTX_ATTN = args.use_ltxattn
    global_args.TRACKADALN_VERSION = args.trackadaln_version
    global_args.TRACK_ROPE_RATIO = args.track_rope_ratio
    global_args.SUMMARY_TOKEN_NUM = args.summary_token_num

    requires_track = args.track_injection is not None or args.use_track_rope
    samples = _read_batch_samples(args.batch_json, require_track=requires_track) if args.batch_json else [{
        "prompt": args.prompt,
        "reference_video": args.video_conditioning[0][0],
        "reference_strength": args.video_conditioning[0][1],
        "track": args.track,
        "output_path": None,
    }]
    if requires_track and not args.batch_json and args.track is None:
        parser.error("The selected Track mode requires --track or per-sample tracks in --batch-json")

    conditioning_attention_mask = None
    conditioning_attention_strength = 1.0
    if args.conditioning_attention_mask is not None:
        mask_path, mask_strength = args.conditioning_attention_mask
        conditioning_attention_strength = mask_strength
        conditioning_attention_mask = _load_mask_video(
            mask_path=mask_path,
            height=args.height // 2,
            width=args.width // 2,
            num_frames=args.num_frames,
        )

    pipeline = ICLoraPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        track_injection=args.track_injection,
        track_conditioning_path=args.track_conditioning_weights,
        simple_summary_path=args.simple_summary_weights,
        use_track_rope=args.use_track_rope,
        apply_track_with_timesteps=args.apply_track_with_timesteps,
        track_ste_kwargs=args.track_ste_kwargs,
        track_rope_kwargs=args.track_rope_kwargs,
    )
    tiling_config = TilingConfig.default()
    result_directory = _result_directory(
        output_root=args.output_path,
        track_injection=args.track_injection,
        track_conditioning_weights=args.track_conditioning_weights,
        simple_summary_weights=args.simple_summary_weights,
        use_track_rope=args.use_track_rope,
        apply_track_with_timesteps=args.apply_track_with_timesteps,
    )
    result_directory.mkdir(parents=True, exist_ok=True)
    generated_directory = result_directory / "generated"
    concat_directory = result_directory / "concat"
    generated_directory.mkdir(parents=True, exist_ok=True)
    concat_directory.mkdir(parents=True, exist_ok=True)
    logging.info("Writing results to %s", result_directory)
    sample_num = args.sample_num if args.sample_num is not None else 1
    use_sample_subdirectories = args.sample_num is not None

    for index, sample in enumerate(samples):
        sample_frames = int(sample.get("num_frames", args.num_frames))
        sample_fps = float(sample.get("frame_rate", args.frame_rate))
        base_seed = int(sample.get("seed", args.seed))
        base_output_path = Path(
            _sample_output_path(
                generated_directory,
                sample.get("output_path"),
                index,
            )
        )

        if use_sample_subdirectories:
            sample_generated_directory = generated_directory / base_output_path.stem
            sample_concat_directory = concat_directory / base_output_path.stem
            sample_generated_directory.mkdir(parents=True, exist_ok=True)
            sample_concat_directory.mkdir(parents=True, exist_ok=True)
        else:
            sample_generated_directory = generated_directory
            sample_concat_directory = concat_directory

        for generation_index in range(sample_num):
            current_seed = base_seed + generation_index

            if use_sample_subdirectories:
                output_path = str(
                    sample_generated_directory
                    / f"seed_{current_seed}{base_output_path.suffix}"
                )
            else:
                output_path = str(base_output_path)

            logging.info(
                "Generating batch sample %d/%d, variation %d/%d, seed=%d -> %s",
                index + 1,
                len(samples),
                generation_index + 1,
                sample_num,
                current_seed,
                output_path,
            )

            video, audio = pipeline(
                prompt=sample["prompt"],
                seed=current_seed,
                height=int(sample.get("height", args.height)),
                width=int(sample.get("width", args.width)),
                num_frames=sample_frames,
                frame_rate=sample_fps,
                images=args.images,
                video_conditioning=[
                    (
                        sample["reference_video"],
                        float(sample.get("reference_strength", 1.0)),
                    )
                ],
                track=sample.get("track"),
                tiling_config=tiling_config,
                conditioning_attention_strength=conditioning_attention_strength,
                skip_stage_2=args.skip_stage_2,
                conditioning_attention_mask=conditioning_attention_mask,
            )

            gen_frames = torch.cat(
                [chunk.to("cpu") for chunk in video],
                dim=0,
            )

            encode_video(
                video=gen_frames,
                fps=sample_fps,
                audio=audio,
                output_path=output_path,
                video_chunks_number=1,
            )

            if args.decode_mode:
                decode_directory = result_directory / "generate_decode"
                if use_sample_subdirectories:
                    decode_directory = decode_directory / base_output_path.stem
                _decode_mono_diff_to_stereo(
                    video_path=output_path,
                    output_dir=decode_directory,
                )

            reference_video = sample.get("reference_video")
            if reference_video:
                ref_frames = _load_reference_frames_for_concat(
                    reference_video=reference_video,
                    target_height=int(gen_frames.shape[1]),
                    frame_cap=int(gen_frames.shape[0]),
                    device=get_device(),
                )
                if ref_frames is not None:
                    n = min(int(gen_frames.shape[0]), int(ref_frames.shape[0]))
                    concat_frames = torch.cat(
                        [ref_frames[:n], gen_frames[:n]],
                        dim=2,
                    )
                    concat_path = _concat_output_path(
                        output_path,
                        sample_concat_directory,
                    )
                    encode_video(
                        video=concat_frames,
                        fps=sample_fps,
                        audio=audio,
                        output_path=concat_path,
                        video_chunks_number=1,
                    )


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return parsed


def _read_batch_samples(path: Path, *, require_track: bool = False) -> list[dict[str, Any]]:
    """Validate and resolve paths in the batch manifest relative to the JSON file."""
    with path.expanduser().open(encoding="utf-8") as handle:
        document = json.load(handle)
    samples = document.get("samples") if isinstance(document, dict) else document
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path} must contain a non-empty samples array")
    base = path.expanduser().resolve().parent
    result = []
    for index, value in enumerate(samples):
        if not isinstance(value, dict):
            raise TypeError(f"Sample {index} in {path} must be an object")
        required = {"prompt", "reference_video"} | ({"track"} if require_track else set())
        missing = {key for key in required if not value.get(key)}
        if missing:
            raise ValueError(f"Sample {index} in {path} is missing: {', '.join(sorted(missing))}")
        sample = dict(value)
        for key in ("reference_video", "track"):
            if sample.get(key) and not Path(sample[key]).expanduser().is_absolute():
                sample[key] = str(base / sample[key])
        result.append(sample)
    return result


def _result_directory(
    *,
    output_root: str | Path,
    track_injection: str | None,
    track_conditioning_weights: str | Path | None,
    simple_summary_weights: str | Path | None,
    use_track_rope: bool = False,
    apply_track_with_timesteps: bool = False,
) -> Path:
    """Build the module-specific result directory below ``output_root``."""
    names = []
    if track_injection is not None:
        names.append(track_injection)
    if track_conditioning_weights is not None:
        names.append(Path(track_conditioning_weights).expanduser().name)
    if simple_summary_weights is not None:
        names.append(Path(simple_summary_weights).expanduser().name)
    if use_track_rope:
        names.append("track_rope")
    if apply_track_with_timesteps:
        names.append("track_timesteps")
    child_name = "__".join(names) if names else "vanilla_results"
    base = Path(output_root).expanduser()
    candidate = base / child_name
    version = 1
    while candidate.exists():
        candidate = base / f"{child_name}_v{version}"
        version += 1
    return candidate


def _sample_output_path(result_directory: Path, requested_path: str | Path | None, index: int) -> str:
    """Return a unique output file inside the selected result directory."""
    if requested_path:
        filename = Path(requested_path).name
        if not filename:
            raise ValueError(f"Sample {index} output_path must include a file name")
    else:
        filename = f"sample_{index:04d}.mp4"
    return str(result_directory / filename)


def _concat_output_path(output_path: str, concat_directory: Path | None = None) -> str:
    """Derive the side-by-side comparison path by appending ``_concat`` to the stem."""
    path = Path(output_path)
    filename = f"{path.stem}_concat{path.suffix}"
    if concat_directory is not None:
        return str(concat_directory / filename)
    return str(path.with_name(filename))


def _load_reference_frames_for_concat(
    reference_video: str,
    target_height: int,
    frame_cap: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Load and format reference video frames for side-by-side concatenation."""
    frames: list[torch.Tensor] = []
    for f in decode_video_by_frame(path=reference_video, frame_cap=frame_cap, device=device):
        src_h, src_w = int(f.shape[1]), int(f.shape[2])
        target_width = max(2, int(round(src_w * target_height / src_h)) // 2 * 2)
        resized = resize_and_center_crop(f.to(torch.float32), target_height, target_width)
        frame = rearrange(resized, "b c f h w -> (b f) h w c").div_(255.0).clamp_(0.0, 1.0)
        frames.append(frame.to("cpu"))
    if not frames:
        return None
    return torch.cat(frames, dim=0)


def _load_mask_video(
    mask_path: str,
    height: int,
    width: int,
    num_frames: int,
) -> torch.Tensor:
    """Load a mask video and return a pixel-space tensor of shape (1, 1, F, H, W)."""
    device = get_device()
    frame_gen = decode_video_by_frame(path=mask_path, frame_cap=num_frames, device=device)
    mask_video = video_preprocess(frame_gen, height, width, torch.bfloat16, device)
    mask = mask_video.mean(dim=1, keepdim=True)
    mask = (mask + 1.0) / 2.0
    return mask.clamp(0.0, 1.0)


def _decode_mono_diff_to_stereo(video_path: str | Path, output_dir: str | Path) -> Path:
    """Decode a video's Mid/Side audio track back into Left/Right stereo."""
    import shutil
    import subprocess
    import tempfile

    import numpy as np
    import soundfile as sf

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for --decode-mode but was not found on PATH.")

    video_path = Path(video_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / video_path.name

    def _run_ffmpeg(cmd: list[str]) -> None:
        completed = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed (returncode={completed.returncode}): {stderr}")

    with tempfile.TemporaryDirectory(prefix="mono_diff_decode_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        extracted_wav = tmpdir_path / "audio_in.wav"
        decoded_wav = tmpdir_path / "audio_out.wav"

        _run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-ac", "2", "-acodec", "pcm_s16le",
            str(extracted_wav),
        ])

        if not extracted_wav.exists() or extracted_wav.stat().st_size == 0:
            raise RuntimeError(f"No usable audio track found in {video_path}")

        audio, sample_rate = sf.read(str(extracted_wav), dtype="float64", always_2d=True)
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError(
                f"Expected stereo (M/D) audio with shape [frames, 2], got {audio.shape}"
            )
        mid = audio[:, 0]
        side = audio[:, 1]
        stereo = np.column_stack((mid + side, mid - side))
        if not np.isfinite(stereo).all():
            raise ValueError("Decoded audio contains NaN or Inf values")
        sf.write(str(decoded_wav), stereo, sample_rate, format="WAV", subtype="PCM_16")

        _run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-i", str(decoded_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest",
            str(output_path),
        ])

    logging.info("[decode-mode] Saved stereo-decoded video to %s", output_path)
    return output_path


if __name__ == "__main__":
    main()
