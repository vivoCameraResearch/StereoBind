from __future__ import annotations

from typing import Any

import torch
from ltx_core.model.transformer.adaln import AdaLayerNormSingle

try:
    from .model import LTXModel
    from .residual_track_rope import ResidualTrackRoPE
    from .spatial_track_encoder import SpatialTrackEncoder_version2
    from .transformer import TransformerConfig
    from .transformer_args_track import (
        TrackMultiModalTransformerArgsPreprocessor,
        TrackMultiModalTransformerArgsPreprocessor_withSTE,
        TrackTransformerArgsPreprocessor,
        TrackTransformerArgsPreprocessor_withSTE,
    )
    from .transformer_track import (
        BasicAVTransformerBlockWithSimpleSummary_version2,
    )
except ImportError:
    # Allows this file to work when the directory itself is placed on PYTHONPATH.
    from model import LTXModel
    from residual_track_rope import ResidualTrackRoPE
    from spatial_track_encoder import SpatialTrackEncoder_version2
    from transformer import TransformerConfig
    from transformer_args_track import (
        TrackMultiModalTransformerArgsPreprocessor,
        TrackMultiModalTransformerArgsPreprocessor_withSTE,
        TrackTransformerArgsPreprocessor,
        TrackTransformerArgsPreprocessor_withSTE,
    )
    from transformer_track import (
        BasicAVTransformerBlockWithSimpleSummary_version2,
    )


class TrackLTXModel(LTXModel):
    """LTXModel variant with optional Track-RoPE and track injection branches."""

    def __init__(
        self,
        *,
        use_track_rope: bool = False,
        track_rope_kwargs: dict[str, Any] | None = None,
        track_injection: str | None = None,
        track_ste_kwargs: dict[str, Any] | None = None,
        apply_track_with_timesteps: bool | None = None,
        **kwargs,
    ) -> None:
        # LTXModel.__init__ calls self._init_transformer_blocks(), so the selected
        # injection mode and timestep flag must exist before super().__init__
        # dispatches there.
        args_from_kwargs = dict(kwargs.get("transformer_args") or {})
        self.track_injection = (
            track_injection
            if track_injection is not None
            else args_from_kwargs.get("track_injection")
        )
        self.apply_track_with_timesteps = bool(
            apply_track_with_timesteps
            if apply_track_with_timesteps is not None
            else args_from_kwargs.get("apply_track_with_timesteps", False)
        )
        self.use_track_rope = bool(
            use_track_rope
            or args_from_kwargs.get("use_track_rope", False)
        )
        resolved_track_rope_kwargs = (
            track_rope_kwargs
            if track_rope_kwargs is not None
            else args_from_kwargs.get("track_rope_kwargs")
        )
        resolved_track_ste_kwargs = (
            track_ste_kwargs
            if track_ste_kwargs is not None
            else args_from_kwargs.get("track_ste_kwargs")
        )
        
        self._track_rope_kwargs = dict(resolved_track_rope_kwargs or {})
        self._track_ste_kwargs = dict(resolved_track_ste_kwargs or {})
        
        
        import ltx_core.model.transformer.global_args as global_args
        self._track_rope_kwargs["track_rope_ratio"] = global_args.TRACK_ROPE_RATIO
        
        super().__init__(**kwargs)

        # Do not register model-level track modules during base LTX checkpoint
        # loading. They are materialized by initialize_track_conditioning_modules()
        # after load_transformer() returns.
        self.residual_track_rope: ResidualTrackRoPE | None = None
        # Track-Ada v2 owns a model-level STE shared by the transformer blocks.
        self.spatial_track_encoder: SpatialTrackEncoder_version2 | None = None
        self.track_timesteps_adaln: AdaLayerNormSingle | None = None
        self._track_conditioning_modules_initialized = False

        if self.track_injection not in (None, "v2"):
            raise ValueError(
                "Unsupported track_injection. Expected None or 'v2', "
                f"got {self.track_injection!r}"
            )

        if self.apply_track_with_timesteps and self.track_injection != "v2":
            raise ValueError(
                "apply_track_with_timesteps=True requires track_injection='v2'"
            )

        if (
            self.use_track_rope
            or self.track_injection == "v2"
        ) and not self.model_type.is_audio_enabled():
            raise ValueError(
                "Track conditioning requires the audio branch to be enabled"
            )

    def initialize_track_conditioning_modules(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Create model-level track modules after base LTX weight loading."""
        if self._track_conditioning_modules_initialized:
            return

        device = torch.device(device)
        if device.type == "meta":
            raise RuntimeError(
                "Track conditioning modules must be initialized on a real device"
            )

        if self.track_injection == "v2":
            self._init_spatial_track_encoder_v2(self._track_ste_kwargs)
            if self.spatial_track_encoder is None:
                raise RuntimeError(
                    "track_injection='v2' failed to initialize its model-level "
                    "SpatialTrackEncoder_version2"
                )
            self.spatial_track_encoder = self.spatial_track_encoder.to(
                device=device,
                dtype=dtype,
            )

        # Optional timestep-conditioned Track-Ada branch. Its output has the
        # same [B, N_or_1, 3 * audio_dim] layout as the v2 STE output.
        if self.track_injection == "v2" and self.apply_track_with_timesteps:
            track_timesteps_adaln = AdaLayerNormSingle(
                self.audio_inner_dim,
                embedding_coefficient=3,
            )

            # Preserve the v2 identity initialization: the timestep branch is
            # trainable, but initially contributes zero.
            if hasattr(track_timesteps_adaln, "linear"):
                torch.nn.init.zeros_(track_timesteps_adaln.linear.weight)
                if track_timesteps_adaln.linear.bias is not None:
                    torch.nn.init.zeros_(track_timesteps_adaln.linear.bias)

            self.track_timesteps_adaln = track_timesteps_adaln.to(
                device=device,
                dtype=dtype,
            )

        if self.use_track_rope:
            residual_track_rope_kwargs = dict(self._track_rope_kwargs)
            residual_track_rope_kwargs.setdefault(
                "num_attention_heads",
                self.audio_num_attention_heads,
            )
            
            self.residual_track_rope = ResidualTrackRoPE(
                **residual_track_rope_kwargs
            ).to(
                device=device,
                dtype=dtype,
            )

        # v2 needs STE output; Track-RoPE needs residual PE.
        if self.use_track_rope or self.track_injection == "v2":
            self._init_track_audio_preprocessor()

        self._track_conditioning_modules_initialized = True

    def refresh_track_conditioning_references(self) -> None:
        """Rebind plain preprocessor references after distributed wrapping.

        FSDP may replace registered children with wrappers, but it cannot
        update the references stored by the non-Module args preprocessors.
        Do not recreate the preprocessor here: its initialization uses an
        isinstance(STE) check that does not recognize an FSDP wrapper.
        This only updates references; it never recreates or registers weights.
        """
        preprocessor = getattr(self, "audio_args_preprocessor", None)
        if self.track_injection == "v2":
            if self.spatial_track_encoder is None:
                raise RuntimeError(
                    "Cannot refresh v2 Track conditioning before STE initialization."
                )
            if preprocessor is None or not hasattr(preprocessor, "spatial_track_encoder"):
                raise RuntimeError(
                    "track_injection='v2' requires an audio preprocessor with STE support."
                )

        if preprocessor is None:
            return

        for name in (
            "spatial_track_encoder",
            "track_timesteps_adaln",
            "residual_track_rope",
        ):
            if hasattr(preprocessor, name):
                setattr(preprocessor, name, getattr(self, name, None))

    def _init_spatial_track_encoder_v2(
        self,
        track_ste_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the model-level STE used only by track_injection='v2'."""
        if self.track_injection != "v2":
            raise RuntimeError(
                "_init_spatial_track_encoder_v2() requires "
                "track_injection='v2'"
            )

        ste_kwargs = dict(track_ste_kwargs or {})
        ste_kwargs.setdefault("audio_dim", self.audio_inner_dim)
        ste_kwargs.setdefault("num_ada_params", 3)
        self.spatial_track_encoder = SpatialTrackEncoder_version2(
            **ste_kwargs
        )

    def _init_track_audio_preprocessor(self) -> None:
        """Replace only the audio args preprocessor with the required track variant."""
        use_ste = isinstance(
            self.spatial_track_encoder,
            SpatialTrackEncoder_version2,
        )
        if self.residual_track_rope is None and not use_ste:
            raise RuntimeError(
                "Track audio preprocessor requires ResidualTrackRoPE or SpatialTrackEncoder_version2"
            )

        common_kwargs = dict(
            patchify_proj=self.audio_patchify_proj,
            adaln=self.audio_adaln_single,
            inner_dim=self.audio_inner_dim,
            max_pos=self.audio_positional_embedding_max_pos,
            num_attention_heads=self.audio_num_attention_heads,
            use_middle_indices_grid=self.use_middle_indices_grid,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            double_precision_rope=self.double_precision_rope,
            positional_embedding_theta=self.positional_embedding_theta,
            rope_type=self.rope_type,
            caption_projection=getattr(self, "audio_caption_projection", None),
            prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            residual_track_rope=self.residual_track_rope,
        )
        if use_ste:
            common_kwargs["spatial_track_encoder"] = self.spatial_track_encoder
            common_kwargs["track_timesteps_adaln"] = self.track_timesteps_adaln

        if self.model_type.is_video_enabled():
            cross_pe_max_pos = max(
                self.positional_embedding_max_pos[0],
                self.audio_positional_embedding_max_pos[0],
            )
            preprocessor_cls = (
                TrackMultiModalTransformerArgsPreprocessor_withSTE
                if use_ste
                else TrackMultiModalTransformerArgsPreprocessor
            )
            self.audio_args_preprocessor = preprocessor_cls(
                **common_kwargs,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                cross_pe_max_pos=cross_pe_max_pos,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                is_count_video_ref=False,
            )
        else:
            preprocessor_cls = (
                TrackTransformerArgsPreprocessor_withSTE
                if use_ste
                else TrackTransformerArgsPreprocessor
            )
            self.audio_args_preprocessor = preprocessor_cls(
                **common_kwargs,
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        ops,
        apply_gated_attention: bool,
        transformer_args: dict = {},
        track_injection: str | None = None,
    ) -> None:
        """Initialize transformer blocks with optional v2 track injection."""
        args = dict(transformer_args or {})
        if getattr(self, "apply_track_with_timesteps", False):
            args["apply_track_with_timesteps"] = True

        selected_track_injection = (
            track_injection
            if track_injection is not None
            else getattr(self, "track_injection", None)
        )
        if selected_track_injection is None:
            selected_track_injection = args.get("track_injection")

        if selected_track_injection is None:
            return super()._init_transformer_blocks(
                num_layers=num_layers,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
                audio_attention_head_dim=audio_attention_head_dim,
                audio_cross_attention_dim=audio_cross_attention_dim,
                norm_eps=norm_eps,
                ops=ops,
                apply_gated_attention=apply_gated_attention,
                transformer_args=args,
            )

        if selected_track_injection != "v2":
            raise ValueError(
                "Unsupported track_injection. Expected None or 'v2', "
                f"got {selected_track_injection!r}"
            )
        if not self.model_type.is_audio_enabled():
            raise ValueError(
                f"track_injection={selected_track_injection!r} requires the audio branch"
            )
        if selected_track_injection == "v2" and not bool(args.get("use_simple_summary", False)):
            raise ValueError(
                "track_injection='v2' only modulates _compute_simple_summary and "
                "therefore requires transformer_args['use_simple_summary']=True"
            )

        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = TransformerConfig(
            dim=self.audio_inner_dim,
            heads=self.audio_num_attention_heads,
            d_head=audio_attention_head_dim,
            context_dim=audio_cross_attention_dim,
            apply_gated_attention=apply_gated_attention,
            cross_attention_adaln=self.cross_attention_adaln,
        )

        block_cls = BasicAVTransformerBlockWithSimpleSummary_version2
        self.transformer_blocks = torch.nn.ModuleList(
            [
                block_cls(
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    ops=ops,
                    transformer_args=args,
                )
                for _ in range(num_layers)
            ]
        )
