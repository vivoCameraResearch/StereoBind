from __future__ import annotations

from dataclasses import replace

import torch

from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.modality import Modality

try:
    from .residual_track_rope import ResidualTrackRoPE
    from .spatial_track_encoder import SpatialTrackEncoder_version2
    from .transformer_args import (
        MultiModalTransformerArgsPreprocessor,
        TransformerArgs,
        TransformerArgsPreprocessor,
    )
except ImportError:
    # Allows the file to be used when copied directly into the transformer package
    # or executed with this directory on PYTHONPATH.
    from residual_track_rope import ResidualTrackRoPE
    from spatial_track_encoder import SpatialTrackEncoder_version2
    from transformer_args import (
        MultiModalTransformerArgsPreprocessor,
        TransformerArgs,
        TransformerArgsPreprocessor,
    )


class _ResidualTrackRoPEPreprocessorMixin:
    """Select composed Audio RoPE when use_track_rope enables the module."""

    residual_track_rope: ResidualTrackRoPE | None

    def _with_residual_track_rope(self, args: TransformerArgs) -> TransformerArgs:
        # TrackLTXModel leaves this module unset when use_track_rope=False.
        if self.residual_track_rope is None:
            return args

        track_xy = args.track_xy
        if track_xy is None:
            return args

        residual_track_pe = self.residual_track_rope(
            args.positional_embeddings,
            track_xy,
        )
        return replace(
            args,
            # Audio self-attention reads this field. The result already combines
            # base Audio RoPE and Track-RoPE; cross-attention PE stays unchanged.
            positional_embeddings=residual_track_pe,
            residual_track_positional_embedding=residual_track_pe,
        )


class TrackTransformerArgsPreprocessor(
    _ResidualTrackRoPEPreprocessorMixin,
    TransformerArgsPreprocessor,
):
    """TransformerArgsPreprocessor with optional Residual Track-RoPE support."""

    def __init__(
        self,
        *args,
        residual_track_rope: ResidualTrackRoPE | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.residual_track_rope = residual_track_rope

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,
    ) -> TransformerArgs:
        args = super().prepare(modality, cross_modality)
        return self._with_residual_track_rope(args)


class TrackMultiModalTransformerArgsPreprocessor(
    _ResidualTrackRoPEPreprocessorMixin,
    MultiModalTransformerArgsPreprocessor,
):
    """MultiModal preprocessor with optional Residual Track-RoPE support."""

    def __init__(
        self,
        *args,
        residual_track_rope: ResidualTrackRoPE | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.residual_track_rope = residual_track_rope

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,
    ) -> TransformerArgs:
        args = super().prepare(modality, cross_modality)
        return self._with_residual_track_rope(args)


class _SpatialTrackEncoderPreprocessorMixin:
    """Run the model-owned v2 STE and attach Track-Ada conditions."""

    spatial_track_encoder: SpatialTrackEncoder_version2
    track_timesteps_adaln: AdaLayerNormSingle | None

    def _get_timestep_preprocessor(self):
        # TransformerArgsPreprocessor exposes _prepare_timestep directly, while
        # MultiModalTransformerArgsPreprocessor owns one as simple_preprocessor.
        return getattr(self, "simple_preprocessor", self)

    def _prepare_track_timesteps_embedding(
        self,
        modality: Modality,
        args: TransformerArgs,
    ) -> torch.Tensor | None:
        if self.track_timesteps_adaln is None:
            return None

        timestep_preprocessor = self._get_timestep_preprocessor()
        track_timesteps_embedding, _ = timestep_preprocessor._prepare_timestep(
            modality.timesteps,
            self.track_timesteps_adaln,
            args.x.shape[0],
            modality.latent.dtype,
        )
        track_timesteps_embedding = track_timesteps_embedding.to(
            device=args.x.device,
            dtype=args.x.dtype,
        )

        expected_dim = 3 * args.x.shape[-1]
        if track_timesteps_embedding.ndim != 3:
            raise RuntimeError(
                "track_timesteps_adaln must return [B, N_or_1, 3*D], "
                f"got {tuple(track_timesteps_embedding.shape)}"
            )
        if track_timesteps_embedding.shape[0] != args.x.shape[0]:
            raise RuntimeError(
                "Track timestep embedding batch size must match audio input: "
                f"track_t={track_timesteps_embedding.shape[0]}, audio={args.x.shape[0]}"
            )
        if track_timesteps_embedding.shape[1] not in (1, args.x.shape[1]):
            raise RuntimeError(
                "Track timestep embedding token count must be 1 or match audio tokens: "
                f"track_t={track_timesteps_embedding.shape[1]}, audio={args.x.shape[1]}"
            )
        if track_timesteps_embedding.shape[-1] != expected_dim:
            raise RuntimeError(
                "Track timestep embedding last dimension must equal 3 * audio_dim: "
                f"expected {expected_dim}, got {track_timesteps_embedding.shape[-1]}"
            )

        return track_timesteps_embedding

    def _with_spatial_track_embedding(
        self,
        args: TransformerArgs,
        modality: Modality,
    ) -> TransformerArgs:
        track_xy = args.track_xy
        if track_xy is None:
            raise ValueError(
                "track_injection='v2' requires track_xy, but TransformerArgs.track_xy is None"
            )

        track_xy = track_xy.to(
            device=args.x.device,
            dtype=args.x.dtype,
        )

        track_embedding = self.spatial_track_encoder(
            track_xy,
            target_length=args.x.shape[1],
        )
        track_embedding = track_embedding.to(
            device=args.x.device,
            dtype=args.x.dtype,
        )

        if track_embedding.ndim != 3:
            raise RuntimeError(
                "SpatialTrackEncoder_version2 must return [B, T_audio, 3*D], "
                f"got {tuple(track_embedding.shape)}"
            )
        if track_embedding.shape[0] != args.x.shape[0]:
            raise RuntimeError(
                "Track embedding batch size must match audio transformer input: "
                f"track={track_embedding.shape[0]}, audio={args.x.shape[0]}"
            )
        if track_embedding.shape[1] != args.x.shape[1]:
            raise RuntimeError(
                "Track embedding temporal length must match audio transformer tokens: "
                f"track={track_embedding.shape[1]}, audio={args.x.shape[1]}"
            )

        track_timesteps_embedding = self._prepare_track_timesteps_embedding(
            modality,
            args,
        )

        return replace(
            args,
            track_embedding=track_embedding,
            track_timesteps_embedding=track_timesteps_embedding,
        )


class TrackTransformerArgsPreprocessor_withSTE(
    _SpatialTrackEncoderPreprocessorMixin,
    TrackTransformerArgsPreprocessor,
):
    """Single-modality track preprocessor with SpatialTrackEncoder_version2."""

    def __init__(
        self,
        *args,
        spatial_track_encoder: SpatialTrackEncoder_version2,
        track_timesteps_adaln: AdaLayerNormSingle | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spatial_track_encoder = spatial_track_encoder
        self.track_timesteps_adaln = track_timesteps_adaln

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,
    ) -> TransformerArgs:
        args = super().prepare(modality, cross_modality)
        return self._with_spatial_track_embedding(args, modality)


class TrackMultiModalTransformerArgsPreprocessor_withSTE(
    _SpatialTrackEncoderPreprocessorMixin,
    TrackMultiModalTransformerArgsPreprocessor,
):
    """Audio-video track preprocessor with SpatialTrackEncoder_version2."""

    def __init__(
        self,
        *args,
        spatial_track_encoder: SpatialTrackEncoder_version2,
        track_timesteps_adaln: AdaLayerNormSingle | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spatial_track_encoder = spatial_track_encoder
        self.track_timesteps_adaln = track_timesteps_adaln

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,
    ) -> TransformerArgs:
        args = super().prepare(modality, cross_modality)
        return self._with_spatial_track_embedding(args, modality)