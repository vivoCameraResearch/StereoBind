from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:
    from .simple_summary_tokens import (
        SimpleTrackSummaryConfig,
        SimpleTrackSummaryOutput,
        SimpleTrackSummaryTokens as _BaseSimpleTrackSummaryTokens,
    )
except ImportError:
    # Allows this file to work when the directory itself is placed on PYTHONPATH.
    from simple_summary_tokens import (
        SimpleTrackSummaryConfig,
        SimpleTrackSummaryOutput,
        SimpleTrackSummaryTokens as _BaseSimpleTrackSummaryTokens,
    )


@dataclass(frozen=True)
class SimpleTrackSummaryTemporalPositionConfig:
    """Temporal-position options for the legacy nn.MultiheadAttention Summary branch.

    This variant is intentionally based on ``SimpleTrackSummaryTokens`` rather
    than ``SimpleTrackSummaryTokensWithLTXAttention``.

    The original class already contains learned ``summary_temporal_positions``.
    Those parameters are retained as learned *summary-slot identities*. This
    module adds a separate deterministic temporal positional encoding that is
    applied to Q/K only in the three MultiheadAttention stages:

        Summary(tau) -> ReferenceVideo(tau)
        Summary(tau) -> TargetVideo(tau)
        Audio(tau)   -> Summary(tau)

    Values do not receive positional encoding.

    ``target_temporal_bins`` and ``reference_temporal_bins`` describe how many
    temporal slices are present in the flattened video token sequences. When
    omitted, target bins default to ``num_summary_tokens`` and reference bins
    are inferred from the target/reference token-count ratio.

    With the repository defaults:

        target:    6144 tokens -> 32 temporal bins -> 192 tokens/bin
        reference: 1536 tokens ->  8 temporal bins -> 192 tokens/bin

    This assumes time-major flattening and the same number of spatial tokens per
    temporal slice in target/reference video tokens.

    Caller-provided normalized temporal positions can override this inference.
    """

    theta: float = 10000.0
    position_span: float | None = None
    position_amplitude: float = 1.0
    target_temporal_bins: int | None = None
    reference_temporal_bins: int | None = None
    strict_temporal_bins: bool = False

    def __post_init__(self) -> None:
        if self.theta <= 1.0:
            raise ValueError(f"theta must be > 1.0, got {self.theta}")
        if self.position_span is not None and self.position_span <= 0.0:
            raise ValueError(
                f"position_span must be positive when provided, got {self.position_span}"
            )
        if self.position_amplitude <= 0.0:
            raise ValueError(
                f"position_amplitude must be positive, got {self.position_amplitude}"
            )
        for name in ("target_temporal_bins", "reference_temporal_bins"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided, got {value}")


class SimpleTrackSummaryTokensWithTemporalPosition(_BaseSimpleTrackSummaryTokens):
    """``SimpleTrackSummaryTokens`` with explicit temporal position encoding.

    This class keeps the original three ``nn.MultiheadAttention`` modules,
    dimensions, residual paths, FFNs, fusion module and zero-initialized final
    projection unchanged.

    The only conceptual change is that the attention Q/K inputs receive an
    explicit 1-D sinusoidal temporal positional encoding:

        1. Summary query PE + Reference-video key PE
        2. Summary query PE + Target-video key PE
        3. Audio query PE   + Summary key PE

    V remains content-only. This preserves the desirable interpretation that
    position changes *where attention reads from* rather than directly changing
    the values being aggregated.

    The existing learned ``summary_temporal_positions`` from the base class are
    still added to the learned Summary tokens. They are treated as slot
    embeddings; the deterministic encoding below is the actual temporal order.
    """

    def __init__(
        self,
        config: SimpleTrackSummaryConfig,
        *,
        temporal_position_config: SimpleTrackSummaryTemporalPositionConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.temporal_position_config = (
            temporal_position_config
            if temporal_position_config is not None
            else SimpleTrackSummaryTemporalPositionConfig()
        )

    @staticmethod
    def _uniform_bin_centers(
        num_bins: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return normalized temporal bin centers in (0, 1), shape [num_bins]."""
        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}")
        return (
            torch.arange(num_bins, device=device, dtype=torch.float32) + 0.5
        ) / float(num_bins)

    def _positions_from_bins(
        self,
        *,
        seq_len: int,
        num_bins: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        """Build [1, seq_len] normalized positions for time-major flattened tokens."""
        if seq_len <= 0:
            raise ValueError(f"{name} sequence length must be positive, got {seq_len}")
        if num_bins <= 0:
            raise ValueError(f"{name} num_bins must be positive, got {num_bins}")

        if seq_len % num_bins != 0:
            if self.temporal_position_config.strict_temporal_bins:
                raise ValueError(
                    f"{name} sequence length {seq_len} is not divisible by "
                    f"temporal bins {num_bins}. Pass the correct bin count or "
                    "set strict_temporal_bins=False."
                )
            # Preserve ordering even when the exact video factorization is unknown.
            return self._uniform_bin_centers(seq_len, device=device).unsqueeze(0)

        tokens_per_bin = seq_len // num_bins
        centers = self._uniform_bin_centers(num_bins, device=device)
        return centers.repeat_interleave(tokens_per_bin).unsqueeze(0)

    @staticmethod
    def _validate_external_positions(
        positions: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        """Validate normalized temporal positions with shape [1|B, T]."""
        if positions.ndim != 2 or positions.shape[1] != seq_len:
            raise ValueError(
                f"{name} must have shape [1|B, {seq_len}], got "
                f"{tuple(positions.shape)}"
            )
        if positions.shape[0] not in (1, batch_size):
            raise ValueError(
                f"{name} batch must be 1 or {batch_size}, got {positions.shape[0]}"
            )
        positions = positions.to(device=device, dtype=torch.float32)
        if not torch.isfinite(positions).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return positions

    def _resolve_temporal_positions(
        self,
        *,
        batch_size: int,
        target_len: int,
        reference_len: int,
        audio_len: int,
        device: torch.device,
        target_temporal_positions: torch.Tensor | None,
        reference_temporal_positions: torch.Tensor | None,
        audio_temporal_positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Summary/target/reference/audio normalized temporal positions."""
        summary_len = self.config.num_summary_tokens
        summary_positions = self._uniform_bin_centers(
            summary_len, device=device
        ).unsqueeze(0)

        target_bins = self.temporal_position_config.target_temporal_bins
        if target_bins is None:
            target_bins = summary_len

        reference_bins = self.temporal_position_config.reference_temporal_bins
        if reference_bins is None:
            configured_target_count = self.config.target_token_count
            configured_reference_count = self.config.reference_token_count
            ratio = configured_reference_count / float(configured_target_count)
            reference_bins = max(1, int(round(target_bins * ratio)))

        if target_temporal_positions is None:
            target_positions = self._positions_from_bins(
                seq_len=target_len,
                num_bins=target_bins,
                device=device,
                name="target",
            )
        else:
            target_positions = self._validate_external_positions(
                target_temporal_positions,
                batch_size=batch_size,
                seq_len=target_len,
                device=device,
                name="target_temporal_positions",
            )

        if reference_temporal_positions is None:
            reference_positions = self._positions_from_bins(
                seq_len=reference_len,
                num_bins=reference_bins,
                device=device,
                name="reference",
            )
        else:
            reference_positions = self._validate_external_positions(
                reference_temporal_positions,
                batch_size=batch_size,
                seq_len=reference_len,
                device=device,
                name="reference_temporal_positions",
            )

        if audio_temporal_positions is None:
            audio_positions = self._uniform_bin_centers(
                audio_len, device=device
            ).unsqueeze(0)
        else:
            audio_positions = self._validate_external_positions(
                audio_temporal_positions,
                batch_size=batch_size,
                seq_len=audio_len,
                device=device,
                name="audio_temporal_positions",
            )

        return (
            summary_positions,
            target_positions,
            reference_positions,
            audio_positions,
        )

    def _sinusoidal_temporal_embedding(
        self,
        positions: torch.Tensor,
        *,
        dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode normalized positions [B,T] as deterministic sinusoidal PE.

        All modalities share the same normalized timeline. ``position_span``
        converts tau in [0,1] to a virtual Transformer position range. By
        default it is ``num_summary_tokens - 1`` so the 32 Summary tokens span
        approximately the same phase range as a standard length-32 sequence.
        """
        if positions.ndim != 2:
            raise ValueError(
                f"positions must have shape [B,T], got {tuple(positions.shape)}"
            )
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        cfg = self.temporal_position_config
        span = (
            float(max(1, self.config.num_summary_tokens - 1))
            if cfg.position_span is None
            else float(cfg.position_span)
        )

        # Compute phases in float32 for numerical stability, then cast back.
        positions_f = positions.float() * span
        even_indices = torch.arange(
            0, dim, 2, device=positions.device, dtype=torch.float32
        )
        inv_freq = cfg.theta ** (-even_indices / float(dim))
        phase = positions_f.unsqueeze(-1) * inv_freq.view(1, 1, -1)

        pe = torch.empty(
            positions.shape[0],
            positions.shape[1],
            dim,
            device=positions.device,
            dtype=torch.float32,
        )
        pe[..., 0::2] = torch.sin(phase)
        if dim > 1:
            pe[..., 1::2] = torch.cos(phase[..., : pe[..., 1::2].shape[-1]])

        pe = pe * float(cfg.position_amplitude)
        return pe.to(dtype=dtype)

    @staticmethod
    def _expand_position_batch(
        pe: torch.Tensor,
        *,
        batch_size: int,
        name: str,
    ) -> torch.Tensor:
        if pe.shape[0] == batch_size:
            return pe
        if pe.shape[0] == 1:
            return pe.expand(batch_size, -1, -1)
        raise ValueError(
            f"{name} batch must be 1 or {batch_size}, got {pe.shape[0]}"
        )

    def forward(
        self,
        *,
        video_hidden: torch.Tensor,
        audio_hidden: torch.Tensor,
        target_token_count: Optional[int] = None,
        reference_token_count: Optional[int] = None,
        target_valid: Optional[torch.Tensor] = None,
        reference_valid: Optional[torch.Tensor] = None,
        need_attention_weights: bool = False,
        target_temporal_positions: torch.Tensor | None = None,
        reference_temporal_positions: torch.Tensor | None = None,
        audio_temporal_positions: torch.Tensor | None = None,
    ) -> SimpleTrackSummaryOutput:
        if audio_hidden.ndim != 3:
            raise ValueError(
                "audio_hidden must have shape [B, N, C], got "
                f"{tuple(audio_hidden.shape)}"
            )
        if audio_hidden.shape[-1] != self.config.audio_dim:
            raise ValueError(
                f"Expected audio_dim={self.config.audio_dim}, got "
                f"{audio_hidden.shape[-1]}"
            )
        if video_hidden.shape[0] != audio_hidden.shape[0]:
            raise ValueError("video_hidden and audio_hidden batch sizes differ")

        target_tokens, reference_tokens = self.split_video_tokens(
            video_hidden,
            target_token_count=target_token_count,
            reference_token_count=reference_token_count,
        )

        batch_size = video_hidden.shape[0]
        summary = self.summary_tokens.expand(batch_size, -1, -1)

        # Keep the original learned per-slot embeddings for compatibility and
        # specialization. The deterministic PE below carries temporal geometry.
        if self.summary_temporal_positions is not None:
            summary = summary + self.summary_temporal_positions

        reference_padding_mask = self._to_key_padding_mask(
            reference_valid,
            expected_length=reference_tokens.shape[1],
            name="reference_valid",
        )
        target_padding_mask = self._to_key_padding_mask(
            target_valid,
            expected_length=target_tokens.shape[1],
            name="target_valid",
        )

        (
            summary_positions,
            target_positions,
            reference_positions,
            audio_positions,
        ) = self._resolve_temporal_positions(
            batch_size=batch_size,
            target_len=target_tokens.shape[1],
            reference_len=reference_tokens.shape[1],
            audio_len=audio_hidden.shape[1],
            device=video_hidden.device,
            target_temporal_positions=target_temporal_positions,
            reference_temporal_positions=reference_temporal_positions,
            audio_temporal_positions=audio_temporal_positions,
        )

        summary_pe = self._sinusoidal_temporal_embedding(
            summary_positions,
            dim=self.config.audio_dim,
            dtype=summary.dtype,
        )
        target_pe = self._sinusoidal_temporal_embedding(
            target_positions,
            dim=self.config.video_dim,
            dtype=target_tokens.dtype,
        )
        reference_pe = self._sinusoidal_temporal_embedding(
            reference_positions,
            dim=self.config.video_dim,
            dtype=reference_tokens.dtype,
        )
        audio_pe = self._sinusoidal_temporal_embedding(
            audio_positions,
            dim=self.config.audio_dim,
            dtype=audio_hidden.dtype,
        )

        summary_pe = self._expand_position_batch(
            summary_pe, batch_size=batch_size, name="summary_pe"
        )
        target_pe = self._expand_position_batch(
            target_pe, batch_size=batch_size, name="target_pe"
        )
        reference_pe = self._expand_position_batch(
            reference_pe, batch_size=batch_size, name="reference_pe"
        )
        audio_pe = self._expand_position_batch(
            audio_pe, batch_size=batch_size, name="audio_pe"
        )

        # --------------------------------------------------------------
        # 1. Summary -> Reference
        # Add PE to Q/K only. Keep V content-only.
        # --------------------------------------------------------------
        reference_kv = self.reference_kv_norm(reference_tokens)
        reference_delta, _ = self.summary_to_reference(
            query=self.summary_query_norm(summary) + summary_pe,
            key=reference_kv + reference_pe,
            value=reference_kv,
            key_padding_mask=reference_padding_mask,
            need_weights=need_attention_weights,
            average_attn_weights=False,
        )
        reference_summary = summary + reference_delta

        if self.reference_ffn is not None:
            reference_summary = reference_summary + self.reference_ffn(
                self.reference_ffn_norm(reference_summary)
            )

        # --------------------------------------------------------------
        # 2. Reference-aware Summary -> Target
        # --------------------------------------------------------------
        target_kv = self.target_kv_norm(target_tokens)
        target_delta, _ = self.summary_to_target(
            query=self.reference_summary_norm(reference_summary) + summary_pe,
            key=target_kv + target_pe,
            value=target_kv,
            key_padding_mask=target_padding_mask,
            need_weights=need_attention_weights,
            average_attn_weights=False,
        )
        target_summary = reference_summary + target_delta

        if self.target_ffn is not None:
            target_summary = target_summary + self.target_ffn(
                self.target_ffn_norm(target_summary)
            )

        if self.summary_fusion is not None:
            final_summary = self.summary_fusion(
                torch.cat(
                    [
                        reference_summary,
                        target_summary,
                        target_summary - reference_summary,
                    ],
                    dim=-1,
                )
            )
        else:
            final_summary = target_summary

        # --------------------------------------------------------------
        # 3. Audio -> Summary
        # Audio tokens and Summary slots now share one normalized timeline.
        # --------------------------------------------------------------
        final_summary_normed = self.final_summary_norm(final_summary)
        audio_delta, _ = self.audio_to_summary(
            query=self.audio_query_norm(audio_hidden) + audio_pe,
            key=final_summary_normed + summary_pe,
            value=final_summary_normed,
            need_weights=need_attention_weights,
            average_attn_weights=False,
        )
        audio_delta = self.audio_output(audio_delta)

        return SimpleTrackSummaryOutput(
            audio_delta=audio_delta,
            summary_tokens=final_summary,
            reference_summary=reference_summary,
            target_summary=target_summary,
            target_tokens=target_tokens,
            reference_tokens=reference_tokens,
        )