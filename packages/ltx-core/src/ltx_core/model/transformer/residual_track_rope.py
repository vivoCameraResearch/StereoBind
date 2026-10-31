from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

try:
    from . import rope as base_rope
except ImportError:  # Allows: python TrackRoPE/residual_track_rope.py
    import rope as base_rope

import ltx_core.model.transformer.global_args as global_args

TrackInputRange = Literal["zero_one", "minus_one_one"]


@dataclass(frozen=True)
class TrackRoPELayout:
    """Runtime allocation of rotary pairs used by the residual track phase."""

    rotary_pairs_per_head: int
    track_pairs: int
    x_pairs: int
    y_pairs: int


class ResidualTrackRoPE(nn.Module):
    """Residual trajectory phase for LTX-2.3 SPLIT Audio RoPE.

    This module *does not replace* the pretrained LTX temporal RoPE. It takes the
    already-computed temporal ``(cos_t, sin_t)`` and composes an additional track
    phase on a subset of rotary pairs:

        theta_new = theta_time + s_x * phi_x(x) + s_y * phi_y(y)

    For LTX-2.3 Audio Self-Attention with 32 heads and 64 dims/head, the base
    SPLIT RoPE tensors are normally ``[B, 32, T_audio, 32]``.

    Args:
        num_attention_heads: Audio attention heads. LTX-2.3 default is 32.
        track_rope_ratio: Fraction of rotary *pairs* per head that receive the
            residual track phase. Temporal RoPE remains present on every pair.
        use_x: Inject horizontal trajectory coordinate.
        use_y: Inject vertical trajectory coordinate.
        track_theta: Frequency-range parameter for the track frequency grid.
            This intentionally defaults much lower than LTX temporal theta=10000
            because normalized image-plane coordinates lie in [-1, 1].
        max_scale: Bound for each learned scale after tanh.
        per_head_scale: If True, learn one x/y scale per attention head;
            otherwise learn one scalar x/y scale shared by all heads.
        input_range: ``zero_one`` for normalized tracks in [0,1], or
            ``minus_one_one`` if tracks are already centered to [-1,1].
    """

    def __init__(
        self,
        *,
        num_attention_heads: int = 32,
        track_rope_ratio: float = 0.25,
        use_x: bool = True,
        use_y: bool = True,
        track_theta: float = 2.0,
        max_scale: float = 1.0,
        per_head_scale: bool = False,
        input_range: TrackInputRange = "zero_one",
    ) -> None:
        super().__init__()

        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")
        if not (0.0 < track_rope_ratio <= 1.0):
            raise ValueError(f"track_rope_ratio must be in (0, 1], got {track_rope_ratio}")
        if not (use_x or use_y):
            raise ValueError("At least one of use_x/use_y must be enabled")
        if track_theta <= 1.0:
            raise ValueError("track_theta must be > 1.0 when using LTX frequency-grid generation")
        if max_scale <= 0.0:
            raise ValueError(f"max_scale must be positive, got {max_scale}")
        if input_range not in ("zero_one", "minus_one_one"):
            raise ValueError(f"Unsupported input_range: {input_range}")

        self.num_attention_heads = num_attention_heads
        self.track_rope_ratio = float(track_rope_ratio)
        self.track_rope_ratio = global_args.TRACK_ROPE_RATIO
        self.use_x = use_x
        self.use_y = use_y
        self.track_theta = float(track_theta)
        self.max_scale = float(max_scale)
        self.per_head_scale = per_head_scale
        self.input_range = input_range

        # FSDP cannot manage zero-dimensional parameters. A one-element vector
        # keeps the shared-scale semantics and the existing phase broadcasting.
        scale_shape = (num_attention_heads,) if per_head_scale else (1,)
        self.raw_scale_x = nn.Parameter(torch.zeros(scale_shape)) if use_x else None
        self.raw_scale_y = nn.Parameter(torch.zeros(scale_shape)) if use_y else None

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Older shared-scale checkpoints stored shape [] instead of [1].
        # Only migrate that representation; per-head shape mismatches must
        # still be reported by the normal PyTorch loading checks.
        if not self.per_head_scale:
            for name in ("raw_scale_x", "raw_scale_y"):
                key = prefix + name
                value = state_dict.get(key)
                if (
                    getattr(self, name) is not None
                    and isinstance(value, torch.Tensor)
                    and value.ndim == 0
                ):
                    state_dict[key] = value.reshape(1)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def effective_scales(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return bounded learned scales actually used by the residual phase."""
        sx = self.max_scale * torch.tanh(self.raw_scale_x) if self.raw_scale_x is not None else None
        sy = self.max_scale * torch.tanh(self.raw_scale_y) if self.raw_scale_y is not None else None
        return sx, sy

    def _layout(self, rotary_pairs_per_head: int) -> TrackRoPELayout:
        if rotary_pairs_per_head <= 0:
            raise ValueError(f"rotary_pairs_per_head must be positive, got {rotary_pairs_per_head}")

        track_pairs = max(1, int(round(rotary_pairs_per_head * self.track_rope_ratio)))
        track_pairs = min(track_pairs, rotary_pairs_per_head)

        if self.use_x and self.use_y:
            # Give the extra pair to x for odd counts; horizontal motion is the
            # more direct cue for ordinary two-channel stereo.
            x_pairs = (track_pairs + 1) // 2
            y_pairs = track_pairs - x_pairs
        elif self.use_x:
            x_pairs, y_pairs = track_pairs, 0
        else:
            x_pairs, y_pairs = 0, track_pairs

        return TrackRoPELayout(
            rotary_pairs_per_head=rotary_pairs_per_head,
            track_pairs=track_pairs,
            x_pairs=x_pairs,
            y_pairs=y_pairs,
        )

    @staticmethod
    def align_track_to_audio_tokens(track_xy: torch.Tensor, target_seq_len: int) -> torch.Tensor:
        """Linearly resample ``[B,T_track,2]`` to ``[B,T_audio,2]``.

        This is appropriate when the video-frame track and audio tokens both span
        the same clip uniformly in time, e.g. 121 video-frame samples -> 126 audio
        tokens. Endpoints are preserved with ``align_corners=True``.
        """
        if track_xy.ndim != 3 or track_xy.shape[-1] != 2:
            raise ValueError(f"track_xy must have shape [B,T,2], got {tuple(track_xy.shape)}")
        if target_seq_len <= 0:
            raise ValueError(f"target_seq_len must be positive, got {target_seq_len}")
        if track_xy.shape[1] == target_seq_len:
            return track_xy

        original_dtype = track_xy.dtype
        # Interpolation in float32 avoids low-precision coordinate drift; track
        # coordinates are conditioning data, not model hidden states.
        aligned = F.interpolate(
            track_xy.float().transpose(1, 2),
            size=target_seq_len,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        return aligned.to(dtype=original_dtype)

    def _normalize_track(self, track_xy: torch.Tensor) -> torch.Tensor:
        if self.input_range == "zero_one":
            return track_xy * 2.0 - 1.0
        return track_xy

    def _track_frequency_grid(self, num_pairs: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if num_pairs == 0:
            return torch.empty(0, device=device, dtype=dtype)

        # Reuse the LTX frequency-grid construction from rope.py. For one spatial
        # axis, inner_dim=2*num_pairs yields exactly num_pairs rotary frequencies.
        omega = base_rope.generate_freq_grid_pytorch(
            self.track_theta,
            1,
            2 * num_pairs,
        )
        return omega.to(device=device, dtype=dtype)

    def _scale_for_phase(self, scale: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        scale = scale.to(device=device, dtype=dtype)
        if self.per_head_scale:
            return scale.view(1, self.num_attention_heads, 1, 1)
        return scale.view(1, 1, 1, 1)

    def build_track_phase(
        self,
        track_xy: torch.Tensor,
        *,
        target_seq_len: int,
        rotary_pairs_per_head: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, TrackRoPELayout]:
        """Build residual track phase.

        Returns:
            phase: ``[B,1,T,R]`` for shared scales or ``[B,H,T,R]`` for
                per-head scales, where ``R = head_dim/2``.
            layout: rotary-pair allocation metadata.
        """
        track_xy = self.align_track_to_audio_tokens(track_xy, target_seq_len)
        track_xy = track_xy.to(device=device, dtype=dtype)
        track_xy = self._normalize_track(track_xy)

        layout = self._layout(rotary_pairs_per_head)
        sx, sy = self.effective_scales()
        parts: list[torch.Tensor] = []

        if layout.x_pairs:
            omega_x = self._track_frequency_grid(layout.x_pairs, device=device, dtype=dtype)
            x = track_xy[..., 0].view(track_xy.shape[0], 1, target_seq_len, 1)
            scale_x = self._scale_for_phase(sx, dtype=dtype, device=device)
            parts.append(scale_x * x * omega_x.view(1, 1, 1, -1))

        if layout.y_pairs:
            omega_y = self._track_frequency_grid(layout.y_pairs, device=device, dtype=dtype)
            y = track_xy[..., 1].view(track_xy.shape[0], 1, target_seq_len, 1)
            scale_y = self._scale_for_phase(sy, dtype=dtype, device=device)
            parts.append(scale_y * y * omega_y.view(1, 1, 1, -1))

        head_axis = self.num_attention_heads if self.per_head_scale else 1
        remaining = rotary_pairs_per_head - layout.track_pairs
        if remaining:
            parts.append(
                torch.zeros(
                    track_xy.shape[0],
                    head_axis,
                    target_seq_len,
                    remaining,
                    device=device,
                    dtype=dtype,
                )
            )

        phase = torch.cat(parts, dim=-1)
        return phase, layout

    def forward(
        self,
        base_freqs_cis: tuple[torch.Tensor, torch.Tensor],
        track_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose residual Track-RoPE with pretrained LTX Audio RoPE.

        Args:
            base_freqs_cis: Original LTX SPLIT Audio RoPE ``(cos_t, sin_t)``,
                each normally ``[B,H,T,D_head/2]``.
            track_xy: Motion track ``[B,T_track,2]``. By default coordinates are
                expected in [0,1]. It is automatically resampled to T.

        Returns:
            ``(cos_new, sin_new)`` with the same head/sequence/rotary shape as
            the base RoPE (batch may broadcast from 1 to track batch size).
        """
        cos_time, sin_time = base_freqs_cis
        if cos_time.shape != sin_time.shape:
            raise ValueError(
                f"cos_time and sin_time must have identical shapes, got {tuple(cos_time.shape)} and {tuple(sin_time.shape)}"
            )
        if cos_time.ndim != 4:
            raise ValueError(
                "ResidualTrackRoPE targets LTX SPLIT RoPE and expects base PE shape "
                f"[B,H,T,D_head/2], got {tuple(cos_time.shape)}"
            )

        base_batch, num_heads, target_seq_len, rotary_pairs_per_head = cos_time.shape
        if num_heads != self.num_attention_heads:
            raise ValueError(
                f"Base RoPE has {num_heads} heads, but module was configured for {self.num_attention_heads}"
            )
        if track_xy.shape[0] != base_batch and base_batch != 1 and track_xy.shape[0] != 1:
            raise ValueError(
                f"Incompatible batch sizes: base RoPE batch={base_batch}, track batch={track_xy.shape[0]}"
            )

        phase_track, _ = self.build_track_phase(
            track_xy,
            target_seq_len=target_seq_len,
            rotary_pairs_per_head=rotary_pairs_per_head,
            dtype=cos_time.dtype,
            device=cos_time.device,
        )

        cos_track = torch.cos(phase_track)
        sin_track = torch.sin(phase_track)

        # theta_new = theta_time + phi_track, composed without recovering theta_time.
        cos_new = cos_time * cos_track - sin_time * sin_track
        sin_new = sin_time * cos_track + cos_time * sin_track
        return cos_new, sin_new


def apply_residual_track_rotary_emb(
    input_tensor: torch.Tensor,
    base_freqs_cis: tuple[torch.Tensor, torch.Tensor],
    track_xy: torch.Tensor,
    track_rope: ResidualTrackRoPE,
    *,
    rope_type: base_rope.LTXRopeType = base_rope.LTXRopeType.SPLIT,
) -> torch.Tensor:
    """Convenience wrapper: compose Track-RoPE and apply the original LTX rotary op."""
    if rope_type != base_rope.LTXRopeType.SPLIT:
        raise ValueError("ResidualTrackRoPE currently supports the LTX-2.3 default SPLIT RoPE only")
    pe = track_rope(base_freqs_cis, track_xy)
    return base_rope.apply_rotary_emb(input_tensor, pe, rope_type)