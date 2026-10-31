from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


TrackCoordinateRange = Literal["zero_one", "minus_one_one"]
SummarySourceMode = Literal[
    "track_xy",
    "reference_tokens",
    "reference_guided_target",
    "hybrid",
]
MissingConditioningPolicy = Literal["error", "skip"]


class ICLoraSummaryOutput(NamedTuple):
    """Fixed-shape return container that remains friendly to torch.compile.

    Attributes:
        summary_tokens: [B, N_audio, D_summary].
        summary_valid: [B, N_audio] boolean mask.
        inferred_track_xy: optional [B, N_audio, 2] in [-1, 1].
    """

    summary_tokens: torch.Tensor
    summary_valid: torch.Tensor
    inferred_track_xy: torch.Tensor | None


@dataclass(frozen=True)
class ICLoraSummaryConfig:
    """Configuration for IC-LoRA-aware spatial summary tokens.

    The official LTX IC-LoRA layout appends reference tokens after target
    tokens.  With a 512x768x121 target and a spatially downsampled x2 reference,
    the usual token layout is:

        target:    16 * 16 * 24 = 6144
        reference: 16 *  8 * 12 = 1536
        total:                     7680

    ``source_mode`` controls where Summary Tokens come from:

    - ``track_xy``:
        Explicit track coordinates guide attention over target tokens.  This is
        the original summary design and requires ``track_xy``.
    - ``reference_tokens``:
        Summary Tokens are extracted directly from the appended IC-LoRA
        reference tokens.  No explicit ``track_xy`` is required.
    - ``reference_guided_target`` (recommended):
        Reference tokens first infer a temporal spatial guide; that guide then
        reads target tokens at the corresponding location.  This uses both the
        clean motion-track reference and the target-video semantics.
    - ``hybrid``:
        Fuse ``reference_guided_target`` with explicit ``track_xy`` guidance.
        This requires both reference tokens and ``track_xy``.

    Audio length is validated against ``expected_audio_tokens`` when
    ``strict_audio_token_count`` is enabled.  For 121 video frames at 24 FPS,
    LTX produces round((121 / 24) * 25) = 126 audio tokens.
    """

    video_dim: int
    summary_dim: int = 256
    source_mode: SummarySourceMode = "reference_guided_target"

    target_t: int = 16
    target_h: int = 16
    target_w: int = 24
    reference_t: int = 16
    reference_h: int = 8
    reference_w: int = 12

    expected_audio_tokens: int | None = 126
    strict_audio_token_count: bool = False

    track_coordinate_range: TrackCoordinateRange = "zero_one"
    target_spatial_sigma: float = 0.18
    target_temporal_sigma: float = 0.08
    reference_temporal_sigma: float = 0.08

    normalize_qk: bool = True
    reference_trackness_bias_scale: float = 1.0
    missing_conditioning_policy: MissingConditioningPolicy = "error"
    eps: float = 1e-6

    @property
    def target_token_count(self) -> int:
        return self.target_t * self.target_h * self.target_w

    @property
    def reference_token_count(self) -> int:
        return self.reference_t * self.reference_h * self.reference_w

    @property
    def expected_total_video_tokens(self) -> int:
        return self.target_token_count + self.reference_token_count

    def __post_init__(self) -> None:
        if self.video_dim <= 0:
            raise ValueError(f"video_dim must be positive, got {self.video_dim}")
        if self.summary_dim <= 0:
            raise ValueError(f"summary_dim must be positive, got {self.summary_dim}")
        if min(
            self.target_t,
            self.target_h,
            self.target_w,
            self.reference_t,
            self.reference_h,
            self.reference_w,
        ) <= 0:
            raise ValueError("All target/reference grid sizes must be positive")
        if self.source_mode not in {
            "track_xy",
            "reference_tokens",
            "reference_guided_target",
            "hybrid",
        }:
            raise ValueError(f"Unsupported source_mode: {self.source_mode!r}")
        if self.track_coordinate_range not in {"zero_one", "minus_one_one"}:
            raise ValueError(
                "track_coordinate_range must be 'zero_one' or 'minus_one_one'"
            )
        if self.target_spatial_sigma <= 0:
            raise ValueError("target_spatial_sigma must be positive")
        if self.target_temporal_sigma <= 0:
            raise ValueError("target_temporal_sigma must be positive")
        if self.reference_temporal_sigma <= 0:
            raise ValueError("reference_temporal_sigma must be positive")
        if self.expected_audio_tokens is not None and self.expected_audio_tokens <= 0:
            raise ValueError("expected_audio_tokens must be positive or None")
        if self.missing_conditioning_policy not in {"error", "skip"}:
            raise ValueError("missing_conditioning_policy must be 'error' or 'skip'")


@dataclass(frozen=True)
class AudioToSummaryConfig:
    """Configuration for the optional Audio Query -> Summary Key/Value layer."""

    audio_dim: int
    summary_dim: int
    attention_dim: int | None = None
    num_heads: int = 8
    temporal_sigma: float | None = 0.08
    normalize_qk: bool = True
    attention_dropout: float = 0.0
    eps: float = 1e-6

    @property
    def resolved_attention_dim(self) -> int:
        return self.audio_dim if self.attention_dim is None else self.attention_dim

    def __post_init__(self) -> None:
        if self.audio_dim <= 0 or self.summary_dim <= 0:
            raise ValueError("audio_dim and summary_dim must be positive")
        if self.resolved_attention_dim <= 0:
            raise ValueError("attention_dim must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.resolved_attention_dim % self.num_heads != 0:
            raise ValueError(
                "resolved attention dimension must be divisible by num_heads, got "
                f"{self.resolved_attention_dim} and {self.num_heads}"
            )
        if self.temporal_sigma is not None and self.temporal_sigma <= 0:
            raise ValueError("temporal_sigma must be positive or None")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")


def build_token_grid_coordinates(
    *,
    batch_size: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build flattened [x, y] and time coordinates for a (t, h, w) grid.

    Returns:
        xy: [B, T*H*W, 2] in [-1, 1].
        time: [B, T*H*W] in [0, 1].
    """

    time = (torch.arange(grid_t, device=device, dtype=dtype) + 0.5) / grid_t
    y = 2.0 * (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) / grid_h - 1.0
    x = 2.0 * (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) / grid_w - 1.0
    tt, yy, xx = torch.meshgrid(time, y, x, indexing="ij")
    xy = torch.stack((xx, yy), dim=-1).reshape(1, -1, 2)
    time = tt.reshape(1, -1)
    return xy.expand(batch_size, -1, -1), time.expand(batch_size, -1)


def build_default_token_time(
    *,
    batch_size: int,
    token_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return (
        (torch.arange(token_count, device=device, dtype=dtype) + 0.5)
        / token_count
    )[None].expand(batch_size, -1)


def normalize_track_coordinates(
    track_xy: torch.Tensor,
    coordinate_range: TrackCoordinateRange,
) -> torch.Tensor:
    if coordinate_range == "zero_one":
        track_xy = track_xy * 2.0 - 1.0
    return track_xy.clamp(-1.0, 1.0)


def resample_track_to_token_count(
    track_xy: torch.Tensor,
    target_tokens: int,
    track_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample [B, T_track, 2] coordinates to the audio-token count."""

    if track_xy.ndim != 3 or track_xy.shape[-1] != 2:
        raise ValueError(f"track_xy must be [B,T,2], got {tuple(track_xy.shape)}")
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    batch_size, source_tokens, _ = track_xy.shape
    if source_tokens <= 0:
        raise ValueError("track_xy contains no time steps")

    if track_valid is None:
        track_valid = torch.ones(
            batch_size,
            source_tokens,
            device=track_xy.device,
            dtype=torch.bool,
        )
    else:
        if track_valid.shape != (batch_size, source_tokens):
            raise ValueError(
                f"track_valid must be {(batch_size, source_tokens)}, got "
                f"{tuple(track_valid.shape)}"
            )
        track_valid = track_valid.to(device=track_xy.device, dtype=torch.bool)

    if source_tokens == target_tokens:
        return track_xy, track_valid

    track_xy = F.interpolate(
        track_xy.transpose(1, 2).float(),
        size=target_tokens,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2).to(dtype=track_xy.dtype)
    track_valid = F.interpolate(
        track_valid[:, None].float(),
        size=target_tokens,
        mode="nearest",
    )[:, 0].to(dtype=torch.bool)
    return track_xy, track_valid


def split_ic_lora_video_tokens(
    video_hidden: torch.Tensor,
    *,
    config: ICLoraSummaryConfig,
    explicit_reference_hidden: torch.Tensor | None = None,
    target_token_count: int | None = None,
    reference_token_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split target and appended reference tokens using the official IC-LoRA order.

    The official ``VideoConditionByReferenceLatent`` appends reference tokens to
    the end of the target sequence.  If ``explicit_reference_hidden`` is
    supplied, ``video_hidden`` may contain only target tokens.
    """

    if video_hidden.ndim != 3:
        raise ValueError(
            f"video_hidden must be [B,N,D], got {tuple(video_hidden.shape)}"
        )

    target_count = config.target_token_count if target_token_count is None else int(target_token_count)
    reference_count = (
        config.reference_token_count
        if reference_token_count is None
        else int(reference_token_count)
    )
    if target_count <= 0 or reference_count <= 0:
        raise ValueError("target/reference token counts must be positive")

    batch_size, total_tokens, hidden_dim = video_hidden.shape
    if hidden_dim != config.video_dim:
        raise ValueError(
            f"Expected video hidden dim {config.video_dim}, got {hidden_dim}"
        )

    if explicit_reference_hidden is not None:
        if explicit_reference_hidden.ndim != 3:
            raise ValueError("explicit_reference_hidden must be [B,N_ref,D]")
        if explicit_reference_hidden.shape != (
            batch_size,
            reference_count,
            hidden_dim,
        ):
            raise ValueError(
                "explicit_reference_hidden must have shape "
                f"{(batch_size, reference_count, hidden_dim)}, got "
                f"{tuple(explicit_reference_hidden.shape)}"
            )
        if total_tokens < target_count:
            raise ValueError(
                f"video_hidden has {total_tokens} tokens, fewer than target_count={target_count}"
            )
        return video_hidden[:, :target_count], explicit_reference_hidden

    if total_tokens == target_count:
        return video_hidden, None

    expected_total = target_count + reference_count
    if total_tokens != expected_total:
        raise ValueError(
            "Unexpected IC-LoRA video token layout. Expected either target-only "
            f"{target_count} tokens or target+reference {expected_total} tokens, "
            f"but received {total_tokens}. For the common setup this should be "
            "6144 target + 1536 reference = 7680."
        )

    target_hidden = video_hidden[:, :target_count]
    reference_hidden = video_hidden[:, target_count:expected_total]
    return target_hidden, reference_hidden


class TrackGuidedTargetSummarizer(nn.Module):
    """Use explicit track coordinates to summarize target-video tokens."""

    def __init__(self, config: ICLoraSummaryConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.summary_dim
        self.track_mlp = nn.Sequential(
            nn.Linear(6, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.to_key = nn.Linear(config.video_dim, dim, bias=False)
        self.to_value = nn.Linear(config.video_dim, dim, bias=False)
        self.query_norm: nn.Module = nn.LayerNorm(dim, eps=config.eps) if config.normalize_qk else nn.Identity()
        self.key_norm: nn.Module = nn.LayerNorm(dim, eps=config.eps) if config.normalize_qk else nn.Identity()
        self.output_proj = nn.Linear(dim, dim)

    def forward(
        self,
        *,
        target_hidden: torch.Tensor,
        track_xy: torch.Tensor,
        target_audio_tokens: int,
        track_valid: torch.Tensor | None,
        audio_time: torch.Tensor,
        target_xy: torch.Tensor,
        target_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        track_xy, track_valid = resample_track_to_token_count(
            track_xy,
            target_audio_tokens,
            track_valid,
        )
        track_xy = normalize_track_coordinates(
            track_xy,
            self.config.track_coordinate_range,
        )

        delta = torch.zeros_like(track_xy)
        delta[:, 1:] = track_xy[:, 1:] - track_xy[:, :-1]
        pair_valid = torch.zeros_like(track_valid)
        pair_valid[:, 0] = track_valid[:, 0]
        pair_valid[:, 1:] = track_valid[:, 1:] & track_valid[:, :-1]
        delta = delta * pair_valid.unsqueeze(-1).to(delta.dtype)
        speed = torch.linalg.vector_norm(delta.float(), dim=-1, keepdim=True).to(track_xy.dtype)
        valid_feature = track_valid.unsqueeze(-1).to(track_xy.dtype)
        features = torch.cat((track_xy, delta, speed, valid_feature), dim=-1)
        features = features * valid_feature

        query = self.query_norm(self.track_mlp(features))
        key = self.key_norm(self.to_key(target_hidden))
        value = self.to_value(target_hidden)
        logits = torch.einsum("bad,bnd->ban", query, key)
        logits = logits * (self.config.summary_dim**-0.5)

        spatial_distance = (
            track_xy[:, :, None, :] - target_xy[:, None, :, :]
        ).float().square().sum(dim=-1)
        temporal_distance = (
            audio_time[:, :, None] - target_time[:, None, :]
        ).float().square()
        logits = logits.float()
        logits = logits - spatial_distance / (2.0 * self.config.target_spatial_sigma**2)
        logits = logits - temporal_distance / (2.0 * self.config.target_temporal_sigma**2)

        valid_query = track_valid[:, :, None]
        logits = torch.where(valid_query, logits, torch.zeros_like(logits))
        attention = torch.softmax(logits, dim=-1).to(value.dtype)
        attention = attention * valid_query.to(attention.dtype)
        summary = torch.einsum("ban,bnd->bad", attention, value)
        summary = self.output_proj(summary) * valid_query.to(summary.dtype)
        return summary, track_valid, track_xy


class ReferenceTokenTemporalSummarizer(nn.Module):
    """Extract one spatially aware Summary Token per audio token from reference tokens.

    No explicit track coordinates are required.  The clean sparse-motion-track
    reference is recognized through a learned trackness bias.  Explicit [x,y,t]
    coordinate embeddings are added to reference keys/values so spatial position
    is not lost when the spatial grid is pooled into a temporal summary sequence.
    """

    def __init__(self, config: ICLoraSummaryConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.summary_dim
        self.time_query = nn.Sequential(
            nn.Linear(3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.to_key = nn.Linear(config.video_dim, dim, bias=False)
        self.to_value = nn.Linear(config.video_dim, dim, bias=False)
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.trackness_norm = nn.LayerNorm(config.video_dim, eps=config.eps)
        self.trackness_head = nn.Linear(config.video_dim, 1)
        self.query_norm: nn.Module = nn.LayerNorm(dim, eps=config.eps) if config.normalize_qk else nn.Identity()
        self.key_norm: nn.Module = nn.LayerNorm(dim, eps=config.eps) if config.normalize_qk else nn.Identity()
        self.output_proj = nn.Linear(dim, dim)

    def forward(
        self,
        *,
        reference_hidden: torch.Tensor,
        target_audio_tokens: int,
        audio_time: torch.Tensor,
        reference_xy: torch.Tensor,
        reference_time: torch.Tensor,
        reference_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_reference_tokens, _ = reference_hidden.shape
        if reference_valid is None:
            reference_valid = torch.ones(
                batch_size,
                num_reference_tokens,
                device=reference_hidden.device,
                dtype=torch.bool,
            )
        else:
            if reference_valid.shape != (batch_size, num_reference_tokens):
                raise ValueError(
                    "reference_valid must be [B,N_reference], got "
                    f"{tuple(reference_valid.shape)}"
                )
            reference_valid = reference_valid.to(reference_hidden.device, torch.bool)

        phase = 2.0 * torch.pi * audio_time.float()
        time_features = torch.stack(
            (audio_time.float(), torch.sin(phase), torch.cos(phase)),
            dim=-1,
        ).to(reference_hidden.dtype)
        query = self.query_norm(self.time_query(time_features))

        coord_features = torch.cat(
            (reference_xy, reference_time.unsqueeze(-1)),
            dim=-1,
        )
        coord_embedding = self.coord_mlp(coord_features)
        key = self.key_norm(self.to_key(reference_hidden) + coord_embedding)
        value = self.to_value(reference_hidden) + coord_embedding

        logits = torch.einsum("bad,bnd->ban", query, key)
        logits = logits * (self.config.summary_dim**-0.5)
        temporal_distance = (
            audio_time[:, :, None] - reference_time[:, None, :]
        ).float().square()
        logits = logits.float() - temporal_distance / (
            2.0 * self.config.reference_temporal_sigma**2
        )

        if self.config.reference_trackness_bias_scale != 0.0:
            trackness = self.trackness_head(
                self.trackness_norm(reference_hidden)
            ).squeeze(-1).float()
            logits = logits + self.config.reference_trackness_bias_scale * trackness[:, None, :]

        any_valid = reference_valid.any(dim=-1)
        safe_valid = reference_valid.clone()
        if (~any_valid).any():
            safe_valid[~any_valid, 0] = True
        logits = logits.masked_fill(
            ~safe_valid[:, None, :],
            torch.finfo(logits.dtype).min,
        )
        attention = torch.softmax(logits, dim=-1)
        attention = attention * reference_valid[:, None, :].to(attention.dtype)
        normalizer = attention.sum(dim=-1, keepdim=True)
        attention = torch.where(
            normalizer > 0,
            attention / normalizer.clamp_min(self.config.eps),
            torch.zeros_like(attention),
        ).to(value.dtype)

        summary = torch.einsum("ban,bnd->bad", attention, value)
        summary = self.output_proj(summary)
        summary_valid = any_valid[:, None].expand(-1, target_audio_tokens)
        summary = summary * summary_valid.unsqueeze(-1).to(summary.dtype)

        inferred_xy = torch.einsum(
            "ban,bnd->bad",
            attention.to(reference_xy.dtype),
            reference_xy,
        )
        inferred_xy = inferred_xy.clamp(-1.0, 1.0)
        inferred_xy = inferred_xy * summary_valid.unsqueeze(-1).to(inferred_xy.dtype)
        return summary, summary_valid, inferred_xy


class ReferenceGuidedTargetSummarizer(nn.Module):
    """Use reference-derived queries and inferred XY to read target tokens."""

    def __init__(self, config: ICLoraSummaryConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.summary_dim
        self.reference_to_query = nn.Linear(dim, dim, bias=False)
        self.target_to_key = nn.Linear(config.video_dim, dim, bias=False)
        self.target_to_value = nn.Linear(config.video_dim, dim, bias=False)
        self.target_coord_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.query_norm: nn.Module = nn.LayerNorm(dim, eps=config.eps) if config.normalize_qk else nn.Identity()
        self.key_norm: nn.Module = nn.LayerNorm(dim, eps=config.eps) if config.normalize_qk else nn.Identity()
        self.fuse = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        *,
        target_hidden: torch.Tensor,
        reference_summary: torch.Tensor,
        summary_valid: torch.Tensor,
        inferred_xy: torch.Tensor,
        audio_time: torch.Tensor,
        target_xy: torch.Tensor,
        target_time: torch.Tensor,
    ) -> torch.Tensor:
        coord_features = torch.cat(
            (target_xy, target_time.unsqueeze(-1)),
            dim=-1,
        )
        coord_embedding = self.target_coord_mlp(coord_features)
        query = self.query_norm(self.reference_to_query(reference_summary))
        key = self.key_norm(self.target_to_key(target_hidden) + coord_embedding)
        value = self.target_to_value(target_hidden) + coord_embedding

        logits = torch.einsum("bad,bnd->ban", query, key)
        logits = logits * (self.config.summary_dim**-0.5)
        spatial_distance = (
            inferred_xy[:, :, None, :] - target_xy[:, None, :, :]
        ).float().square().sum(dim=-1)
        temporal_distance = (
            audio_time[:, :, None] - target_time[:, None, :]
        ).float().square()
        logits = logits.float()
        logits = logits - spatial_distance / (2.0 * self.config.target_spatial_sigma**2)
        logits = logits - temporal_distance / (2.0 * self.config.target_temporal_sigma**2)

        valid_query = summary_valid[:, :, None]
        logits = torch.where(valid_query, logits, torch.zeros_like(logits))
        attention = torch.softmax(logits, dim=-1).to(value.dtype)
        attention = attention * valid_query.to(attention.dtype)
        target_summary = torch.einsum("ban,bnd->bad", attention, value)
        fused = self.fuse(torch.cat((reference_summary, target_summary), dim=-1))
        fused = fused * valid_query.to(fused.dtype)
        return fused


class ICLoraSpatialSummary(nn.Module):
    """IC-LoRA-aware Summary Token generator.

    The module can consume the concatenated LTX sequence directly, or accept an
    explicit separate reference tensor.  In the normal LTX IC-LoRA path, pass
    the complete [target, reference] sequence and leave
    ``explicit_reference_hidden=None``.
    """

    def __init__(self, config: ICLoraSummaryConfig) -> None:
        super().__init__()
        self.config = config
        self.track_summarizer = TrackGuidedTargetSummarizer(config)
        self.reference_summarizer = ReferenceTokenTemporalSummarizer(config)
        self.reference_guided_target_summarizer = ReferenceGuidedTargetSummarizer(config)
        self.hybrid_fuse = nn.Sequential(
            nn.Linear(2 * config.summary_dim, config.summary_dim),
            nn.SiLU(),
            nn.Linear(config.summary_dim, config.summary_dim),
        )

    def _handle_missing(self, message: str) -> None:
        if self.config.missing_conditioning_policy == "error":
            raise ValueError(message)

    def _validate_audio_tokens(self, actual_tokens: int) -> None:
        expected = self.config.expected_audio_tokens
        if (
            self.config.strict_audio_token_count
            and expected is not None
            and actual_tokens != expected
        ):
            raise ValueError(
                f"Expected {expected} audio tokens but received {actual_tokens}. "
                "For 121 frames at 24 FPS LTX-2.3 normally uses 126 tokens. "
                "Set strict_audio_token_count=False only when intentionally using "
                "a different video duration or frame rate."
            )

    def forward(
        self,
        *,
        video_hidden: torch.Tensor,
        target_audio_tokens: int,
        track_xy: torch.Tensor | None = None,
        track_valid: torch.Tensor | None = None,
        audio_time: torch.Tensor | None = None,
        explicit_reference_hidden: torch.Tensor | None = None,
        target_token_count: int | None = None,
        reference_token_count: int | None = None,
        reference_valid: torch.Tensor | None = None,
    ) -> ICLoraSummaryOutput | None:
        self._validate_audio_tokens(target_audio_tokens)
        target_hidden, reference_hidden = split_ic_lora_video_tokens(
            video_hidden,
            config=self.config,
            explicit_reference_hidden=explicit_reference_hidden,
            target_token_count=target_token_count,
            reference_token_count=reference_token_count,
        )
        batch_size = video_hidden.shape[0]
        if audio_time is None:
            audio_time = build_default_token_time(
                batch_size=batch_size,
                token_count=target_audio_tokens,
                device=video_hidden.device,
                dtype=video_hidden.dtype,
            )
        else:
            if audio_time.shape != (batch_size, target_audio_tokens):
                raise ValueError(
                    "audio_time must be [B,N_audio], got "
                    f"{tuple(audio_time.shape)}"
                )
            audio_time = audio_time.to(video_hidden.device, video_hidden.dtype)

        target_xy, target_time = build_token_grid_coordinates(
            batch_size=batch_size,
            grid_t=self.config.target_t,
            grid_h=self.config.target_h,
            grid_w=self.config.target_w,
            device=video_hidden.device,
            dtype=video_hidden.dtype,
        )

        mode = self.config.source_mode
        if mode == "track_xy":
            if track_xy is None:
                self._handle_missing("source_mode='track_xy' requires audio.track_xy")
                return None
            summary, valid, normalized_track = self.track_summarizer(
                target_hidden=target_hidden,
                track_xy=track_xy,
                target_audio_tokens=target_audio_tokens,
                track_valid=track_valid,
                audio_time=audio_time,
                target_xy=target_xy,
                target_time=target_time,
            )
            return ICLoraSummaryOutput(
                summary,
                valid,
                normalized_track,
            )

        if reference_hidden is None:
            self._handle_missing(
                f"source_mode={mode!r} requires appended or explicit reference tokens"
            )
            return None

        reference_xy, reference_time = build_token_grid_coordinates(
            batch_size=batch_size,
            grid_t=self.config.reference_t,
            grid_h=self.config.reference_h,
            grid_w=self.config.reference_w,
            device=video_hidden.device,
            dtype=video_hidden.dtype,
        )
        reference_summary, reference_summary_valid, inferred_xy = (
            self.reference_summarizer(
                reference_hidden=reference_hidden,
                target_audio_tokens=target_audio_tokens,
                audio_time=audio_time,
                reference_xy=reference_xy,
                reference_time=reference_time,
                reference_valid=reference_valid,
            )
        )

        if mode == "reference_tokens":
            return ICLoraSummaryOutput(
                reference_summary,
                reference_summary_valid,
                inferred_xy,
            )

        reference_guided_summary = self.reference_guided_target_summarizer(
            target_hidden=target_hidden,
            reference_summary=reference_summary,
            summary_valid=reference_summary_valid,
            inferred_xy=inferred_xy,
            audio_time=audio_time,
            target_xy=target_xy,
            target_time=target_time,
        )

        if mode == "reference_guided_target":
            return ICLoraSummaryOutput(
                reference_guided_summary,
                reference_summary_valid,
                inferred_xy,
            )

        if mode != "hybrid":
            raise RuntimeError(f"Unhandled source_mode: {mode}")
        if track_xy is None:
            self._handle_missing("source_mode='hybrid' requires audio.track_xy")
            return None

        track_summary, track_summary_valid, normalized_track = (
            self.track_summarizer(
                target_hidden=target_hidden,
                track_xy=track_xy,
                target_audio_tokens=target_audio_tokens,
                track_valid=track_valid,
                audio_time=audio_time,
                target_xy=target_xy,
                target_time=target_time,
            )
        )
        valid = reference_summary_valid & track_summary_valid
        hybrid_summary = self.hybrid_fuse(
            torch.cat((reference_guided_summary, track_summary), dim=-1)
        )
        hybrid_summary = hybrid_summary * valid.unsqueeze(-1).to(hybrid_summary.dtype)
        return ICLoraSummaryOutput(
            hybrid_summary,
            valid,
            normalized_track,
        )


class AudioToSummaryAttention(nn.Module):
    """Second layer: current Audio hidden states query Summary Tokens."""

    def __init__(self, config: AudioToSummaryConfig) -> None:
        super().__init__()
        self.config = config
        self.attention_dim = config.resolved_attention_dim
        self.head_dim = self.attention_dim // config.num_heads

        self.to_query = nn.Linear(config.audio_dim, self.attention_dim, bias=False)
        self.to_key = nn.Linear(config.summary_dim, self.attention_dim, bias=False)
        self.to_value = nn.Linear(config.summary_dim, self.attention_dim, bias=False)
        self.query_norm: nn.Module = nn.LayerNorm(self.attention_dim, eps=config.eps) if config.normalize_qk else nn.Identity()
        self.key_norm: nn.Module = nn.LayerNorm(self.attention_dim, eps=config.eps) if config.normalize_qk else nn.Identity()
        self.output_proj = nn.Linear(self.attention_dim, config.audio_dim)
        # Exact no-op at initialization, while output_proj receives gradient on
        # the first optimization step.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        *,
        audio_hidden: torch.Tensor,
        summary_tokens: torch.Tensor,
        summary_valid: torch.Tensor | None = None,
        audio_time: torch.Tensor | None = None,
        summary_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if audio_hidden.ndim != 3 or summary_tokens.ndim != 3:
            raise ValueError("audio_hidden and summary_tokens must both be rank-3")
        batch_size, num_audio_tokens, audio_dim = audio_hidden.shape
        summary_batch, num_summary_tokens, summary_dim = summary_tokens.shape
        if summary_batch != batch_size:
            raise ValueError("Audio and summary batch sizes do not match")
        if audio_dim != self.config.audio_dim:
            raise ValueError(
                f"Expected audio dim {self.config.audio_dim}, got {audio_dim}"
            )
        if summary_dim != self.config.summary_dim:
            raise ValueError(
                f"Expected summary dim {self.config.summary_dim}, got {summary_dim}"
            )

        if summary_valid is None:
            summary_valid = torch.ones(
                batch_size,
                num_summary_tokens,
                device=summary_tokens.device,
                dtype=torch.bool,
            )
        else:
            if summary_valid.shape != (batch_size, num_summary_tokens):
                raise ValueError("summary_valid must be [B,N_summary]")
            summary_valid = summary_valid.to(summary_tokens.device, torch.bool)

        if audio_time is None:
            audio_time = build_default_token_time(
                batch_size=batch_size,
                token_count=num_audio_tokens,
                device=audio_hidden.device,
                dtype=audio_hidden.dtype,
            )
        if summary_time is None:
            summary_time = build_default_token_time(
                batch_size=batch_size,
                token_count=num_summary_tokens,
                device=audio_hidden.device,
                dtype=audio_hidden.dtype,
            )
        audio_time = audio_time.to(audio_hidden.device, audio_hidden.dtype)
        summary_time = summary_time.to(audio_hidden.device, audio_hidden.dtype)
        summary_tokens = summary_tokens.to(audio_hidden.device, audio_hidden.dtype)
        summary_valid = summary_valid.to(audio_hidden.device)

        query = self.query_norm(self.to_query(audio_hidden))
        key = self.key_norm(self.to_key(summary_tokens))
        value = self.to_value(summary_tokens)
        query = query.view(
            batch_size,
            num_audio_tokens,
            self.config.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = key.view(
            batch_size,
            num_summary_tokens,
            self.config.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            num_summary_tokens,
            self.config.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        logits = torch.matmul(query, key.transpose(-1, -2)) * (self.head_dim**-0.5)
        if self.config.temporal_sigma is not None:
            temporal_distance = (
                audio_time[:, :, None] - summary_time[:, None, :]
            ).float().square()
            logits = logits.float() - temporal_distance[:, None] / (
                2.0 * self.config.temporal_sigma**2
            )
        else:
            logits = logits.float()

        any_valid = summary_valid.any(dim=-1)
        safe_valid = summary_valid.clone()
        if (~any_valid).any():
            safe_valid[~any_valid, 0] = True
        logits = logits.masked_fill(
            ~safe_valid[:, None, None, :],
            torch.finfo(logits.dtype).min,
        )
        attention = torch.softmax(logits, dim=-1)
        attention = attention * summary_valid[:, None, None, :].to(attention.dtype)
        normalizer = attention.sum(dim=-1, keepdim=True)
        attention = torch.where(
            normalizer > 0,
            attention / normalizer.clamp_min(self.config.eps),
            torch.zeros_like(attention),
        )
        attention_for_value = F.dropout(
            attention,
            p=self.config.attention_dropout,
            training=self.training,
        ) if self.config.attention_dropout > 0 else attention

        output = torch.matmul(attention_for_value.to(value.dtype), value)
        output = output.transpose(1, 2).reshape(
            batch_size,
            num_audio_tokens,
            self.attention_dim,
        )
        output = self.output_proj(output)
        output = output * any_valid[:, None, None].to(output.dtype)
        return output
