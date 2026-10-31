from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from dataclasses import replace
if TYPE_CHECKING:
    from ltx_core.model.transformer.attention import AttentionOps


@dataclass(frozen=True)
class SimpleTrackSummaryConfig:
    """Configuration for the lightweight IC-LoRA summary-token branch.

    The module implements:

        learned summary queries -> reference video tokens
        reference-aware summary -> target video tokens
        audio tokens -> final summary tokens

    Expected default LTX-2.3 shapes:

        target video:    [B, 6144, 4096]
        reference video: [B, 1536, 4096]
        audio:           [B,  122, 2048]
        summary:         [B,   32, 2048]
    """

    video_dim: int = 4096
    audio_dim: int = 2048
    num_summary_tokens: int = 32
    num_heads: int = 16
    dropout: float = 0.0

    target_token_count: int = 6144
    reference_token_count: int = 1536

    use_temporal_summary_positions: bool = True
    use_summary_ffn: bool = True
    use_residual_fusion: bool = True
    ffn_multiplier: int = 4
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.video_dim <= 0 or self.audio_dim <= 0:
            raise ValueError("video_dim and audio_dim must be positive")
        if self.num_summary_tokens <= 0:
            raise ValueError("num_summary_tokens must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.audio_dim % self.num_heads != 0:
            raise ValueError(
                f"audio_dim={self.audio_dim} must be divisible by "
                f"num_heads={self.num_heads}"
            )
        if self.target_token_count <= 0:
            raise ValueError("target_token_count must be positive")
        if self.reference_token_count <= 0:
            raise ValueError("reference_token_count must be positive")


@dataclass
class SimpleTrackSummaryOutput:
    """Outputs useful for both training and optional debugging."""

    audio_delta: torch.Tensor
    summary_tokens: torch.Tensor
    reference_summary: torch.Tensor
    target_summary: torch.Tensor
    target_tokens: torch.Tensor
    reference_tokens: torch.Tensor


class _SummaryFFN(nn.Module):
    def __init__(self, dim: int, multiplier: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = dim * multiplier
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleTrackSummaryTokens(nn.Module):
    """Lightweight summary-token adapter for stereo audio-video generation.

    The three attention stages do NOT share parameters:

    1. ``summary_to_reference`` extracts motion/spatial information from the
       appended IC-LoRA reference tokens.
    2. ``summary_to_target`` grounds the reference summary in the current target
       video state.
    3. ``audio_to_summary`` lets every audio token retrieve the spatial summary
       relevant to its own current hidden state.

    The final projection is zero-initialized. Therefore, immediately after
    initialization this adapter contributes an exact zero residual and preserves
    the pretrained BasicAVTransformerBlock behavior.
    """

    def __init__(self, config: SimpleTrackSummaryConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.audio_dim
        
        import ltx_core.model.transformer.global_args as global_args
        config = replace(
            config,
            num_summary_tokens=global_args.SUMMARY_TOKEN_NUM,
        )
        
        self.summary_tokens = nn.Parameter(
            torch.empty(1, config.num_summary_tokens, dim)
        )
        if config.use_temporal_summary_positions:
            self.summary_temporal_positions = nn.Parameter(
                torch.empty(1, config.num_summary_tokens, dim)
            )
        else:
            self.register_parameter("summary_temporal_positions", None)

        # Attention 1: Q=summary, K/V=reference video.
        self.summary_to_reference = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            kdim=config.video_dim,
            vdim=config.video_dim,
            batch_first=True,
        )

        # Attention 2: Q=reference summary, K/V=target video.
        # This is intentionally a separate parameter set.
        self.summary_to_target = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            kdim=config.video_dim,
            vdim=config.video_dim,
            batch_first=True,
        )

        # Final injection: Q=audio, K/V=summary.
        self.audio_to_summary = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.summary_query_norm = nn.LayerNorm(dim, eps=config.norm_eps)
        self.reference_kv_norm = nn.LayerNorm(
            config.video_dim, eps=config.norm_eps
        )
        self.reference_summary_norm = nn.LayerNorm(dim, eps=config.norm_eps)
        self.target_kv_norm = nn.LayerNorm(
            config.video_dim, eps=config.norm_eps
        )
        self.audio_query_norm = nn.LayerNorm(dim, eps=config.norm_eps)
        self.final_summary_norm = nn.LayerNorm(dim, eps=config.norm_eps)

        if config.use_summary_ffn:
            self.reference_ffn_norm = nn.LayerNorm(dim, eps=config.norm_eps)
            self.reference_ffn = _SummaryFFN(
                dim, config.ffn_multiplier, config.dropout
            )
            self.target_ffn_norm = nn.LayerNorm(dim, eps=config.norm_eps)
            self.target_ffn = _SummaryFFN(
                dim, config.ffn_multiplier, config.dropout
            )
        else:
            self.reference_ffn_norm = None
            self.reference_ffn = None
            self.target_ffn_norm = None
            self.target_ffn = None

        if config.use_residual_fusion:
            self.summary_fusion = nn.Sequential(
                nn.LayerNorm(dim * 3, eps=config.norm_eps),
                nn.Linear(dim * 3, dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(dim, dim),
            )
        else:
            self.summary_fusion = None

        # A separate zero-initialized projection is preferable to a zero scalar
        # gate: gradients immediately reach this projection, while the base
        # transformer output remains exactly unchanged at initialization.
        self.audio_output = nn.Linear(dim, dim)
        nn.init.zeros_(self.audio_output.weight)
        nn.init.zeros_(self.audio_output.bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.summary_tokens, mean=0.0, std=0.02)
        if self.summary_temporal_positions is not None:
            nn.init.normal_(
                self.summary_temporal_positions, mean=0.0, std=0.02
            )

    def _resolve_counts(
        self,
        video_hidden: torch.Tensor,
        target_token_count: Optional[int],
        reference_token_count: Optional[int],
    ) -> tuple[int, int]:
        total_tokens = video_hidden.shape[1]
        target_count = (
            self.config.target_token_count
            if target_token_count is None
            else int(target_token_count)
        )
        reference_count = (
            self.config.reference_token_count
            if reference_token_count is None
            else int(reference_token_count)
        )

        if target_count <= 0 or reference_count <= 0:
            raise ValueError(
                "target_token_count and reference_token_count must be positive"
            )
        if target_count + reference_count > total_tokens:
            raise ValueError(
                "IC-LoRA token split exceeds video sequence length: "
                f"target={target_count}, reference={reference_count}, "
                f"total={total_tokens}"
            )
        return target_count, reference_count

    def split_video_tokens(
        self,
        video_hidden: torch.Tensor,
        *,
        target_token_count: Optional[int] = None,
        reference_token_count: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split official IC-LoRA order: [target tokens, reference tokens]."""

        if video_hidden.ndim != 3:
            raise ValueError(
                "video_hidden must have shape [B, N, C], got "
                f"{tuple(video_hidden.shape)}"
            )
        if video_hidden.shape[-1] != self.config.video_dim:
            raise ValueError(
                f"Expected video_dim={self.config.video_dim}, got "
                f"{video_hidden.shape[-1]}"
            )

        target_count, reference_count = self._resolve_counts(
            video_hidden,
            target_token_count,
            reference_token_count,
        )
        target_tokens = video_hidden[:, :target_count]
        reference_start = target_count
        reference_end = target_count + reference_count
        reference_tokens = video_hidden[:, reference_start:reference_end]
        return target_tokens, reference_tokens

    @staticmethod
    def _to_key_padding_mask(
        valid_mask: Optional[torch.Tensor],
        *,
        expected_length: int,
        name: str,
    ) -> Optional[torch.Tensor]:
        """Convert a valid-token mask to MultiheadAttention padding semantics."""

        if valid_mask is None:
            return None
        if valid_mask.ndim != 2 or valid_mask.shape[1] != expected_length:
            raise ValueError(
                f"{name} must have shape [B, {expected_length}], got "
                f"{tuple(valid_mask.shape)}"
            )
        # MultiheadAttention uses True for positions that must be ignored.
        return ~valid_mask.to(dtype=torch.bool)

    def forward(
        self,
        *,
        video_hidden: torch.Tensor,
        audio_hidden: torch.Tensor,
        target_token_count: Optional[int] = None,
        reference_token_count: Optional[int] = None,
        target_valid: Optional[torch.Tensor] = None,
        reference_valid: Optional[torch.Tensor] = None,
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
        
        reference_delta, _ = self.summary_to_reference(
            query=self.summary_query_norm(summary),
            key=self.reference_kv_norm(reference_tokens),
            value=self.reference_kv_norm(reference_tokens),
            key_padding_mask=reference_padding_mask,
            need_weights=False,
            average_attn_weights=False,
        )
        reference_summary = summary + reference_delta
        if self.reference_ffn is not None:
            reference_summary = reference_summary + self.reference_ffn(
                self.reference_ffn_norm(reference_summary)
            )

        target_delta, _ = self.summary_to_target(
            query=self.reference_summary_norm(reference_summary),
            key=self.target_kv_norm(target_tokens),
            value=self.target_kv_norm(target_tokens),
            key_padding_mask=target_padding_mask,
            need_weights=False,
            average_attn_weights=False,
        )
        target_summary = reference_summary + target_delta
        if self.target_ffn is not None:
            target_summary = target_summary + self.target_ffn(
                self.target_ffn_norm(target_summary)
            )

        if self.summary_fusion is not None:
            # Explicitly expose the reference-target residual requested by the
            # spatial-audio task.
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

        audio_delta, _ = self.audio_to_summary(
            query=self.audio_query_norm(audio_hidden),
            key=self.final_summary_norm(final_summary),
            value=self.final_summary_norm(final_summary),
            need_weights=False,
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


@dataclass(frozen=True)
class SimpleTrackSummaryBottleneckConfig(SimpleTrackSummaryConfig):
    """512-wide Summary with unchanged video/audio interfaces.

    The inherited FFN multiplier is four and residual fusion remains enabled.
    Only summary_dim, not audio_dim, must be divisible by num_heads. The
    original SimpleTrackSummaryConfig and MHA implementation stay unchanged.
    """

    num_heads: int = 8
    summary_dim: int = 512

    def __post_init__(self) -> None:
        for name in (
            "video_dim", "audio_dim", "summary_dim", "num_summary_tokens",
            "num_heads", "target_token_count", "reference_token_count",
            "ffn_multiplier",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.summary_dim % self.num_heads != 0:
            raise ValueError(
                f"summary_dim={self.summary_dim} must be divisible by "
                f"num_heads={self.num_heads}"
            )
        if self.dropout != 0.0:
            raise ValueError("LTX Summary requires dropout=0.0")
        if not self.norm_eps > 0:
            raise ValueError("norm_eps must be positive")


class SimpleTrackSummaryTokensWithLTXAttention(SimpleTrackSummaryTokens):
    """Bottleneck Summary adapter using the same Attention as transformer.py.

    Reuses the legacy configuration, output, token splitting and query
    initialization, but constructs three independent LTX Attention modules
    directly. Pass ``ops=transformer_ops.attention_ops`` to use the block's
    backend choices; ``None`` uses LTX's default AttentionOps.

    Unlike nn.MultiheadAttention, LTX Attention applies learned Q/K RMSNorm
    and returns only the output tensor. It has no attention dropout or
    attention-weight output, so ``config.dropout`` must be zero.
    No RoPE is applied: the learned
    summary positions are retained, and no ``pe``/``k_pe`` are supplied.

    Summary queries, both FFNs and fusion operate at summary_dim (default 512).
    Audio queries keep audio_dim; the final Attention uses a narrow inner
    dimension and a zero-initialized summary_dim -> audio_dim output projection.
    No additional audio_dim -> audio_dim projection is allocated. Both legacy
    MHA and the previous full-width LTX Summary checkpoints are incompatible.
    """

    def __init__(
        self,
        config: SimpleTrackSummaryConfig,
        *,
        ops: Optional[AttentionOps] = None,
        apply_gated_attention: bool = False,
    ) -> None:
        # Do not call the legacy constructor: it would allocate three unused
        # MultiheadAttention modules before replacing them.
        nn.Module.__init__(self)
        if not isinstance(config, SimpleTrackSummaryBottleneckConfig):
            # Existing explicit config objects retain their options while
            # gaining the 512-wide bottleneck. The old MHA class is unaffected.
            config = SimpleTrackSummaryBottleneckConfig(**vars(config))
        if config.dropout != 0.0:
            raise ValueError(
                "LTX Attention does not implement attention dropout; "
                "SimpleTrackSummaryTokensWithLTXAttention requires dropout=0.0"
            )

        # Lazy import keeps the original class usable without an LTX install.
        from ltx_core.model.transformer.attention import Attention, AttentionOps

        if ops is None:
            ops = AttentionOps()
        self.config = config
        dim = config.summary_dim
        dim_head = dim // config.num_heads

        self.summary_tokens = nn.Parameter(
            torch.empty(1, config.num_summary_tokens, dim)
        )
        if config.use_temporal_summary_positions:
            self.summary_temporal_positions = nn.Parameter(
                torch.empty(1, config.num_summary_tokens, dim)
            )
        else:
            self.register_parameter("summary_temporal_positions", None)

        # Q=summary, K/V=reference video.
        self.summary_to_reference = Attention(
            query_dim=dim,
            context_dim=config.video_dim,
            heads=config.num_heads,
            dim_head=dim_head,
            norm_eps=config.norm_eps,
            ops=ops,
            apply_gated_attention=apply_gated_attention,
        )
        # Q=reference-aware summary, K/V=target video (separate parameters).
        self.summary_to_target = Attention(
            query_dim=dim,
            context_dim=config.video_dim,
            heads=config.num_heads,
            dim_head=dim_head,
            norm_eps=config.norm_eps,
            ops=ops,
            apply_gated_attention=apply_gated_attention,
        )
        # Q=audio, K/V=final summary.
        self.audio_to_summary = Attention(
            query_dim=config.audio_dim,
            context_dim=dim,
            heads=config.num_heads,
            dim_head=dim_head,
            norm_eps=config.norm_eps,
            ops=ops,
            apply_gated_attention=apply_gated_attention,
        )

        self.summary_query_norm = nn.LayerNorm(dim, eps=config.norm_eps)
        self.reference_kv_norm = nn.LayerNorm(config.video_dim, eps=config.norm_eps)
        self.reference_summary_norm = nn.LayerNorm(dim, eps=config.norm_eps)
        self.target_kv_norm = nn.LayerNorm(config.video_dim, eps=config.norm_eps)
        self.audio_query_norm = nn.LayerNorm(config.audio_dim, eps=config.norm_eps)
        self.final_summary_norm = nn.LayerNorm(dim, eps=config.norm_eps)

        if config.use_summary_ffn:
            self.reference_ffn_norm = nn.LayerNorm(dim, eps=config.norm_eps)
            self.reference_ffn = _SummaryFFN(dim, config.ffn_multiplier, config.dropout)
            self.target_ffn_norm = nn.LayerNorm(dim, eps=config.norm_eps)
            self.target_ffn = _SummaryFFN(dim, config.ffn_multiplier, config.dropout)
        else:
            self.reference_ffn_norm = None
            self.reference_ffn = None
            self.target_ffn_norm = None
            self.target_ffn = None

        if config.use_residual_fusion:
            self.summary_fusion = nn.Sequential(
                nn.LayerNorm(dim * 3, eps=config.norm_eps),
                nn.Linear(dim * 3, dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(dim, dim),
            )
        else:
            self.summary_fusion = None

        # Preserve a zero initial residual without an extra audio-width matrix.
        nn.init.zeros_(self.audio_to_summary.to_out[0].weight)
        nn.init.zeros_(self.audio_to_summary.to_out[0].bias)
        self.audio_output = nn.Identity()
        self.reset_parameters()

    @staticmethod
    def _to_attention_mask(
        valid_mask: Optional[torch.Tensor],
        *,
        context: torch.Tensor,
        name: str,
    ) -> Optional[torch.Tensor]:
        """Convert [B,S] valid tokens to an additive [B,1,1,S] bias.

        Zero keeps a key and -inf excludes it, consistently for LTX's SDPA
        and xFormers paths. The caller does not allocate a per-head or
        per-query mask. Reject empty context rows rather than relying on
        backend-specific behavior for an all-masked softmax.
        """
        if valid_mask is None:
            return None
        batch_size, length = context.shape[:2]
        if valid_mask.ndim != 2 or tuple(valid_mask.shape) != (batch_size, length):
            raise ValueError(
                f"{name} must have shape [{batch_size}, {length}], got "
                f"{tuple(valid_mask.shape)}"
            )
        valid = valid_mask.to(device=context.device, dtype=torch.bool)
        if not valid.any(dim=-1).all():
            raise ValueError(f"{name} must contain at least one valid token per sample")
        mask = context.new_zeros((batch_size, 1, 1, length))
        return mask.masked_fill_(~valid[:, None, None, :], float("-inf"))

    def forward(
        self,
        *,
        video_hidden: torch.Tensor,
        audio_hidden: torch.Tensor,
        target_token_count: Optional[int] = None,
        reference_token_count: Optional[int] = None,
        target_valid: Optional[torch.Tensor] = None,
        reference_valid: Optional[torch.Tensor] = None,
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
        target_tokens, reference_tokens = self.split_video_tokens(
            video_hidden,
            target_token_count=target_token_count,
            reference_token_count=reference_token_count,
        )
        if video_hidden.shape[0] != audio_hidden.shape[0]:
            raise ValueError("video_hidden and audio_hidden batch sizes differ")

        batch_size = video_hidden.shape[0]
        summary = self.summary_tokens.expand(batch_size, -1, -1)
        if self.summary_temporal_positions is not None:
            summary = summary + self.summary_temporal_positions

        reference_mask = self._to_attention_mask(
            reference_valid, context=reference_tokens, name="reference_valid"
        )
        target_mask = self._to_attention_mask(
            target_valid, context=target_tokens, name="target_valid"
        )

        # A single normalized context supplies both K and V in LTX Attention.
        reference_delta = self.summary_to_reference(
            self.summary_query_norm(summary),
            context=self.reference_kv_norm(reference_tokens),
            mask=reference_mask,
        )
        reference_summary = summary + reference_delta
        if self.reference_ffn is not None:
            reference_summary = reference_summary + self.reference_ffn(
                self.reference_ffn_norm(reference_summary)
            )

        target_delta = self.summary_to_target(
            self.reference_summary_norm(reference_summary),
            context=self.target_kv_norm(target_tokens),
            mask=target_mask,
        )
        target_summary = reference_summary + target_delta
        if self.target_ffn is not None:
            target_summary = target_summary + self.target_ffn(
                self.target_ffn_norm(target_summary)
            )

        if self.summary_fusion is not None:
            final_summary = self.summary_fusion(
                torch.cat(
                    [reference_summary, target_summary, target_summary - reference_summary],
                    dim=-1,
                )
            )
        else:
            final_summary = target_summary

        audio_delta = self.audio_to_summary(
            self.audio_query_norm(audio_hidden),
            context=self.final_summary_norm(final_summary),
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
