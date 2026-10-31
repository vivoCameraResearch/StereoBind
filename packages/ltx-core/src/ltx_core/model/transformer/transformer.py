from dataclasses import dataclass, field, replace

import torch
from torch import nn
from typing import Optional

from ltx_core.model.transformer.adaln import adaln_embedding_coefficient
from ltx_core.model.transformer.attention import (
    Attention,
    AttentionCallable,
    AttentionFunction,
    AttentionOps,
    MaskedAttentionCallable,
    MaskedAttentionFunction,
)
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.ops import (
    AdaZeroCallable,
    GatedAttentionCallable,
    PostSACallable,
    PreAttentionCallable,
    PytorchAdaZeroFunction,
    PytorchGatedAttention,
    PytorchPostSAFunction,
    PytorchPreAttention,
)
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer_args import TransformerArgs

from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.model.transformer.summary_tokens import (
    AudioToSummaryAttention,
    AudioToSummaryConfig,
    ICLoraSpatialSummary,
    ICLoraSummaryConfig,
)

from ltx_core.model.transformer.simple_summary_tokens import (
    SimpleTrackSummaryBottleneckConfig,
    SimpleTrackSummaryConfig,
    SimpleTrackSummaryOutput,
    SimpleTrackSummaryTokensWithLTXAttention,
    SimpleTrackSummaryTokens
)


from ltx_core.model.transformer.simple_summary_tokens_temporal_rope import (
    SimpleTrackSummaryTokensWithTemporalPosition
)



@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False


@dataclass(frozen=True)
class TransformerOpsConfig:
    """Pluggable ops for :class:`BasicAVTransformerBlock`.
    Use :meth:`from_functions` to construct from enum values or partial overrides
    without spelling out a full :class:`AttentionOps`.
    """

    attention_ops: AttentionOps = field(default_factory=AttentionOps)
    ada_zero_function: AdaZeroCallable = field(default_factory=PytorchAdaZeroFunction)
    post_sa_function: PostSACallable = field(default_factory=PytorchPostSAFunction)

    @classmethod
    def from_functions(
        cls,
        attention: AttentionFunction | AttentionCallable = AttentionFunction.AUTOMATIC,
        masked_attention: MaskedAttentionFunction | MaskedAttentionCallable = MaskedAttentionFunction.AUTOMATIC,
        preattention: PreAttentionCallable | None = None,
        gated_attention: GatedAttentionCallable | None = None,
        ada_zero: AdaZeroCallable | None = None,
        post_sa: PostSACallable | None = None,
    ) -> "TransformerOpsConfig":
        """Build a config from individual functions or enums. Each *None* slot
        falls back to the standard PyTorch implementation."""
        attention_callable = attention.to_callable() if isinstance(attention, AttentionFunction) else attention
        masked_callable = (
            masked_attention.to_callable()
            if isinstance(masked_attention, MaskedAttentionFunction)
            else masked_attention
        )
        attention_ops = AttentionOps(
            attention_function=attention_callable,
            masked_attention_function=masked_callable,
            preattention_function=preattention if preattention is not None else PytorchPreAttention(),
            gated_attention_function=(gated_attention if gated_attention is not None else PytorchGatedAttention()),
        )
        return cls(
            attention_ops=attention_ops,
            ada_zero_function=ada_zero if ada_zero is not None else PytorchAdaZeroFunction(),
            post_sa_function=post_sa if post_sa is not None else PytorchPostSAFunction(),
        )


# Frozen, so safe to share as a default argument across callers that want the
# stock PyTorch ops without explicit construction.
DEFAULT_TRANSFORMER_OPS = TransformerOpsConfig()


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        ops: TransformerOpsConfig | None = None,
        transformer_args: dict = None,
    ):
        super().__init__()
        if ops is None:
            ops = TransformerOpsConfig()
        self.ada_zero_function = ops.ada_zero_function
        self.post_sa_function = ops.post_sa_function
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            video_sst_size = adaln_embedding_coefficient(video.cross_attention_adaln)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(video_sst_size, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            audio_sst_size = adaln_embedding_coefficient(audio.cross_attention_adaln)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(audio_sst_size, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.cross_attention_adaln = (video is not None and video.cross_attention_adaln) or (
            audio is not None and audio.cross_attention_adaln
        )

        if self.cross_attention_adaln and video is not None:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
        if self.cross_attention_adaln and audio is not None:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))

        self.norm_eps = norm_eps

        # =======================
        self.audio_model_dim = audio.dim if audio is not None else None
        # =======================


    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]
        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        scale_shift_indices: slice,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, scale_shift_indices
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale, shift = (t.squeeze(2) for t in scale_shift_ada_values)
        (gate,) = (t.squeeze(2) for t in gate_ada_values)

        return scale, shift, gate

    def _apply_text_cross_attention(
        self,
        x_normed: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        cross_attention_adaln: bool = False,
    ) -> torch.Tensor:
        """Apply text cross-attention, with optional AdaLN modulation.
        ``x_normed`` is the RMS-normalized self-attention output produced by
        ``post_sa_function`` -- this method does not normalize again.
        """
        if cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, x_normed.shape[0], timestep, slice(6, 9))
            return apply_cross_attention_adaln(
                x_normed,
                context,
                attn,
                shift_q,
                scale_q,
                gate,
                prompt_scale_shift_table,
                prompt_timestep,
                context_mask,
            )
        return attn(x_normed, context=context, mask=context_mask)

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            norm_vx = self.ada_zero_function(vx, self.norm_eps, vscale_msa, vshift_msa)
            del vshift_msa, vscale_msa

            vx_msa_out = self.attn1(
                norm_vx,
                pe=video.positional_embeddings,
                mask=video.self_attention_mask,
                perturbation_mask=video.self_attn_perturbation_mask,
                all_perturbed=video.self_attn_all_perturbed,
            )
            vx, vx_normed = self.post_sa_function(vx, vx_msa_out, None, self.norm_eps, vgate_msa)
            del vgate_msa, norm_vx, vx_msa_out
            vx = vx + self._apply_text_cross_attention(
                vx_normed,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            del vx_normed

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )

            norm_ax = self.ada_zero_function(ax, self.norm_eps, ascale_msa, ashift_msa)
            del ashift_msa, ascale_msa
            ax_msa_out = self.audio_attn1(
                norm_ax,
                pe=audio.positional_embeddings,
                mask=audio.self_attention_mask,
                perturbation_mask=audio.self_attn_perturbation_mask,
                all_perturbed=audio.self_attn_all_perturbed,
            )

            ax, ax_normed = self.post_sa_function(ax, ax_msa_out, None, self.norm_eps, agate_msa)
            del agate_msa, norm_ax, ax_msa_out
            ax = ax + self._apply_text_cross_attention(
                ax_normed,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            del ax_normed

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            # Snapshot vx/ax before A2V mutates vx; V2A's video keys/values must
            # use the pre-A2V state so direction order doesn't bias the result.
            vx_pre_av = vx
            ax_pre_av = ax
            if run_a2v and not video.cross_attn_skip_all:
                scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_vx_scaled = self.ada_zero_function(vx_pre_av, self.norm_eps, scale_ca_video_a2v, shift_ca_video_a2v)
                del scale_ca_video_a2v, shift_ca_video_a2v

                scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_ax_scaled = self.ada_zero_function(ax_pre_av, self.norm_eps, scale_ca_audio_a2v, shift_ca_audio_a2v)
                del scale_ca_audio_a2v, shift_ca_audio_a2v
                
                vx = vx + (
                    self.audio_to_video_attn(
                        a2v_vx_scaled,
                        context=a2v_ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                    * video.cross_attn_perturbation_mask
                )
                del gate_out_a2v, a2v_vx_scaled, a2v_ax_scaled

            if run_v2a and not audio.cross_attn_skip_all:
                scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_ax_scaled = self.ada_zero_function(ax_pre_av, self.norm_eps, scale_ca_audio_v2a, shift_ca_audio_v2a)
                del scale_ca_audio_v2a, shift_ca_audio_v2a
                scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_vx_scaled = self.ada_zero_function(vx_pre_av, self.norm_eps, scale_ca_video_v2a, shift_ca_video_v2a)
                del scale_ca_video_v2a, shift_ca_video_v2a
                ax = ax + (
                    self.video_to_audio_attn(
                        v2a_ax_scaled,
                        context=v2a_vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                    * audio.cross_attn_perturbation_mask
                )
                del gate_out_v2a, v2a_vx_scaled, v2a_ax_scaled
            del vx_pre_av, ax_pre_av

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = self.ada_zero_function(vx, self.norm_eps, vscale_mlp, vshift_mlp)
            vx = vx + self.ff(vx_scaled) * vgate_mlp

            del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ax_scaled = self.ada_zero_function(ax, self.norm_eps, ascale_mlp, ashift_mlp)
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

            del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


def apply_cross_attention_adaln(
    x_normed: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor,
    context_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply query/key AdaLN modulation then cross-attention.
    ``x_normed`` is already RMS-normalized by ``post_sa_function``; this only
    applies the affine (scale/shift) modulation, so the normalization is not
    repeated here.
    """
    batch_size = x_normed.shape[0]
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x_normed.device, dtype=x_normed.dtype)
        + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    attn_input = x_normed * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate




class BasicAVTransformerBlockWithSimpleSummary(BasicAVTransformerBlock):
    """BasicAVTransformerBlock with an in-block IC-LoRA Summary branch.

    Execution order:

        1. video self-attention
        2. video text cross-attention
        3. audio self-attention
        4. audio text cross-attention
        5. Simple Summary:
             learned summary Q -> reference video K/V
             reference summary Q -> target video K/V
             audio Q -> final summary K/V
        6. direct Simple Summary residual on the audio stream
        7. original A2V attention
        8. original V2A attention
        9. video/audio FFN

    Summary is computed from ``vx`` and ``ax`` after both modality branches
    complete self-attention and text cross-attention. Its gated residual is
    added directly to ``ax`` before the original A2V and V2A modules consume
    the updated audio hidden state. The Summary residual uses the audio
    perturbation mask without applying the original V2A AdaLN gate again.
    """

    def __init__(
        self,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        ops: TransformerOpsConfig | None = None,
        transformer_args: Optional[dict] = None,
        *,
        simple_summary_config: SimpleTrackSummaryConfig | None = None,
    ) -> None:
        args = transformer_args or {}
        ops = DEFAULT_TRANSFORMER_OPS if ops is None else ops
        super().__init__(
            video=video,
            audio=audio,
            rope_type=rope_type,
            norm_eps=norm_eps,
            ops=ops,
            transformer_args=args,
        )

        self.use_simple_summary = bool(args.get("use_simple_summary", False))
        if self.use_simple_summary and (video is None or audio is None):
            raise ValueError(
                "Simple Summary requires both video and audio configurations"
            )

        # Only remember dimensions here. The new trainable module is created
        # after loading the pretrained checkpoint and leaving the meta device.
        self._simple_summary_video_dim = video.dim if video is not None else None
        self._simple_summary_audio_dim = audio.dim if audio is not None else None
        self._simple_summary_attention_ops = ops.attention_ops
        self._simple_summary_norm_eps = norm_eps
        self._simple_summary_config_arg = simple_summary_config

        # self.simple_summary: SimpleTrackSummaryTokensWithLTXAttention | None = None
        self.simple_summary: SimpleTrackSummaryTokens | None = None
        self.simple_summary_scale: torch.nn.Parameter | None = None
        self.simple_summary_config: SimpleTrackSummaryConfig | None = None

        # Detached debugging snapshots.
        self.last_simple_summary: torch.Tensor | None = None
        self.last_reference_summary: torch.Tensor | None = None
        self.last_target_summary: torch.Tensor | None = None


    def initialize_simple_summary(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Create Summary parameters after loading the base LTX checkpoint."""

        if not self.use_simple_summary:
            return
        if self.simple_summary is not None:
            return

        device = torch.device(device)
        if device.type == "meta":
            raise RuntimeError(
                "Simple Summary cannot be initialized on the meta device. "
                "Call initialize_simple_summary() after load_transformer()."
            )

        if (
            self._simple_summary_video_dim is None
            or self._simple_summary_audio_dim is None
        ):
            raise RuntimeError(
                "Simple Summary dimensions were not captured during block construction"
            )

        config = self._simple_summary_config_arg
        if config is None:
            config = SimpleTrackSummaryBottleneckConfig(
                video_dim=self._simple_summary_video_dim,
                audio_dim=self._simple_summary_audio_dim,
                norm_eps=self._simple_summary_norm_eps,
            )
        else:
            if config.video_dim != self._simple_summary_video_dim:
                raise ValueError(
                    "simple_summary_config.video_dim must equal video.dim: "
                    f"{config.video_dim} vs {self._simple_summary_video_dim}"
                )
            if config.audio_dim != self._simple_summary_audio_dim:
                raise ValueError(
                    "simple_summary_config.audio_dim must equal audio.dim: "
                    f"{config.audio_dim} vs {self._simple_summary_audio_dim}"
                )

        import ltx_core.model.transformer.global_args as global_args
        if global_args.USE_LTX_ATTN:
            self.simple_summary = SimpleTrackSummaryTokensWithLTXAttention(
                config, ops=self._simple_summary_attention_ops,
            ).to(
                device=device,
                dtype=dtype,
            )
        else:
            self.simple_summary = SimpleTrackSummaryTokens(
                config,
            ).to(
                device=device,
                dtype=dtype,
            )

        self.simple_summary_config = self.simple_summary.config

        # Use [1] rather than a scalar for FSDP compatibility. The Summary
        # module's final audio projection is zero-initialized, so scale=1 keeps
        # the pretrained block exactly unchanged at initialization.
        self.simple_summary_scale = torch.nn.Parameter(
            torch.ones(1, device=device, dtype=dtype)
        )

    def _compute_simple_summary(
        self,
        *,
        video: TransformerArgs,
        audio: TransformerArgs,
        video_hidden: torch.Tensor,
        audio_hidden: torch.Tensor,
    ) -> SimpleTrackSummaryOutput | None:
        if not self.use_simple_summary:
            return None
        if self.simple_summary is None or self.simple_summary_scale is None:
            raise RuntimeError(
                "use_simple_summary=True, but initialize_simple_summary() "
                "has not been called"
            )

        return self.simple_summary(
            video_hidden=video_hidden,
            audio_hidden=audio_hidden,
            target_token_count=getattr(video, "target_token_count", None),
            reference_token_count=getattr(video, "reference_token_count", None),
            target_valid=getattr(video, "target_token_valid", None),
            reference_valid=getattr(video, "reference_token_valid", None),
        )

    def _clear_summary_debug_state(self) -> None:
        self.last_simple_summary = None
        self.last_reference_summary = None
        self.last_target_summary = None

    def forward_version3(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """Inject the gated Summary delta directly into ``ax`` before A2V/V2A.

        Unlike V1, Summary is never merged into ``v2a_delta``.  Unlike V2, the
        Summary residual is committed to the audio stream immediately after the
        Summary computation, so the following A2V/V2A modules consume the
        Summary-updated audio hidden state.  Track-Ada v2 already folds
        ``track_gate`` into ``summary_output.audio_delta``.

        ``post_sa_function`` is intentionally not used for Simple Summary; it
        remains only on the actual self-attention residuals.
        """
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = (
            video is not None
            and video.enabled
            and vx is not None
            and vx.numel() > 0
        )
        run_ax = (
            audio is not None
            and audio.enabled
            and ax is not None
            and ax.numel() > 0
        )
        run_a2v = (
            run_vx
            and audio is not None
            and ax is not None
            and ax.numel() > 0
        )
        run_v2a = (
            run_ax
            and video is not None
            and vx is not None
            and vx.numel() > 0
        )
        # ==============================================================
        # 1. VIDEO BRANCH: self-attention -> text cross-attention
        # ==============================================================
        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table,
                vx.shape[0],
                video.timesteps,
                slice(0, 3),
            )
            norm_vx = self.ada_zero_function(
                vx,
                self.norm_eps,
                vscale_msa,
                vshift_msa,
            )
            del vshift_msa, vscale_msa

            vx_msa_out = self.attn1(
                norm_vx,
                pe=video.positional_embeddings,
                mask=video.self_attention_mask,
                perturbation_mask=video.self_attn_perturbation_mask,
                all_perturbed=video.self_attn_all_perturbed,
            )
            vx, vx_normed = self.post_sa_function(
                vx,
                vx_msa_out,
                None,
                self.norm_eps,
                vgate_msa,
            )
            del vgate_msa, norm_vx, vx_msa_out

            vx = vx + self._apply_text_cross_attention(
                vx_normed,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            del vx_normed

        # ==============================================================
        # 2. AUDIO BRANCH: self-attention -> text cross-attention
        # ==============================================================
        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table,
                ax.shape[0],
                audio.timesteps,
                slice(0, 3),
            )
            norm_ax = self.ada_zero_function(
                ax,
                self.norm_eps,
                ascale_msa,
                ashift_msa,
            )
            del ashift_msa, ascale_msa

            ax_msa_out = self.audio_attn1(
                norm_ax,
                pe=audio.positional_embeddings,
                mask=audio.self_attention_mask,
                perturbation_mask=audio.self_attn_perturbation_mask,
                all_perturbed=audio.self_attn_all_perturbed,
            )
            ax, ax_normed = self.post_sa_function(
                ax,
                ax_msa_out,
                None,
                self.norm_eps,
                agate_msa,
            )
            del agate_msa, norm_ax, ax_msa_out

            ax = ax + self._apply_text_cross_attention(
                ax_normed,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            del ax_normed

        # ==============================================================
        # 3. SIMPLE SUMMARY -> DIRECT AUDIO RESIDUAL
        # ==============================================================
        summary_delta: torch.Tensor | None = None
        if (
            run_v2a
            and not audio.cross_attn_skip_all
            and self.use_simple_summary
        ):
            summary_output = self._compute_simple_summary(
                video=video,
                audio=audio,
                video_hidden=vx,
                audio_hidden=ax,
            )
            if summary_output is not None:
                summary_delta = summary_output.audio_delta
                summary_delta = summary_delta * self.simple_summary_scale.to(
                    device=summary_delta.device,
                    dtype=summary_delta.dtype,
                )

                if summary_delta.shape != ax.shape:
                    raise RuntimeError(
                        "Simple Summary output shape must equal audio hidden "
                        f"shape, got {tuple(summary_delta.shape)} vs "
                        f"{tuple(ax.shape)}"
                    )

                # The Track-Ada v2 override has already applied track_gate to
                # audio_delta.  Add this branch directly to the audio stream,
                # analogous to x = x + branch_output * branch_mask.
                ax = ax + (
                    summary_delta
                    * audio.cross_attn_perturbation_mask
                )

                self.last_simple_summary = summary_output.summary_tokens.detach()
                self.last_reference_summary = (
                    summary_output.reference_summary.detach()
                )
                self.last_target_summary = summary_output.target_summary.detach()
        else:
            self._clear_summary_debug_state()

        del summary_delta

        # ==============================================================
        # 4. ORIGINAL AUDIO <-> VIDEO CROSS-ATTENTION
        #
        # V3 snapshots are taken after the direct Summary residual, so both
        # A2V and V2A see the Summary-updated audio hidden state.
        # ==============================================================
        if run_a2v or run_v2a:
            vx_pre_av = vx
            ax_pre_av = ax

            if run_a2v and not video.cross_attn_skip_all:
                (
                    scale_ca_video_a2v,
                    shift_ca_video_a2v,
                    gate_out_a2v,
                ) = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_vx_scaled = self.ada_zero_function(
                    vx_pre_av,
                    self.norm_eps,
                    scale_ca_video_a2v,
                    shift_ca_video_a2v,
                )
                del scale_ca_video_a2v, shift_ca_video_a2v

                (
                    scale_ca_audio_a2v,
                    shift_ca_audio_a2v,
                    _,
                ) = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_ax_scaled = self.ada_zero_function(
                    ax_pre_av,
                    self.norm_eps,
                    scale_ca_audio_a2v,
                    shift_ca_audio_a2v,
                )
                del scale_ca_audio_a2v, shift_ca_audio_a2v

                vx = vx + (
                    self.audio_to_video_attn(
                        a2v_vx_scaled,
                        context=a2v_ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                    * video.cross_attn_perturbation_mask
                )
                del gate_out_a2v, a2v_vx_scaled, a2v_ax_scaled

            if run_v2a and not audio.cross_attn_skip_all:
                (
                    scale_ca_audio_v2a,
                    shift_ca_audio_v2a,
                    gate_out_v2a,
                ) = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_ax_scaled = self.ada_zero_function(
                    ax_pre_av,
                    self.norm_eps,
                    scale_ca_audio_v2a,
                    shift_ca_audio_v2a,
                )
                del scale_ca_audio_v2a, shift_ca_audio_v2a

                (
                    scale_ca_video_v2a,
                    shift_ca_video_v2a,
                    _,
                ) = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_vx_scaled = self.ada_zero_function(
                    vx_pre_av,
                    self.norm_eps,
                    scale_ca_video_v2a,
                    shift_ca_video_v2a,
                )
                del scale_ca_video_v2a, shift_ca_video_v2a

                ax = ax + (
                    self.video_to_audio_attn(
                        v2a_ax_scaled,
                        context=v2a_vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                    * audio.cross_attn_perturbation_mask
                )
                del gate_out_v2a, v2a_vx_scaled, v2a_ax_scaled

            del vx_pre_av, ax_pre_av

        # ==============================================================
        # 5. ORIGINAL VIDEO/AUDIO FFN
        # ==============================================================
        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table,
                vx.shape[0],
                video.timesteps,
                slice(3, 6),
            )
            vx_scaled = self.ada_zero_function(
                vx,
                self.norm_eps,
                vscale_mlp,
                vshift_mlp,
            )
            vx = vx + self.ff(vx_scaled) * vgate_mlp
            del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table,
                ax.shape[0],
                audio.timesteps,
                slice(3, 6),
            )
            ax_scaled = self.ada_zero_function(
                ax,
                self.norm_eps,
                ascale_mlp,
                ashift_mlp,
            )
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp
            del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

        return (
            replace(video, x=vx) if video is not None else None,
            replace(audio, x=ax) if audio is not None else None,
        )

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        return self.forward_version3(video, audio)
