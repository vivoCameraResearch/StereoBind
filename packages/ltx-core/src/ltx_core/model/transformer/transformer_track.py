from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch

from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer_args import TransformerArgs

try:
    from .simple_summary_tokens import SimpleTrackSummaryConfig
    from .transformer import (
        BasicAVTransformerBlockWithSimpleSummary,
        TransformerConfig,
        TransformerOpsConfig,
    )
except ImportError:
    # Allows the file to work when this directory itself is placed on PYTHONPATH.
    from simple_summary_tokens import SimpleTrackSummaryConfig
    from transformer import (
        BasicAVTransformerBlockWithSimpleSummary,
        TransformerConfig,
        TransformerOpsConfig,
    )


class BasicAVTransformerBlockWithSimpleSummary_version2(
    BasicAVTransformerBlockWithSimpleSummary
):
    """Simple-Summary block with a dedicated Track-Ada modulation branch.

    Track modulation applies only to the Simple-Summary branch input, using the
    same AdaZero operation as the base transformer (RMSNorm, then scale/shift).
    The helper returns a gated residual contribution, not the updated main stream:

        audio_delta = SimpleSummary(AdaZero(audio_hidden, scale, shift)) * gate

    The parent forward applies the Summary scale and shared V2A gate/mask, then
    adds the contribution to the main audio stream exactly once. The Summary
    module's internal query LayerNorm remains unchanged.

    Without timestep conditioning:

        track_embedding + scale_shift_table_audio_track
            -> track_shift / track_scale / track_gate

    With ``apply_track_with_timesteps=True``:

        track_embedding + track_timesteps_embedding
            + scale_shift_table_audio_track
            -> track_shift / track_scale / track_gate

    The table has only three rows because v2 has one injection site:
    shift, scale and gate. It is initialized to (0, 0, 1), while the STE output
    projection and optional track timestep Ada projection are initialized to zero.
    Zero shift/scale therefore gives RMSNorm(audio_hidden), not an identity input.
    A freshly initialized Simple Summary still contributes zero residual because
    its final audio output projection is zero-initialized.
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
        super().__init__(
            video=video,
            audio=audio,
            rope_type=rope_type,
            norm_eps=norm_eps,
            ops=ops,
            transformer_args=args,
            simple_summary_config=simple_summary_config,
        )

        if audio is None:
            raise ValueError(
                "BasicAVTransformerBlockWithSimpleSummary_version2 requires an audio configuration"
            )
        if not self.use_simple_summary:
            raise ValueError(
                "track_injection='v2' requires transformer_args['use_simple_summary']=True"
            )

        self.apply_track_with_timesteps = bool(
            args.get("apply_track_with_timesteps", False)
        )

        # Keep only construction metadata here. The v2 Track-Ada parameter is
        # created only after the base LTX checkpoint has finished loading.
        self._track_ada_audio_dim = audio.dim
        self.scale_shift_table_audio_track: torch.nn.Parameter | None = None

    def initialize_track_injection_modules(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Create v2-only modules after the base LTX checkpoint is loaded."""
        if self.scale_shift_table_audio_track is not None:
            return

        device = torch.device(device)
        if device.type == "meta":
            raise RuntimeError(
                "v2 track modules must be initialized on a real device"
            )

        track_ada_table = torch.zeros(
            3,
            self._track_ada_audio_dim,
            device=device,
            dtype=dtype,
        )
        track_ada_table[2].fill_(1.0)
        self.scale_shift_table_audio_track = torch.nn.Parameter(
            track_ada_table
        )

    def get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
        indices: slice,
    ) -> tuple[torch.Tensor, ...]:
        """Same AdaLN calculation as the parent, without the debug breakpoint."""
        num_ada_params = scale_shift_table.shape[0]
        ada_values = (
            scale_shift_table[indices]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(
                batch_size,
                timestep.shape[1],
                num_ada_params,
                -1,
            )[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_track_ada_values(
        self,
        track_embedding: torch.Tensor,
        batch_size: int,
        track_timesteps_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if track_embedding.ndim != 3:
            raise ValueError(
                "track_embedding must have shape [B, T_audio, 3*audio_dim], "
                f"got {tuple(track_embedding.shape)}"
            )
        if track_embedding.shape[0] != batch_size:
            raise ValueError(
                "track_embedding batch size must match audio hidden batch size: "
                f"track={track_embedding.shape[0]}, audio={batch_size}"
            )

        scale_shift_table_audio_track = self.scale_shift_table_audio_track
        if scale_shift_table_audio_track is None:
            raise RuntimeError(
                "v2 track modules are not initialized. Call "
                "initialize_track_injection_modules() after load_transformer()."
            )

        num_ada_params, audio_dim = scale_shift_table_audio_track.shape
        expected_dim = num_ada_params * audio_dim
        if track_embedding.shape[-1] != expected_dim:
            raise ValueError(
                "track_embedding last dimension must equal 3 * audio_dim: "
                f"expected {expected_dim}, got {track_embedding.shape[-1]}"
            )

        track_ada = (
            scale_shift_table_audio_track
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=track_embedding.device, dtype=track_embedding.dtype)
            + track_embedding.reshape(
                batch_size,
                track_embedding.shape[1],
                num_ada_params,
                audio_dim,
            )
        )

        if track_timesteps_embedding is not None:
            if track_timesteps_embedding.ndim != 3:
                raise ValueError(
                    "track_timesteps_embedding must have shape [B, N_or_1, 3*audio_dim], "
                    f"got {tuple(track_timesteps_embedding.shape)}"
                )
            if track_timesteps_embedding.shape[0] != batch_size:
                raise ValueError(
                    "track_timesteps_embedding batch size must match audio batch: "
                    f"track_t={track_timesteps_embedding.shape[0]}, audio={batch_size}"
                )
            if track_timesteps_embedding.shape[1] not in (1, track_embedding.shape[1]):
                raise ValueError(
                    "track_timesteps_embedding token count must be 1 or match track_embedding: "
                    f"track_t={track_timesteps_embedding.shape[1]}, "
                    f"track={track_embedding.shape[1]}"
                )
            if track_timesteps_embedding.shape[-1] != expected_dim:
                raise ValueError(
                    "track_timesteps_embedding last dimension must equal 3 * audio_dim: "
                    f"expected {expected_dim}, got {track_timesteps_embedding.shape[-1]}"
                )

            track_timestep_ada = track_timesteps_embedding.to(
                device=track_embedding.device,
                dtype=track_embedding.dtype,
            ).reshape(
                batch_size,
                track_timesteps_embedding.shape[1],
                num_ada_params,
                audio_dim,
            )
            # The token dimension may be 1. PyTorch broadcasting then applies
            # the same denoising condition to every audio token, matching LTX's
            # normal global-timestep behavior.
            track_ada = track_ada + track_timestep_ada

        track_shift, track_scale, track_gate = track_ada.unbind(dim=2)
        return track_shift, track_scale, track_gate

    def _compute_simple_summary(
        self,
        *,
        video: TransformerArgs,
        audio: TransformerArgs,
        video_hidden: torch.Tensor,
        audio_hidden: torch.Tensor,
    ):
        track_embedding = getattr(audio, "track_embedding", None)
        if track_embedding is None:
            raise ValueError(
                "track_injection='v2' requires audio.track_embedding. "
                "Use TrackTransformerArgsPreprocessor_withSTE or "
                "TrackMultiModalTransformerArgsPreprocessor_withSTE."
            )
        if track_embedding.shape[1] != audio_hidden.shape[1]:
            raise ValueError(
                "track_embedding temporal length must match audio_hidden: "
                f"track={track_embedding.shape[1]}, audio={audio_hidden.shape[1]}"
            )

        track_timesteps_embedding = getattr(
            audio,
            "track_timesteps_embedding",
            None,
        )
        if self.apply_track_with_timesteps and track_timesteps_embedding is None:
            raise ValueError(
                "apply_track_with_timesteps=True requires "
                "audio.track_timesteps_embedding, but it is None"
            )
        if not self.apply_track_with_timesteps:
            track_timesteps_embedding = None

        track_shift, track_scale, track_gate = self.get_track_ada_values(
            track_embedding,
            batch_size=audio_hidden.shape[0],
            track_timesteps_embedding=track_timesteps_embedding,
        )
        track_shift = track_shift.to(
            device=audio_hidden.device,
            dtype=audio_hidden.dtype,
        )
        track_scale = track_scale.to(
            device=audio_hidden.device,
            dtype=audio_hidden.dtype,
        )
        track_gate = track_gate.to(
            device=audio_hidden.device,
            dtype=audio_hidden.dtype,
        )

        # Match the base transformer's pre-branch AdaZero modulation. Use the
        # configured operation (PytorchAdaZeroFunction by default).
        audio_hidden_for_summary = self.ada_zero_function(
            audio_hidden,
            self.norm_eps,
            track_scale,
            track_shift,
        )

        summary_output = super()._compute_simple_summary(
            video=video,
            audio=audio,
            video_hidden=video_hidden,
            audio_hidden=audio_hidden_for_summary,
        )
        if summary_output is None:
            return None

        if summary_output.audio_delta.shape != track_gate.shape:
            raise RuntimeError(
                "Track gate must match Simple-Summary audio_delta shape: "
                f"gate={tuple(track_gate.shape)}, "
                f"audio_delta={tuple(summary_output.audio_delta.shape)}"
            )

        # Gate only the branch delta, as in x + branch(AdaZero(x)) * gate.
        # The parent forward owns the residual addition; adding audio_hidden
        # here would incorrectly add the main stream a second time.
        return replace(
            summary_output,
            audio_delta=summary_output.audio_delta * track_gate,
        )
