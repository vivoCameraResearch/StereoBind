import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTrackEncoder_version1(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        audio_dim: int = 2048,
    ):
        super().__init__()

        self.audio_dim = audio_dim

        # [x, y] -> hidden feature
        self.track_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # hidden feature -> scale
        self.scale_head = nn.Linear(
            hidden_dim,
            audio_dim,
        )

        # hidden feature -> shift
        self.shift_head = nn.Linear(
            hidden_dim,
            audio_dim,
        )

        # 初始化为 0：
        # scale = 0
        # shift = 0
        # 初始时不会改变原始 audio latent
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

        nn.init.zeros_(self.shift_head.weight)
        nn.init.zeros_(self.shift_head.bias)

    def forward(
        self,
        track: torch.Tensor,
        audio_length: int,
    ):
        """
        Args:
            track:
                [121, 2]
                or
                [B, 121, 2]

            audio_length:
                audio latent 的时间长度 T

        Returns:
            scale:
                [B, T, 2048]

            shift:
                [B, T, 2048]
        """

        # 支持输入 [121, 2]
        if track.ndim == 2:
            track = track.unsqueeze(0)

        if track.ndim != 3 or track.shape[-1] != 2:
            raise ValueError(
                f"track should have shape [121, 2] or [B, 121, 2], "
                f"but got {track.shape}"
            )

        # --------------------------------
        # 1. encode track
        # --------------------------------
        # [B, 121, 2]
        # ->
        # [B, 121, hidden_dim]
        x = self.track_encoder(track)

        # --------------------------------
        # 2. 对齐到 audio token 数量 T
        # --------------------------------
        if x.shape[1] != audio_length:
            # interpolate 输入要求 [B, C, T]
            x = x.transpose(1, 2)

            x = F.interpolate(
                x,
                size=audio_length,
                mode="linear",
                align_corners=True,
            )

            x = x.transpose(1, 2)

        # x:
        # [B, T, hidden_dim]

        # --------------------------------
        # 3. generate scale / shift
        # --------------------------------
        scale = self.scale_head(x)
        shift = self.shift_head(x)

        # [B, T, 2048]

        return scale, shift
    

class SpatialTrackEncoder_version2(nn.Module):
    """
    Spatial Track Encoder version 2.

    将二维空间轨迹:
        track_xy: [B, T_track, 2]

    编码为与 LTX AdaLN timestep embedding 对齐的 track embedding:
        track_embeddings: [B, T_audio, num_ada_params * audio_dim]

    默认:
        T_track = 121
        T_audio = 122
        audio_dim = 2048
        num_ada_params = 6

    因此:
        input : [B, 121, 2]
        output: [B, 122, 6 * 2048]
              = [B, 122, 12288]

    输出不会在这里直接拆成 shift / scale / gate。

    在后续 Transformer Block 中，应当执行:

        track_ada =
            track_shift_scale_table
            + track_embeddings

    然后 reshape 为:

        [B, T_audio, num_ada_params, audio_dim]

    再拆成:

        0 -> shift_msa
        1 -> scale_msa
        2 -> gate_msa

        3 -> shift_mlp
        4 -> scale_mlp
        5 -> gate_mlp
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        audio_dim: int = 2048,
        num_ada_params: int = 6,
        input_num_frames: int = 121,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.audio_dim = audio_dim
        self.num_ada_params = num_ada_params
        self.input_num_frames = input_num_frames

        self.output_dim = num_ada_params * audio_dim

        # ---------------------------------------------------------
        # 1. 将二维轨迹坐标投影到 hidden space
        # ---------------------------------------------------------
        self.input_proj = nn.Linear(
            input_dim,
            hidden_dim,
        )

        # ---------------------------------------------------------
        # 2. 可学习的 temporal positional embedding
        #
        # shape:
        #   [1, 121, hidden_dim]
        # ---------------------------------------------------------
        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                input_num_frames,
                hidden_dim,
            )
        )

        # ---------------------------------------------------------
        # 3. Temporal Transformer
        #
        # 用于理解整个 motion trajectory，而不是逐帧独立 MLP。
        # ---------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # ---------------------------------------------------------
        # 4. 输出前 normalization
        # ---------------------------------------------------------
        self.norm = nn.LayerNorm(hidden_dim)

        # ---------------------------------------------------------
        # 5. 类似 LTX AdaLayerNormSingle 的最终 projection
        #
        # hidden_dim
        #       ↓
        # num_ada_params * audio_dim
        #
        # 默认:
        # 256 -> 6 * 2048 = 12288
        # ---------------------------------------------------------
        self.ada_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                self.output_dim,
            ),
        )

        self._init_weights()

    def _init_weights(self):
        """
        初始化新增加的 STE。

        这里不把最后的 ada_proj 强制 zero-init。

        原因:
        track_shift_scale_table + track_embedding
        本身就是一个独立的新 conditioning branch。

        如果后续希望整个 track injection 在训练开始时
        严格为 0，更推荐在 Transformer Block 侧给
        track branch 增加 zero-init gain，而不是让 STE
        自己完全输出 0。
        """

        nn.init.normal_(
            self.pos_embedding,
            mean=0.0,
            std=0.02,
        )

        nn.init.xavier_uniform_(
            self.input_proj.weight
        )

        if self.input_proj.bias is not None:
            nn.init.zeros_(
                self.input_proj.bias
            )

        final_linear = self.ada_proj[-1]

        nn.init.xavier_uniform_(
            final_linear.weight,
            gain=0.02,
        )

        if final_linear.bias is not None:
            nn.init.zeros_(
                final_linear.bias
            )

    def _get_pos_embedding(
        self,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        当输入不是严格 121 帧时，对 positional embedding
        做 temporal interpolation。

        正常 LTX-2.3 121-frame training/inference 下不会触发。
        """

        if num_frames == self.input_num_frames:
            return self.pos_embedding.to(
                device=device,
                dtype=dtype,
            )

        # [1, T, C]
        # ->
        # [1, C, T]
        pos = self.pos_embedding.transpose(1, 2)

        pos = F.interpolate(
            pos,
            size=num_frames,
            mode="linear",
            align_corners=False,
        )

        # [1, C, T]
        # ->
        # [1, T, C]
        pos = pos.transpose(1, 2)

        return pos.to(
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _temporal_resample(
        x: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """
        将 track temporal length 对齐到 audio token length。

        输入:
            x: [B, T_track, C]

        输出:
            [B, T_audio, C]

        例如:
            [B, 121, 256]
                ->
            [B, 122, 256]
        """

        if x.shape[1] == target_length:
            return x

        # [B, T, C]
        # ->
        # [B, C, T]
        x = x.transpose(1, 2)

        x = F.interpolate(
            x,
            size=target_length,
            mode="linear",
            align_corners=False,
        )

        # [B, C, T_audio]
        # ->
        # [B, T_audio, C]
        x = x.transpose(1, 2)

        return x

    def forward(
        self,
        track_xy: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """
        Args
        ----
        track_xy:
            shape:
                [B, T_track, 2]

            推荐输入已经归一化，例如:
                x, y ∈ [-1, 1]

            或:
                x, y ∈ [0, 1]

            但训练和推理必须保持完全一致。

        target_length:
            audio Transformer token 数量。

            例如 LTX-2.3 当前 sample:
                122

        Returns
        -------
        track_embeddings:
            shape:
                [
                    B,
                    target_length,
                    num_ada_params * audio_dim
                ]

            默认:
                [B, 122, 12288]
        """

        if track_xy.ndim != 3:
            raise ValueError(
                "track_xy must have shape "
                "[B, T_track, 2], "
                f"got {tuple(track_xy.shape)}."
            )

        if track_xy.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected track_xy last dim = {self.input_dim}, "
                f"got {track_xy.shape[-1]}."
            )

        if target_length <= 0:
            raise ValueError(
                f"target_length must be > 0, "
                f"got {target_length}."
            )

        # ---------------------------------------------------------
        # [B, T_track, 2]
        # ->
        # [B, T_track, hidden_dim]
        # ---------------------------------------------------------
        x = self.input_proj(track_xy)

        # ---------------------------------------------------------
        # temporal positional information
        # ---------------------------------------------------------
        pos = self._get_pos_embedding(
            num_frames=x.shape[1],
            dtype=x.dtype,
            device=x.device,
        )

        x = x + pos

        # ---------------------------------------------------------
        # model the whole trajectory
        #
        # [B, T_track, hidden_dim]
        # ---------------------------------------------------------
        x = self.transformer(x)

        x = self.norm(x)

        # ---------------------------------------------------------
        # 121 track frames
        # ->
        # audio token length
        #
        # [B, 121, hidden_dim]
        # ->
        # [B, T_audio, hidden_dim]
        # ---------------------------------------------------------
        x = self._temporal_resample(
            x,
            target_length=target_length,
        )

        # ---------------------------------------------------------
        # 与 LTX timestep Ada embedding 维度对齐
        #
        # [B, T_audio, hidden_dim]
        # ->
        # [B, T_audio, num_ada_params * audio_dim]
        # ---------------------------------------------------------
        track_embeddings = self.ada_proj(x)

        return track_embeddings