from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch

from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.simple_summary_tokens import (
    SimpleTrackSummaryConfig,
    SimpleTrackSummaryOutput,
    SimpleTrackSummaryTokens,
)
from ltx_core.model.transformer.transformer import (
    BasicAVTransformerBlock,
    TransformerConfig,
    TransformerOpsConfig,
)
from ltx_core.model.transformer.transformer_args import TransformerArgs
