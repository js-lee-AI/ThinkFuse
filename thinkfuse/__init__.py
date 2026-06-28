from .alignment import select_aligned_prefix
from .fusion import FusionResult, ThinkFuse
from .model_client import MockModelClient, ModelClient
from .scoring import select_by_primary_ppl
from .tags import (
    THINK_TAG_MAP,
    PhaseDetector,
    adapt_think_tags_for_model,
    get_think_tags,
    normalize_think_tags,
)
from .threshold import AdaptiveThreshold, SoftFusionBudget
from .uncertainty import UncertaintyCalculator

__all__ = [
    "ThinkFuse",
    "FusionResult",
    "ModelClient",
    "MockModelClient",
    "UncertaintyCalculator",
    "AdaptiveThreshold",
    "SoftFusionBudget",
    "PhaseDetector",
    "THINK_TAG_MAP",
    "get_think_tags",
    "normalize_think_tags",
    "adapt_think_tags_for_model",
    "select_aligned_prefix",
    "select_by_primary_ppl",
]

__version__ = "0.1.0"
