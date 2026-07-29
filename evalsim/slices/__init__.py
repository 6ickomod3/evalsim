"""Source-only scenario slices used by the M5 scorecards."""

from .m5 import (
    M5_SLICE_NAMES,
    M5_SLICE_REGISTRY,
    M5_SLICE_SPECS,
    M5_SLICE_VERSION,
    SliceRegistry,
    SliceResult,
    SliceSpec,
    evaluate_m5_slices,
)

__all__ = [
    "M5_SLICE_NAMES",
    "M5_SLICE_REGISTRY",
    "M5_SLICE_SPECS",
    "M5_SLICE_VERSION",
    "SliceRegistry",
    "SliceResult",
    "SliceSpec",
    "evaluate_m5_slices",
]
