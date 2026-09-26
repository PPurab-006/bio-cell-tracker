"""3D cell detection module."""

from .base import BaseDetector, DetectionResult
from .classical_dog import AnisotropicDoGDetector
from .local_maxima import extract_3d_local_maxima
from .adaptive_dog import AdaptiveDoGDetector
from .temporal_observability import (
    extract_physical_patch,
    compute_intensity_statistics,
    compute_dog_observability,
    classify_observability,
    analyze_ground_truth_observability,
)

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "AnisotropicDoGDetector",
    "AdaptiveDoGDetector",
    "extract_3d_local_maxima",
    "extract_physical_patch",
    "compute_intensity_statistics",
    "compute_dog_observability",
    "classify_observability",
    "analyze_ground_truth_observability",
]
