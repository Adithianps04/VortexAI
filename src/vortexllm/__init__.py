"""
VortexLLM - A quantum-inspired language model with phase synchronization
"""

from .vortex_ai import VortexAI
from .ahamov_net import AhamovNet
from .brain_memory import BrainMemory
from .performance_monitor import PerformanceMonitor
from .resonance_optimizer import ResonanceOptimizer

__version__ = "0.1.0"
__all__ = [
    "VortexAI",
    "AhamovNet",
    "BrainMemory", 
    "PerformanceMonitor",
    "ResonanceOptimizer"
]
