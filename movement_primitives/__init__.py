"""
Movement Primitives Playground

A comprehensive Python framework for implementing various Movement Primitives algorithms
including DMP, ProDMP, ProMP, CNMP, and other classic movement primitive approaches.
"""

__version__ = "0.1.0"
__author__ = "Movement Primitives Contributors"

# Import main classes
from .base import BaseMovementPrimitive, MovementPrimitiveConfig
from .dmp import DMP, DMPConfig
from .prodmp import ProDMP, ProDMPConfig
from .promp import ProMP, ProMPConfig
from .cnmp import CNMP, CNMPConfig

# Import utilities
from .utils import (
    GaussianBasis,
    RadialBasisFunction,
    PhaseGenerator,
    TrajectoryGenerator,
    plot_trajectory,
    save_trajectory,
    load_trajectory,
)

__all__ = [
    # Base classes
    "BaseMovementPrimitive",
    "MovementPrimitiveConfig",
    
    # Movement Primitive implementations
    "DMP",
    "DMPConfig",
    "ProDMP", 
    "ProDMPConfig",
    "ProMP",
    "ProMPConfig",
    "CNMP",
    "CNMPConfig",
    
    # Utilities
    "GaussianBasis",
    "RadialBasisFunction", 
    "PhaseGenerator",
    "TrajectoryGenerator",
    "plot_trajectory",
    "save_trajectory",
    "load_trajectory",
]
