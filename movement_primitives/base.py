"""
Base classes and interfaces for Movement Primitives.

This module provides the foundational classes that all movement primitive
implementations should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MovementPrimitiveConfig:
    """Base configuration class for movement primitives."""
    
    # Common parameters
    n_dims: int = 1
    dt: float = 0.01
    execution_time: float = 1.0
    
    # Phase parameters
    alpha: float = 25.0  # Phase convergence rate
    beta: float = 6.25   # Phase convergence rate
    
    # Basis function parameters
    n_basis: int = 50
    basis_width: float = 0.1
    
    # Learning parameters
    regularization: float = 1e-6
    
    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_dims <= 0:
            raise ValueError("n_dims must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.execution_time <= 0:
            raise ValueError("execution_time must be positive")
        if self.n_basis <= 0:
            raise ValueError("n_basis must be positive")


class BaseMovementPrimitive(ABC):
    """
    Abstract base class for all movement primitive implementations.
    
    This class defines the common interface that all movement primitives
    should implement, including training, execution, and utility methods.
    """
    
    def __init__(self, config: MovementPrimitiveConfig):
        """
        Initialize the movement primitive.
        
        Args:
            config: Configuration object containing all parameters
        """
        self.config = config
        self.is_trained = False
        self.training_data = None
        
    @abstractmethod
    def fit(self, trajectories: Union[np.ndarray, List[np.ndarray]], 
            **kwargs) -> 'BaseMovementPrimitive':
        """
        Train the movement primitive on demonstration data.
        
        Args:
            trajectories: Demonstration trajectories. Can be a single trajectory
                        (shape: [T, n_dims]) or a list of trajectories
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def generate(self, start: Optional[np.ndarray] = None,
                 goal: Optional[np.ndarray] = None,
                 execution_time: Optional[float] = None,
                 **kwargs) -> np.ndarray:
        """
        Generate a trajectory using the trained movement primitive.
        
        Args:
            start: Starting position (shape: [n_dims])
            goal: Goal position (shape: [n_dims])
            execution_time: Execution time for the trajectory
            **kwargs: Additional generation parameters
            
        Returns:
            Generated trajectory (shape: [T, n_dims])
        """
        pass
    
    @abstractmethod
    def get_weights(self) -> np.ndarray:
        """
        Get the learned weights of the movement primitive.
        
        Returns:
            Weight vector
        """
        pass
    
    @abstractmethod
    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set the weights of the movement primitive.
        
        Args:
            weights: Weight vector to set
        """
        pass
    
    def is_ready(self) -> bool:
        """
        Check if the movement primitive is ready for execution.
        
        Returns:
            True if trained and ready, False otherwise
        """
        return self.is_trained
    
    def get_config(self) -> MovementPrimitiveConfig:
        """
        Get the configuration of the movement primitive.
        
        Returns:
            Configuration object
        """
        return self.config
    
    def get_training_data(self) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
        """
        Get the training data used to train the movement primitive.
        
        Returns:
            Training trajectories or None if not trained
        """
        return self.training_data
    
    def save(self, filepath: str) -> None:
        """
        Save the movement primitive to a file.
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        
        data = {
            'config': self.config,
            'weights': self.get_weights(),
            'is_trained': self.is_trained,
            'training_data': self.training_data
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str) -> None:
        """
        Load a movement primitive from a file.
        
        Args:
            filepath: Path to load the model from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.config = data['config']
        self.set_weights(data['weights'])
        self.is_trained = data['is_trained']
        self.training_data = data['training_data']
    
    def reset(self) -> None:
        """Reset the movement primitive to untrained state."""
        self.is_trained = False
        self.training_data = None
        # Reset weights to zero or random initialization
        self.set_weights(np.zeros(self._get_weight_shape()))
    
    @abstractmethod
    def _get_weight_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the weight vector.
        
        Returns:
            Shape tuple for the weight vector
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the movement primitive."""
        return f"{self.__class__.__name__}(config={self.config}, trained={self.is_trained})"
