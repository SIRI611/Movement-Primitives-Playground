"""
Dynamic Movement Primitives (DMP) implementation.

DMPs are a framework for learning and reproducing complex motor behaviors
based on dynamical systems theory.
"""

import numpy as np
from typing import Union, List, Optional, Tuple
from dataclasses import dataclass
from .base import BaseMovementPrimitive, MovementPrimitiveConfig
from .utils import PhaseGenerator, GaussianBasis, TrajectoryGenerator


@dataclass
class DMPConfig(MovementPrimitiveConfig):
    """Configuration for Dynamic Movement Primitives."""
    
    # DMP-specific parameters
    alpha: float = 25.0  # Phase convergence rate
    beta: float = 6.25   # Phase convergence rate
    tau: float = 1.0     # Temporal scaling factor
    
    # Canonical system parameters
    canonical_alpha: float = 25.0
    
    # Transformation system parameters
    transformation_alpha: float = 25.0
    transformation_beta: float = 6.25
    
    # Basis function parameters
    n_basis: int = 50
    basis_width: float = 0.1
    
    # Learning parameters
    regularization: float = 1e-6


class DMP(BaseMovementPrimitive):
    """
    Dynamic Movement Primitives implementation.
    
    DMPs encode movements as a set of nonlinear differential equations
    that can be learned from demonstrations and reproduced with different
    goals and temporal scales.
    """
    
    def __init__(self, config: DMPConfig):
        """
        Initialize DMP.
        
        Args:
            config: DMP configuration
        """
        super().__init__(config)
        self.config: DMPConfig = config
        
        # Initialize components
        self.phase_generator = PhaseGenerator(config.canonical_alpha)
        self.basis_functions = GaussianBasis(config.n_basis, config.basis_width)
        
        # DMP parameters
        self.weights = None
        self.start = None
        self.goal = None
        
        # Weight shape for each dimension
        self._weight_shape_per_dim = (config.n_basis,)
    
    def fit(self, trajectories: Union[np.ndarray, List[np.ndarray]], 
            **kwargs) -> 'DMP':
        """
        Train DMP on demonstration trajectories.
        
        Args:
            trajectories: Demonstration trajectories
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        # Handle single trajectory
        if isinstance(trajectories, np.ndarray):
            trajectories = [trajectories]
        
        # Validate trajectories
        for traj in trajectories:
            if traj.shape[1] != self.config.n_dims:
                raise ValueError(f"Trajectory dimension mismatch: expected {self.config.n_dims}, got {traj.shape[1]}")
        
        # Store training data
        self.training_data = trajectories
        
        # Learn weights for each dimension
        self.weights = np.zeros((self.config.n_dims, self.config.n_basis))
        
        for dim in range(self.config.n_dims):
            self.weights[dim] = self._learn_weights_dimension(trajectories, dim)
        
        self.is_trained = True
        return self
    
    def _learn_weights_dimension(self, trajectories: List[np.ndarray], dim: int) -> np.ndarray:
        """
        Learn weights for a specific dimension.
        
        Args:
            trajectories: List of demonstration trajectories
            dim: Dimension index
            
        Returns:
            Learned weights for the dimension
        """
        # Collect training data
        phases = []
        forcing_terms = []
        
        for traj in trajectories:
            # Generate phase
            phase = self.phase_generator.generate(self.config.dt, self.config.execution_time)
            
            # Compute forcing term
            forcing_term = self._compute_forcing_term(traj[:, dim], phase)
            
            phases.append(phase)
            forcing_terms.append(forcing_term)
        
        # Concatenate all trajectories
        all_phases = np.concatenate(phases)
        all_forcing_terms = np.concatenate(forcing_terms)
        
        # Evaluate basis functions
        basis_values = self.basis_functions(all_phases)
        
        # Solve linear system: F = Phi * w
        # Using ridge regression for regularization
        Phi_T_Phi = basis_values.T @ basis_values
        Phi_T_F = basis_values.T @ all_forcing_terms
        
        # Add regularization
        Phi_T_Phi += self.config.regularization * np.eye(self.config.n_basis)
        
        # Solve for weights
        weights = np.linalg.solve(Phi_T_Phi, Phi_T_F)
        
        return weights
    
    def _compute_forcing_term(self, trajectory: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """
        Compute forcing term from trajectory.
        
        Args:
            trajectory: Position trajectory for one dimension
            phase: Phase values
            
        Returns:
            Forcing term values
        """
        # Compute derivatives
        velocity = TrajectoryGenerator.compute_velocity(trajectory, self.config.dt)
        acceleration = TrajectoryGenerator.compute_acceleration(trajectory, self.config.dt)
        
        # Compute phase derivative
        phase_dot = self.phase_generator.get_phase_derivative(phase, self.config.dt)
        
        # Compute forcing term
        # F = tau^2 * y_ddot - alpha * (beta * (g - y) - tau * y_dot)
        start = trajectory[0]
        goal = trajectory[-1]
        
        forcing_term = (
            self.config.tau**2 * acceleration -
            self.config.transformation_alpha * (
                self.config.transformation_beta * (goal - trajectory) -
                self.config.tau * velocity
            )
        )
        
        # Scale by phase
        forcing_term = forcing_term / (phase * self.config.tau**2)
        
        return forcing_term
    
    def generate(self, start: Optional[np.ndarray] = None,
                 goal: Optional[np.ndarray] = None,
                 execution_time: Optional[float] = None,
                 **kwargs) -> np.ndarray:
        """
        Generate trajectory using trained DMP.
        
        Args:
            start: Starting position
            goal: Goal position
            execution_time: Execution time
            **kwargs: Additional parameters
            
        Returns:
            Generated trajectory
        """
        if not self.is_trained:
            raise RuntimeError("DMP must be trained before generating trajectories")
        
        # Set default values
        if start is None:
            start = np.zeros(self.config.n_dims)
        if goal is None:
            goal = np.ones(self.config.n_dims)
        if execution_time is None:
            execution_time = self.config.execution_time
        
        # Store start and goal
        self.start = np.array(start)
        self.goal = np.array(goal)
        
        # Generate phase
        phase = self.phase_generator.generate(self.config.dt, execution_time)
        
        # Generate trajectory for each dimension
        trajectory = np.zeros((len(phase), self.config.n_dims))
        
        for dim in range(self.config.n_dims):
            trajectory[:, dim] = self._generate_dimension(dim, phase, start[dim], goal[dim])
        
        return trajectory
    
    def _generate_dimension(self, dim: int, phase: np.ndarray, 
                           start: float, goal: float) -> np.ndarray:
        """
        Generate trajectory for a specific dimension.
        
        Args:
            dim: Dimension index
            phase: Phase values
            start: Starting position
            goal: Goal position
            
        Returns:
            Generated trajectory for the dimension
        """
        n_steps = len(phase)
        trajectory = np.zeros(n_steps)
        velocity = np.zeros(n_steps)
        
        # Initialize
        trajectory[0] = start
        velocity[0] = 0.0
        
        # Evaluate basis functions
        basis_values = self.basis_functions(phase)
        
        # Generate trajectory step by step
        for t in range(1, n_steps):
            dt = self.config.dt
            
            # Compute forcing term
            forcing_term = np.dot(basis_values[t], self.weights[dim])
            
            # Scale by phase
            forcing_term *= phase[t]
            
            # Compute acceleration
            acceleration = (
                self.config.transformation_alpha * (
                    self.config.transformation_beta * (goal - trajectory[t-1]) -
                    velocity[t-1]
                ) + forcing_term
            ) / self.config.tau**2
            
            # Integrate
            velocity[t] = velocity[t-1] + acceleration * dt
            trajectory[t] = trajectory[t-1] + velocity[t] * dt
        
        return trajectory
    
    def get_weights(self) -> np.ndarray:
        """Get the learned weights."""
        if not self.is_trained:
            raise RuntimeError("DMP must be trained before accessing weights")
        return self.weights.copy()
    
    def set_weights(self, weights: np.ndarray) -> None:
        """Set the weights."""
        if weights.shape != (self.config.n_dims, self.config.n_basis):
            raise ValueError(f"Weight shape mismatch: expected {(self.config.n_dims, self.config.n_basis)}, got {weights.shape}")
        self.weights = weights.copy()
        self.is_trained = True
    
    def _get_weight_shape(self) -> Tuple[int, ...]:
        """Get the shape of the weight vector."""
        return (self.config.n_dims, self.config.n_basis)
    
    def modify_goal(self, new_goal: np.ndarray) -> None:
        """
        Modify the goal position.
        
        Args:
            new_goal: New goal position
        """
        if new_goal.shape != (self.config.n_dims,):
            raise ValueError(f"Goal shape mismatch: expected {(self.config.n_dims,)}, got {new_goal.shape}")
        self.goal = np.array(new_goal)
    
    def modify_start(self, new_start: np.ndarray) -> None:
        """
        Modify the start position.
        
        Args:
            new_start: New start position
        """
        if new_start.shape != (self.config.n_dims,):
            raise ValueError(f"Start shape mismatch: expected {(self.config.n_dims,)}, got {new_start.shape}")
        self.start = np.array(new_start)
    
    def get_phase(self, execution_time: Optional[float] = None) -> np.ndarray:
        """
        Get phase values for given execution time.
        
        Args:
            execution_time: Execution time
            
        Returns:
            Phase values
        """
        if execution_time is None:
            execution_time = self.config.execution_time
        return self.phase_generator.generate(self.config.dt, execution_time)
    
    def get_basis_functions(self, phase: np.ndarray) -> np.ndarray:
        """
        Get basis function values for given phase.
        
        Args:
            phase: Phase values
            
        Returns:
            Basis function values
        """
        return self.basis_functions(phase)
