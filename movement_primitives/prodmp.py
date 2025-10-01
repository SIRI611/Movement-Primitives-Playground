"""
Probabilistic Dynamic Movement Primitives (ProDMP) implementation.

ProDMPs combine the benefits of DMPs and ProMPs by incorporating
probabilistic modeling into the DMP framework.
"""

import numpy as np
from typing import Union, List, Optional, Tuple, Dict
from dataclasses import dataclass
from scipy.stats import multivariate_normal
from .base import BaseMovementPrimitive, MovementPrimitiveConfig
from .utils import PhaseGenerator, GaussianBasis, TrajectoryGenerator


@dataclass
class ProDMPConfig(MovementPrimitiveConfig):
    """Configuration for Probabilistic Dynamic Movement Primitives."""
    
    # ProDMP-specific parameters
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
    
    # Probabilistic parameters
    noise_variance: float = 1e-6  # Observation noise variance
    prior_variance: float = 1.0   # Prior variance for weights
    
    # ProDMP-specific parameters
    use_velocity: bool = True     # Whether to use velocity information
    use_acceleration: bool = True  # Whether to use acceleration information


class ProDMP(BaseMovementPrimitive):
    """
    Probabilistic Dynamic Movement Primitives implementation.
    
    ProDMPs extend DMPs with probabilistic modeling capabilities,
    allowing for uncertainty quantification and probabilistic inference.
    """
    
    def __init__(self, config: ProDMPConfig):
        """
        Initialize ProDMP.
        
        Args:
            config: ProDMP configuration
        """
        super().__init__(config)
        self.config: ProDMPConfig = config
        
        # Initialize components
        self.phase_generator = PhaseGenerator(config.canonical_alpha)
        self.basis_functions = GaussianBasis(config.n_basis, config.basis_width)
        
        # ProDMP parameters
        self.weight_mean = None  # Mean of weight distribution
        self.weight_cov = None   # Covariance of weight distribution
        self.start = None
        self.goal = None
        
        # Weight shape for each dimension
        self._weight_shape_per_dim = (config.n_basis,)
    
    def fit(self, trajectories: Union[np.ndarray, List[np.ndarray]], 
            **kwargs) -> 'ProDMP':
        """
        Train ProDMP on demonstration trajectories.
        
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
        
        # Learn probabilistic weights for each dimension
        self.weight_mean = np.zeros((self.config.n_dims, self.config.n_basis))
        self.weight_cov = np.zeros((self.config.n_dims, self.config.n_basis, self.config.n_basis))
        
        for dim in range(self.config.n_dims):
            mean, cov = self._learn_probabilistic_weights_dimension(trajectories, dim)
            self.weight_mean[dim] = mean
            self.weight_cov[dim] = cov
        
        self.is_trained = True
        return self
    
    def _learn_probabilistic_weights_dimension(self, trajectories: List[np.ndarray], 
                                             dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Learn probabilistic weights for a specific dimension.
        
        Args:
            trajectories: List of demonstration trajectories
            dim: Dimension index
            
        Returns:
            Tuple of (mean_weights, covariance_matrix)
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
        
        # Bayesian linear regression
        # Prior: p(w) = N(0, alpha^(-1) * I)
        # Likelihood: p(y|w) = N(Phi * w, beta^(-1) * I)
        # Posterior: p(w|y) = N(mu, S)
        
        # Prior precision
        alpha = 1.0 / self.config.prior_variance
        
        # Likelihood precision
        beta = 1.0 / self.config.noise_variance
        
        # Posterior precision
        S_inv = alpha * np.eye(self.config.n_basis) + beta * basis_values.T @ basis_values
        
        # Posterior covariance
        S = np.linalg.inv(S_inv)
        
        # Posterior mean
        mu = beta * S @ basis_values.T @ all_forcing_terms
        
        return mu, S
    
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
                 sample: bool = False,
                 **kwargs) -> np.ndarray:
        """
        Generate trajectory using trained ProDMP.
        
        Args:
            start: Starting position
            goal: Goal position
            execution_time: Execution time
            sample: Whether to sample from the distribution
            **kwargs: Additional parameters
            
        Returns:
            Generated trajectory
        """
        if not self.is_trained:
            raise RuntimeError("ProDMP must be trained before generating trajectories")
        
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
            if sample:
                # Sample weights from the distribution
                weights = np.random.multivariate_normal(
                    self.weight_mean[dim], 
                    self.weight_cov[dim]
                )
            else:
                # Use mean weights
                weights = self.weight_mean[dim]
            
            trajectory[:, dim] = self._generate_dimension(dim, phase, start[dim], goal[dim], weights)
        
        return trajectory
    
    def _generate_dimension(self, dim: int, phase: np.ndarray, 
                           start: float, goal: float, weights: np.ndarray) -> np.ndarray:
        """
        Generate trajectory for a specific dimension.
        
        Args:
            dim: Dimension index
            phase: Phase values
            start: Starting position
            goal: Goal position
            weights: Weight vector
            
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
            forcing_term = np.dot(basis_values[t], weights)
            
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
        """Get the mean weights."""
        if not self.is_trained:
            raise RuntimeError("ProDMP must be trained before accessing weights")
        return self.weight_mean.copy()
    
    def set_weights(self, weights: np.ndarray) -> None:
        """Set the mean weights."""
        if weights.shape != (self.config.n_dims, self.config.n_basis):
            raise ValueError(f"Weight shape mismatch: expected {(self.config.n_dims, self.config.n_basis)}, got {weights.shape}")
        self.weight_mean = weights.copy()
        self.is_trained = True
    
    def _get_weight_shape(self) -> Tuple[int, ...]:
        """Get the shape of the weight vector."""
        return (self.config.n_dims, self.config.n_basis)
    
    def get_weight_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the weight distribution parameters.
        
        Returns:
            Tuple of (mean, covariance)
        """
        if not self.is_trained:
            raise RuntimeError("ProDMP must be trained before accessing weight distribution")
        return self.weight_mean.copy(), self.weight_cov.copy()
    
    def set_weight_distribution(self, mean: np.ndarray, cov: np.ndarray) -> None:
        """
        Set the weight distribution parameters.
        
        Args:
            mean: Mean weights
            cov: Covariance matrix
        """
        if mean.shape != (self.config.n_dims, self.config.n_basis):
            raise ValueError(f"Mean shape mismatch: expected {(self.config.n_dims, self.config.n_basis)}, got {mean.shape}")
        if cov.shape != (self.config.n_dims, self.config.n_basis, self.config.n_basis):
            raise ValueError(f"Covariance shape mismatch: expected {(self.config.n_dims, self.config.n_basis, self.config.n_basis)}, got {cov.shape}")
        
        self.weight_mean = mean.copy()
        self.weight_cov = cov.copy()
        self.is_trained = True
    
    def condition_on_waypoint(self, waypoint: np.ndarray, 
                             waypoint_time: float,
                             execution_time: Optional[float] = None) -> 'ProDMP':
        """
        Condition the ProDMP on a waypoint.
        
        Args:
            waypoint: Waypoint position
            waypoint_time: Time at which waypoint should be reached
            execution_time: Total execution time
            
        Returns:
            Conditioned ProDMP
        """
        if not self.is_trained:
            raise RuntimeError("ProDMP must be trained before conditioning")
        
        if execution_time is None:
            execution_time = self.config.execution_time
        
        # Generate phase at waypoint time
        phase = self.phase_generator.generate(self.config.dt, execution_time)
        waypoint_idx = int(waypoint_time / self.config.dt)
        
        if waypoint_idx >= len(phase):
            raise ValueError("Waypoint time exceeds execution time")
        
        # Create conditioned ProDMP
        conditioned_prodmp = ProDMP(self.config)
        conditioned_prodmp.weight_mean = self.weight_mean.copy()
        conditioned_prodmp.weight_cov = self.weight_cov.copy()
        conditioned_prodmp.is_trained = True
        
        # Condition on waypoint for each dimension
        for dim in range(self.config.n_dims):
            conditioned_prodmp.weight_mean[dim], conditioned_prodmp.weight_cov[dim] = \
                self._condition_dimension_on_waypoint(dim, waypoint[dim], waypoint_idx, phase)
        
        return conditioned_prodmp
    
    def _condition_dimension_on_waypoint(self, dim: int, waypoint: float, 
                                       waypoint_idx: int, phase: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Condition a dimension on a waypoint.
        
        Args:
            dim: Dimension index
            waypoint: Waypoint position
            waypoint_idx: Waypoint time index
            phase: Phase values
            
        Returns:
            Tuple of (conditioned_mean, conditioned_covariance)
        """
        # Get basis functions at waypoint
        basis_at_waypoint = self.basis_functions(phase[waypoint_idx:waypoint_idx+1])[0]
        
        # Compute observation matrix
        H = basis_at_waypoint.reshape(1, -1)
        
        # Observation noise
        R = self.config.noise_variance
        
        # Current distribution
        mu = self.weight_mean[dim]
        S = self.weight_cov[dim]
        
        # Kalman filter update
        S_inv = np.linalg.inv(S)
        K = S @ H.T @ np.linalg.inv(H @ S @ H.T + R)
        
        # Update mean and covariance
        mu_new = mu + K @ (waypoint - H @ mu)
        S_new = (np.eye(self.config.n_basis) - K @ H) @ S
        
        return mu_new, S_new
    
    def get_trajectory_variance(self, execution_time: Optional[float] = None) -> np.ndarray:
        """
        Get the variance of the trajectory at each time step.
        
        Args:
            execution_time: Execution time
            
        Returns:
            Trajectory variance (shape: [T, n_dims])
        """
        if not self.is_trained:
            raise RuntimeError("ProDMP must be trained before computing trajectory variance")
        
        if execution_time is None:
            execution_time = self.config.execution_time
        
        # Generate phase
        phase = self.phase_generator.generate(self.config.dt, execution_time)
        
        # Evaluate basis functions
        basis_values = self.basis_functions(phase)
        
        # Compute variance for each dimension
        trajectory_variance = np.zeros((len(phase), self.config.n_dims))
        
        for dim in range(self.config.n_dims):
            for t in range(len(phase)):
                phi_t = basis_values[t]
                trajectory_variance[t, dim] = phi_t.T @ self.weight_cov[dim] @ phi_t
        
        return trajectory_variance
    
    def sample_trajectories(self, n_samples: int, 
                           start: Optional[np.ndarray] = None,
                           goal: Optional[np.ndarray] = None,
                           execution_time: Optional[float] = None) -> List[np.ndarray]:
        """
        Sample multiple trajectories from the ProDMP.
        
        Args:
            n_samples: Number of trajectories to sample
            start: Starting position
            goal: Goal position
            execution_time: Execution time
            
        Returns:
            List of sampled trajectories
        """
        trajectories = []
        for _ in range(n_samples):
            traj = self.generate(start=start, goal=goal, execution_time=execution_time, sample=True)
            trajectories.append(traj)
        return trajectories
    
    def get_velocity_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Get velocity trajectory from position trajectory.
        
        Args:
            trajectory: Position trajectory
            
        Returns:
            Velocity trajectory
        """
        return TrajectoryGenerator.compute_velocity(trajectory, self.config.dt)
    
    def get_acceleration_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Get acceleration trajectory from position trajectory.
        
        Args:
            trajectory: Position trajectory
            
        Returns:
            Acceleration trajectory
        """
        return TrajectoryGenerator.compute_acceleration(trajectory, self.config.dt)
    
    def get_full_trajectory(self, start: Optional[np.ndarray] = None,
                           goal: Optional[np.ndarray] = None,
                           execution_time: Optional[float] = None,
                           sample: bool = False) -> Dict[str, np.ndarray]:
        """
        Get full trajectory including position, velocity, and acceleration.
        
        Args:
            start: Starting position
            goal: Goal position
            execution_time: Execution time
            sample: Whether to sample from the distribution
            
        Returns:
            Dictionary containing position, velocity, and acceleration trajectories
        """
        position = self.generate(start=start, goal=goal, execution_time=execution_time, sample=sample)
        velocity = self.get_velocity_trajectory(position)
        acceleration = self.get_acceleration_trajectory(position)
        
        return {
            'position': position,
            'velocity': velocity,
            'acceleration': acceleration
        }
