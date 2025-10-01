"""
Conditional Neural Movement Primitives (CNMP) implementation.

CNMPs use neural networks to learn complex movement patterns conditioned
on contextual information, enabling more flexible and expressive movement generation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Union, List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from .base import BaseMovementPrimitive, MovementPrimitiveConfig
from .utils import PhaseGenerator, GaussianBasis, TrajectoryGenerator


@dataclass
class CNMPConfig(MovementPrimitiveConfig):
    """Configuration for Conditional Neural Movement Primitives."""
    
    # CNMP-specific parameters
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
    
    # Neural network parameters
    context_dim: int = 2  # Dimension of context vector
    hidden_dim: int = 128  # Hidden layer dimension
    n_hidden_layers: int = 2  # Number of hidden layers
    activation: str = 'relu'  # Activation function
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    n_epochs: int = 1000
    validation_split: float = 0.2
    
    # Device
    device: str = 'cpu'


class CNMPNetwork(nn.Module):
    """
    Neural network for CNMP.
    
    The network takes phase and context as input and outputs
    the forcing term for the movement primitive.
    """
    
    def __init__(self, phase_dim: int, context_dim: int, 
                 hidden_dim: int, n_hidden_layers: int,
                 activation: str = 'relu'):
        """
        Initialize CNMP network.
        
        Args:
            phase_dim: Dimension of phase input
            context_dim: Dimension of context input
            hidden_dim: Hidden layer dimension
            n_hidden_layers: Number of hidden layers
            activation: Activation function name
        """
        super().__init__()
        
        self.phase_dim = phase_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        
        # Input layer
        input_dim = phase_dim + context_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) 
            for _ in range(n_hidden_layers - 1)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, phase: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            phase: Phase input (shape: [batch_size, phase_dim])
            context: Context input (shape: [batch_size, context_dim])
            
        Returns:
            Forcing term output (shape: [batch_size, 1])
        """
        # Concatenate phase and context
        x = torch.cat([phase, context], dim=1)
        
        # Input layer
        x = self.activation(self.input_layer(x))
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        # Output layer
        output = self.output_layer(x)
        
        return output


class CNMP(BaseMovementPrimitive):
    """
    Conditional Neural Movement Primitives implementation.
    
    CNMPs use neural networks to learn complex movement patterns
    conditioned on contextual information.
    """
    
    def __init__(self, config: CNMPConfig):
        """
        Initialize CNMP.
        
        Args:
            config: CNMP configuration
        """
        super().__init__(config)
        self.config: CNMPConfig = config
        
        # Initialize components
        self.phase_generator = PhaseGenerator(config.canonical_alpha)
        self.basis_functions = GaussianBasis(config.n_basis, config.basis_width)
        
        # Neural network
        self.network = CNMPNetwork(
            phase_dim=config.n_basis,
            context_dim=config.context_dim,
            hidden_dim=config.hidden_dim,
            n_hidden_layers=config.n_hidden_layers,
            activation=config.activation
        ).to(config.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # CNMP parameters
        self.start = None
        self.goal = None
        self.context = None
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def fit(self, trajectories: Union[np.ndarray, List[np.ndarray]], 
            contexts: Union[np.ndarray, List[np.ndarray]],
            **kwargs) -> 'CNMP':
        """
        Train CNMP on demonstration trajectories and contexts.
        
        Args:
            trajectories: Demonstration trajectories
            contexts: Context vectors for each trajectory
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        # Handle single trajectory
        if isinstance(trajectories, np.ndarray):
            trajectories = [trajectories]
        if isinstance(contexts, np.ndarray):
            contexts = [contexts]
        
        # Validate inputs
        if len(trajectories) != len(contexts):
            raise ValueError("Number of trajectories and contexts must match")
        
        for traj in trajectories:
            if traj.shape[1] != self.config.n_dims:
                raise ValueError(f"Trajectory dimension mismatch: expected {self.config.n_dims}, got {traj.shape[1]}")
        
        for ctx in contexts:
            if ctx.shape[0] != self.config.context_dim:
                raise ValueError(f"Context dimension mismatch: expected {self.config.context_dim}, got {ctx.shape[0]}")
        
        # Store training data
        self.training_data = (trajectories, contexts)
        
        # Prepare training data
        X_phase, X_context, y = self._prepare_training_data(trajectories, contexts)
        
        # Split into train and validation
        n_samples = len(X_phase)
        n_train = int(n_samples * (1 - self.config.validation_split))
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_phase_train = X_phase[train_indices]
        X_context_train = X_context[train_indices]
        y_train = y[train_indices]
        
        X_phase_val = X_phase[val_indices]
        X_context_val = X_context[val_indices]
        y_val = y[val_indices]
        
        # Convert to tensors
        X_phase_train = torch.FloatTensor(X_phase_train).to(self.config.device)
        X_context_train = torch.FloatTensor(X_context_train).to(self.config.device)
        y_train = torch.FloatTensor(y_train).to(self.config.device)
        
        X_phase_val = torch.FloatTensor(X_phase_val).to(self.config.device)
        X_context_val = torch.FloatTensor(X_context_val).to(self.config.device)
        y_val = torch.FloatTensor(y_val).to(self.config.device)
        
        # Training loop
        for epoch in range(self.config.n_epochs):
            # Training
            self.network.train()
            train_loss = 0.0
            
            # Mini-batch training
            n_batches = len(X_phase_train) // self.config.batch_size
            for i in range(n_batches):
                start_idx = i * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                
                batch_phase = X_phase_train[start_idx:end_idx]
                batch_context = X_context_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.network(batch_phase, batch_context)
                loss = self.criterion(output, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.network.eval()
            with torch.no_grad():
                val_output = self.network(X_phase_val, X_context_val)
                val_loss = self.criterion(val_output, y_val).item()
            
            # Store history
            self.training_history['train_loss'].append(train_loss / n_batches)
            self.training_history['val_loss'].append(val_loss)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss / n_batches:.6f}, Val Loss = {val_loss:.6f}")
        
        self.is_trained = True
        return self
    
    def _prepare_training_data(self, trajectories: List[np.ndarray], 
                              contexts: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data for neural network.
        
        Args:
            trajectories: List of demonstration trajectories
            contexts: List of context vectors
            
        Returns:
            Tuple of (phase_features, context_features, forcing_terms)
        """
        phases = []
        forcing_terms = []
        context_features = []
        
        for traj, ctx in zip(trajectories, contexts):
            # Generate phase
            phase = self.phase_generator.generate(self.config.dt, self.config.execution_time)
            
            # Compute forcing term for each dimension
            for dim in range(self.config.n_dims):
                forcing_term = self._compute_forcing_term(traj[:, dim], phase)
                
                # Evaluate basis functions
                basis_values = self.basis_functions(phase)
                
                phases.append(basis_values)
                forcing_terms.append(forcing_term)
                context_features.append(ctx)
        
        # Concatenate all data
        X_phase = np.concatenate(phases, axis=0)
        X_context = np.array(context_features)
        y = np.concatenate(forcing_terms, axis=0)
        
        return X_phase, X_context, y
    
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
                 context: Optional[np.ndarray] = None,
                 execution_time: Optional[float] = None,
                 **kwargs) -> np.ndarray:
        """
        Generate trajectory using trained CNMP.
        
        Args:
            start: Starting position
            goal: Goal position
            context: Context vector
            execution_time: Execution time
            **kwargs: Additional parameters
            
        Returns:
            Generated trajectory
        """
        if not self.is_trained:
            raise RuntimeError("CNMP must be trained before generating trajectories")
        
        # Set default values
        if start is None:
            start = np.zeros(self.config.n_dims)
        if goal is None:
            goal = np.ones(self.config.n_dims)
        if context is None:
            context = np.zeros(self.config.context_dim)
        if execution_time is None:
            execution_time = self.config.execution_time
        
        # Store parameters
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.context = np.array(context)
        
        # Generate phase
        phase = self.phase_generator.generate(self.config.dt, execution_time)
        
        # Generate trajectory for each dimension
        trajectory = np.zeros((len(phase), self.config.n_dims))
        
        for dim in range(self.config.n_dims):
            trajectory[:, dim] = self._generate_dimension(dim, phase, start[dim], goal[dim], context)
        
        return trajectory
    
    def _generate_dimension(self, dim: int, phase: np.ndarray, 
                           start: float, goal: float, context: np.ndarray) -> np.ndarray:
        """
        Generate trajectory for a specific dimension.
        
        Args:
            dim: Dimension index
            phase: Phase values
            start: Starting position
            goal: Goal position
            context: Context vector
            
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
        
        # Convert to tensors
        basis_tensor = torch.FloatTensor(basis_values).to(self.config.device)
        context_tensor = torch.FloatTensor(context).unsqueeze(0).repeat(n_steps, 1).to(self.config.device)
        
        # Generate trajectory step by step
        self.network.eval()
        with torch.no_grad():
            # Get forcing terms from network
            forcing_terms = self.network(basis_tensor, context_tensor).cpu().numpy().flatten()
        
        for t in range(1, n_steps):
            dt = self.config.dt
            
            # Compute forcing term
            forcing_term = forcing_terms[t]
            
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
        """Get the network weights (not applicable for CNMP)."""
        raise NotImplementedError("CNMP uses neural networks, weights are not directly accessible")
    
    def set_weights(self, weights: np.ndarray) -> None:
        """Set the network weights (not applicable for CNMP)."""
        raise NotImplementedError("CNMP uses neural networks, weights cannot be set directly")
    
    def _get_weight_shape(self) -> Tuple[int, ...]:
        """Get the shape of the weight vector (not applicable for CNMP)."""
        raise NotImplementedError("CNMP uses neural networks, weight shape is not applicable")
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("CNMP must be trained before saving")
        
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.training_history = checkpoint['training_history']
        
        self.is_trained = True
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.
        
        Returns:
            Dictionary containing training and validation losses
        """
        return self.training_history.copy()
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['train_loss'], label='Training Loss')
        plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('CNMP Training History')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def modify_context(self, new_context: np.ndarray) -> None:
        """
        Modify the context vector.
        
        Args:
            new_context: New context vector
        """
        if new_context.shape != (self.config.context_dim,):
            raise ValueError(f"Context shape mismatch: expected {(self.config.context_dim,)}, got {new_context.shape}")
        self.context = np.array(new_context)
    
    def get_context(self) -> Optional[np.ndarray]:
        """
        Get the current context vector.
        
        Returns:
            Current context vector or None
        """
        return self.context.copy() if self.context is not None else None
