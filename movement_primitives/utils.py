"""
Utility classes and functions for Movement Primitives.

This module provides common utilities including basis functions,
phase generators, and trajectory handling functions.
"""

import numpy as np
from typing import Union, List, Optional, Tuple
import matplotlib.pyplot as plt
import h5py
import pickle


class PhaseGenerator:
    """
    Phase generator for movement primitives.
    
    Generates phase variables that monotonically decrease from 1 to 0
    during trajectory execution.
    """
    
    def __init__(self, alpha: float = 25.0):
        """
        Initialize phase generator.
        
        Args:
            alpha: Phase convergence rate
        """
        self.alpha = alpha
    
    def generate(self, dt: float, execution_time: float) -> np.ndarray:
        """
        Generate phase values for the given time parameters.
        
        Args:
            dt: Time step
            execution_time: Total execution time
            
        Returns:
            Phase values (shape: [T])
        """
        n_steps = int(execution_time / dt)
        phase = np.zeros(n_steps)
        phase[0] = 1.0
        
        for t in range(1, n_steps):
            phase[t] = phase[t-1] * np.exp(-self.alpha * dt)
        
        return phase
    
    def get_phase_derivative(self, phase: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute phase derivative.
        
        Args:
            phase: Phase values
            dt: Time step
            
        Returns:
            Phase derivative values
        """
        return -self.alpha * phase


class GaussianBasis:
    """
    Gaussian basis functions for movement primitives.
    """
    
    def __init__(self, n_basis: int, basis_width: float = 0.1):
        """
        Initialize Gaussian basis functions.
        
        Args:
            n_basis: Number of basis functions
            basis_width: Width of each basis function
        """
        self.n_basis = n_basis
        self.basis_width = basis_width
        self.centers = np.linspace(0, 1, n_basis)
    
    def __call__(self, phase: np.ndarray) -> np.ndarray:
        """
        Evaluate basis functions at given phase values.
        
        Args:
            phase: Phase values (shape: [T])
            
        Returns:
            Basis function values (shape: [T, n_basis])
        """
        phase = np.atleast_1d(phase)
        if phase.ndim == 0:
            phase = phase.reshape(1)
        
        basis_values = np.zeros((len(phase), self.n_basis))
        
        for i, center in enumerate(self.centers):
            basis_values[:, i] = np.exp(-0.5 * ((phase - center) / self.basis_width) ** 2)
        
        return basis_values


class RadialBasisFunction:
    """
    Radial basis function implementation.
    """
    
    def __init__(self, centers: np.ndarray, widths: Union[float, np.ndarray]):
        """
        Initialize RBF.
        
        Args:
            centers: Center points (shape: [n_basis, n_dims])
            widths: Width parameters (scalar or array)
        """
        self.centers = np.array(centers)
        self.widths = np.array(widths) if np.isscalar(widths) else widths
        
        if self.widths.ndim == 0:
            self.widths = np.full(self.centers.shape[0], self.widths)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate RBF at given points.
        
        Args:
            x: Input points (shape: [T, n_dims])
            
        Returns:
            RBF values (shape: [T, n_basis])
        """
        x = np.atleast_2d(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n_points, n_dims = x.shape
        n_basis = len(self.centers)
        
        rbf_values = np.zeros((n_points, n_basis))
        
        for i in range(n_basis):
            diff = x - self.centers[i]
            dist_sq = np.sum(diff ** 2, axis=1)
            rbf_values[:, i] = np.exp(-dist_sq / (2 * self.widths[i] ** 2))
        
        return rbf_values


class TrajectoryGenerator:
    """
    Utility class for generating and manipulating trajectories.
    """
    
    @staticmethod
    def resample_trajectory(trajectory: np.ndarray, 
                           new_length: int) -> np.ndarray:
        """
        Resample trajectory to new length using linear interpolation.
        
        Args:
            trajectory: Input trajectory (shape: [T, n_dims])
            new_length: Desired new length
            
        Returns:
            Resampled trajectory (shape: [new_length, n_dims])
        """
        old_length = len(trajectory)
        old_indices = np.linspace(0, old_length - 1, old_length)
        new_indices = np.linspace(0, old_length - 1, new_length)
        
        resampled = np.zeros((new_length, trajectory.shape[1]))
        for dim in range(trajectory.shape[1]):
            resampled[:, dim] = np.interp(new_indices, old_indices, trajectory[:, dim])
        
        return resampled
    
    @staticmethod
    def normalize_trajectory(trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize trajectory to [0, 1] range.
        
        Args:
            trajectory: Input trajectory (shape: [T, n_dims])
            
        Returns:
            Tuple of (normalized_trajectory, min_vals, max_vals)
        """
        min_vals = np.min(trajectory, axis=0)
        max_vals = np.max(trajectory, axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        normalized = (trajectory - min_vals) / range_vals
        return normalized, min_vals, max_vals
    
    @staticmethod
    def denormalize_trajectory(normalized_trajectory: np.ndarray,
                              min_vals: np.ndarray,
                              max_vals: np.ndarray) -> np.ndarray:
        """
        Denormalize trajectory from [0, 1] range.
        
        Args:
            normalized_trajectory: Normalized trajectory (shape: [T, n_dims])
            min_vals: Minimum values for each dimension
            max_vals: Maximum values for each dimension
            
        Returns:
            Denormalized trajectory
        """
        range_vals = max_vals - min_vals
        return normalized_trajectory * range_vals + min_vals
    
    @staticmethod
    def compute_velocity(trajectory: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute velocity from position trajectory.
        
        Args:
            trajectory: Position trajectory (shape: [T, n_dims])
            dt: Time step
            
        Returns:
            Velocity trajectory (shape: [T, n_dims])
        """
        velocity = np.gradient(trajectory, dt, axis=0)
        return velocity
    
    @staticmethod
    def compute_acceleration(trajectory: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute acceleration from position trajectory.
        
        Args:
            trajectory: Position trajectory (shape: [T, n_dims])
            dt: Time step
            
        Returns:
            Acceleration trajectory (shape: [T, n_dims])
        """
        velocity = TrajectoryGenerator.compute_velocity(trajectory, dt)
        acceleration = np.gradient(velocity, dt, axis=0)
        return acceleration


def plot_trajectory(trajectory: np.ndarray, 
                   title: str = "Trajectory",
                   labels: Optional[List[str]] = None,
                   figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot a trajectory.
    
    Args:
        trajectory: Trajectory to plot (shape: [T, n_dims])
        title: Plot title
        labels: Labels for each dimension
        figsize: Figure size
    """
    n_dims = trajectory.shape[1]
    
    if n_dims == 1:
        plt.figure(figsize=figsize)
        plt.plot(trajectory[:, 0])
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("Position")
        plt.grid(True)
        plt.show()
    
    elif n_dims == 2:
        plt.figure(figsize=figsize)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8, label='End')
        plt.title(title)
        plt.xlabel(labels[0] if labels else "X")
        plt.ylabel(labels[1] if labels else "Y")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    
    else:
        fig, axes = plt.subplots(n_dims, 1, figsize=(figsize[0], figsize[1] * n_dims))
        if n_dims == 1:
            axes = [axes]
        
        for i in range(n_dims):
            axes[i].plot(trajectory[:, i])
            axes[i].set_title(f"{labels[i] if labels else f'Dimension {i+1}'}")
            axes[i].set_xlabel("Time Step")
            axes[i].set_ylabel("Position")
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()


def save_trajectory(trajectory: np.ndarray, filepath: str) -> None:
    """
    Save trajectory to file.
    
    Args:
        trajectory: Trajectory to save
        filepath: Path to save file
    """
    if filepath.endswith('.h5') or filepath.endswith('.hdf5'):
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('trajectory', data=trajectory)
    else:
        np.save(filepath, trajectory)


def load_trajectory(filepath: str) -> np.ndarray:
    """
    Load trajectory from file.
    
    Args:
        filepath: Path to load file from
        
    Returns:
        Loaded trajectory
    """
    if filepath.endswith('.h5') or filepath.endswith('.hdf5'):
        with h5py.File(filepath, 'r') as f:
            return f['trajectory'][:]
    else:
        return np.load(filepath)
