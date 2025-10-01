"""
Unit tests for Movement Primitives implementations.

This module contains comprehensive tests for all MP implementations.
"""

import pytest
import numpy as np
from movement_primitives import (
    DMP, DMPConfig, ProMP, ProMPConfig, ProDMP, ProDMPConfig, CNMP, CNMPConfig,
    PhaseGenerator, GaussianBasis, TrajectoryGenerator
)


class TestPhaseGenerator:
    """Test PhaseGenerator class."""
    
    def test_phase_generation(self):
        """Test phase generation."""
        generator = PhaseGenerator(alpha=25.0)
        phase = generator.generate(dt=0.01, execution_time=1.0)
        
        assert len(phase) == 100
        assert phase[0] == 1.0
        assert phase[-1] < 0.1
        assert np.all(phase[:-1] > phase[1:])  # Monotonically decreasing
    
    def test_phase_derivative(self):
        """Test phase derivative computation."""
        generator = PhaseGenerator(alpha=25.0)
        phase = generator.generate(dt=0.01, execution_time=1.0)
        phase_dot = generator.get_phase_derivative(phase, dt=0.01)
        
        assert len(phase_dot) == len(phase)
        assert np.all(phase_dot < 0)  # Negative derivative


class TestGaussianBasis:
    """Test GaussianBasis class."""
    
    def test_basis_evaluation(self):
        """Test basis function evaluation."""
        basis = GaussianBasis(n_basis=10, basis_width=0.1)
        phase = np.linspace(0, 1, 50)
        basis_values = basis(phase)
        
        assert basis_values.shape == (50, 10)
        assert np.all(basis_values >= 0)
        assert np.all(basis_values <= 1)
    
    def test_basis_centers(self):
        """Test basis function centers."""
        basis = GaussianBasis(n_basis=5, basis_width=0.1)
        expected_centers = np.linspace(0, 1, 5)
        np.testing.assert_array_almost_equal(basis.centers, expected_centers)


class TestTrajectoryGenerator:
    """Test TrajectoryGenerator class."""
    
    def test_resample_trajectory(self):
        """Test trajectory resampling."""
        trajectory = np.random.rand(100, 2)
        resampled = TrajectoryGenerator.resample_trajectory(trajectory, 50)
        
        assert resampled.shape == (50, 2)
    
    def test_normalize_trajectory(self):
        """Test trajectory normalization."""
        trajectory = np.random.rand(100, 2) * 10
        normalized, min_vals, max_vals = TrajectoryGenerator.normalize_trajectory(trajectory)
        
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
        assert len(min_vals) == 2
        assert len(max_vals) == 2
    
    def test_denormalize_trajectory(self):
        """Test trajectory denormalization."""
        trajectory = np.random.rand(100, 2)
        min_vals = np.array([0, 0])
        max_vals = np.array([10, 20])
        
        denormalized = TrajectoryGenerator.denormalize_trajectory(trajectory, min_vals, max_vals)
        
        assert denormalized.shape == trajectory.shape
        assert np.all(denormalized >= min_vals)
        assert np.all(denormalized <= max_vals)
    
    def test_velocity_computation(self):
        """Test velocity computation."""
        t = np.linspace(0, 1, 100)
        trajectory = np.column_stack([t, t**2])
        velocity = TrajectoryGenerator.compute_velocity(trajectory, dt=0.01)
        
        assert velocity.shape == trajectory.shape
        assert np.allclose(velocity[:, 0], 1.0)  # Constant velocity in x
        assert np.allclose(velocity[:, 1], 2*t)  # Linear velocity in y
    
    def test_acceleration_computation(self):
        """Test acceleration computation."""
        t = np.linspace(0, 1, 100)
        trajectory = np.column_stack([t, t**2])
        acceleration = TrajectoryGenerator.compute_acceleration(trajectory, dt=0.01)
        
        assert acceleration.shape == trajectory.shape
        assert np.allclose(acceleration[:, 0], 0.0)  # Zero acceleration in x
        assert np.allclose(acceleration[:, 1], 2.0)    # Constant acceleration in y


class TestDMP:
    """Test DMP implementation."""
    
    def test_dmp_initialization(self):
        """Test DMP initialization."""
        config = DMPConfig(n_dims=2, dt=0.01, execution_time=1.0)
        dmp = DMP(config)
        
        assert dmp.config == config
        assert not dmp.is_trained
        assert dmp.training_data is None
    
    def test_dmp_training(self):
        """Test DMP training."""
        config = DMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=20)
        dmp = DMP(config)
        
        # Create simple trajectory
        t = np.linspace(0, 1, 100)
        trajectory = np.column_stack([t, 0.5 * np.sin(2 * np.pi * t)])
        
        dmp.fit([trajectory])
        
        assert dmp.is_trained
        assert dmp.training_data is not None
        assert dmp.get_weights().shape == (2, 20)
    
    def test_dmp_generation(self):
        """Test DMP trajectory generation."""
        config = DMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=20)
        dmp = DMP(config)
        
        # Create simple trajectory
        t = np.linspace(0, 1, 100)
        trajectory = np.column_stack([t, 0.5 * np.sin(2 * np.pi * t)])
        
        dmp.fit([trajectory])
        
        # Generate trajectory
        start = np.array([0.0, 0.0])
        goal = np.array([1.0, 0.0])
        generated = dmp.generate(start=start, goal=goal)
        
        assert generated.shape[1] == 2
        assert len(generated) == 100
        assert np.allclose(generated[0], start, atol=1e-6)
        assert np.allclose(generated[-1], goal, atol=1e-2)
    
    def test_dmp_different_goals(self):
        """Test DMP with different goals."""
        config = DMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=20)
        dmp = DMP(config)
        
        # Create simple trajectory
        t = np.linspace(0, 1, 100)
        trajectory = np.column_stack([t, 0.5 * np.sin(2 * np.pi * t)])
        
        dmp.fit([trajectory])
        
        # Test different goals
        goals = [
            np.array([1.0, 0.0]),
            np.array([1.0, 0.5]),
            np.array([1.0, -0.5]),
        ]
        
        for goal in goals:
            generated = dmp.generate(start=np.array([0.0, 0.0]), goal=goal)
            assert np.allclose(generated[-1], goal, atol=1e-2)
    
    def test_dmp_different_times(self):
        """Test DMP with different execution times."""
        config = DMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=20)
        dmp = DMP(config)
        
        # Create simple trajectory
        t = np.linspace(0, 1, 100)
        trajectory = np.column_stack([t, 0.5 * np.sin(2 * np.pi * t)])
        
        dmp.fit([trajectory])
        
        # Test different execution times
        times = [0.5, 1.0, 1.5, 2.0]
        
        for exec_time in times:
            generated = dmp.generate(
                start=np.array([0.0, 0.0]),
                goal=np.array([1.0, 0.0]),
                execution_time=exec_time
            )
            expected_length = int(exec_time / config.dt)
            assert len(generated) == expected_length


class TestProMP:
    """Test ProMP implementation."""
    
    def test_promp_initialization(self):
        """Test ProMP initialization."""
        config = ProMPConfig(n_dims=2, dt=0.01, execution_time=1.0)
        promp = ProMP(config)
        
        assert promp.config == config
        assert not promp.is_trained
        assert promp.training_data is None
    
    def test_promp_training(self):
        """Test ProMP training."""
        config = ProMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=20)
        promp = ProMP(config)
        
        # Create multiple trajectories
        trajectories = []
        for i in range(3):
            t = np.linspace(0, 1, 100)
            trajectory = np.column_stack([t, 0.5 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, 100)])
            trajectories.append(trajectory)
        
        promp.fit(trajectories)
        
        assert promp.is_trained
        assert promp.training_data is not None
        weight_mean, weight_cov = promp.get_weight_distribution()
        assert weight_mean.shape == (2, 20)
        assert weight_cov.shape == (2, 20, 20)
    
    def test_promp_generation(self):
        """Test ProMP trajectory generation."""
        config = ProMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=20)
        promp = ProMP(config)
        
        # Create multiple trajectories
        trajectories = []
        for i in range(3):
            t = np.linspace(0, 1, 100)
            trajectory = np.column_stack([t, 0.5 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, 100)])
            trajectories.append(trajectory)
        
        promp.fit(trajectories)
        
        # Generate trajectory
        start = np.array([0.0, 0.0])
        goal = np.array([1.0, 0.0])
        generated = promp.generate(start=start, goal=goal)
        
        assert generated.shape[1] == 2
        assert len(generated) == 100
        assert np.allclose(generated[0], start, atol=1e-6)
        assert np.allclose(generated[-1], goal, atol=1e-2)
    
    def test_promp_sampling(self):
        """Test ProMP trajectory sampling."""
        config = ProMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=20)
        promp = ProMP(config)
        
        # Create multiple trajectories
        trajectories = []
        for i in range(3):
            t = np.linspace(0, 1, 100)
            trajectory = np.column_stack([t, 0.5 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, 100)])
            trajectories.append(trajectory)
        
        promp.fit(trajectories)
        
        # Sample trajectories
        samples = promp.sample_trajectories(5)
        
        assert len(samples) == 5
        for sample in samples:
            assert sample.shape[1] == 2
            assert len(sample) == 100
    
    def test_promp_variance(self):
        """Test ProMP trajectory variance computation."""
        config = ProMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=20)
        promp = ProMP(config)
        
        # Create multiple trajectories
        trajectories = []
        for i in range(3):
            t = np.linspace(0, 1, 100)
            trajectory = np.column_stack([t, 0.5 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, 100)])
            trajectories.append(trajectory)
        
        promp.fit(trajectories)
        
        # Compute variance
        variance = promp.get_trajectory_variance()
        
        assert variance.shape == (100, 2)
        assert np.all(variance >= 0)


class TestProDMP:
    """Test ProDMP implementation."""
    
    def test_prodmp_initialization(self):
        """Test ProDMP initialization."""
        config = ProDMPConfig(n_dims=2, dt=0.01, execution_time=1.0)
        prodmp = ProDMP(config)
        
        assert prodmp.config == config
        assert not prodmp.is_trained
        assert prodmp.training_data is None
    
    def test_prodmp_training(self):
        """Test ProDMP training."""
        config = ProDMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=20)
        prodmp = ProDMP(config)
        
        # Create multiple trajectories
        trajectories = []
        for i in range(3):
            t = np.linspace(0, 1, 100)
            trajectory = np.column_stack([t, 0.5 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, 100)])
            trajectories.append(trajectory)
        
        prodmp.fit(trajectories)
        
        assert prodmp.is_trained
        assert prodmp.training_data is not None
        weight_mean, weight_cov = prodmp.get_weight_distribution()
        assert weight_mean.shape == (2, 20)
        assert weight_cov.shape == (2, 20, 20)
    
    def test_prodmp_full_trajectory(self):
        """Test ProDMP full trajectory generation."""
        config = ProDMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=20)
        prodmp = ProDMP(config)
        
        # Create multiple trajectories
        trajectories = []
        for i in range(3):
            t = np.linspace(0, 1, 100)
            trajectory = np.column_stack([t, 0.5 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, 100)])
            trajectories.append(trajectory)
        
        prodmp.fit(trajectories)
        
        # Generate full trajectory
        full_traj = prodmp.get_full_trajectory()
        
        assert 'position' in full_traj
        assert 'velocity' in full_traj
        assert 'acceleration' in full_traj
        
        for key in full_traj:
            assert full_traj[key].shape[1] == 2
            assert len(full_traj[key]) == 100


class TestCNMP:
    """Test CNMP implementation."""
    
    def test_cnmp_initialization(self):
        """Test CNMP initialization."""
        config = CNMPConfig(n_dims=2, dt=0.01, execution_time=1.0, context_dim=2)
        cnmp = CNMP(config)
        
        assert cnmp.config == config
        assert not cnmp.is_trained
        assert cnmp.training_data is None
    
    def test_cnmp_training(self):
        """Test CNMP training."""
        config = CNMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=20, 
                           context_dim=2, n_epochs=10)  # Reduced epochs for testing
        cnmp = CNMP(config)
        
        # Create contextual trajectories
        trajectories = []
        contexts = []
        for i in range(3):
            t = np.linspace(0, 1, 100)
            amp = 0.3 + i * 0.2
            freq = 1.0 + i * 0.5
            trajectory = np.column_stack([t, amp * np.sin(2 * np.pi * freq * t)])
            trajectories.append(trajectory)
            contexts.append(np.array([amp, freq]))
        
        cnmp.fit(trajectories, contexts)
        
        assert cnmp.is_trained
        assert cnmp.training_data is not None
    
    def test_cnmp_generation(self):
        """Test CNMP trajectory generation."""
        config = CNMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=20, 
                           context_dim=2, n_epochs=10)  # Reduced epochs for testing
        cnmp = CNMP(config)
        
        # Create contextual trajectories
        trajectories = []
        contexts = []
        for i in range(3):
            t = np.linspace(0, 1, 100)
            amp = 0.3 + i * 0.2
            freq = 1.0 + i * 0.5
            trajectory = np.column_stack([t, amp * np.sin(2 * np.pi * freq * t)])
            trajectories.append(trajectory)
            contexts.append(np.array([amp, freq]))
        
        cnmp.fit(trajectories, contexts)
        
        # Generate trajectory
        start = np.array([0.0, 0.0])
        goal = np.array([1.0, 0.0])
        context = np.array([0.5, 1.5])
        generated = cnmp.generate(start=start, goal=goal, context=context)
        
        assert generated.shape[1] == 2
        assert len(generated) == 100
        assert np.allclose(generated[0], start, atol=1e-6)
        assert np.allclose(generated[-1], goal, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
