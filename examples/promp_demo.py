#!/usr/bin/env python3
"""
Example script demonstrating Probabilistic Movement Primitives (ProMP).

This script shows how to:
1. Create demonstration trajectories with variations
2. Train a ProMP on the demonstrations
3. Generate trajectories with uncertainty quantification
4. Sample multiple trajectories from the distribution
5. Condition on waypoints
"""

import numpy as np
import matplotlib.pyplot as plt
from movement_primitives import ProMP, ProMPConfig, plot_trajectory


def create_demonstration_trajectories():
    """Create multiple demonstration trajectories with variations."""
    trajectories = []
    
    for i in range(10):
        t = np.linspace(0, 1, 100)
        
        # Base trajectory
        x = t
        y_base = 0.5 * np.sin(2 * np.pi * t) + 0.1 * np.sin(8 * np.pi * t)
        
        # Add variations
        variation = np.random.normal(0, 0.1, len(t))
        y = y_base + variation
        
        trajectory = np.column_stack([x, y])
        trajectories.append(trajectory)
    
    return trajectories


def main():
    """Main demonstration function."""
    print("Probabilistic Movement Primitives (ProMP) Demo")
    print("=" * 50)
    
    # Create demonstration trajectories
    print("Creating demonstration trajectories with variations...")
    trajectories = create_demonstration_trajectories()
    
    # Plot original demonstrations
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    for i, traj in enumerate(trajectories):
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.6, linewidth=1)
    plt.title('Demonstration Trajectories')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    
    # Configure and train ProMP
    print("Training ProMP...")
    config = ProMPConfig(
        n_dims=2,
        dt=0.01,
        execution_time=1.0,
        n_basis=50,
        basis_width=0.1,
        noise_variance=1e-4,
        prior_variance=1.0
    )
    
    promp = ProMP(config)
    promp.fit(trajectories)
    
    print(f"ProMP trained successfully!")
    weight_mean, weight_cov = promp.get_weight_distribution()
    print(f"Weight mean shape: {weight_mean.shape}")
    print(f"Weight covariance shape: {weight_cov.shape}")
    
    # Generate mean trajectory
    print("Generating mean trajectory...")
    traj_mean = promp.generate(start=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0]))
    
    # Sample multiple trajectories
    print("Sampling multiple trajectories...")
    n_samples = 20
    sampled_trajectories = promp.sample_trajectories(
        n_samples=n_samples,
        start=np.array([0.0, 0.0]),
        goal=np.array([1.0, 0.0])
    )
    
    plt.subplot(2, 3, 2)
    for traj in sampled_trajectories:
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.3, color='blue', linewidth=0.5)
    plt.plot(traj_mean[:, 0], traj_mean[:, 1], 'r-', linewidth=3, label='Mean')
    plt.title('Sampled Trajectories')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Show trajectory variance
    print("Computing trajectory variance...")
    trajectory_variance = promp.get_trajectory_variance()
    
    plt.subplot(2, 3, 3)
    plt.plot(trajectory_variance[:, 0], label='X variance')
    plt.plot(trajectory_variance[:, 1], label='Y variance')
    plt.title('Trajectory Variance')
    plt.xlabel('Time Step')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)
    
    # Show uncertainty bands
    plt.subplot(2, 3, 4)
    std_x = np.sqrt(trajectory_variance[:, 0])
    std_y = np.sqrt(trajectory_variance[:, 1])
    
    plt.plot(traj_mean[:, 0], traj_mean[:, 1], 'r-', linewidth=2, label='Mean')
    plt.fill_between(traj_mean[:, 0], 
                     traj_mean[:, 1] - 2*std_y, 
                     traj_mean[:, 1] + 2*std_y, 
                     alpha=0.3, color='red', label='±2σ')
    plt.title('Trajectory with Uncertainty Bands')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Test waypoint conditioning
    print("Testing waypoint conditioning...")
    waypoint = np.array([0.5, 0.3])
    waypoint_time = 0.5
    
    conditioned_promp = promp.condition_on_waypoint(waypoint, waypoint_time)
    
    # Generate trajectories with and without conditioning
    traj_unconditioned = promp.generate(start=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0]))
    traj_conditioned = conditioned_promp.generate(start=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0]))
    
    plt.subplot(2, 3, 5)
    plt.plot(traj_unconditioned[:, 0], traj_unconditioned[:, 1], 'b-', linewidth=2, label='Unconditioned')
    plt.plot(traj_conditioned[:, 0], traj_conditioned[:, 1], 'r-', linewidth=2, label='Conditioned')
    plt.plot(waypoint[0], waypoint[1], 'go', markersize=10, label='Waypoint')
    plt.title('Waypoint Conditioning')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Compare weight distributions
    plt.subplot(2, 3, 6)
    original_mean, original_cov = promp.get_weight_distribution()
    conditioned_mean, conditioned_cov = conditioned_promp.get_weight_distribution()
    
    # Plot weight means for first dimension
    plt.plot(original_mean[0], 'b-', linewidth=2, label='Original')
    plt.plot(conditioned_mean[0], 'r-', linewidth=2, label='Conditioned')
    plt.title('Weight Distribution Comparison')
    plt.xlabel('Basis Function Index')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    print("\nAdditional Analysis:")
    print(f"Number of demonstrations: {len(trajectories)}")
    print(f"Number of basis functions: {config.n_basis}")
    print(f"Noise variance: {config.noise_variance}")
    print(f"Prior variance: {config.prior_variance}")
    
    # Test different goals
    print("\nTesting different goals...")
    goals = [
        np.array([1.0, 0.5]),
        np.array([1.0, -0.5]),
        np.array([1.2, 0.0]),
    ]
    
    plt.figure(figsize=(12, 4))
    for i, goal in enumerate(goals):
        plt.subplot(1, 3, i+1)
        traj = promp.generate(start=np.array([0.0, 0.0]), goal=goal)
        plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2)
        plt.plot(goal[0], goal[1], 'ro', markersize=8, label='Goal')
        plt.title(f'Goal: ({goal[0]:.1f}, {goal[1]:.1f})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    print("\nProMP Demo completed successfully!")


if __name__ == "__main__":
    main()
