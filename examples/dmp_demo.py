#!/usr/bin/env python3
"""
Example script demonstrating Dynamic Movement Primitives (DMP).

This script shows how to:
1. Create demonstration trajectories
2. Train a DMP on the demonstrations
3. Generate new trajectories with different goals and execution times
4. Visualize the results
"""

import numpy as np
import matplotlib.pyplot as plt
from movement_primitives import DMP, DMPConfig, plot_trajectory


def create_demonstration_trajectory():
    """Create a simple demonstration trajectory."""
    t = np.linspace(0, 1, 100)
    
    # Create a smooth trajectory with some curvature
    x = t
    y = 0.5 * np.sin(2 * np.pi * t) + 0.1 * np.sin(8 * np.pi * t)
    
    trajectory = np.column_stack([x, y])
    return trajectory


def create_multiple_demonstrations():
    """Create multiple demonstration trajectories with variations."""
    trajectories = []
    
    for i in range(5):
        t = np.linspace(0, 1, 100)
        
        # Add some variation to the trajectory
        noise = np.random.normal(0, 0.05, len(t))
        
        x = t
        y = 0.5 * np.sin(2 * np.pi * t) + 0.1 * np.sin(8 * np.pi * t) + noise
        
        trajectory = np.column_stack([x, y])
        trajectories.append(trajectory)
    
    return trajectories


def main():
    """Main demonstration function."""
    print("Dynamic Movement Primitives (DMP) Demo")
    print("=" * 40)
    
    # Create demonstration trajectories
    print("Creating demonstration trajectories...")
    trajectories = create_multiple_demonstrations()
    
    # Plot original demonstrations
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for i, traj in enumerate(trajectories):
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.7, label=f'Demo {i+1}')
    plt.title('Demonstration Trajectories')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Configure and train DMP
    print("Training DMP...")
    config = DMPConfig(
        n_dims=2,
        dt=0.01,
        execution_time=1.0,
        n_basis=50,
        basis_width=0.1
    )
    
    dmp = DMP(config)
    dmp.fit(trajectories)
    
    print(f"DMP trained successfully!")
    print(f"Weight shape: {dmp.get_weights().shape}")
    
    # Generate trajectories with different goals
    print("Generating trajectories with different goals...")
    
    # Original goal
    original_goal = np.array([1.0, 0.0])
    traj_original = dmp.generate(start=np.array([0.0, 0.0]), goal=original_goal)
    
    # Modified goals
    goals = [
        np.array([1.0, 0.5]),   # Higher goal
        np.array([1.0, -0.5]),  # Lower goal
        np.array([1.2, 0.0]),   # Further goal
        np.array([0.8, 0.0]),   # Closer goal
    ]
    
    plt.subplot(2, 2, 2)
    plt.plot(traj_original[:, 0], traj_original[:, 1], 'b-', linewidth=2, label='Original Goal')
    
    for i, goal in enumerate(goals):
        traj = dmp.generate(start=np.array([0.0, 0.0]), goal=goal)
        plt.plot(traj[:, 0], traj[:, 1], '--', linewidth=2, label=f'Goal {i+1}')
    
    plt.title('DMP with Different Goals')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Generate trajectories with different execution times
    print("Generating trajectories with different execution times...")
    
    execution_times = [0.5, 1.0, 1.5, 2.0]
    
    plt.subplot(2, 2, 3)
    for i, exec_time in enumerate(execution_times):
        traj = dmp.generate(
            start=np.array([0.0, 0.0]), 
            goal=np.array([1.0, 0.0]),
            execution_time=exec_time
        )
        plt.plot(traj[:, 0], traj[:, 1], linewidth=2, label=f'Time {exec_time}s')
    
    plt.title('DMP with Different Execution Times')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Show phase and basis functions
    print("Visualizing phase and basis functions...")
    
    phase = dmp.get_phase()
    basis_values = dmp.get_basis_functions(phase)
    
    plt.subplot(2, 2, 4)
    plt.plot(phase, label='Phase')
    plt.title('Phase Evolution')
    plt.xlabel('Time Step')
    plt.ylabel('Phase Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    print("\nAdditional Analysis:")
    print(f"Phase starts at: {phase[0]:.3f}")
    print(f"Phase ends at: {phase[-1]:.3f}")
    print(f"Number of basis functions: {basis_values.shape[1]}")
    print(f"Basis function centers: {dmp.basis_functions.centers[:5]}...")  # Show first 5
    
    # Test waypoint conditioning (if available)
    print("\nTesting waypoint conditioning...")
    try:
        # This would require implementing waypoint conditioning in DMP
        print("Waypoint conditioning not implemented in basic DMP")
    except Exception as e:
        print(f"Waypoint conditioning error: {e}")
    
    print("\nDMP Demo completed successfully!")


if __name__ == "__main__":
    main()
