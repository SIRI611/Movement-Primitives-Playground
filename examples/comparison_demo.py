#!/usr/bin/env python3
"""
Comprehensive comparison of all Movement Primitives implementations.

This script demonstrates and compares:
- Dynamic Movement Primitives (DMP)
- Probabilistic Movement Primitives (ProMP)
- Probabilistic Dynamic Movement Primitives (ProDMP)
- Conditional Neural Movement Primitives (CNMP)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from movement_primitives import DMP, DMPConfig, ProMP, ProMPConfig, ProDMP, ProDMPConfig, CNMP, CNMPConfig


def create_demonstration_trajectories():
    """Create demonstration trajectories for comparison."""
    trajectories = []
    
    for i in range(6):
        t = np.linspace(0, 1, 100)
        
        # Base trajectory
        x = t
        y_base = 0.4 * np.sin(2 * np.pi * t) + 0.1 * np.sin(8 * np.pi * t)
        
        # Add variations
        variation = np.random.normal(0, 0.08, len(t))
        y = y_base + variation
        
        trajectory = np.column_stack([x, y])
        trajectories.append(trajectory)
    
    return trajectories


def create_contextual_demonstrations():
    """Create contextual demonstrations for CNMP."""
    trajectories = []
    contexts = []
    
    # Different contexts: [amplitude, frequency]
    context_params = [
        [0.3, 1.0], [0.5, 1.0], [0.7, 1.0],
        [0.3, 2.0], [0.5, 2.0], [0.7, 2.0],
    ]
    
    for amp, freq in context_params:
        t = np.linspace(0, 1, 100)
        
        x = t
        y = amp * np.sin(2 * np.pi * freq * t) + 0.1 * np.sin(8 * np.pi * t)
        noise = np.random.normal(0, 0.05, len(t))
        y += noise
        
        trajectory = np.column_stack([x, y])
        trajectories.append(trajectory)
        contexts.append(np.array([amp, freq]))
    
    return trajectories, contexts


def compare_training_time():
    """Compare training times of different MPs."""
    print("Comparing Training Times")
    print("=" * 30)
    
    trajectories = create_demonstration_trajectories()
    
    # DMP
    start_time = time.time()
    dmp_config = DMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=50)
    dmp = DMP(dmp_config)
    dmp.fit(trajectories)
    dmp_time = time.time() - start_time
    
    # ProMP
    start_time = time.time()
    promp_config = ProMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=50)
    promp = ProMP(promp_config)
    promp.fit(trajectories)
    promp_time = time.time() - start_time
    
    # ProDMP
    start_time = time.time()
    prodmp_config = ProDMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=50)
    prodmp = ProDMP(prodmp_config)
    prodmp.fit(trajectories)
    prodmp_time = time.time() - start_time
    
    # CNMP
    trajectories_cnmp, contexts_cnmp = create_contextual_demonstrations()
    start_time = time.time()
    cnmp_config = CNMPConfig(n_dims=2, dt=0.01, execution_time=1.0, n_basis=50, 
                           context_dim=2, n_epochs=200)  # Reduced epochs for comparison
    cnmp = CNMP(cnmp_config)
    cnmp.fit(trajectories_cnmp, contexts_cnmp)
    cnmp_time = time.time() - start_time
    
    print(f"DMP Training Time: {dmp_time:.4f} seconds")
    print(f"ProMP Training Time: {promp_time:.4f} seconds")
    print(f"ProDMP Training Time: {prodmp_time:.4f} seconds")
    print(f"CNMP Training Time: {cnmp_time:.4f} seconds")
    
    return dmp, promp, prodmp, cnmp


def compare_generation_time(dmp, promp, prodmp, cnmp):
    """Compare generation times of different MPs."""
    print("\nComparing Generation Times")
    print("=" * 30)
    
    n_runs = 100
    
    # DMP
    start_time = time.time()
    for _ in range(n_runs):
        dmp.generate(start=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0]))
    dmp_gen_time = (time.time() - start_time) / n_runs
    
    # ProMP
    start_time = time.time()
    for _ in range(n_runs):
        promp.generate(start=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0]))
    promp_gen_time = (time.time() - start_time) / n_runs
    
    # ProDMP
    start_time = time.time()
    for _ in range(n_runs):
        prodmp.generate(start=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0]))
    prodmp_gen_time = (time.time() - start_time) / n_runs
    
    # CNMP
    context = np.array([0.5, 1.5])
    start_time = time.time()
    for _ in range(n_runs):
        cnmp.generate(start=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0]), context=context)
    cnmp_gen_time = (time.time() - start_time) / n_runs
    
    print(f"DMP Generation Time: {dmp_gen_time*1000:.2f} ms")
    print(f"ProMP Generation Time: {promp_gen_time*1000:.2f} ms")
    print(f"ProDMP Generation Time: {prodmp_gen_time*1000:.2f} ms")
    print(f"CNMP Generation Time: {cnmp_gen_time*1000:.2f} ms")


def visualize_comparison(dmp, promp, prodmp, cnmp):
    """Visualize comparison of different MPs."""
    print("\nGenerating Comparison Visualizations")
    print("=" * 40)
    
    plt.figure(figsize=(16, 12))
    
    # Generate trajectories
    start = np.array([0.0, 0.0])
    goal = np.array([1.0, 0.0])
    context = np.array([0.5, 1.5])
    
    traj_dmp = dmp.generate(start=start, goal=goal)
    traj_promp = promp.generate(start=start, goal=goal)
    traj_prodmp = prodmp.generate(start=start, goal=goal)
    traj_cnmp = cnmp.generate(start=start, goal=goal, context=context)
    
    # Plot 1: Basic trajectory comparison
    plt.subplot(3, 3, 1)
    plt.plot(traj_dmp[:, 0], traj_dmp[:, 1], 'b-', linewidth=2, label='DMP')
    plt.plot(traj_promp[:, 0], traj_promp[:, 1], 'r-', linewidth=2, label='ProMP')
    plt.plot(traj_prodmp[:, 0], traj_prodmp[:, 1], 'g-', linewidth=2, label='ProDMP')
    plt.plot(traj_cnmp[:, 0], traj_cnmp[:, 1], 'm-', linewidth=2, label='CNMP')
    plt.title('Trajectory Comparison')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Plot 2: ProMP sampling
    plt.subplot(3, 3, 2)
    sampled_trajs = promp.sample_trajectories(10, start=start, goal=goal)
    for traj in sampled_trajs:
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.3, color='red')
    plt.plot(traj_promp[:, 0], traj_promp[:, 1], 'r-', linewidth=3, label='Mean')
    plt.title('ProMP Sampling')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Plot 3: ProDMP sampling
    plt.subplot(3, 3, 3)
    sampled_trajs = prodmp.sample_trajectories(10, start=start, goal=goal)
    for traj in sampled_trajs:
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.3, color='green')
    plt.plot(traj_prodmp[:, 0], traj_prodmp[:, 1], 'g-', linewidth=3, label='Mean')
    plt.title('ProDMP Sampling')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Plot 4: CNMP context variation
    plt.subplot(3, 3, 4)
    contexts = [
        np.array([0.3, 1.0]),
        np.array([0.5, 1.5]),
        np.array([0.7, 2.0]),
    ]
    colors = ['blue', 'green', 'red']
    for ctx, color in zip(contexts, colors):
        traj = cnmp.generate(start=start, goal=goal, context=ctx)
        plt.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, 
                label=f'Amp={ctx[0]:.1f}, Freq={ctx[1]:.1f}')
    plt.title('CNMP Context Variation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Plot 5: Different goals
    plt.subplot(3, 3, 5)
    goals = [
        np.array([1.0, 0.0]),
        np.array([1.0, 0.3]),
        np.array([1.0, -0.3]),
    ]
    for i, goal in enumerate(goals):
        traj = dmp.generate(start=start, goal=goal)
        plt.plot(traj[:, 0], traj[:, 1], linewidth=2, label=f'Goal {i+1}')
    plt.title('DMP Different Goals')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Plot 6: Different execution times
    plt.subplot(3, 3, 6)
    times = [0.5, 1.0, 1.5, 2.0]
    for i, exec_time in enumerate(times):
        traj = dmp.generate(start=start, goal=goal, execution_time=exec_time)
        plt.plot(traj[:, 0], traj[:, 1], linewidth=2, label=f'{exec_time}s')
    plt.title('DMP Different Times')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Plot 7: ProMP uncertainty
    plt.subplot(3, 3, 7)
    variance = promp.get_trajectory_variance()
    std = np.sqrt(variance[:, 1])
    plt.plot(traj_promp[:, 0], traj_promp[:, 1], 'r-', linewidth=2, label='Mean')
    plt.fill_between(traj_promp[:, 0], 
                     traj_promp[:, 1] - 2*std, 
                     traj_promp[:, 1] + 2*std, 
                     alpha=0.3, color='red', label='±2σ')
    plt.title('ProMP Uncertainty')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Plot 8: ProDMP full trajectory
    plt.subplot(3, 3, 8)
    full_traj = prodmp.get_full_trajectory(start=start, goal=goal)
    plt.plot(full_traj['position'][:, 0], full_traj['position'][:, 1], 'g-', linewidth=2, label='Position')
    plt.title('ProDMP Full Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Plot 9: CNMP training history
    plt.subplot(3, 3, 9)
    history = cnmp.get_training_history()
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('CNMP Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def feature_comparison():
    """Compare features of different MPs."""
    print("\nFeature Comparison")
    print("=" * 20)
    
    features = {
        'DMP': {
            'Probabilistic': 'No',
            'Neural Network': 'No',
            'Context Conditioning': 'No',
            'Uncertainty Quantification': 'No',
            'Waypoint Conditioning': 'No',
            'Velocity/Acceleration': 'Yes',
            'Temporal Scaling': 'Yes',
            'Goal Modification': 'Yes',
        },
        'ProMP': {
            'Probabilistic': 'Yes',
            'Neural Network': 'No',
            'Context Conditioning': 'No',
            'Uncertainty Quantification': 'Yes',
            'Waypoint Conditioning': 'Yes',
            'Velocity/Acceleration': 'Yes',
            'Temporal Scaling': 'Yes',
            'Goal Modification': 'Yes',
        },
        'ProDMP': {
            'Probabilistic': 'Yes',
            'Neural Network': 'No',
            'Context Conditioning': 'No',
            'Uncertainty Quantification': 'Yes',
            'Waypoint Conditioning': 'Yes',
            'Velocity/Acceleration': 'Yes',
            'Temporal Scaling': 'Yes',
            'Goal Modification': 'Yes',
        },
        'CNMP': {
            'Probabilistic': 'No',
            'Neural Network': 'Yes',
            'Context Conditioning': 'Yes',
            'Uncertainty Quantification': 'No',
            'Waypoint Conditioning': 'No',
            'Velocity/Acceleration': 'Yes',
            'Temporal Scaling': 'Yes',
            'Goal Modification': 'Yes',
        },
    }
    
    # Print comparison table
    print(f"{'Feature':<25} {'DMP':<8} {'ProMP':<8} {'ProDMP':<8} {'CNMP':<8}")
    print("-" * 65)
    
    for feature in features['DMP'].keys():
        print(f"{feature:<25} {features['DMP'][feature]:<8} {features['ProMP'][feature]:<8} {features['ProDMP'][feature]:<8} {features['CNMP'][feature]:<8}")


def main():
    """Main comparison function."""
    print("Movement Primitives Comprehensive Comparison")
    print("=" * 50)
    
    # Compare training times
    dmp, promp, prodmp, cnmp = compare_training_time()
    
    # Compare generation times
    compare_generation_time(dmp, promp, prodmp, cnmp)
    
    # Feature comparison
    feature_comparison()
    
    # Visualize comparison
    visualize_comparison(dmp, promp, prodmp, cnmp)
    
    print("\nComparison completed successfully!")
    print("\nSummary:")
    print("- DMP: Fast, deterministic, good for basic trajectory learning")
    print("- ProMP: Probabilistic, uncertainty quantification, waypoint conditioning")
    print("- ProDMP: Combines DMP and ProMP benefits, full trajectory analysis")
    print("- CNMP: Neural network-based, context conditioning, most flexible")


if __name__ == "__main__":
    main()
