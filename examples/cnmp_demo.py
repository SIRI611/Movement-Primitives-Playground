#!/usr/bin/env python3
"""
Example script demonstrating Conditional Neural Movement Primitives (CNMP).

This script shows how to:
1. Create demonstration trajectories with different contexts
2. Train a CNMP on the demonstrations and contexts
3. Generate trajectories conditioned on new contexts
4. Visualize the training process and results
"""

import numpy as np
import matplotlib.pyplot as plt
from movement_primitives import CNMP, CNMPConfig, plot_trajectory


def create_contextual_demonstrations():
    """Create demonstration trajectories with different contexts."""
    trajectories = []
    contexts = []
    
    # Different contexts: [amplitude, frequency]
    context_params = [
        [0.3, 1.0],   # Low amplitude, low frequency
        [0.5, 1.0],   # Medium amplitude, low frequency
        [0.7, 1.0],   # High amplitude, low frequency
        [0.3, 2.0],   # Low amplitude, high frequency
        [0.5, 2.0],   # Medium amplitude, high frequency
        [0.7, 2.0],   # High amplitude, high frequency
        [0.4, 1.5],   # Medium amplitude, medium frequency
        [0.6, 1.5],   # High amplitude, medium frequency
    ]
    
    for amp, freq in context_params:
        t = np.linspace(0, 1, 100)
        
        # Create trajectory based on context
        x = t
        y = amp * np.sin(2 * np.pi * freq * t) + 0.1 * np.sin(8 * np.pi * t)
        
        # Add some noise
        noise = np.random.normal(0, 0.05, len(t))
        y += noise
        
        trajectory = np.column_stack([x, y])
        trajectories.append(trajectory)
        contexts.append(np.array([amp, freq]))
    
    return trajectories, contexts


def main():
    """Main demonstration function."""
    print("Conditional Neural Movement Primitives (CNMP) Demo")
    print("=" * 60)
    
    # Create demonstration trajectories with contexts
    print("Creating contextual demonstration trajectories...")
    trajectories, contexts = create_contextual_demonstrations()
    
    # Plot original demonstrations
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    for i, (traj, ctx) in enumerate(zip(trajectories, contexts)):
        plt.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, 
                label=f'Amp={ctx[0]:.1f}, Freq={ctx[1]:.1f}')
    plt.title('Demonstration Trajectories')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.axis('equal')
    
    # Configure and train CNMP
    print("Training CNMP...")
    config = CNMPConfig(
        n_dims=2,
        dt=0.01,
        execution_time=1.0,
        n_basis=50,
        basis_width=0.1,
        context_dim=2,
        hidden_dim=128,
        n_hidden_layers=2,
        activation='relu',
        learning_rate=1e-3,
        batch_size=16,
        n_epochs=500,
        device='cpu'
    )
    
    cnmp = CNMP(config)
    cnmp.fit(trajectories, contexts)
    
    print(f"CNMP trained successfully!")
    
    # Plot training history
    plt.subplot(2, 3, 2)
    history = cnmp.get_training_history()
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Test on training contexts
    print("Testing on training contexts...")
    plt.subplot(2, 3, 3)
    for i, (traj, ctx) in enumerate(zip(trajectories, contexts)):
        # Generate trajectory with same context
        generated_traj = cnmp.generate(
            start=np.array([0.0, 0.0]),
            goal=np.array([1.0, 0.0]),
            context=ctx
        )
        
        plt.plot(traj[:, 0], traj[:, 1], '--', color=colors[i], alpha=0.7, linewidth=1)
        plt.plot(generated_traj[:, 0], generated_traj[:, 1], '-', color=colors[i], linewidth=2)
    
    plt.title('Training Contexts: Original (--) vs Generated (-)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    
    # Test on new contexts
    print("Testing on new contexts...")
    new_contexts = [
        [0.4, 1.0],   # Medium amplitude, low frequency
        [0.6, 1.0],   # High amplitude, low frequency
        [0.3, 1.5],   # Low amplitude, medium frequency
        [0.5, 1.5],   # Medium amplitude, medium frequency
        [0.7, 1.5],   # High amplitude, medium frequency
    ]
    
    plt.subplot(2, 3, 4)
    new_colors = plt.cm.plasma(np.linspace(0, 1, len(new_contexts)))
    for i, ctx in enumerate(new_contexts):
        generated_traj = cnmp.generate(
            start=np.array([0.0, 0.0]),
            goal=np.array([1.0, 0.0]),
            context=np.array(ctx)
        )
        
        plt.plot(generated_traj[:, 0], generated_traj[:, 1], 
                color=new_colors[i], linewidth=2,
                label=f'Amp={ctx[0]:.1f}, Freq={ctx[1]:.1f}')
    
    plt.title('New Contexts')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Show context interpolation
    print("Testing context interpolation...")
    plt.subplot(2, 3, 5)
    
    # Interpolate between two contexts
    ctx1 = np.array([0.3, 1.0])
    ctx2 = np.array([0.7, 2.0])
    
    n_interp = 5
    for i in range(n_interp):
        alpha = i / (n_interp - 1)
        interp_ctx = (1 - alpha) * ctx1 + alpha * ctx2
        
        generated_traj = cnmp.generate(
            start=np.array([0.0, 0.0]),
            goal=np.array([1.0, 0.0]),
            context=interp_ctx
        )
        
        plt.plot(generated_traj[:, 0], generated_traj[:, 1], 
                linewidth=2, alpha=0.7,
                label=f'α={alpha:.1f}')
    
    plt.title('Context Interpolation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Show context space
    plt.subplot(2, 3, 6)
    context_array = np.array(contexts)
    plt.scatter(context_array[:, 0], context_array[:, 1], 
               c=range(len(contexts)), cmap='viridis', s=100)
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    plt.title('Context Space')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    print("\nAdditional Analysis:")
    print(f"Number of demonstrations: {len(trajectories)}")
    print(f"Context dimension: {config.context_dim}")
    print(f"Hidden dimension: {config.hidden_dim}")
    print(f"Number of hidden layers: {config.n_hidden_layers}")
    print(f"Final training loss: {history['train_loss'][-1]:.6f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    
    # Test different goals with same context
    print("\nTesting different goals with same context...")
    context = np.array([0.5, 1.5])
    goals = [
        np.array([1.0, 0.0]),
        np.array([1.0, 0.3]),
        np.array([1.0, -0.3]),
        np.array([1.2, 0.0]),
    ]
    
    plt.figure(figsize=(12, 4))
    for i, goal in enumerate(goals):
        plt.subplot(1, 4, i+1)
        traj = cnmp.generate(
            start=np.array([0.0, 0.0]),
            goal=goal,
            context=context
        )
        plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2)
        plt.plot(goal[0], goal[1], 'ro', markersize=8, label='Goal')
        plt.title(f'Goal: ({goal[0]:.1f}, {goal[1]:.1f})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
    
    plt.suptitle(f'Different Goals with Context: Amp={context[0]:.1f}, Freq={context[1]:.1f}')
    plt.tight_layout()
    plt.show()
    
    # Test different execution times
    print("\nTesting different execution times...")
    execution_times = [0.5, 1.0, 1.5, 2.0]
    
    plt.figure(figsize=(12, 4))
    for i, exec_time in enumerate(execution_times):
        plt.subplot(1, 4, i+1)
        traj = cnmp.generate(
            start=np.array([0.0, 0.0]),
            goal=np.array([1.0, 0.0]),
            context=context,
            execution_time=exec_time
        )
        plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2)
        plt.title(f'Execution Time: {exec_time}s')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')
    
    plt.suptitle(f'Different Execution Times with Context: Amp={context[0]:.1f}, Freq={context[1]:.1f}')
    plt.tight_layout()
    plt.show()
    
    print("\nCNMP Demo completed successfully!")


if __name__ == "__main__":
    main()
