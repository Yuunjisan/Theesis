"""
This script visualizes the TabPFN surrogate model and acquisition function values
to verify that the model is working correctly.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from Algorithms.BayesianOptimization.TabPFN_BO.TabPFN_BO import TabPFN_BO, TabPFNSurrogateModel
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from scipy.stats import norm

# Define a simple 1D test function
def f(x):
    """A 1D test function with multiple local minima."""
    return np.sin(3 * x) * x**2 + 0.1 * np.cos(10 * x)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def visualize_model_and_acquisition(n_points=5, n_iterations=3, acq_function="log_ei"):
    """
    Visualize TabPFN surrogate model and acquisition function during optimization.
    
    Args:
        n_points: Number of initial points
        n_iterations: Number of BO iterations to visualize
        acq_function: Acquisition function to use - "ei" or "log_ei" (recommended)
    """
    # Create figure for visualization
    fig, axes = plt.subplots(n_iterations, 2, figsize=(12, 4*n_iterations))
    
    # Generate test points for visualization
    x_test = np.linspace(-3, 3, 300).reshape(-1, 1)
    y_test = np.array([f(xi) for xi in x_test])
    
    # Generate initial points
    x_train = np.random.uniform(-3, 3, (n_points, 1))
    y_train = np.array([f(xi) for xi in x_train])
    
    # Create bounds
    bounds = np.array([[-3, 3]])
    
    # For each iteration
    for i in range(n_iterations):
        print(f"\nIteration {i+1}/{n_iterations}")
        
        # Create TabPFN surrogate model with fit_with_cache mode
        model = TabPFNSurrogateModel(
            train_X=x_train, 
            train_Y=y_train, 
            n_estimators=16, 
            fit_mode="fit_with_cache",
            device=device
        )
        
        # Get posterior predictions
        x_tensor = torch.tensor(x_test, dtype=torch.float64).to(device).unsqueeze(1)  # Add batch dimension and move to device
        with torch.no_grad():
            posterior = model.posterior(x_tensor)
            mean = posterior.mean.squeeze().cpu().numpy()
            
            # Calculate standard deviation (sqrt of diagonal of covariance)
            if hasattr(posterior, 'covariance_matrix'):
                std = torch.sqrt(torch.diagonal(posterior.covariance_matrix, dim1=-2, dim2=-1)).squeeze().cpu().numpy()
            else:
                # For MVN with scale_tril
                std = posterior.stddev.squeeze().cpu().numpy()
                
        # Find current best observed value
        best_y = np.min(y_train)
        
        # Create acquisition function
        if acq_function == "ei":
            acq = ExpectedImprovement(model, best_f=torch.tensor(best_y, dtype=torch.float64).to(device), maximize=False)
        elif acq_function == "log_ei":
            acq = LogExpectedImprovement(model, best_f=torch.tensor(best_y, dtype=torch.float64).to(device), maximize=False)
        else:
            raise ValueError(f"Unknown acquisition function: {acq_function}")
        
        # Compute acquisition function values
        with torch.no_grad():
            acq_values = acq(x_tensor).squeeze().cpu().numpy()
        
        # Plot surrogate model
        ax1 = axes[i, 0]
        ax1.plot(x_test, y_test, 'k--', label='True Function')
        ax1.plot(x_test, mean, 'b-', label='TabPFN Mean')
        ax1.fill_between(x_test.flatten(), 
                        mean - 1.96 * std, 
                        mean + 1.96 * std, 
                        alpha=0.3, color='b',
                        label='95% Confidence')
        
        # Color observations based on their sequence to show history
        if i > 0:  # If not the first iteration
            # Use a colormap to show the sequence of evaluations
            cmap = plt.cm.viridis
            colors = cmap(np.linspace(0, 1, len(x_train)-1))
            
            # Plot each observation with color based on evaluation order
            for j in range(len(x_train)-1):
                ax1.scatter(x_train[j], y_train[j], 
                           color=colors[j], 
                           marker='o', 
                           s=50, 
                           edgecolors='black', 
                           linewidth=1,
                           alpha=0.7)
            
            # Add colorbar to show sequence
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(x_train)-2))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax1)
            cbar.set_label('Evaluation Order')
        else:
            # For first iteration, just use regular scatter
            ax1.scatter(x_train, y_train, c='r', marker='o', label='Initial Points')
        
        ax1.axhline(best_y, color='g', linestyle='--', label='Best Value')
        ax1.set_title(f'Iteration {i+1}/{n_iterations}: TabPFN Surrogate Model\nBest so far: y = {best_y:.4f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot acquisition function
        ax2 = axes[i, 1]
        acq_label = "Log Expected Improvement" if acq_function == "log_ei" else "Expected Improvement"
        ax2.plot(x_test, acq_values, 'r-', label=acq_label)
        
        # Highlight the maximum acquisition value
        max_idx = np.argmax(acq_values)
        ax2.scatter([x_test[max_idx]], [acq_values[max_idx]], 
                   color='k', marker='*', s=100,
                   label='Max Acquisition')
                   
        # Now that max_idx is defined, we can use it in the title
        ax2.set_title(f'Iteration {i+1}: {acq_label}\nNext point selected at x = {x_test[max_idx][0]:.4f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Acquisition Value')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Find next point to sample (maximum of acquisition function)
        next_x = x_test[max_idx]
        next_y = f(next_x)
        
        # Ensure next_y is a scalar
        if isinstance(next_y, np.ndarray):
            next_y = float(next_y.item())
        
        # Add new observation
        x_train = np.vstack([x_train, [next_x]])
        y_train = np.append(y_train, next_y)
        
        # Add marker for next evaluation point on the surrogate model plot
        ax1.scatter([next_x], [next_y], color='red', marker='*', s=200, 
                   label='Next evaluation point', zorder=10)
        
        # Add vertical line connecting both plots at the next evaluation point
        ax1.axvline(x=next_x[0], color='red', linestyle='--', alpha=0.5)
        ax2.axvline(x=next_x[0], color='red', linestyle='--', alpha=0.5)
        
        # Add text annotation indicating the next point
        ax1.annotate(f"Next point: x={next_x[0]:.2f}, y={next_y:.2f}", 
                    xy=(next_x[0], next_y),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"))
        
        print(f"  Selected x = {next_x[0]:.4f}, y = {next_y:.4f}")
        print(f"  Acquisition max value: {acq_values[max_idx]:.6f}")
        
    plt.tight_layout()
    plt.savefig('tabpfn_acquisition_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    print("Visualizing TabPFN model and acquisition function...")
    visualize_model_and_acquisition(
        n_points=5, 
        n_iterations=20,
        acq_function="log_ei"  # Use LogExpectedImprovement to avoid warnings
    )
    print("Done! Visualization saved to 'tabpfn_acquisition_visualization.png'") 