import numpy as np
from matplotlib import pyplot as plt
import os

# Scikit Learn Libraries (For handling basic Regression Processes)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from typing import Callable
from pyDOE import lhs

def f(X:np.ndarray):
    x = X.copy()
    return x**2 * np.sin(x) + 5

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

def generate_gp_plot(n_samples, alpha, xi, fi, save_dir):
    """Generate and save a GP plot with the specified number of samples and noise parameter"""
    print(f"Generating GP plot with {n_samples} samples and noise alpha={alpha}...")
    
    lb = -6.0
    ub = 6.0

    # Controls how fast the function is expected to change
    L = 0.3

    # Define the kernel (Matern with length scale L)
    kernel = C(1.0) * Matern(length_scale=L, nu=1.5)

    # Create and fit the GP model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)
    gp.fit(xi, fi)

    # Define points for predictions (fine grid for smooth visualization)
    X_pred = np.atleast_2d(np.linspace(lb, ub, 100)).T

    # Make predictions
    y_pred, sigma = gp.predict(X_pred, return_std=True)

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Plot the true function
    X_true = np.atleast_2d(np.linspace(lb, ub, 500)).T
    plt.plot(X_true, f(X_true), 'r--', label='True function')

    # Plot multiple confidence intervals with decreasing opacity
    alphas = [0.05, 0.1, 0.15, 0.2]
    stds = [4, 3, 2, 1]
    for std, alpha_val in zip(stds, alphas):
        plt.fill_between(X_pred.ravel(), 
                        y_pred - std*sigma, 
                        y_pred + std*sigma, 
                        alpha=alpha_val, color='purple', 
                        label=f'μ ± {std}σ' if std == 4 else None)

    # Plot the predicted mean
    plt.plot(X_pred, y_pred, color='darkorange', label='GP mean')

    # Plot training data
    plt.scatter(xi, fi, color='navy', label='Training points')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'GP Regression with {n_samples} Training Points (Noise α={alpha})')
    plt.legend()
    plt.grid(True)
    
    # Set consistent y-axis limits
    plt.ylim(-40, 60)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f"gp_samples_{n_samples}_noise_{alpha:.3f}.png"), dpi=300)
    plt.close()
    
    # Return optimized kernel parameters
    return gp.kernel_

# Create plots directory
plots_dir = ensure_dir("gp_plots")

# Set seed for reproducible training point generation
seed = 42  # Can be changed to any integer for different random samples
np.random.seed(seed)
print(f"Using seed {seed} for training point generation")

# Generate plots for different sample sizes and noise values
sample_sizes = [10]
noise_values = [0.250,0.500,0.750,1,1.250,1.5,2]  # Small, medium, and large noise values

results = {}
for n_samples in sample_sizes:
    # Generate training data once per sample size
    lb = -6.0
    ub = 6.0
    samples_normalized = lhs(1, samples=n_samples)
    xi = lb + (ub - lb) * samples_normalized
    fi = f(xi).ravel()
    
    # Use the same training data for all noise levels
    for alpha in noise_values:
        kernel_params = generate_gp_plot(n_samples, alpha, xi, fi, plots_dir)
        results[(n_samples, alpha)] = str(kernel_params)

# Print summary of results
print("\nOptimized kernel parameters for different configurations:")
for (n_samples, alpha), kernel_str in results.items():
    print(f"Samples: {n_samples}, Noise α: {alpha:.3f}, Kernel: {kernel_str}")

print(f"\nAll plots have been saved to the '{plots_dir}' directory.")

