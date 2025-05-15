import numpy as np
from matplotlib import pyplot as plt
from tabpfn import TabPFNRegressor
from pyDOE import lhs
import os

# Set random seed for reproducibility
np.random.seed(42)

def f(X:np.ndarray):
    x = X.copy()
    return x**2 * np.sin(x) + 5

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

def generate_tabpfn_plot(n_samples, softmax_temp, n_estimators, save_dir, xi=None, fi=None):
    """Generate and save a TabPFN plot with specified samples, softmax temperature, and number of estimators"""
    print(f"Generating TabPFN plot with {n_samples} samples, temperature={softmax_temp}, n_estimators={n_estimators}...")
    
    # Define bounds
    lb = -6.0
    ub = 6.0

    # Use provided samples or generate new ones
    if xi is None or fi is None:
        # Sample points using Latin Hypercube Sampling
        random_state = np.random.RandomState(42)  # Use fixed seed 42
        # Use the random_state parameter in lhs function (requires pyDOE2)
        # If using original pyDOE which doesn't have random_state param:
        np.random.seed(42)  # Set seed right before LHS call
        samples_normalized = lhs(1, samples=n_samples, criterion="center")
        xi = lb + (ub - lb) * samples_normalized
        fi = f(xi).ravel()

    # Create and fit the TabPFN model
    model = TabPFNRegressor(n_estimators=n_estimators, softmax_temperature=softmax_temp)
    model.fit(xi, fi)

    # Define points for predictions (fine grid for smooth visualization)
    X_pred = np.atleast_2d(np.linspace(lb, ub, 100)).T

    # Get predictions and quantiles from TabPFN
    predictions = model.predict(X_pred, output_type="main")
    mean = predictions["mean"]
    quantiles = predictions["quantiles"]
    
    # Calculate median (which is quantile at 0.5 - index 4)
    median = quantiles[4]

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Plot the true function
    X_true = np.atleast_2d(np.linspace(lb, ub, 500)).T
    plt.plot(X_true, f(X_true), 'r--', label='True function')

    # Plot multiple confidence intervals with decreasing opacity
    alphas = [0.05, 0.1, 0.15, 0.2]
    # Use quantile pairs that roughly correspond to 4σ, 3σ, 2σ, 1σ intervals
    # For a normal distribution: 4σ ≈ 99.99%, 3σ ≈ 99.7%, 2σ ≈ 95.4%, 1σ ≈ 68.3%
    quantile_pairs = [(0, 8), (1, 7), (2, 6), (3, 5)]  # These correspond to 10-90%, 20-80%, 30-70%, 40-60%
    
    for (lower_idx, upper_idx), alpha_val in zip(quantile_pairs, alphas):
        plt.fill_between(X_pred.ravel(), 
                        quantiles[lower_idx], 
                        quantiles[upper_idx], 
                        alpha=alpha_val, color='purple', 
                        label=f'Confidence region' if lower_idx == 0 else None)

    # Plot the predicted mean
    plt.plot(X_pred, mean, color='darkorange', label='TabPFN mean')
    
    # Plot the predicted median
    plt.plot(X_pred, median, color='green', linestyle='--', label='TabPFN median')

    # Plot training data
    plt.scatter(xi, fi, color='navy', label='Training points')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'TabPFN Regression (n={n_samples}, temp={softmax_temp}, estimators={n_estimators})')
    plt.legend()
    plt.grid(True)
    
    # Set consistent y-axis limits
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f"tabpfn_samples_{n_samples}_temp_{softmax_temp:.2f}_estimators_{n_estimators}.png"), dpi=300)
    plt.close()

# Create plots directory
plots_dir = ensure_dir("tabpfn_plots")

# Define sample sizes and temperatures to test
sample_sizes = [5]
temperatures = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
n_estimators = [16]

# Calculate global y-axis limits based on the true function
lb = -6.0
ub = 6.0
X_true = np.atleast_2d(np.linspace(lb, ub, 500)).T
y_true = f(X_true)
y_min = -40  # Match GP plot range
y_max = 60   # Match GP plot range

# Generate plots for each combination
for n_samples in sample_sizes:
    # Generate samples once for each sample size
    np.random.seed(42)
    samples_normalized = lhs(1, samples=n_samples, criterion="center")
    xi = lb + (ub - lb) * samples_normalized
    fi = f(xi).ravel()
    
    for temp in temperatures:
        for n_est in n_estimators:
            generate_tabpfn_plot(n_samples, temp, n_est, plots_dir, xi=xi, fi=fi)

print(f"\nAll plots have been saved to the '{plots_dir}' directory.")