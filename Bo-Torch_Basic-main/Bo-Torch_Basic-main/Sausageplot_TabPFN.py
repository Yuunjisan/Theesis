import numpy as np
from matplotlib import pyplot as plt
from tabpfn import TabPFNRegressor
from pyDOE import lhs
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# Set random seed for reproducibility
np.random.seed(42)

def f(X:np.ndarray):
    x = X.copy()
    return np.sin(x) + np.sin(2.2 * x) + np.sin(3.5 * x) + 0.05 * x**2

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

def generate_tabpfn_plot(n_samples, softmax_temp, n_estimators, save_dir, xi=None, fi=None, y_min=None, y_max=None):
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
    model = TabPFNRegressor(n_estimators=n_estimators, softmax_temperature=softmax_temp, fit_mode="fit_preprocessors")
    model.fit(xi, fi)

    # Define points for predictions (fine grid for smooth visualization)
    X_pred = np.atleast_2d(np.linspace(lb, ub, 100)).T

    # Get predictions and quantiles from TabPFN
    predictions = model.predict(X_pred, output_type="main")
    mean = predictions["mean"]
    quantiles = predictions["quantiles"]
    
    # Calculate median (which is quantile at 0.5 - index 4)
    median = quantiles[4]

    # Calculate evaluation metrics on training data
    y_train_pred = model.predict(xi)
    train_mse = mean_squared_error(fi, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(fi, y_train_pred)
    train_r2 = r2_score(fi, y_train_pred)
    train_evs = explained_variance_score(fi, y_train_pred)
    
    # Calculate metrics on test data (using true function values)
    y_test_true = f(X_pred)
    test_mse = mean_squared_error(y_test_true, mean)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_true, mean)
    test_r2 = r2_score(y_test_true, mean)
    test_evs = explained_variance_score(y_test_true, mean)
    
    # Create a dictionary with all metrics
    metrics = {
        'train': {
            'MSE': train_mse,
            'RMSE': train_rmse,
            'MAE': train_mae,
            'R²': train_r2,
            'EVS': train_evs
        },
        'test': {
            'MSE': test_mse,
            'RMSE': test_rmse,
            'MAE': test_mae,
            'R²': test_r2,
            'EVS': test_evs
        }
    }

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
    
    # Add metrics text box
    metrics_text = (
        f"Test Metrics:\n"
        f"MSE: {test_mse:.3f}\n"
        f"RMSE: {test_rmse:.3f}\n"
        f"MAE: {test_mae:.3f}\n"
        f"R²: {test_r2:.3f}\n"
        f"EVS: {test_evs:.3f}"
    )
    plt.figtext(0.02, 0.02, metrics_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.legend()
    plt.grid(True)
    
    # Set y-axis limits if provided
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f"tabpfn_samples_{n_samples}_temp_{softmax_temp:.2f}_estimators_{n_estimators}.png"), dpi=300)
    plt.close()
    
    # Print metrics for reference
    print(f"Test metrics for n={n_samples}, temp={softmax_temp}, estimators={n_estimators}:")
    print(f"  MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}, EVS: {test_evs:.4f}")
    
    # Return metrics for potential further analysis
    return metrics

# Create plots directory
plots_dir = ensure_dir("tabpfn_plots")

# Define sample sizes and temperatures to test
sample_sizes = [3,4,5,6,7,8,9,10]
temperatures = [1]
n_estimators = [16]

# Calculate y-axis limits based on the true function with a 20% margin
lb = -6.0
ub = 6.0
X_true = np.atleast_2d(np.linspace(lb, ub, 500)).T
y_true = f(X_true)
# Calculate min and max with 20% margin
y_min_val = y_true.min()
y_max_val = y_true.max()
y_range = y_max_val - y_min_val
margin = 0.2 * y_range
y_min = y_min_val - margin
y_max = y_max_val + margin

print(f"Auto-calculated y-axis limits: [{y_min:.3f}, {y_max:.3f}]")

# Store all metrics for comparison
all_metrics = {}

# Generate plots for each combination
for n_samples in sample_sizes:
    # Generate samples once for each sample size
    np.random.seed(42)
    samples_normalized = lhs(1, samples=n_samples, criterion="center")
    xi = lb + (ub - lb) * samples_normalized
    fi = f(xi).ravel()
    
    for temp in temperatures:
        for n_est in n_estimators:
            metric_key = f"samples_{n_samples}_temp_{temp:.2f}_estimators_{n_est}"
            all_metrics[metric_key] = generate_tabpfn_plot(n_samples, temp, n_est, plots_dir, xi=xi, fi=fi, y_min=y_min, y_max=y_max)

print(f"\nAll plots have been saved to the '{plots_dir}' directory.")

# Create a summary file with all metrics
summary_path = os.path.join(plots_dir, "metrics_summary.csv")
with open(summary_path, "w") as f:
    # Write header
    f.write("Samples,Temperature,Estimators,Test_MSE,Test_RMSE,Test_MAE,Test_R2,Test_EVS\n")
    
    # Write data for each configuration
    for n_samples in sample_sizes:
        for temp in temperatures:
            for n_est in n_estimators:
                metric_key = f"samples_{n_samples}_temp_{temp:.2f}_estimators_{n_est}"
                metrics = all_metrics[metric_key]
                f.write(f"{n_samples},{temp},{n_est},"
                        f"{metrics['test']['MSE']:.6f},"
                        f"{metrics['test']['RMSE']:.6f},"
                        f"{metrics['test']['MAE']:.6f},"
                        f"{metrics['test']['R²']:.6f},"
                        f"{metrics['test']['EVS']:.6f}\n")

print(f"Metrics summary saved to {summary_path}")