import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import gpytorch
from ioh import get_problem, ProblemClass
import os
from scipy import stats
from scipy.stats import qmc

# Create a directory to save plots
def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

# Create plots directory
plots_dir = ensure_dir("bbob_plots")

# Create the problem
problem = get_problem(1, 1, 2, problem_class=ProblemClass.BBOB)

# Define a function to evaluate the BBOB function
def evaluate_function(x):
    """
    Evaluate the BBOB function at point x
    x: numpy array of shape (2,) for 2D
    """
    value = problem(x)
    return value

# Define the domain (search space) - typically [-5, 5] for most BBOB functions
lower_bound = -5
upper_bound = 5               

def generate_dataset(n_samples):
    """
    Generate a dataset of n_samples points and their function values using Sobol sampling
    """
    # Create a Sobol sequence generator for 2D
    sampler = qmc.Sobol(d=2, scramble=True)
    
    # Generate samples in [0, 1]^2
    samples = sampler.random(n=n_samples)
    
    # Scale samples to the domain [lower_bound, upper_bound]^2
    X = qmc.scale(samples, lower_bound, upper_bound)
    
    # Evaluate function at each point
    y = np.array([evaluate_function(x) for x in X])
    
    return X, y

# Visualize the true function
def plot_true_function():
    """
    Create a contour plot of the true BBOB function
    """
    # Create a grid of points with 501x501 resolution
    resolution = 501
    x = np.linspace(lower_bound, upper_bound, resolution)
    y = np.linspace(lower_bound, upper_bound, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate function at each grid point
    Z = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = evaluate_function(np.array([X[i, j], Y[i, j]]))
    
    # Create custom colormap: blue -> purple -> pink -> red -> yellow
    colors = ['#000080', '#4B0082', '#800080', '#C71585', '#FF1493', 
              '#FF69B4', '#FF6347', '#FF4500', '#FFA500', '#FFD700']
    custom_cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom_cmap', colors)
    
    # Create contour plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create filled contour with custom colormap
    levels = 15  # Number of contour levels
    contourf = ax.contourf(X, Y, Z, levels, cmap=custom_cmap, alpha=0.7)
    
    # Add contour lines with labels
    contour_lines = ax.contour(X, Y, Z, levels, colors='k', alpha=0.5, linewidths=0.5)
    
    # Add contour labels
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Set title and axis labels
    ax.set_title('Level sets (bbob f₁, 2-D, inst. 1)', fontsize=16)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    
    # Find the minimum point (should be around 0,0 for f1)
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    min_x, min_y = X[min_idx], Y[min_idx]
    
    # Mark the minimum point with an X
    ax.plot(min_x, min_y, 'kx', markersize=10, markeredgewidth=2)
    
    # Add text at the bottom showing grid size
    ax.text(0.5, -0.1, f'{resolution}×{resolution} grid', 
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=10,
            color='#8B4513')  # Use brown color as in the reference image
    
    # Save the figure
    plt.savefig(os.path.join(plots_dir, "bbob_function_levelsets.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    
    return X, Y, Z  # Return grid for comparison later

# Generate and plot the true function
X_grid, Y_grid, Z_true = plot_true_function()

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.LinearMean(input_size=2)
        
        # Covariance module
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_and_predict_gp(train_X, train_y, grid_X, grid_Y):
    """
    Train a GP model on training data and predict on a grid
    """
    # Convert numpy arrays to torch tensors
    train_X_tensor = torch.tensor(train_X, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
    
    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.raw_noise.requires_grad_(False)
    likelihood.noise_covar.raw_noise.fill_(-10.0)  # log(1e-4) ≈ -9.2
    model = ExactGPModel(train_X_tensor, train_y_tensor, likelihood)
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    # Use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # "Loss" for GP learning is the negative marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # If using spectral mixture kernel, initialize it
    if isinstance(model.covar_module.base_kernel, gpytorch.kernels.SpectralMixtureKernel):
        # Initialize the spectral mixture kernel with data
        with torch.no_grad():
            model.covar_module.base_kernel.initialize_from_data(train_X_tensor, train_y_tensor)
    
    # Training loop - increase iterations for more complex kernels
    training_iterations = 200  # Increase from 100 to 200
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_X_tensor)
        loss = -mll(output, train_y_tensor)
        loss.backward()
        optimizer.step()
    
    # Make predictions with the model
    model.eval()
    likelihood.eval()
    
    # Prepare grid data for prediction
    grid_size = grid_X.shape[0]
    grid_points = np.column_stack((grid_X.flatten(), grid_Y.flatten()))
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    
    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(grid_tensor))
        mean = predictions.mean.numpy()
        variance = predictions.variance.numpy()
    
    # Reshape for plotting
    mean_grid = mean.reshape(grid_size, grid_size)
    variance_grid = variance.reshape(grid_size, grid_size)
    
    return mean_grid, variance_grid

def plot_approximation(sample_size, train_X, mean_grid, variance_grid, Z_true):
    """
    Plot the GP approximation alongside the true function
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot the GP mean prediction
    # First, filled contours
    contour1 = axes[0].contourf(X_grid, Y_grid, mean_grid, 50, cmap='viridis', alpha=0.7)
    # Then, add labeled contour lines
    contour_lines = axes[0].contour(X_grid, Y_grid, mean_grid, 10, colors='black', linewidths=0.5)
    axes[0].clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")
    plt.colorbar(contour1, ax=axes[0])
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_title(f'GP Mean Prediction (n={sample_size})')
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    
    # Plot the GP uncertainty (variance)
    # First, filled contours
    contour2 = axes[1].contourf(X_grid, Y_grid, variance_grid, 50, cmap='plasma', alpha=0.7)
    # Then, add labeled contour lines
    contour_lines2 = axes[1].contour(X_grid, Y_grid, variance_grid, 10, colors='black', linewidths=0.5)
    axes[1].clabel(contour_lines2, inline=True, fontsize=8, fmt="%.2e")
    plt.colorbar(contour2, ax=axes[1])
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_title(f'GP Prediction Uncertainty (n={sample_size})')
    axes[1].set_xlabel('x1')
    axes[1].set_ylabel('x2')
    
    # Plot the error between prediction and true function
    error = np.abs(mean_grid - Z_true)
    # First, filled contours
    contour3 = axes[2].contourf(X_grid, Y_grid, error, 50, cmap='Reds', alpha=0.7)
    # Then, add labeled contour lines
    contour_lines3 = axes[2].contour(X_grid, Y_grid, error, 10, colors='black', linewidths=0.5)
    axes[2].clabel(contour_lines3, inline=True, fontsize=8, fmt="%.2f")
    plt.colorbar(contour3, ax=axes[2])
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].set_title(f'Absolute Error (n={sample_size})')
    axes[2].set_xlabel('x1')
    axes[2].set_ylabel('x2')
    
    # Add training points to all plots
    for ax in axes:
        ax.scatter(train_X[:, 0], train_X[:, 1], c='black', marker='x', s=30, label='Training points')
    
    # Calculate error metrics
    mean_error = np.mean(error)
    max_error = np.max(error)
    
    # Add the text about error metrics
    plt.figtext(0.5, 0.01, f'Mean Absolute Error: {mean_error:.4f}, Max Absolute Error: {max_error:.4f}', 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Gaussian Process Approximation with {sample_size} Training Points', fontsize=16)
    
    # Save the figure
    plt.savefig(os.path.join(plots_dir, f"approximation_n{sample_size}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    
    return mean_error, max_error

# Sample sizes to test
sample_sizes = [10, 50, 100, 1000]
results = {}

# For each sample size, generate data, train model, and visualize
for n_samples in sample_sizes:
    print(f"\nProcessing {n_samples} samples...")
    
    # Generate dataset
    train_X, train_y = generate_dataset(n_samples)
    
    # Train GP model and get predictions
    mean_grid, variance_grid = train_and_predict_gp(train_X, train_y, X_grid, Y_grid)
    
    # Plot and get error metrics
    mean_error, max_error = plot_approximation(n_samples, train_X, mean_grid, variance_grid, Z_true)
    
    # Store results
    results[n_samples] = {
        'mean_error': mean_error,
        'max_error': max_error
    }

# Summarize results
print("\nSummary of Approximation Errors:")
print("Sample Size | Mean Absolute Error | Max Absolute Error")
print("-" * 60)
for n, metrics in results.items():
    print(f"{n:11d} | {metrics['mean_error']:18.4f} | {metrics['max_error']:17.4f}")

# Save the results to a text file
with open(os.path.join(plots_dir, "results_summary.txt"), 'w') as f:
    f.write("Summary of Approximation Errors:\n")
    f.write("Sample Size | Mean Absolute Error | Max Absolute Error\n")
    f.write("-" * 60 + "\n")
    for n, metrics in results.items():
        f.write(f"{n:11d} | {metrics['mean_error']:18.4f} | {metrics['max_error']:17.4f}\n")

print(f"\nAll plots and results have been saved to the '{plots_dir}' directory.")