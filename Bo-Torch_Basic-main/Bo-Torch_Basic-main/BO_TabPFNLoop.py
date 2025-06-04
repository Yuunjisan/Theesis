"""
This script visualizes the Bayesian optimization process for 1D functions using TabPFN as a surrogate model,
showing both the TabPFN surrogate model (sausage plots) and acquisition function.

Adapted from Sausage_Acq_plot_BO.py but using TabPFN_BO instead of Vanilla_BO.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import os
from pathlib import Path
import pandas as pd
from scipy import stats

# BoTorch imports
from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound

# Import TabPFN_BO
from Algorithms.BayesianOptimization.TabPFN_BO.TabPFN_BO import TabPFN_BO

# Define a simple 1D test function
def f(x):
    """Test function to optimize: f(x) = x^2 * sin(x) + 5"""
    if isinstance(x, np.ndarray):
        if x.ndim > 1:
            x = x.ravel()
    return x**2 * np.sin(x) + 5

class TabPFNVisualizationBO(TabPFN_BO):
    """
    Extension of TabPFN_BO that adds visualization capabilities for the surrogate model
    and acquisition function during optimization.
    """
    def __init__(self, budget, n_DoE=4, acquisition_function="expected_improvement", 
                 random_seed=42, save_plots=True, plots_dir="tabpfn_bo_visualizations",
                 n_estimators=8, fit_mode="fit_with_cache", device="cpu", **kwargs):
        super().__init__(budget, n_DoE, acquisition_function, random_seed, 
                         n_estimators=n_estimators, fit_mode=fit_mode, device="cpu", **kwargs)
        
        self.device = torch.device('cpu')
        # Visualization settings
        self.save_plots = save_plots
        self.plots_dir = plots_dir
        self.iteration_count = 0
        self.next_point_tensor = None  # Store the next point for accurate visualization
        
        # Plot style settings
        self.plot_colors = {
            'true_function': 'r--',
            'tabpfn_mean': 'darkorange',
            'confidence': 'purple',
            'points': 'navy',
            'next_point': 'green',
            'best_point': 'red',
            'acquisition': 'g-'
        }
        
        self.plot_markers = {
            'points': 'o',
            'next_point': '*',
            'best_point': 'x'
        }
        
        # For confidence band visualization
        self.confidence_alphas = [0.05, 0.1, 0.15, 0.2]
        self.quantile_pairs = [(0, 8), (1, 7), (2, 6), (3, 5)]
        
        # Create plots directory
        if self.save_plots:
            os.makedirs(self.plots_dir, exist_ok=True)
    
    def __call__(self, problem, bounds=None, **kwargs):
        """
        Run Bayesian optimization with visualization at each iteration.
        For 1D problems with visualization of the surrogate model and acquisition function.
        """
        # Initialize the optimization
        self._initialize(problem, bounds, **kwargs)
        
        # Visualize the initial surrogate model
        self.visualize_initial_model()
        
        # Run the optimization loop
        self._run_optimization_loop(problem, **kwargs)
        
        # Final visualization
        self.visualize_final_result()
    
    def _initialize(self, problem, bounds, **kwargs):
        """Initialize the optimization process."""
        # Set default bounds if not provided
        if bounds is None:
            bounds = np.array([[-6.0, 6.0]])  # Wider bounds for showing more of the function
        
        # Force 1D
        dim = 1
        
        # Call the superclass __call__ method with initial sampling
        super(TabPFN_BO, self).__call__(problem, dim, bounds, **kwargs)
        
        # Initialize the TabPFN model
        self._initialize_model()
    
    def _run_optimization_loop(self, problem, **kwargs):
        """Run the main optimization loop with visualization at each step."""
        # Get beta for UCB if needed
        beta = kwargs.get("beta", 0.2)
        
        for cur_iteration in range(self.budget - self.n_DoE):
            self.iteration_count = cur_iteration + 1
            
            # Set up the acquisition function based on name
            if self.acquistion_function_name == "upper_confidence_bound":
                self.acquisition_function = self.acquisition_function_class(
                    model=self.model,
                    beta=beta,
                    maximize=self.maximisation
                )
            else:
                self.acquisition_function = self.acquisition_function_class(
                    model=self.model,
                    best_f=self.current_best,
                    maximize=self.maximisation
                )
            
            # Get the next point
            new_x = self.optimize_acqf_and_get_observation()
            
            # Store the selected point for visualization
            self.next_point_tensor = new_x.clone()
            
            # Visualize after we know the next point to evaluate
            self.visualize_iteration()
            
            # Evaluate function - ensure we have a flat vector
            if new_x.dim() > 2:
                new_x = new_x.squeeze(1)  # Remove extra dimension if present
            
            new_x_numpy = new_x.detach().squeeze().numpy()
            
            # Handle both scalar and vector inputs correctly
            if new_x_numpy.ndim == 0:  # Handle scalar case
                new_x_numpy = float(new_x_numpy)
                new_f_eval = float(problem(new_x_numpy))
            else:  # Handle vector case
                new_f_eval = float(problem(new_x_numpy))  # Ensure result is a float
            
            # Append evaluations - ensure consistent data types
            self.x_evals.append(new_x_numpy)
            self.f_evals.append(new_f_eval)
            self.number_of_function_evaluations += 1
            
            # Force complete model reinitialization with all data points instead of in-place updates
            self._initialize_model()
            
            # Assign new best
            self.assign_new_best()
            
            # Print progress if verbose
            if self.verbose:
                print(f"Iteration {cur_iteration+1}/{self.budget - self.n_DoE}: Best value = {self.current_best}")
        
        print("Optimization Process finalized!")
    
    def _plot_true_function(self, ax, x_range=None):
        """Plot the true function."""
        if x_range is None:
            x_range = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
        
        y_true = np.array([f(x_i) for x_i in x_range])
        ax.plot(x_range, y_true, self.plot_colors['true_function'], label='True function')
        return x_range, y_true
    
    def _plot_tabpfn_predictions(self, ax, x_range):
        """Plot TabPFN predictions with uncertainty bands."""
        # Create tensor input for TabPFN
        X_pred = torch.tensor(x_range.reshape(-1, 1), dtype=torch.float64)
        
        # Get predictions from TabPFN - use cached predictions if available
        with torch.no_grad():
            # Cache key based on model state and input
            if hasattr(self, '_prediction_cache'):
                # Check if predictions are already cached for this x_range and model state
                current_data_size = len(self.x_evals)
                cache_key = (current_data_size, x_range.shape[0], x_range[0], x_range[-1])
                
                if cache_key in self._prediction_cache and self.iteration_count > 0:
                    # Reuse cached predictions if data size hasn't changed
                    print(f"Using cached predictions for iteration {self.iteration_count}")
                    mean = self._prediction_cache[cache_key]['mean']
                    quantiles = self._prediction_cache[cache_key]['quantiles']
                    posterior = None  # Not needed when using cache
                else:
                    # The posterior method in TabPFNSurrogateModel returns MVN and gives us mean/covariance
                    posterior = self.model.posterior(X_pred)
                    mean = posterior.mean.squeeze(-1).numpy()
                    
                    # We need to get quantiles for confidence bands - directly from the model
                    predictions = self.model.model.predict(X_pred.cpu().numpy(), output_type="main")
                    quantiles = predictions["quantiles"]
                    
                    # Cache the predictions
                    self._prediction_cache[cache_key] = {
                        'mean': mean,
                        'quantiles': quantiles
                    }
            else:
                # Initialize cache on first run
                self._prediction_cache = {}
                
                # The posterior method in TabPFNSurrogateModel returns MVN and gives us mean/covariance
                posterior = self.model.posterior(X_pred)
                mean = posterior.mean.squeeze(-1).numpy()
                
                # We need to get quantiles for confidence bands - directly from the model
                predictions = self.model.model.predict(X_pred.cpu().numpy(), output_type="main")
                quantiles = predictions["quantiles"]
                
                # Cache the predictions
                cache_key = (len(self.x_evals), x_range.shape[0], x_range[0], x_range[-1])
                self._prediction_cache[cache_key] = {
                    'mean': mean,
                    'quantiles': quantiles
                }
            
            # Get median (50th percentile) from quantiles
            median = quantiles[4]  # Index 4 corresponds to 50th percentile
        
        # Plot mean prediction
        ax.plot(x_range, mean, color=self.plot_colors['tabpfn_mean'], label='TabPFN mean')
        
        # Plot median prediction
        ax.plot(x_range, median, color='green', linestyle='--', label='TabPFN median')
        
        # Plot confidence bands using quantiles
        for (lower_idx, upper_idx), alpha in zip(self.quantile_pairs, self.confidence_alphas):
            # Get lower and upper quantiles for this band
            q_low = quantiles[lower_idx] 
            q_high = quantiles[upper_idx]
            
            # Plot the confidence band
            ax.fill_between(
                x_range, 
                q_low, 
                q_high, 
                alpha=alpha, 
                color=self.plot_colors['confidence'], 
                label=f'Quantile {lower_idx+1}0-{upper_idx+1}0%' if lower_idx == 0 else None
            )
        
        return mean, quantiles
    
    def _plot_evaluated_points(self, ax, highlight_best=True, highlight_next=None):
        """Plot all evaluated points and optionally highlight the best and/or next point."""
        # Convert points to arrays with consistent shapes
        if len(self.x_evals) > 0:
            # Ensure x_evals are properly shaped
            x_eval = np.array([float(x) if np.isscalar(x) else float(x[0]) for x in self.x_evals])
            # Ensure f_evals are properly shaped
            y_eval = np.array([float(y) for y in self.f_evals])
            
            # Plot all points
            ax.scatter(
                x_eval, y_eval, 
                c=self.plot_colors['points'], 
                marker=self.plot_markers['points'], 
                label='Evaluated points'
            )
            
            # Highlight the best point
            if highlight_best and len(self.x_evals) > 0:
                best_idx = np.argmin(y_eval) if not self.maximisation else np.argmax(y_eval)
                ax.scatter(
                    x_eval[best_idx], y_eval[best_idx], 
                    c=self.plot_colors['best_point'], 
                    marker=self.plot_markers['best_point'], 
                    s=150, 
                    label='Best point'
                )
            
            # Highlight the next point (if provided)
            if highlight_next is not None:
                next_x, next_y = highlight_next
                # Ensure next point values are scalars
                next_x = float(next_x) if np.isscalar(next_x) else float(next_x[0])
                next_y = float(next_y)
                ax.scatter(
                    next_x, next_y, 
                    c=self.plot_colors['next_point'], 
                    marker=self.plot_markers['next_point'], 
                    s=200, 
                    label='Next point'
                )
    
    def _save_plot(self, fig, filename):
        """Save the plot to a file if save_plots is enabled."""
        if self.save_plots:
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=300)
        plt.close(fig)
    
    def visualize_initial_model(self):
        """Visualize the initial surrogate model with the DoE points."""
        # Create a figure for the initial state
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot the true function
        self._plot_true_function(ax)
        
        # Plot the initial points
        self._plot_evaluated_points(ax, highlight_best=True, highlight_next=None)
        
        # If we already have some initial points, we can plot the initial model
        if len(self.x_evals) >= 2:  # Need at least 2 points for a TabPFN model
            # Generate x points for prediction
            x_pred = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
            
            # Plot TabPFN predictions
            self._plot_tabpfn_predictions(ax, x_pred)
        
        # Set title and labels
        ax.set_title(f'Initial Design of Experiments (DoE), n={len(self.x_evals)}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True)
        
        self._save_plot(fig, 'initial_doe.png')
    
    def visualize_iteration(self):
        """Visualize the current surrogate model and acquisition function."""
        # Create figure with two subplots: surrogate model and acquisition function
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # 1. Plot the surrogate model (sausage plot)
        # -----------------------------------------
        
        # Generate x points for prediction
        x_pred = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
        
        # Plot the true function
        self._plot_true_function(ax1)
        
        # Plot TabPFN predictions
        mean, quantiles = self._plot_tabpfn_predictions(ax1, x_pred)
        
        # 2. Calculate acquisition function values
        # ----------------------------------------------------------------
        # Evaluate the acquisition function at each x point - similar to BO_GP.py
        acq_values = []
        for x in x_pred:
            # Reshape for single point evaluation
            x_tensor = torch.tensor([[x]], dtype=torch.float64)
            with torch.no_grad():
                acq_val = self.acquisition_function(x_tensor)
            acq_values.append(acq_val.item())
        
        # Find the maximum of the acquisition function based on our grid evaluation
        max_idx = np.argmax(acq_values)
        grid_max_x = x_pred[max_idx]
        
        # Get the actual next point we'll evaluate (if available)
        if self.next_point_tensor is not None:
            # Convert the next point tensor to a scalar
            next_x_np = self.next_point_tensor.detach().squeeze().cpu().numpy()
            
            # Handle different array dimensions properly
            if np.isscalar(next_x_np) or next_x_np.ndim == 0:
                next_x = float(next_x_np)
            else:
                next_x = float(next_x_np[0])
                
            # Get median prediction for this point
            closest_idx = np.abs(x_pred - next_x).argmin()
            next_median = quantiles[4][closest_idx]  # Index 4 corresponds to 50th percentile
        else:
            # If no next point is provided, use the grid maximum
            next_x = grid_max_x
            next_median = quantiles[4][max_idx]
        
        # Plot all evaluated points and highlight the best and next points
        self._plot_evaluated_points(ax1, highlight_best=True, highlight_next=(next_x, next_median))
        
        # Set title and labels for the surrogate plot
        ax1.set_title(f'Iteration {self.iteration_count}: TabPFN Surrogate Model')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.legend(loc='best')
        ax1.grid(True)
        
        # 3. Plot the acquisition function
        # --------------------------------
        
        # Create a second y-axis for a different scale view
        ax3 = ax2.twinx()
        
        # For the second y-axis, clip extreme values for better visualization
        acq_array = np.array(acq_values)
        clipped_acq = np.clip(acq_array, 
                             np.percentile(acq_array, 5),  # 5th percentile
                             np.percentile(acq_array, 95))  # 95th percentile
        ax3.plot(x_pred, clipped_acq, self.plot_colors['acquisition'], linewidth=2, label='Acquisition function')
        
        # If we have an actual next point from the optimizer, mark it
        if self.next_point_tensor is not None:
            # Find the closest grid point for a clean visualization
            closest_idx = np.abs(x_pred - next_x).argmin()
            ax3.scatter(
                next_x, clipped_acq[closest_idx], 
                c=self.plot_colors['next_point'], 
                marker=self.plot_markers['next_point'], 
                s=200, 
                label='Next point'
            )
        else:
            # Just mark the grid maximum
            ax3.scatter(
                x_pred[max_idx], clipped_acq[max_idx], 
                c=self.plot_colors['next_point'], 
                marker=self.plot_markers['next_point'], 
                s=200, 
                label='Next point'
            )
        
        ax3.set_ylabel('Acquisition value', color='g')
        ax3.tick_params(axis='y', colors='g')
        ax3.legend(loc='lower right')
        
        ax2.set_title(f'Acquisition Function ({self.acquistion_function_name})')
        ax2.set_xlabel('x')
        ax2.get_yaxis().set_visible(False)  # Hide the left y-axis
        ax2.grid(True)
        
        plt.tight_layout()
        
        self._save_plot(fig, f'iteration_{self.iteration_count:03d}.png')
    
    def visualize_final_result(self):
        """Visualize the final surrogate model."""
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Generate x points for prediction
        x_pred = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
        
        # Plot the true function
        self._plot_true_function(ax)
        
        # Plot TabPFN predictions
        self._plot_tabpfn_predictions(ax, x_pred)
        
        # Plot all evaluated points and highlight the best point
        self._plot_evaluated_points(ax, highlight_best=True)
        
        # Set title and labels
        ax.set_title(f'Final TabPFN Surrogate Model after {self.budget} evaluations')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend(loc='best')
        ax.grid(True)
        plt.tight_layout()
        
        self._save_plot(fig, 'final_model.png')

def run_1d_tabpfn_visualization_bo(acquisition_function="expected_improvement", budget=15, 
                                  n_DoE=3, bounds=None, save_plots=True, n_estimators=16):
    """
    Run Bayesian optimization with TabPFN visualization on a 1D test function.
    
    Args:
        acquisition_function: Which acquisition function to use
        budget: Total function evaluation budget
        n_DoE: Number of initial points
        bounds: Problem bounds (default: [-6.0, 6.0])
        save_plots: Whether to save plots to disk
        n_estimators: Number of TabPFN estimators
    """
    # Set default bounds
    if bounds is None:
        bounds = np.array([[-6.0, 6.0]])  # Wider bounds for showing more of the function
    
    print(f"Starting TabPFN Bayesian Optimization visualization with {acquisition_function}")
    
    try:
        # Create directory for plots
        plots_dir = f"tabpfn_bo_visualizations/{acquisition_function}"
        
        # Set up the TabPFN-based Bayesian optimizer with visualization
        optimizer = TabPFNVisualizationBO(
            budget=budget,
            n_DoE=n_DoE,
            acquisition_function=acquisition_function,
            random_seed=42,
            verbose=True,
            maximisation=False,
            save_plots=save_plots,
            plots_dir=plots_dir,
            n_estimators=n_estimators,
            device="cpu"  # Force CPU
        )
        
        # Run optimization with visualization
        optimizer(
            problem=f,
            bounds=bounds
        )
        
        print(f"Optimization completed successfully.")
        print(f"Best point found: x = {float(optimizer.x_evals[optimizer.current_best_index]) if np.isscalar(optimizer.x_evals[optimizer.current_best_index]) else float(optimizer.x_evals[optimizer.current_best_index][0]):.4f}, f(x) = {float(optimizer.current_best):.4f}")
        
    except Exception as e:
        print(f"ERROR: Run failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run with different acquisition functions
    for acq_func in ["expected_improvement"]:
        # For single visualization run
        run_1d_tabpfn_visualization_bo(
            acquisition_function=acq_func,
            budget=25,
            n_DoE=4,
            n_estimators=8
        )