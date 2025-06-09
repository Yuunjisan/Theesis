"""
This script visualizes the Bayesian optimization process for 1D functions,
showing both the surrogate model (sausage plots) and acquisition function.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import os
from pathlib import Path
import pandas as pd

# BoTorch imports
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel

# Import from Algorithms
from Algorithms import Vanilla_BO

# Define a simple 1D test function
def f(x):
    """Simple 1D test function: f(x) = x * np.cos(x) + 10"""
    if isinstance(x, np.ndarray):
        if x.ndim > 1:
            x = x.ravel()
    return x**2 * np.sin(x) + 5

class VisualizationBO(Vanilla_BO):
    """
    Extension of Vanilla_BO that adds visualization capabilities for the surrogate model
    and acquisition function during optimization.
    """
    def __init__(self, budget, n_DoE=0, acquisition_function="expected_improvement", 
                 random_seed=42, save_plots=True, plots_dir="bo_visualizations", **kwargs):
        # Force CPU usage
        kwargs['device'] = "cpu"
        super().__init__(budget, n_DoE, acquisition_function, random_seed, **kwargs)
        
        # Visualization settings
        self.save_plots = save_plots
        self.plots_dir = plots_dir
        self.iteration_count = 0
        
        # Plot style settings
        self.plot_colors = {
            'true_function': 'r--',
            'gp_mean': 'darkorange',
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
        
        self.confidence_alphas = [0.05, 0.1, 0.15, 0.2]
        self.confidence_stds = [4, 3, 2, 1]
        
        # Create plots directory
        if self.save_plots:
            os.makedirs(self.plots_dir, exist_ok=True)
    
    def __call__(self, problem, bounds=None, **kwargs):
        """
        Run Bayesian optimization with visualization at each iteration.
        For 1D problems with visualization of the surrogate model and acquisition function.
        """
        # Initialize
        self._initialize(problem, bounds, **kwargs)
        
        # Visualize the initial surrogate model
        self.visualize_initial_model()
        
        # Run the optimization loop
        self._run_optimization_loop(problem, **kwargs)
        
        # Final visualization
        self.visualize_final_result()
    
    def _initialize(self, problem, bounds, **kwargs):
        """Initialize the optimization process"""
        # Set default bounds if not provided
        if bounds is None:
            bounds = np.array([[-6.0, 6.0]])  # Wider bounds for showing more of the function
        
        # Force 1D
        dim = 1
        
        # Call the superclass to run the initial sampling of the problem
        super(Vanilla_BO, self).__call__(problem, dim, bounds, **kwargs)
        
        # Run the model initialization
        self._initialise_model(**kwargs)
    
    def _run_optimization_loop(self, problem, **kwargs):
        """Run the main optimization loop with visualization at each step"""
        for cur_iteration in range(self.budget - self.n_DoE):
            self.iteration_count = cur_iteration + 1
            
            # Set up the acquisition function
            self._setup_acquisition_function()
            
            # Visualize the surrogate model and acquisition function before selecting the next point
            self.visualize_iteration()
            
            # Get and evaluate the next point
            self._evaluate_next_point(problem)
            
            # Print progress if verbose
            if self.verbose:
                print(f"Current Iteration: {cur_iteration + 1}",
                      f"Current Best: x: {self.x_evals[self.current_best_index]} y: {self.current_best}",
                      flush=True)
            
            # Re-fit the model
            self._initialise_model()
        
        print("Optimization Process finalized!")
    
    def _setup_acquisition_function(self):
        """Set up the acquisition function for the current iteration"""
        self.acquisition_function = ExpectedImprovement(
            model=self._Vanilla_BO__model_obj, 
            best_f=self.current_best, 
            maximize=self.maximisation
        )
    
    def _evaluate_next_point(self, problem):
        """Get and evaluate the next point"""
        new_x = self.optimize_acqf_and_get_observation()
        
        # Append the new values
        for _, new_x_arr in enumerate(new_x):
            new_x_arr_numpy = new_x_arr.detach().numpy().ravel()
            
            # Append the new value
            self.x_evals.append(new_x_arr_numpy)
            
            # Evaluate the function
            new_f_eval = problem(new_x_arr_numpy)
            
            # Append the function evaluation
            self.f_evals.append(new_f_eval)
            
            # Increment the number of function evaluations
            self.number_of_function_evaluations += 1
        
        # Assign the new best
        self.assign_new_best()
    
    def _plot_true_function(self, ax, x_range=None):
        """Plot the true function"""
        if x_range is None:
            x_range = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
        
        y_true = np.array([f(x_i) for x_i in x_range])
        ax.plot(x_range, y_true, self.plot_colors['true_function'], label='True function')
        return x_range, y_true
    
    def _plot_confidence_bands(self, ax, x_pred, mean, std):
        """Plot confidence bands around the mean prediction"""
        for std_factor, alpha in zip(self.confidence_stds, self.confidence_alphas):
            ax.fill_between(
                x_pred, 
                mean - std_factor * std,
                mean + std_factor * std, 
                alpha=alpha, color=self.plot_colors['confidence'], 
                label=f'μ ± {std_factor}σ' if std_factor == self.confidence_stds[0] else None
            )
    
    def _plot_evaluated_points(self, ax, highlight_best=True, highlight_next=None):
        """Plot all evaluated points and optionally highlight the best and/or next point"""
        x_eval = np.array([x[0] for x in self.x_evals])
        y_eval = np.array(self.f_evals)
        
        # Plot all points
        ax.scatter(
            x_eval, y_eval, 
            c=self.plot_colors['points'], 
            marker=self.plot_markers['points'], 
            label='Evaluated points'
        )
        
        # Highlight the best point
        if highlight_best:
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
            ax.scatter(
                next_x, next_y, 
                c=self.plot_colors['next_point'], 
                marker=self.plot_markers['next_point'], 
                s=200, 
                label='Next point'
            )
    
    def _save_plot(self, fig, filename):
        """Save the plot to a file if save_plots is enabled"""
        if self.save_plots:
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=300)
        plt.close(fig)
    
    def visualize_initial_model(self):
        """Visualize the initial surrogate model with the DoE points"""
        # Create a figure for the initial state
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot the true function
        self._plot_true_function(ax)
        
        # Plot the initial points
        self._plot_evaluated_points(ax, highlight_best=True, highlight_next=None)
        
        # Set title and labels
        ax.set_title(f'Initial Design of Experiments (DoE), n={len(self.x_evals)}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True)
        
        self._save_plot(fig, 'initial_doe.png')
    
    def visualize_iteration(self):
        """Visualize the current surrogate model and acquisition function"""
        # Extract the GP model
        model = self._Vanilla_BO__model_obj
        
        # Create figure with two subplots: surrogate model and acquisition function
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # 1. Plot the surrogate model (sausage plot)
        # -----------------------------------------
        
        # Generate x points for prediction
        x_pred = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
        X_pred = torch.tensor(x_pred).view(-1, 1).double()
        
        # Get predictions from the model
        with torch.no_grad():
            posterior = model.posterior(X_pred)
            mean = posterior.mean.numpy().flatten()
            lower, upper = posterior.mvn.confidence_region()
            lower = lower.numpy().flatten()
            upper = upper.numpy().flatten()
        
        # Plot the true function
        self._plot_true_function(ax1)
        
        # Plot mean prediction
        ax1.plot(x_pred, mean, color=self.plot_colors['gp_mean'], label='GP mean')
        
        # Plot confidence intervals
        std = (upper - lower) / 4  # Approximate standard deviation
        self._plot_confidence_bands(ax1, x_pred, mean, std)
        
        # 2. Calculate acquisition function values to find the next point
        # ----------------------------------------------------------------
        acq_values = []
        for x in X_pred:
            x_tensor = x.reshape(1, 1)
            with torch.no_grad():
                acq_val = self.acquisition_function(x_tensor)
            acq_values.append(acq_val.item())
            
        # Find the maximum of the acquisition function - this is the next point
        max_idx = np.argmax(acq_values)
        next_x = x_pred[max_idx]
        
        # Predict the function value at the next point
        next_x_tensor = torch.tensor([[next_x]]).double()
        with torch.no_grad():
            next_mean = model.posterior(next_x_tensor).mean.item()
            
        # Plot all evaluated points and highlight the best and next points
        self._plot_evaluated_points(ax1, highlight_best=True, highlight_next=(next_x, next_mean))
        
        # Set title and labels for the surrogate plot
        ax1.set_title(f'Iteration {self.iteration_count}: Surrogate Model (GP)')
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
        
        # Add the "Next point" marker on the same axis as the green line
        ax3.scatter(
            x_pred[max_idx], clipped_acq[max_idx], 
            c=self.plot_colors['next_point'], 
            marker=self.plot_markers['next_point'], 
            s=200, 
            label='Next point'
        )
        
        ax3.set_ylabel('Acquisition value', color='g')
        ax3.tick_params(axis='y', colors='g')
        # Move legend to lower right corner to avoid covering peaks in the acquisition function
        ax3.legend(loc='lower right')
        
        ax2.set_title('Acquisition Function')
        ax2.set_xlabel('x')
        # Since we removed the lines from ax2, don't label its y-axis
        ax2.get_yaxis().set_visible(False)  # Hide the left y-axis
        # No longer need ax2's legend since we moved the star to ax3
        ax2.grid(True)
        
        plt.tight_layout()
        
        self._save_plot(fig, f'iteration_{self.iteration_count:03d}.png')
    
    def visualize_final_result(self):
        """Visualize the final surrogate model"""
        # Extract the GP model
        model = self._Vanilla_BO__model_obj
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Generate x points for prediction
        x_pred = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
        X_pred = torch.tensor(x_pred).view(-1, 1).double()
        
        # Get predictions from the model
        with torch.no_grad():
            posterior = model.posterior(X_pred)
            mean = posterior.mean.numpy().flatten()
            lower, upper = posterior.mvn.confidence_region()
            lower = lower.numpy().flatten()
            upper = upper.numpy().flatten()
        
        # Plot the true function
        self._plot_true_function(ax)
        
        # Plot mean prediction
        ax.plot(x_pred, mean, color=self.plot_colors['gp_mean'], label='GP mean')
        
        # Plot confidence intervals
        std = (upper - lower) / 4  # Approximate standard deviation
        self._plot_confidence_bands(ax, x_pred, mean, std)
        
        # Plot all evaluated points and highlight the best point
        self._plot_evaluated_points(ax, highlight_best=True)
        
        # Set title and labels
        ax.set_title(f'Final Surrogate Model after {self.budget} evaluations')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend(loc='best')
        ax.grid(True)
        plt.tight_layout()
        
        self._save_plot(fig, 'final_model.png')

def run_1d_visualization_bo(acquisition_function="expected_improvement", budget=15, 
                          n_DoE=3, bounds=None, save_plots=True):
    """
    Run Bayesian optimization with visualization on a 1D test function.
    
    Args:
        acquisition_function: Which acquisition function to use
        budget: Total function evaluation budget
        n_DoE: Number of initial points
        bounds: Problem bounds (default: [-10.0, 10.0])
        save_plots: Whether to save plots to disk
    """
    # Set default bounds
    if bounds is None:
        bounds = np.array([[-6.0, 6.0]])  # Wider bounds for showing more of the function
    
    print(f"Starting Bayesian Optimization visualization with {acquisition_function}")
    
    try:
        # Create directory for plots
        plots_dir = f"bo_visualizations/{acquisition_function}"
        
        # Set up the Bayesian optimizer with visualization
        optimizer = VisualizationBO(
            budget=budget,
            n_DoE=n_DoE,
            acquisition_function=acquisition_function,
            random_seed=42,
            verbose=True,
            maximisation=False,
            save_plots=save_plots,
            plots_dir=plots_dir
        )
        
        # Run optimization with visualization
        optimizer(
            problem=f,
            bounds=bounds
        )
        
        print(f"Optimization completed successfully.")
        print(f"Best point found: x = {float(optimizer.x_evals[optimizer.current_best_index][0]):.4f}, f(x) = {float(optimizer.current_best):.4f}")
        
    except Exception as e:
        print(f"ERROR: Run failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run with different acquisition functions
    for acq_func in ["expected_improvement"]:
        # For single visualization run
        run_1d_visualization_bo(
            acquisition_function=acq_func,
            budget=10,
            n_DoE=3
        ) 