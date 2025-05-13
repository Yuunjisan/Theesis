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
    return x * np.cos(x) + 10

class VisualizationBO(Vanilla_BO):
    """
    Extension of Vanilla_BO that adds visualization capabilities for the surrogate model
    and acquisition function during optimization.
    """
    def __init__(self, budget, n_DoE=0, acquisition_function="expected_improvement", 
                 random_seed=43, save_plots=True, plots_dir="bo_visualizations", **kwargs):
        super().__init__(budget, n_DoE, acquisition_function, random_seed, **kwargs)
        
        # Visualization settings
        self.save_plots = save_plots
        self.plots_dir = plots_dir
        self.iteration_count = 0
        
        # Create plots directory
        if self.save_plots:
            os.makedirs(self.plots_dir, exist_ok=True)
    
    def __call__(self, problem, bounds=None, **kwargs):
        """
        Run Bayesian optimization with visualization at each iteration.
        For 1D problems with visualization of the surrogate model and acquisition function.
        """
        # Set default bounds if not provided
        if bounds is None:
            bounds = np.array([[-5.0, 5.0]])  # Wider bounds for showing more of the function
        
        # Force 1D
        dim = 1
        
        # Call the superclass to run the initial sampling of the problem
        super(Vanilla_BO, self).__call__(problem, dim, bounds, **kwargs)
        
        # Visualize the initial surrogate model
        self.visualize_initial_model()
        
        # Get a default beta (for UCB)
        beta = kwargs.pop("beta", 0.2)
        
        # Run the model initialization
        self._initialise_model(**kwargs)
        
        # Start the optimization loop
        for cur_iteration in range(self.budget - self.n_DoE):
            self.iteration_count = cur_iteration + 1
            
            # Set up the acquisition function based on its type
            if self.acquistion_function_name == "upper_confidence_bound":
                # UCB uses beta but not best_f
                self.acquisition_function = self.acquisition_function_class(
                    model=self._Vanilla_BO__model_obj,
                    beta=beta,
                    maximize=self.maximisation
                )
            else:
                # EI and PI use best_f but not beta
                self.acquisition_function = self.acquisition_function_class(
                    model=self._Vanilla_BO__model_obj,
                    best_f=self.current_best,
                    maximize=self.maximisation
                )
            
            # Visualize the surrogate model and acquisition function before selecting the next point
            self.visualize_iteration()
            
            # Get the next point to evaluate
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
            
            # Print best to screen if verbose
            if self.verbose:
                print(f"Current Iteration: {cur_iteration + 1}",
                       f"Current Best: x: {self.x_evals[self.current_best_index]} y: {self.current_best}",
                      flush=True)
            
            # Re-fit the GPR
            self._initialise_model()
        
        print("Optimization Process finalized!")
        
        # Final visualization
        self.visualize_final_result()
    
    def visualize_initial_model(self):
        """Visualize the initial surrogate model with the DoE points"""
        # Create a figure for the initial state
        plt.figure(figsize=(10, 6))
        
        # Plot the true function over the full range
        x_true = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 500)
        y_true = np.array([f(x_i) for x_i in x_true])
        plt.plot(x_true, y_true, 'r--', label='True function')
        
        # Plot the initial points
        x_init = np.array([x[0] for x in self.x_evals])
        y_init = np.array(self.f_evals)
        plt.scatter(x_init, y_init, c='navy', marker='o', label='Initial points')
        
        plt.title(f'Initial Design of Experiments (DoE), n={len(self.x_evals)}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plots_dir, 'initial_doe.png'), dpi=300)
        plt.close()
    
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
        x_true = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 500)
        y_true = np.array([f(x_i) for x_i in x_true])
        ax1.plot(x_true, y_true, 'r--', label='True function')
        
        # Plot mean prediction
        ax1.plot(x_pred, mean, color='darkorange', label='GP mean')
        
        # Plot confidence intervals
        std = (upper - lower) / 4  # Approximate standard deviation
        
        # Plot multiple confidence intervals with decreasing opacity
        alphas = [0.05, 0.1, 0.15, 0.2]
        stds = [4, 3, 2, 1]
        
        for std_factor, alpha in zip(stds, alphas):
            ax1.fill_between(
                x_pred, 
                mean - std_factor * std,
                mean + std_factor * std, 
                alpha=alpha, color='purple', 
                label=f'μ ± {std_factor}σ' if std_factor == stds[0] else None
            )
        
        # Plot all evaluated points
        x_eval = np.array([x[0] for x in self.x_evals])
        y_eval = np.array(self.f_evals)
        ax1.scatter(x_eval, y_eval, c='navy', marker='o', label='Evaluated points')
        
        # Highlight the most recent point
        if len(self.x_evals) > self.n_DoE:
            ax1.scatter(
                x_eval[-1], y_eval[-1], 
                c='green', marker='*', s=200, 
                label='Latest point'
            )
        
        # Highlight the best point
        best_idx = np.argmin(y_eval) if not self.maximisation else np.argmax(y_eval)
        ax1.scatter(
            x_eval[best_idx], y_eval[best_idx], 
            c='red', marker='x', s=150, 
            label='Best point'
        )
        
        ax1.set_title(f'Iteration {self.iteration_count}: Surrogate Model (GP)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.legend(loc='best')
        ax1.grid(True)
        
        # 2. Plot the acquisition function
        # --------------------------------
        
        # Generate acquisition function values
        acq_values = []
        for x in X_pred:
            x_tensor = x.reshape(1, 1)
            with torch.no_grad():
                acq_val = self.acquisition_function(x_tensor)
            acq_values.append(acq_val.item())
        
        # Plot acquisition function
        ax2.plot(x_pred, acq_values, 'b-', label=f'Acquisition ({self.acquistion_function_name.replace("_", " ").title()})')
        
        # Find and mark the maximum of the acquisition function
        max_idx = np.argmax(acq_values)
        ax2.scatter(
            x_pred[max_idx], acq_values[max_idx], 
            c='green', marker='*', s=200, 
            label='Next point'
        )
        
        ax2.set_title('Acquisition Function')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Acquisition Value')
        ax2.legend(loc='best')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plots_dir, f'iteration_{self.iteration_count:03d}.png'), dpi=300)
        plt.close()
    
    def visualize_final_result(self):
        """Visualize the final surrogate model"""
        # Extract the GP model
        model = self._Vanilla_BO__model_obj
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
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
        x_true = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 500)
        y_true = np.array([f(x_i) for x_i in x_true])
        plt.plot(x_true, y_true, 'r--', label='True function')
        
        # Plot mean prediction
        plt.plot(x_pred, mean, color='darkorange', label='GP mean')
        
        # Plot confidence intervals
        std = (upper - lower) / 4  # Approximate standard deviation
        
        # Plot multiple confidence intervals with decreasing opacity
        alphas = [0.05, 0.1, 0.15, 0.2]
        stds = [4, 3, 2, 1]
        
        for std_factor, alpha in zip(stds, alphas):
            plt.fill_between(
                x_pred, 
                mean - std_factor * std,
                mean + std_factor * std, 
                alpha=alpha, color='purple', 
                label=f'μ ± {std_factor}σ' if std_factor == stds[0] else None
            )
        
        # Plot all evaluated points
        x_eval = np.array([x[0] for x in self.x_evals])
        y_eval = np.array(self.f_evals)
        plt.scatter(x_eval, y_eval, c='navy', marker='o', label='Evaluated points')
        
        # Highlight the best point
        best_idx = np.argmin(y_eval) if not self.maximisation else np.argmax(y_eval)
        plt.scatter(
            x_eval[best_idx], y_eval[best_idx], 
            c='red', marker='x', s=150, 
            label='Best point'
        )
        
        plt.title(f'Final Surrogate Model after {self.budget} evaluations')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plots_dir, 'final_model.png'), dpi=300)
        plt.close()

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
        bounds = np.array([[-5.0, 5.0]])  # Wider bounds for showing more of the function
    
    print(f"Starting Bayesian Optimization visualization with {acquisition_function}")
    
    try:
        # Create directory for plots
        plots_dir = f"bo_visualizations/{acquisition_function}"
        
        # Set up the Bayesian optimizer with visualization
        optimizer = VisualizationBO(
            budget=budget,
            n_DoE=n_DoE,
            acquisition_function=acquisition_function,
            random_seed=43,
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
    for acq_func in ["expected_improvement", "probability_of_improvement", "upper_confidence_bound"]:
        run_1d_visualization_bo(
            acquisition_function=acq_func,
            budget=15,
            n_DoE=6
        ) 