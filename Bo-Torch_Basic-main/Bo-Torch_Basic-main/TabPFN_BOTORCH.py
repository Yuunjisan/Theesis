import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from typing import Optional, List, Union, Any, Dict, Tuple
from botorch.models.model import Model
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement, LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.posteriors import Posterior
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor
from torch.distributions import MultivariateNormal
from tabpfn import TabPFNRegressor
from pyDOE import lhs
from scipy import stats

class TabPFNModel(Model):
    """
    A BoTorch-compatible wrapper for the TabPFN model.
    """
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        n_estimators: int = 16,
        use_gaussian: bool = True,
    ):
        super().__init__()
        self.train_X = train_X
        self.train_Y = train_Y
        self.n_estimators = n_estimators
        self.use_gaussian = use_gaussian
        
        # Calculate normalization constants for X to prevent overflow
        self.X_mean = train_X.mean(dim=0, keepdim=True)
        self.X_std = train_X.std(dim=0, keepdim=True).clamp(min=1e-8)  # Prevent division by zero
        
        # Calculate normalization constants for Y
        self.Y_mean = train_Y.mean(dim=0, keepdim=True)
        self.Y_std = train_Y.std(dim=0, keepdim=True).clamp(min=1e-8)  # Prevent division by zero
        
        # Convert to numpy for TabPFN
        X_np = self._normalize_X(train_X).numpy()
        y_np = self._normalize_Y(train_Y).numpy()
        
        # Initialize and fit TabPFN
        self.model = TabPFNRegressor(n_estimators=n_estimators, softmax_temperature=1)
        self.model.fit(X_np, y_np)
        
    def _normalize_X(self, X):
        """Normalize X values to have zero mean and unit variance."""
        return (X - self.X_mean) / self.X_std
    
    def _normalize_Y(self, Y):
        """Normalize Y values to have zero mean and unit variance."""
        return (Y - self.Y_mean) / self.Y_std
    
    def _unnormalize_Y(self, Y):
        """Unnormalize Y values."""
        return Y * self.Y_std + self.Y_mean
        
    @property
    def batch_shape(self):
        return torch.Size([])  #lets BoTorch know it cannot process parallel.
        
    @property
    def num_outputs(self):
        return 1  #Indicating doing single output regression.
        
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Any] = None,
        return_tabpfn_quantiles: bool = False,
        **kwargs: Any,
    ) -> Union[Posterior, Tuple[Posterior, Dict]]:
        #convert to numpy and reshape for TabPFN
        #X shape is [batch_shape, q=1, d=1]
        #We need to reshape to [batch_shape, d]
        X_np = X.squeeze(1).detach().numpy()  # Remove q dimension and detach
        
        #reshape to 2D array for TabPFN
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
            
        # Normalize X input to prevent overflow
        X_tensor = torch.tensor(X_np, dtype=self.train_X.dtype, device=self.train_X.device)
        X_normalized = self._normalize_X(X_tensor)
        X_np = X_normalized.numpy()
        
        try:
            #get predictions from TabPFN
            predictions = self.model.predict(X_np, output_type="main")
            mean = predictions["mean"]
            quantiles = predictions["quantiles"]
            
            # Unnormalize the predictions to the original scale
            mean = self._unnormalize_Y(torch.tensor(mean, dtype=X.dtype, device=X.device)).numpy().ravel()
            unnormalized_quantiles = []
            for q in quantiles:
                unnormalized_q = self._unnormalize_Y(torch.tensor(q, dtype=X.dtype, device=X.device)).numpy().ravel()
                unnormalized_quantiles.append(unnormalized_q)
            
            # Store raw TabPFN outputs for visualization
            tabpfn_outputs = {
                "mean": mean,
                "quantiles": unnormalized_quantiles
            }
        except Exception as e:
            # Gracefully handle any remaining errors
            print(f"Error in TabPFN prediction: {e}")
            # Provide fallback predictions to prevent crashes
            batch_size = X_np.shape[0]
            mean = np.zeros(batch_size)
            fallback_quantiles = [np.zeros(batch_size) for _ in range(9)]
            tabpfn_outputs = {
                "mean": mean,
                "quantiles": fallback_quantiles
            }
        
        #Calculate standard deviation from quantiles
        #TabPFN returns 9 quantiles from 0.1 to 0.9
        #We'll use symmetric pairs around the median
        std_estimates = []
        quantile_pairs = [(0, 8), (1, 7), (2, 6), (3, 5)]
        alphas = [0.1, 0.2, 0.3, 0.4]  #corresponding to the quantile pairs
        
        try:
            for (lower_idx, upper_idx), alpha in zip(quantile_pairs, alphas):
                q_low = tabpfn_outputs["quantiles"][lower_idx]
                q_high = tabpfn_outputs["quantiles"][upper_idx]
                # For normal distribution: (q_high - q_low) / (2 * ppf(1-alpha))
                z_score = 2 * np.abs(stats.norm.ppf(alpha))
                std = (q_high - q_low) / z_score
                std_estimates.append(std)
            
            # Average the standard deviation estimates
            std = np.mean(std_estimates, axis=0)
            
            # Ensure std is positive and handle any numerical issues
            std = np.maximum(std, 1e-6)
        except Exception as e:
            # Fallback if there's an issue with std calculation
            print(f"Error calculating std: {e}")
            std = np.ones_like(mean) * 0.1  # Default std
        
        # Convert to tensors with correct shapes
        # mean should be [batch_shape x 1], std should be [batch_shape x 1]
        mean = torch.tensor(mean, dtype=X.dtype, device=X.device).unsqueeze(-1)
        std = torch.tensor(std, dtype=X.dtype, device=X.device).unsqueeze(-1)
        
        # Create scale_tril [batch_shape x 1 x 1]
        # We connect it to X to maintain the gradient graph
        dummy = X.sum() * 0  # Creates a zero tensor connected to X's graph
        scale_tril = (std.unsqueeze(-1) + dummy).abs()  # Ensure positive and connect to graph
        mean = mean + dummy  # Connect mean to the graph
        
        if return_tabpfn_quantiles:
            return MultivariateNormal(mean, scale_tril=scale_tril), tabpfn_outputs
        
        return MultivariateNormal(mean, scale_tril=scale_tril)
        
    def subset_output(self, idcs: List[int]) -> Model:
        if idcs != [0]:
            raise NotImplementedError("TabPFNModel only supports single output")
        return self
        
    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        **kwargs: Any,
    ) -> Model:
        # Combine new observations with existing training data
        new_train_X = torch.cat([self.train_X, X], dim=0)
        new_train_Y = torch.cat([self.train_Y, Y], dim=0)
        
        return TabPFNModel(new_train_X, new_train_Y, self.n_estimators, self.use_gaussian)

class TabPFNVisualizationBO:
    """
    Extension of TabPFNModel that adds visualization capabilities for the surrogate model
    and acquisition function during optimization.
    """
    def __init__(self, budget, n_DoE=0, acquisition_function="expected_improvement", 
                 random_seed=42, save_plots=True, plots_dir="tabpfn_bo_visualizations", 
                 show_both_distributions=True, **kwargs):
        self.budget = budget
        self.n_DoE = n_DoE
        self.acquisition_function = acquisition_function
        self.random_seed = random_seed
        self.save_plots = save_plots
        self.plots_dir = plots_dir
        self.iteration_count = 0
        self.show_both_distributions = show_both_distributions
        
        # Plot style settings
        self.plot_colors = {
            'true_function': 'r--',
            'tabpfn_mean': 'b-',
            'tabpfn_median': 'green',
            'confidence': 'purple',
            'points': 'navy',
            'next_point': 'red',
            'best_point': 'red',
            'acquisition': 'g-',
            'acquisition_log': 'g--'
        }
        
        self.plot_markers = {
            'points': 'o',
            'next_point': '*',
            'best_point': 'x'
        }
        
        self.plot_sizes = {
            'points': 30,
            'next_point': 200,
            'best_point': 200
        }
        
        self.confidence_alphas = [0.05, 0.1, 0.15, 0.2]
        self.quantile_pairs = [(0, 8), (1, 7), (2, 6), (3, 5)]
        
        # Axis limits for consistent visualization
        self.x_lim = (-6.0, 6.0)
        self.y_lim = (-40, 60)
        
        # Create plots directory
        if self.save_plots:
            os.makedirs(self.plots_dir, exist_ok=True)
            
        # Store evaluated points
        self.x_evals = []
        self.f_evals = []
        self.current_best = float('inf')
        self.current_best_index = -1
        self.number_of_function_evaluations = 0
        
        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
    def __call__(self, problem, bounds=None, **kwargs):
        """
        Run Bayesian optimization with visualization at each iteration.
        """
        # Initialize optimization
        self._initialize(problem, bounds)
        
        # Visualize initial model
        self.visualize_initial_model()
        
        # Run optimization loop
        self._run_optimization_loop(problem, **kwargs)
        
        # Final visualization
        self.visualize_final_result()
        
        # Print results
        self._print_results()
    
    def _initialize(self, problem, bounds):
        """Initialize the optimization process"""
        # Set default bounds if not provided
        if bounds is None:
            bounds = np.array([self.x_lim])
            
        # Ensure bounds are in the correct format for BoTorch (2 x d)
        bounds = torch.tensor(bounds, dtype=torch.float64).T
        
        # Store bounds
        self.bounds = bounds
        self.problem = problem
        
        print(f"Starting Bayesian Optimization visualization with {self.acquisition_function}")
        
        # Initial sampling
        self._initial_sampling(problem, bounds)
    
    def _run_optimization_loop(self, problem, **kwargs):
        """Run the main optimization loop"""
        for cur_iteration in range(self.budget - self.n_DoE):
            self.iteration_count = cur_iteration + 1
            
            # Create and fit model and acquisition function
            model = self._create_model(**kwargs)
            acq_func = self._create_acquisition_function(model, **kwargs)
            
            # Get next point to evaluate
            next_x, next_acq_value = self._get_next_point(acq_func)
            
            # Visualize current state
            self.visualize_iteration(model, acq_func, next_x, next_acq_value)
            
            # Evaluate and update
            self._evaluate_point(problem, next_x)
            
            # Print progress
            self._print_progress(cur_iteration)
    
    def _create_model(self, **kwargs):
        """Create and fit a TabPFN model with current data"""
        train_X = torch.tensor(self.x_evals, dtype=torch.float64)
        train_Y = torch.tensor(self.f_evals, dtype=torch.float64)
        return TabPFNModel(train_X, train_Y, n_estimators=kwargs.get("n_estimators", 16))
    
    def _create_acquisition_function(self, model, **kwargs):
        """Create the appropriate acquisition function"""
        if self.acquisition_function == "upper_confidence_bound":
            return UpperConfidenceBound(
                model=model,
                beta=kwargs.get("beta", 0.2),
                maximize=False
            )
        elif self.acquisition_function == "probability_of_improvement":
            return ProbabilityOfImprovement(
                model=model,
                best_f=self.current_best,
                maximize=False
            )
        else:  # expected_improvement
            return LogExpectedImprovement(
                model=model,
                best_f=self.current_best,
                maximize=False
            )
    
    def _get_next_point(self, acq_func):
        """Optimize acquisition function to get next evaluation point"""
        candidate, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
        )
        
        # Extract and return the point and its acquisition value
        next_x = candidate.detach().numpy().item()
        next_acq_value = acq_value.item()
        return next_x, next_acq_value
    
    def _evaluate_point(self, problem, next_x):
        """Evaluate problem at the given point and update best values"""
        # Convert to numpy array in correct shape
        new_x = np.array([[next_x]])
        
        # Evaluate function
        new_f = problem(new_x.ravel())
        
        # Update evaluations
        self.x_evals = np.vstack((self.x_evals, new_x))
        self.f_evals = np.append(self.f_evals, new_f)
        self.number_of_function_evaluations += 1
        
        # Update best
        if new_f < self.current_best:
            self.current_best = new_f
            self.current_best_index = len(self.f_evals) - 1
    
    def _print_progress(self, iteration):
        """Print current optimization progress"""
        print(f"Current Iteration: {iteration + 1}",
              f"Current Best: x: {self.x_evals[self.current_best_index]} y: {self.current_best}",
              flush=True)
    
    def _print_results(self):
        """Print final optimization results"""
        print("Optimization Process finalized!")
        print(f"Optimization completed successfully.")
        print(f"Best point found: x = {float(self.x_evals[self.current_best_index][0]):.4f}, f(x) = {float(self.current_best):.4f}")
    
    def _initial_sampling(self, problem, bounds):
        """Perform initial sampling using LHS with fixed seed."""
        # Set seed directly before sampling
        np.random.seed(self.random_seed)
        
        # Use static bounds for consistency
        lb, ub = self.x_lim
        
        # Generate samples with LHS
        samples = lhs(1, samples=self.n_DoE, criterion="center")
        xi = lb + (ub - lb) * samples
        
        # Evaluate function at sampled points
        fi = np.array([problem(x) for x in xi])
        
        # Store as numpy arrays with consistent shape
        self.x_evals = xi
        self.f_evals = fi
        self.current_best = min(fi)
        self.current_best_index = np.argmin(fi)
        self.number_of_function_evaluations = self.n_DoE
    
    def _save_plot(self, fig, filename):
        """Save the plot to a file if save_plots is enabled"""
        if self.save_plots:
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=300)
        plt.close(fig)
    
    def _plot_true_function(self, ax, resolution=100):
        """Plot the true function"""
        x_true = np.linspace(self.x_lim[0], self.x_lim[1], resolution)
        y_true = f(x_true)
        ax.plot(x_true, y_true, self.plot_colors['true_function'], label='True function')
        return x_true, y_true
    
    def _plot_tabpfn_predictions(self, ax, model, x_range, include_median=True):
        """Plot TabPFN predictions including mean, median, and confidence bands"""
        # Convert to tensor for TabPFN
        X_pred = torch.tensor(x_range.reshape(-1, 1), dtype=torch.float64)
        
        with torch.no_grad():
            # Get predictions with quantiles
            posterior, tabpfn_outputs = model.posterior(X_pred, return_tabpfn_quantiles=True)
            mean = posterior.mean.numpy()
            quantiles = tabpfn_outputs["quantiles"]
            
            # Plot mean
            ax.plot(x_range, mean, self.plot_colors['tabpfn_mean'], label='TabPFN mean')
            
            # Plot median (optional)
            if include_median:
                median = quantiles[4]  # Quantile at 0.5 (index 4)
                ax.plot(x_range, median, 
                        color=self.plot_colors['tabpfn_median'], 
                        linestyle='--', 
                        label='TabPFN median')
            
            # Plot confidence bands
            for (lower_idx, upper_idx), alpha_val in zip(self.quantile_pairs, self.confidence_alphas):
                ax.fill_between(
                    x_range, 
                    quantiles[lower_idx], 
                    quantiles[upper_idx], 
                    alpha=alpha_val, 
                    color=self.plot_colors['confidence'], 
                    label=f'Confidence region' if lower_idx == 0 else None
                )
        
        return mean, quantiles
    
    def _plot_acquisition_function(self, ax, acq_func, x_range, next_x, next_acq_value):
        """Plot acquisition function values"""
        # Prepare input tensor for acquisition function
        X_pred = torch.tensor(x_range.reshape(-1, 1), dtype=torch.float64)
        X_pred_batch = X_pred.unsqueeze(1)  # Add q dimension
        
        try:
            # Compute acquisition values
            acq_values = acq_func(X_pred_batch).detach().numpy().flatten()
            
            # Handle NaN values
            if np.isnan(acq_values).any():
                print("Warning: NaN values in acquisition function. Replacing with zeros.")
                acq_values = np.nan_to_num(acq_values, nan=0.0)
            
            # Find the exact acquisition value at the optimized point
            next_x_idx = np.abs(x_range - next_x).argmin()
            plot_next_acq = acq_values[next_x_idx]
            
            # Plot the acquisition function
            ax.plot(x_range, acq_values, self.plot_colors['acquisition'], 
                    label=f'Acquisition ({self.acquisition_function})')
            
            # Mark the next point
            ax.scatter(
                next_x, plot_next_acq, 
                c=self.plot_colors['next_point'], 
                marker=self.plot_markers['next_point'], 
                s=self.plot_sizes['next_point'], 
                label='Next point'
            )
            
            return acq_values
        except Exception as e:
            print(f"Error computing acquisition function values: {e}")
            return np.zeros_like(x_range)
    
    def _plot_evaluated_points(self, ax, highlight_best=True, highlight_next=None):
        """Plot all evaluated points and optionally highlight best/next points"""
        # Plot all evaluated points
        ax.scatter(
            self.x_evals, self.f_evals, 
            c=self.plot_colors['points'], 
            marker=self.plot_markers['points'], 
            s=self.plot_sizes['points'], 
            label='Evaluated points'
        )
        
        # Highlight the best point if requested
        if highlight_best:
            ax.scatter(
                self.x_evals[self.current_best_index], 
                self.f_evals[self.current_best_index],
                c=self.plot_colors['best_point'], 
                marker=self.plot_markers['best_point'], 
                s=self.plot_sizes['best_point'], 
                label='Best point'
            )
        
        # Highlight the next point if provided
        if highlight_next is not None:
            next_x, next_f = highlight_next
            ax.scatter(
                next_x, next_f, 
                c=self.plot_colors['next_point'], 
                marker=self.plot_markers['next_point'], 
                s=self.plot_sizes['next_point'], 
                label='Next point'
            )
    
    def _setup_plot_axes(self, ax, title, set_limits=True):
        """Set up axis with common formatting"""
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True)
        
        if set_limits:
            ax.set_xlim(self.x_lim)
            ax.set_ylim(self.y_lim)
    
    def visualize_initial_model(self):
        """Visualize the initial surrogate model."""
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot true function
        self._plot_true_function(ax)
        
        # Plot initial points
        self._plot_evaluated_points(ax, highlight_best=True)
        
        # Set up axis properties
        self._setup_plot_axes(ax, f'Initial Design of Experiments (DoE), n={len(self.x_evals)}')
        
        # Save plot
        self._save_plot(fig, 'initial_doe.png')
    
    def visualize_iteration(self, model, acq_func, next_x, next_acq_value):
        """Visualize the current surrogate model and acquisition function."""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Create prediction grids
        x_pred_surrogate = np.linspace(self.x_lim[0], self.x_lim[1], 100)
        x_pred_acq = np.linspace(self.x_lim[0], self.x_lim[1], 500)
        
        # 1. Plot surrogate model
        # Plot the true function
        self._plot_true_function(ax1)
        
        # Plot TabPFN predictions
        self._plot_tabpfn_predictions(ax1, model, x_pred_surrogate)
        
        # Compute function value at next point for visualization
        next_f = f(next_x)
        
        # Plot evaluated points and highlight next point
        self._plot_evaluated_points(ax1, highlight_best=True, highlight_next=(next_x, next_f))
        
        # Set up upper axis
        self._setup_plot_axes(ax1, f'Iteration {self.iteration_count}: Surrogate Model')
        
        # 2. Plot acquisition function
        acq_values = self._plot_acquisition_function(ax2, acq_func, x_pred_acq, next_x, next_acq_value)
        
        # Set up lower axis
        self._setup_plot_axes(ax2, 'Acquisition Function', set_limits=False)
        ax2.set_xlim(self.x_lim)
        ax2.set_ylabel('Acquisition Value')
        
        # Add logarithmic scale for better visualization of peaks
        if np.max(acq_values) > 0.001:
            ax3 = ax2.twinx()
            
            # Create log values, avoiding non-positive values
            min_positive = max(np.min(acq_values[acq_values > 0]), 1e-10)
            log_values = np.copy(acq_values)
            log_values[log_values <= 0] = min_positive
            
            # Plot on log scale
            ax3.semilogy(
                x_pred_acq, log_values, 
                self.plot_colors['acquisition_log'], 
                alpha=0.3, linewidth=0.5
            )
            ax3.set_ylabel('Log Acquisition Value')
            ax3.tick_params(axis='y', colors='green')
        
        plt.tight_layout()
        
        # Save plot
        self._save_plot(fig, f'iteration_{self.iteration_count:03d}.png')
    
    def visualize_final_result(self):
        """Visualize the final surrogate model."""
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Create model
        model = self._create_model()
        
        # Create prediction grid
        x_pred = np.linspace(self.x_lim[0], self.x_lim[1], 100)
        
        # Plot true function
        self._plot_true_function(ax)
        
        # Plot TabPFN predictions
        self._plot_tabpfn_predictions(ax, model, x_pred)
        
        # Plot evaluated points
        self._plot_evaluated_points(ax, highlight_best=True)
        
        # Set up axis properties
        self._setup_plot_axes(ax, f'Final Surrogate Model after {self.budget} evaluations')
        
        # Save plot
        self._save_plot(fig, 'final_model.png')

def f(x):
    """Test function to optimize."""
    if isinstance(x, np.ndarray):
        if x.ndim > 1:
            x = x.ravel()
    return x**2 * np.sin(x) + 5

def run_tabpfn_optimization(acquisition_function="expected_improvement", budget=25, 
                           n_DoE=5, bounds=None, save_plots=True):
    """
    Run TabPFN Bayesian optimization with visualization on a 1D test function.
    
    Args:
        acquisition_function: Which acquisition function to use
        budget: Total function evaluation budget
        n_DoE: Number of initial points
        bounds: Problem bounds (default: [-6.0, 6.0])
        save_plots: Whether to save plots to disk
    """
    # Set default bounds
    if bounds is None:
        bounds = np.array([[-6.0, 6.0]])
    
    print(f"Starting TabPFN Bayesian Optimization with {acquisition_function}")
    
    try:
        # Create directory for plots
        plots_dir = f"tabpfn_bo_visualizations/{acquisition_function}"
        
        # Set up optimizer
        optimizer = TabPFNVisualizationBO(
            budget=budget,
            n_DoE=n_DoE,
            acquisition_function=acquisition_function,
            random_seed=42,
            save_plots=save_plots,
            plots_dir=plots_dir
        )
        
        # Run optimization
        optimizer(
            problem=f,
            bounds=bounds
        )
        
        return optimizer
        
    except Exception as e:
        print(f"ERROR: Run failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run with a specific acquisition function
    acq_func = "expected_improvement"
    run_tabpfn_optimization(
        acquisition_function=acq_func,
        budget=25,
        n_DoE=5
    )