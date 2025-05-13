import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from typing import Optional, List, Union, Any
from botorch.models.model import Model
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
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
    ):
        super().__init__()
        self.train_X = train_X
        self.train_Y = train_Y
        self.n_estimators = n_estimators
        
        # Convert to numpy for TabPFN
        X_np = train_X.numpy()
        y_np = train_Y.numpy()
        
        # Initialize and fit TabPFN
        self.model = TabPFNRegressor(n_estimators=n_estimators, softmax_temperature=1)
        self.model.fit(X_np, y_np)
        
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
        **kwargs: Any,
    ) -> Posterior:
        #convert to numpy and reshape for TabPFN
        #X shape is [batch_shape, q=1, d=1]
        #We need to reshape to [batch_shape, d]
        X_np = X.squeeze(1).detach().numpy()  # Remove q dimension and detach
        
        #reshape to 2D array for TabPFN
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        
        #get predictions from TabPFN
        predictions = self.model.predict(X_np, output_type="main")
        mean = predictions["mean"]
        quantiles = predictions["quantiles"]
        
        #Calculate standard deviation from quantiles
        #TabPFN returns 9 quantiles from 0.1 to 0.9
        #We'll use symmetric pairs around the median
        std_estimates = []
        quantile_pairs = [(0, 8), (1, 7), (2, 6), (3, 5)]
        alphas = [0.1, 0.2, 0.3, 0.4]  #corresponding to the quantile pairs
        
        for (lower_idx, upper_idx), alpha in zip(quantile_pairs, alphas):
            q_low = quantiles[lower_idx]
            q_high = quantiles[upper_idx]
            # For normal distribution: (q_high - q_low) / (2 * ppf(1-alpha))
            z_score = 2 * np.abs(stats.norm.ppf(alpha))
            std = (q_high - q_low) / z_score
            std_estimates.append(std)
        
        # Average the standard deviation estimates
        std = np.mean(std_estimates, axis=0)
        
        # Convert to tensors with correct shapes
        # mean should be [batch_shape x 1], std should be [batch_shape x 1]
        mean = torch.tensor(mean, dtype=X.dtype, device=X.device).unsqueeze(-1)
        std = torch.tensor(std, dtype=X.dtype, device=X.device).unsqueeze(-1)
        
        # Create scale_tril [batch_shape x 1 x 1]
        # We connect it to X to maintain the gradient graph
        dummy = X.sum() * 0  # Creates a zero tensor connected to X's graph
        scale_tril = (std.unsqueeze(-1) + dummy).abs()  # Ensure positive and connect to graph
        mean = mean + dummy  # Connect mean to the graph
        
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
        
        return TabPFNModel(new_train_X, new_train_Y, self.n_estimators,)

class TabPFNVisualizationBO:
    """
    Extension of TabPFNModel that adds visualization capabilities for the surrogate model
    and acquisition function during optimization.
    """
    def __init__(self, budget, n_DoE=0, acquisition_function="expected_improvement", 
                 random_seed=42, save_plots=True, plots_dir="tabpfn_bo_visualizations", **kwargs):
        self.budget = budget
        self.n_DoE = n_DoE
        self.acquisition_function = acquisition_function
        self.random_seed = random_seed
        self.save_plots = save_plots
        self.plots_dir = plots_dir
        self.iteration_count = 0
        
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
        # Set default bounds if not provided
        if bounds is None:
            bounds = np.array([[-8.0, 8.0]])
            
        # Ensure bounds are in the correct format for BoTorch (2 x d)
        bounds = torch.tensor(bounds, dtype=torch.float64).T
            
        # Force 1D
        dim = 1
        
        # Initial sampling
        self._initial_sampling(problem, bounds)
        
        # Visualize initial model
        self.visualize_initial_model()
        
        # Start optimization loop
        for cur_iteration in range(self.budget - self.n_DoE):
            self.iteration_count = cur_iteration + 1
            
            # Create and fit TabPFN model
            train_X = torch.tensor(self.x_evals, dtype=torch.float64)
            train_Y = torch.tensor(self.f_evals, dtype=torch.float64)
            model = TabPFNModel(train_X, train_Y)
            
            # Set up acquisition function
            if self.acquisition_function == "upper_confidence_bound":
                acq_func = UpperConfidenceBound(
                    model=model,
                    beta=kwargs.get("beta", 0.2),
                    maximize=False
                )
            elif self.acquisition_function == "probability_of_improvement":
                acq_func = ProbabilityOfImprovement(
                    model=model,
                    best_f=self.current_best,
                    maximize=False
                )
            else:  # expected_improvement
                acq_func = ExpectedImprovement(
                    model=model,
                    best_f=self.current_best,
                    maximize=False
                )
            
            # Visualize current state
            self.visualize_iteration(model, acq_func)
            
            # Optimize acquisition function
            candidate, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=20,
            )
            
            # Evaluate new point
            new_x = candidate.detach().numpy()
            new_f = problem(new_x.ravel())
            
            # Update evaluations
            self.x_evals = np.vstack((self.x_evals, new_x))
            self.f_evals = np.append(self.f_evals, new_f)
            self.number_of_function_evaluations += 1
            
            # Update best
            if new_f < self.current_best:
                self.current_best = new_f
                self.current_best_index = len(self.f_evals) - 1
            
            # Print progress
            print(f"Iteration {cur_iteration + 1}: x = {float(new_x.ravel()[0]):.4f}, f(x) = {float(new_f):.4f}")
            
        # Final visualization
        self.visualize_final_result()
        
    def _initial_sampling(self, problem, bounds):
        """Perform initial sampling using LHS."""
        samples = lhs(1, samples=self.n_DoE)
        # For transposed bounds: bounds[0] is lower bound, bounds[1] is upper bound
        xi = bounds[0] + (bounds[1] - bounds[0]) * samples
        fi = np.array([problem(x) for x in xi])
        
        # Store as numpy arrays with consistent shape
        self.x_evals = xi
        self.f_evals = fi
        self.current_best = min(fi)
        self.current_best_index = np.argmin(fi)
        self.number_of_function_evaluations = self.n_DoE
        
    def visualize_initial_model(self):
        """Visualize the initial surrogate model."""
        train_X = torch.tensor(self.x_evals, dtype=torch.float64)
        train_Y = torch.tensor(self.f_evals, dtype=torch.float64)
        model = TabPFNModel(train_X, train_Y)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot true function
        x_true = np.linspace(self.x_evals.min(), self.x_evals.max(), 500)
        y_true = f(x_true)
        plt.plot(x_true, y_true, 'r--', label='True function')
        
        # Plot initial points
        plt.scatter(self.x_evals, self.f_evals, c='navy', marker='o', label='Initial points')
        
        plt.title(f'Initial Design of Experiments (DoE), n={len(self.x_evals)}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plots_dir, 'initial_doe.png'), dpi=300)
        plt.close()
        
    def visualize_iteration(self, model, acq_func):
        """Visualize the current surrogate model and acquisition function."""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Create consistent x points for both plots
        x_pred = np.linspace(self.x_evals.min(), self.x_evals.max(), 100)
        X_pred = torch.tensor(x_pred.reshape(-1, 1), dtype=torch.float64)
        X_pred_batch = X_pred.unsqueeze(1)  # Add q dimension for acquisition function
        
        with torch.no_grad():
            # Get surrogate model predictions
            posterior = model.posterior(X_pred)
            mean = posterior.mean.numpy()
            std = posterior.variance.sqrt().numpy()
            
            # Get acquisition function values
            acq_values = acq_func(X_pred_batch)
        
        # Plot true function
        y_true = f(x_pred)
        ax1.plot(x_pred, y_true, 'r--', label='True function')
        
        # Plot mean and confidence intervals with multiple bounds
        ax1.plot(x_pred, mean, 'b-', label='TabPFN mean')
        
        # Plot multiple confidence intervals with decreasing opacity
        alphas = [0.05, 0.1, 0.15, 0.2]
        # Use quantile pairs that roughly correspond to 4σ, 3σ, 2σ, 1σ intervals
        quantile_pairs = [(0, 8), (1, 7), (2, 6), (3, 5)]  # These correspond to 10-90%, 20-80%, 30-70%, 40-60%
        
        for (lower_idx, upper_idx), alpha_val in zip(quantile_pairs, alphas):
            ax1.fill_between(x_pred, 
                           mean.ravel() - (4 - lower_idx) * std.ravel(),
                           mean.ravel() + (4 - lower_idx) * std.ravel(),
                           alpha=alpha_val, color='purple', 
                           label=f'Confidence region' if lower_idx == 0 else None)
        
        # Plot evaluated points
        ax1.scatter(self.x_evals, self.f_evals, c='navy', marker='o', label='Evaluated points')
        
        # Plot acquisition function
        ax2.plot(x_pred, acq_values.numpy(), 'g-', label=f'Acquisition ({self.acquisition_function})')
        
        # Find and mark the maximum
        max_idx = np.argmax(acq_values.numpy())
        next_x = x_pred[max_idx]
        next_acq = acq_values.numpy()[max_idx]
        
        # Mark the next point on both plots
        ax1.scatter(next_x, f(next_x), c='red', marker='*', s=200, label='Next point')
        ax2.scatter(next_x, next_acq, c='red', marker='*', s=200, label='Next point')
        
        ax1.set_title(f'Iteration {self.iteration_count}: Surrogate Model')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Acquisition Function')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Acquisition Value')
        ax2.legend()
        ax2.grid(True)
        
        # Ensure both plots have the same x-axis limits
        ax1.set_xlim(x_pred.min(), x_pred.max())
        ax2.set_xlim(x_pred.min(), x_pred.max())
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plots_dir, f'iteration_{self.iteration_count:03d}.png'), dpi=300)
        plt.close()
        
    def visualize_final_result(self):
        """Visualize the final surrogate model."""
        train_X = torch.tensor(self.x_evals, dtype=torch.float64)
        train_Y = torch.tensor(self.f_evals, dtype=torch.float64)
        model = TabPFNModel(train_X, train_Y)
        
        plt.figure(figsize=(10, 6))
        
        # Plot true function
        x_true = np.linspace(self.x_evals.min(), self.x_evals.max(), 500)
        y_true = f(x_true)
        plt.plot(x_true, y_true, 'r--', label='True function')
        
        # Plot final model
        x_pred = np.linspace(self.x_evals.min(), self.x_evals.max(), 100)
        X_pred = torch.tensor(x_pred.reshape(-1, 1), dtype=torch.float64)
        
        with torch.no_grad():
            posterior = model.posterior(X_pred)
            mean = posterior.mean.numpy()
            std = posterior.variance.sqrt().numpy()
        
        plt.plot(x_pred, mean, 'b-', label='TabPFN mean')
        
        # Plot multiple confidence intervals with decreasing opacity
        alphas = [0.05, 0.1, 0.15, 0.2]
        # Use quantile pairs that roughly correspond to 4σ, 3σ, 2σ, 1σ intervals
        quantile_pairs = [(0, 8), (1, 7), (2, 6), (3, 5)]  # These correspond to 10-90%, 20-80%, 30-70%, 40-60%
        
        for (lower_idx, upper_idx), alpha_val in zip(quantile_pairs, alphas):
            plt.fill_between(x_pred, 
                           mean.ravel() - (4 - lower_idx) * std.ravel(),
                           mean.ravel() + (4 - lower_idx) * std.ravel(),
                           alpha=alpha_val, color='purple', 
                           label=f'Confidence region' if lower_idx == 0 else None)
        
        # Plot all evaluated points
        plt.scatter(self.x_evals, self.f_evals, c='navy', marker='o', label='Evaluated points')
        
        # Highlight best point
        plt.scatter(self.x_evals[self.current_best_index], 
                   self.f_evals[self.current_best_index],
                   c='red', marker='*', s=200, label='Best point')
        
        plt.title(f'Final Surrogate Model after {self.budget} evaluations')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plots_dir, 'final_model.png'), dpi=300)
        plt.close()

def f(x):
    """Test function to optimize."""
    return x**2 * np.sin(x) + 5

if __name__ == "__main__":
    # Create the optimizer
    optimizer = TabPFNVisualizationBO(
        budget=25,
        n_DoE=10,
        acquisition_function="expected_improvement",
        save_plots=True
    )
    
    # Run the optimization
    optimizer(f, bounds=np.array([[-6.0, 6.0]]))
