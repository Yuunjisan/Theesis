from ..AbstractBayesianOptimizer import AbstractBayesianOptimizer
from typing import Union, Callable, Optional, List, Dict, Any, Tuple
from ioh.iohcpp.problem import RealSingleObjective
import numpy as np
import torch
import os
from torch import Tensor
from botorch.models.model import Model
from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound, AnalyticAcquisitionFunction, LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.posteriors import Posterior
from botorch.utils.transforms import normalize, unnormalize
from torch.distributions import MultivariateNormal
from tabpfn import TabPFNRegressor


ALLOWED_ACQUISITION_FUNCTION_STRINGS:tuple = ("expected_improvement",
                                              "probability_of_improvement",
                                              "upper_confidence_bound",
                                              "log_expected_improvement")

ALLOWED_SHORTHAND_ACQUISITION_FUNCTION_STRINGS:dict = {"EI":"expected_improvement",
                                                       "PI":"probability_of_improvement",
                                                       "UCB":"upper_confidence_bound",
                                                       "LogEI":"log_expected_improvement"}

class TabPFNSurrogateModel(Model):
    """
    A simplified BoTorch-compatible wrapper for TabPFN, closely following TabPFN_BOTORCH.py.
    """
    def __init__(
        self,
        train_X: np.ndarray,
        train_Y: np.ndarray,
        n_estimators: int = 16,
        fit_mode: str = "fit_with_cache",
        device: str = "auto"
    ):
        super().__init__()
        
        # Convert numpy arrays to tensors
        self.train_X = torch.tensor(train_X, dtype=torch.float64)
        self.train_Y = torch.tensor(train_Y, dtype=torch.float64).view(-1, 1)
        self.n_estimators = n_estimators
        self.fit_mode = fit_mode
        self.device = device
        
        # CRITICAL FIX: Adaptive configuration for high dimensions
        dimension = train_X.shape[1] if len(train_X.shape) > 1 else 1
        data_size = train_X.shape[0]
        
        # Adaptive temperature: higher for high dimensions
        if dimension >= 20:
            adaptive_temperature = min(2.0, 1.0 + 0.05 * dimension)
        elif dimension >= 10:
            adaptive_temperature = 1.2
        else:
            adaptive_temperature = 0.9
            
        # Adaptive ensemble size: more estimators for high dimensions
        if dimension >= 30:
            adaptive_n_estimators = min(32, max(16, n_estimators + dimension // 5))
        elif dimension >= 15:
            adaptive_n_estimators = max(16, n_estimators + 4)
        else:
            adaptive_n_estimators = n_estimators
            
        print(f"TabPFN Config: dim={dimension}, temp={adaptive_temperature:.2f}, estimators={adaptive_n_estimators}")
        
        # Initialize and fit TabPFN with adaptive parameters
        self.model = TabPFNRegressor(
            n_estimators=adaptive_n_estimators, 
            softmax_temperature=adaptive_temperature,  # FIXED: Adaptive temperature
            fit_mode=fit_mode,
            device=device
        )

        self.model.fit(train_X, train_Y.ravel())
    
    @property
    def num_outputs(self) -> int:
        return 1
        
    @property
    def batch_shape(self):
        return torch.Size([])
    
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        **kwargs
    ) -> Posterior:
        """
        Returns the posterior at X using TabPFN's native BarDistribution mean and variance.
        X shape: [batch_shape, q, d] where q is typically 1 for acquisition functions
        """
        # Extract original batch shape and q value for proper reshaping
        batch_shape = X.shape[:-2]  # Everything before the last two dimensions
        q = X.shape[-2]  # Number of points per batch
        d = X.shape[-1]  # Dimension of each point
        
        # Reshape X for TabPFN input (samples, features) while preserving batch structure
        if len(batch_shape) > 0:
            # For batched case
            batch_size = torch.prod(torch.tensor(batch_shape)).item()
            X_reshaped = X.view(batch_size * q, d).detach().cpu().numpy()
        else:
            # For non-batched case
            X_reshaped = X.view(q, d).detach().cpu().numpy()
        
        # CRITICAL FIX: Apply same preprocessing as training
        if hasattr(self, 'use_pca') and self.use_pca:
            X_scaled = self.scaler.transform(X_reshaped)
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_reshaped
        
        # Get predictions from TabPFN using full output to access BarDistribution
        try:
            # Use output_type="full" to get logits and criterion (BarDistribution)
            output = self.model.predict(X_processed, output_type="full")
            logits = output["logits"]
            criterion = output["criterion"]  # This is the BarDistribution
            
            # Extract mean and variance directly from BarDistribution
            mean = criterion.mean(logits).detach().cpu().numpy()  # Shape: [n_points]
            variance = criterion.variance(logits).detach().cpu().numpy()  # Shape: [n_points]
            
            # CRITICAL FIX: Adaptive variance scaling for high dimensions
            dimension = X.shape[-1]
            if dimension >= 20:
                # Scale variance up for high dimensions to account for increased uncertainty
                variance_scale = 1.0 + 0.1 * np.log(dimension / 10)
                variance = variance * variance_scale
            
            # Ensure positive variance
            variance = np.maximum(variance, 1e-8)
            std = np.sqrt(variance)
            
        except Exception as e:
            print(f"Error in TabPFN BarDistribution prediction: {e}")
            # Fallback to zeros
            sample_size = X_processed.shape[0]
            mean = np.zeros(sample_size)
            std = np.ones(sample_size) * 1e-3  # Small positive std
        
        # Convert to tensors with correct shape matching X's dtype and device
        mean_tensor = torch.tensor(mean, dtype=X.dtype, device=X.device)
        std_tensor = torch.tensor(std, dtype=X.dtype, device=X.device)
        
        # Reshape output for BoTorch
        # For batched case, we need to ensure the output has shape [batch_shape, q, 1]
        if len(batch_shape) > 0:
            # First reshape back to match the batch structure
            output_shape = batch_shape + (q, 1)
            mean_out = mean_tensor.view(output_shape)
            std_out = std_tensor.view(output_shape)
        else:
            # For non-batched case, reshape to [q, 1]
            mean_out = mean_tensor.view(q, 1)
            std_out = std_tensor.view(q, 1)
        
        # Create a dummy tensor connected to X's computation graph for gradient tracking
        dummy = X.sum() * 0
        
        # Connect tensors to input graph
        mean_out = mean_out + dummy
        std_out = std_out + dummy
        
        # Create MVN distribution with appropriate dimension handling
        return MultivariateNormal(
            mean_out,
            scale_tril=std_out.unsqueeze(-1)  # Make scale_tril have shape [..., q, 1, 1]
        )
    
    def condition_on_observations(self, X: Tensor, Y: Tensor, update_in_place: bool = False, verbose: bool = False, **kwargs) -> Model:
        """Update the model with new observations.
        
        Args:
            X: New input points
            Y: New observations
            update_in_place: If True, update this model in-place instead of creating a new one (recommended for performance)
            verbose: If True, print update information
            **kwargs: Additional parameters
            
        Returns:
            Updated model (self if update_in_place=True, otherwise a new model)
        """
        # Handle various possible tensor shapes for X and Y
        if X.dim() > 2:
            # Handle batched inputs: squeeze out the q dimension
            X_to_add = X.squeeze(-2)
        else:
            X_to_add = X
            
        # Make sure Y is properly shaped
        if Y.dim() > 2:
            Y_to_add = Y.squeeze(-1)
        else:
            Y_to_add = Y

        # Create a flat view of the tensors
        X_flat = X_to_add.reshape(-1, X_to_add.shape[-1])
        Y_flat = Y_to_add.reshape(-1)
        
        # Combine with existing training data
        new_train_X = torch.cat([self.train_X, X_flat], dim=0).cpu().numpy()
        new_train_Y = torch.cat([self.train_Y.view(-1), Y_flat], dim=0).cpu().numpy()
        
        if update_in_place:
            # Update this model in-place
            if verbose:
                print(f"Updating TabPFN model in-place: {len(self.train_X)} -> {len(new_train_X)} points")
            
            self.train_X = torch.tensor(new_train_X, dtype=torch.float64)
            self.train_Y = torch.tensor(new_train_Y, dtype=torch.float64).view(-1, 1)
            

            self.model.fit(new_train_X, new_train_Y)
            return self
        else:
            # Create a new model with updated data (original behavior - less efficient)
            if verbose:
                print(f"Creating new TabPFN model: {len(self.train_X)} -> {len(new_train_X)} points")
                
            return TabPFNSurrogateModel(
                train_X=new_train_X,
                train_Y=new_train_Y,
                n_estimators=self.n_estimators,
                fit_mode=self.fit_mode,
                device=self.device
            )


class TabPFN_BO(AbstractBayesianOptimizer):
    def __init__(self, budget, n_DoE=0, acquisition_function:str="expected_improvement",
                 random_seed:int=43, n_estimators:int=16, fit_mode:str="fit_with_cache", 
                 device:str="auto", **kwargs):
        """
        Bayesian Optimization using TabPFN as a surrogate model.
        
        Args:
            budget: Total number of function evaluations allowed
            n_DoE: Number of initial design points
            acquisition_function: Acquisition function to use
            random_seed: Random seed for reproducibility
            n_estimators: Number of TabPFN estimators in the ensemble
            fit_mode: Mode for TabPFN fitting, one of "low_memory", "fit_preprocessors", or "fit_with_cache"
            device: Device to use for TabPFN, "auto", "cpu", or "cuda"
            **kwargs: Additional parameters
        """
        # Call the superclass
        super().__init__(budget, n_DoE, random_seed, **kwargs)
        
        # TabPFN specific parameters
        self.n_estimators = n_estimators
        self.fit_mode = fit_mode
        self.device = device
        
        # Set up the acquisition function
        self.__acq_func = None
        self.acquistion_function_name = acquisition_function
        
        # TabPFN surrogate model
        self.__model_obj = None
        
    def __str__(self):
        return "This is an instance of TabPFN BO Optimizer"

    def __call__(self, problem:Union[RealSingleObjective,Callable], 
                 dim:Optional[int]=-1, 
                 bounds:Optional[np.ndarray]=None, 
                 **kwargs)-> None:
        """
        Run Bayesian optimization using TabPFN as surrogate model.
        """
        # Call the superclass to run the initial sampling of the problem
        super().__call__(problem, dim, bounds, **kwargs)
        
        # CRITICAL FIX: Adaptive initialization for high dimensions
        if self.dimension >= 20:
            # For high dimensions, ensure we have enough initial points
            min_initial_points = max(self.n_DoE, 2 * self.dimension)
            if len(self.x_evals) < min_initial_points:
                print(f"High dimension detected ({self.dimension}D). Adding more initial points: {len(self.x_evals)} -> {min_initial_points}")
                
                # Generate additional Latin Hypercube samples
                from scipy.stats import qmc
                sampler = qmc.LatinHypercube(d=self.dimension, seed=self.random_seed)
                additional_points = min_initial_points - len(self.x_evals)
                lhs_samples = sampler.random(n=additional_points)
                
                # Scale to bounds
                lb = self.bounds[:, 0]
                ub = self.bounds[:, 1]
                scaled_samples = lb + lhs_samples * (ub - lb)
                
                # Evaluate additional points
                for sample in scaled_samples:
                    f_val = float(problem(sample))
                    self.x_evals.append(sample)
                    self.f_evals.append(f_val)
                    self.number_of_function_evaluations += 1
        
        # Get beta for UCB
        beta = kwargs.pop("beta", 0.2)
        
        # Initialize model once
        self._initialize_model()
        
        # Start the optimization loop
        for cur_iteration in range(self.budget - self.n_DoE):
            # Set up the acquisition function
            if self.__acquisition_function_name == "upper_confidence_bound":
                self.acquisition_function = self.acquisition_function_class(
                    model=self.__model_obj,
                    beta=beta,
                    maximize=self.maximisation
                )
            else:
                self.acquisition_function = self.acquisition_function_class(
                    model=self.__model_obj,
                    best_f=self.current_best,
                    maximize=self.maximisation
                )
            
            # Get next point
            new_x = self.optimize_acqf_and_get_observation()
            
            # Evaluate function - ensure proper dimensionality
            if new_x.dim() > 2:
                new_x = new_x.squeeze(1)  # Remove extra dimension if present
            
            # Properly extract numpy array maintaining all dimensions
            new_x_numpy = new_x.detach().cpu().numpy()
            
            # Handle scalar vs vector correctly without dimension reduction
            if new_x_numpy.ndim == 1 and new_x_numpy.size == 1:  # Handle 1D scalar case
                new_x_numpy = float(new_x_numpy[0])
                new_f_eval = float(problem(new_x_numpy))
            elif new_x_numpy.ndim == 2 and new_x_numpy.shape[0] == 1:  # Handle single row vector
                new_x_numpy = new_x_numpy.reshape(-1)  # Convert to 1D array
                new_f_eval = float(problem(new_x_numpy))
            else:  # Handle general vector case
                new_f_eval = float(problem(new_x_numpy.squeeze()))  # Ensure result is a float
            
            # Append evaluations - ensure consistent data types
            self.x_evals.append(new_x_numpy)
            self.f_evals.append(new_f_eval)
            self.number_of_function_evaluations += 1
            
            # Update model - now using in-place update for performance
            X_new = new_x
            Y_new = torch.tensor([new_f_eval], dtype=torch.float64).view(1, 1)
            # Using in-place update significantly improves performance by avoiding recreating the model
            self.__model_obj.condition_on_observations(X=X_new, Y=Y_new, update_in_place=True, verbose=self.verbose)
            
            # Assign new best
            self.assign_new_best()
            
            # Print progress
            if self.verbose:
                print(f"Iteration {cur_iteration+1}/{self.budget - self.n_DoE}: Best value = {self.current_best}")
        
        print("Optimization finished!")

    def assign_new_best(self):
        # Call the super class
        super().assign_new_best()
    
    def _initialize_model(self):
        """Initialize the TabPFN surrogate model with current data."""
        # Create properly shaped arrays with consistent dimensions
        # FIXED: Properly handle multi-dimensional inputs without dimensionality reduction
        if len(self.x_evals) > 0:
            # Handle the case where inputs might be mixed scalar and array values
            if np.isscalar(self.x_evals[0]) or (isinstance(self.x_evals[0], np.ndarray) and self.x_evals[0].size == 1):
                # For scalar inputs, reshape to column vector
                train_x = np.array([float(x) for x in self.x_evals]).reshape((-1, 1))
            else:
                # For vector inputs, preserve all dimensions
                train_x = np.array([x if isinstance(x, np.ndarray) else np.array(x) for x in self.x_evals])
                # Ensure consistent shape
                if train_x.ndim == 1:
                    train_x = train_x.reshape(-1, 1)
        else:
            # Empty dataset case
            train_x = np.array([]).reshape(0, self.dimension)
            
        train_obj = np.array([float(y) for y in self.f_evals]).reshape(-1)
        
        self.__model_obj = TabPFNSurrogateModel(
            train_X=train_x,
            train_Y=train_obj,
            n_estimators=self.n_estimators,
            fit_mode=self.fit_mode,
            device=self.device
        )
    
    def optimize_acqf_and_get_observation(self):
        """Optimizes the acquisition function and returns a new candidate."""
        bounds_tensor = torch.tensor(self.bounds.transpose(), dtype=torch.float64)
        
        # CRITICAL FIX: Adaptive optimization parameters for high dimensions
        dimension = self.dimension
        if dimension >= 30:
            num_restarts = 20
            raw_samples = 200
        elif dimension >= 15:
            num_restarts = 15
            raw_samples = 150
        else:
            num_restarts = 10
            raw_samples = 100
        
        try:
            # Print diagnostic info
            print(f"Attempting to optimize acquisition function: {self.__acquisition_function_name}")
            print(f"Dimension: {dimension}, Restarts: {num_restarts}, Raw samples: {raw_samples}")
            
            candidates, acq_value = optimize_acqf(
                acq_function=self.acquisition_function,
                bounds=bounds_tensor,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )
            
            # Ensure shape is correct (n x d tensor where n=1)
            if candidates.dim() == 3:
                candidates = candidates.squeeze(1)
            
            print(f"Acquisition function optimization SUCCEEDED with value: {acq_value.item():.6f}")
            return candidates
            
        except Exception as e:
            print(f"ERROR: Acquisition function optimization FAILED: {e}")
            print("Falling back to random sampling!")
            
            # Fallback to random sampling - ensure correct shape with proper dimensionality
            lb = self.bounds[:, 0]
            ub = self.bounds[:, 1]
            random_point = lb + np.random.rand(self.dimension) * (ub - lb)
            return torch.tensor(random_point.reshape(1, self.dimension), dtype=torch.float64)
    
    def set_acquisition_function_subclass(self) -> None:
        """Set the acquisition function class based on the name"""
        if self.__acquisition_function_name == ALLOWED_ACQUISITION_FUNCTION_STRINGS[0]:
            self.__acq_func_class = ExpectedImprovement
        elif self.__acquisition_function_name == ALLOWED_ACQUISITION_FUNCTION_STRINGS[1]:
            self.__acq_func_class = ProbabilityOfImprovement
        elif self.__acquisition_function_name == ALLOWED_ACQUISITION_FUNCTION_STRINGS[2]:
            self.__acq_func_class = UpperConfidenceBound
        elif self.__acquisition_function_name == ALLOWED_ACQUISITION_FUNCTION_STRINGS[3]:
            self.__acq_func_class = LogExpectedImprovement
    
    def __repr__(self):
        return super().__repr__()
    
    def reset(self):
        return super().reset()
    
    @property
    def acquistion_function_name(self) -> str:
        return self.__acquisition_function_name
    
    @acquistion_function_name.setter
    def acquistion_function_name(self, new_name: str) -> None:
        # Remove some spaces and convert to lower case
        new_name = new_name.strip().lower()
        
        # Start with a dummy variable
        dummy_var: str = ""
        
        # Check in the reduced
        if new_name in [*ALLOWED_SHORTHAND_ACQUISITION_FUNCTION_STRINGS]:
            # Assign the name
            dummy_var = ALLOWED_SHORTHAND_ACQUISITION_FUNCTION_STRINGS[new_name]
        else:
            if new_name in ALLOWED_ACQUISITION_FUNCTION_STRINGS:
                dummy_var = new_name
            else:
                raise ValueError(f"Invalid acquisition function name: {new_name}. Must be one of {ALLOWED_ACQUISITION_FUNCTION_STRINGS}")
        
        self.__acquisition_function_name = dummy_var
        # Run to set up the acquisition function subclass
        self.set_acquisition_function_subclass()
    
    @property
    def acquisition_function_class(self) -> Callable:
        return self.__acq_func_class
    
    @property
    def acquisition_function(self) -> AnalyticAcquisitionFunction:
        """Returns the acquisition function"""
        return self.__acq_func
    
    @acquisition_function.setter
    def acquisition_function(self, new_acquisition_function: AnalyticAcquisitionFunction) -> None:
        """Sets the acquisition function"""
        if issubclass(type(new_acquisition_function), AnalyticAcquisitionFunction):
            self.__acq_func = new_acquisition_function
        else:
            raise AttributeError("Cannot assign the acquisition function as this does not inherit from the class `AnalyticAcquisitionFunction`",
                                name="acquisition_function",
                                obj=self.__acq_func)
    
    @property
    def model(self):
        """Get the TabPFN surrogate model."""
        return self.__model_obj 