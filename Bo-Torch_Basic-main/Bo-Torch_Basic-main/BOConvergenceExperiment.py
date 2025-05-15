"""
This script runs Bayesian Optimization with different surrogate models and creates 
convergence plots similar to those in Convergence.py.

It supports:
- Vanilla BO with Gaussian Process (from Sausage_Acq_plot_BO.py)
- TabPFN BO (from TabPFN_BOTORCH.py)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import traceback

# Import optimization implementations
from Sausage_Acq_plot_BO import VisualizationBO
from TabPFN_BOTORCH import TabPFNVisualizationBO

# Define the test function
def f(x):
    """Test function to optimize: f(x) = x^2 * sin(x) + 5"""
    if isinstance(x, np.ndarray):
        if x.ndim > 1:
            x = x.ravel()
    return x**2 * np.sin(x) + 5 + np.sin(x) + np.cos(x)

# Define experiment folder
experiment_folder = "bo_convergence_experiment"

def run_optimization(optimizer_type="gp", problem_id=1, acquisition_function="expected_improvement", dimension=1, bounds=None, budget=25, n_doe=5, n_runs=5):
    """
    Run Bayesian Optimization on the specified problem multiple times with
    either GP or TabPFN surrogate model.
    
    Args:
        optimizer_type: Either "gp" or "tabpfn"
        problem_id: Problem identifier (for naming only)
        acquisition_function: Acquisition function to use
        dimension: Problem dimension
        bounds: Problem bounds
        budget: Total evaluation budget
        n_doe: Number of initial design points
        n_runs: Number of independent runs
        
    Results are saved to .dat files in the experiment folder.
    """
    # Set default bounds if not provided
    if bounds is None:
        bounds = np.array([[-6.0, 6.0]])
    
    # Create experiment directory
    os.makedirs(experiment_folder, exist_ok=True)
    
    # Dictionary to collect results for plotting
    convergence_data = []
    
    # Run optimization for each run with a different seed
    for run in range(1, n_runs+1):
        print(f"\nStarting run {run}/{n_runs} for {optimizer_type.upper()}, {acquisition_function}")
        
        try:
            # Set seed
            random_seed = 42 + run
            
            # Initialize the appropriate optimizer
            if optimizer_type.lower() == "gp":
                optimizer = VisualizationBO(
                    budget=budget,
                    n_DoE=n_doe,
                    acquisition_function=acquisition_function,
                    random_seed=random_seed,
                    verbose=False,
                    maximisation=False,
                    save_plots=False
                )
                
                # Run optimization
                optimizer(
                    problem=f,
                    bounds=bounds
                )
                
                # Extract evaluations
                f_evals = optimizer.f_evals
                
            elif optimizer_type.lower() == "tabpfn":
                optimizer = TabPFNVisualizationBO(
                    budget=budget,
                    n_DoE=n_doe,
                    acquisition_function=acquisition_function,
                    random_seed=random_seed,
                    save_plots=False
                )
                
                # Run optimization
                optimizer(
                    problem=f,
                    bounds=bounds
                )
                
                # Extract evaluations
                f_evals = optimizer.f_evals
            
            # Calculate best-so-far values
            best_so_far = []
            best_value = float('inf')
            for val in f_evals:
                best_value = min(best_value, val)
                best_so_far.append(best_value)
            
            # Create run data structure similar to what Convergence.py expects
            run_data = {
                'run': run,
                'evals': list(range(1, len(best_so_far) + 1)),
                'best_so_far': best_so_far,
                'final_regret': best_so_far[-1]
            }
            
            # Add to the data collection
            convergence_data.append(run_data)
            
            # Save to .dat file with format similar to IOH profiler
            dat_dir = Path(f"{experiment_folder}/{optimizer_type}_{acquisition_function}")
            dat_dir.mkdir(exist_ok=True)
            
            # Create dataframe for .dat file (similar format to IOH profiler)
            df = pd.DataFrame({
                'evaluations': run_data['evals'],
                'raw_y_best': run_data['best_so_far']
            })
            
            # Save dataframe
            dat_file = dat_dir / f"run_{run}_f{problem_id}_DIM{dimension}.dat"
            df.to_csv(dat_file, sep=' ', index=False)
            
            print(f"Run {run} completed successfully.")
            print(f"  - Final best value: {best_so_far[-1]:.6e}")
            
        except Exception as e:
            print(f"ERROR: Run {run} failed with error: {e}")
            traceback.print_exc()
            continue
    
    return convergence_data

def plot_convergence(convergence_data, label, problem_id, dimension, save_path=None):
    """
    Plot convergence curves from optimization runs.
    This is a simplified version of the plot_convergence function from Convergence.py.
    """
    # Sort convergence data by run number
    convergence_data = sorted(convergence_data, key=lambda x: x['run'])
    
    # Create the convergence plot
    plt.figure(figsize=(10, 6))
    
    # Use color cycle for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(convergence_data))))
    
    # Plot individual runs
    for i, run_data in enumerate(convergence_data):
        plt.plot(
            run_data['evals'], 
            run_data['best_so_far'], 
            color=colors[i % len(colors)],
            linewidth=1.5,
            alpha=0.7,
            label=f"Run {run_data['run']}"
        )
    
    # Calculate and plot the mean
    max_length = max(len(data['best_so_far']) for data in convergence_data)
    mean_values = []
    
    for i in range(max_length):
        values_at_i = [data['best_so_far'][i] if i < len(data['best_so_far']) else data['best_so_far'][-1] 
                      for data in convergence_data]
        mean_values.append(np.mean(values_at_i))
    
    # Plot mean curve
    plt.plot(
        range(1, max_length + 1), 
        mean_values, 
        'k-', 
        linewidth=2.5,
        label='Mean'
    )
    
    # Set y-ticks at 0.5 second intervals
    min_y = min(min(data['best_so_far']) for data in convergence_data)
    max_y = max(max(data['best_so_far']) for data in convergence_data)
    # Add a small buffer to the min/max
    min_y = max(0, min_y - 0.5)  # Ensure we don't go below 0
    max_y = max_y + 0.5
    # Create ticks at 0.5 intervals
    y_ticks = np.arange(min_y, max_y + 0.5, 0.5)
    plt.yticks(y_ticks)
    
    # Customize plot
    plt.title(f"Convergence - {label}, Problem {problem_id}, Dimension {dimension} ({len(convergence_data)} runs)")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Best Function Value")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    # Print summary statistics
    final_values = [data['best_so_far'][-1] for data in convergence_data]
    print(f"\nMean final value: {np.mean(final_values):.6e} ± {np.std(final_values):.6e}")
    
    return plt

def plot_convergence_with_error_bands(convergence_data, label, problem_id, dimension, save_path=None):
    """
    Plot convergence curves with mean and standard error bands.
    This is a simplified version of the plot_convergence_with_error_bands function from Convergence.py.
    """
    # Sort convergence data by run number
    convergence_data = sorted(convergence_data, key=lambda x: x['run'])
    
    # Create the convergence plot
    plt.figure(figsize=(10, 6))
    
    # Calculate mean and standard error at each evaluation point
    max_length = max(len(data['best_so_far']) for data in convergence_data)
    mean_values = []
    stderr_values = []
    
    for i in range(max_length):
        values_at_i = [data['best_so_far'][i] if i < len(data['best_so_far']) else data['best_so_far'][-1] 
                      for data in convergence_data]
        mean_values.append(np.mean(values_at_i))
        stderr_values.append(np.std(values_at_i) / np.sqrt(len(values_at_i)))
    
    # Generate x-axis values
    x_values = range(1, max_length + 1)
    
    # Plot mean curve
    plt.plot(
        x_values, 
        mean_values, 
        'b-', 
        linewidth=2.0,
        label='Mean'
    )
    
    # Calculate upper and lower bounds for error bands
    upper_bound = [mean + stderr for mean, stderr in zip(mean_values, stderr_values)]
    lower_bound = [mean - stderr for mean, stderr in zip(mean_values, stderr_values)]
    
    # Plot standard error bands
    plt.fill_between(
        x_values,
        lower_bound,
        upper_bound,
        color='blue',
        alpha=0.2,
        label='Standard Error'
    )
    
    # Set y-ticks at 0.5 second intervals
    min_y = min(min(lower_bound), min(mean_values))
    max_y = max(max(upper_bound), max(mean_values))
    # Add a small buffer to the min/max
    min_y = max(0, min_y - 0.5)  # Ensure we don't go below 0
    max_y = max_y + 0.5
    # Create ticks at 0.5 intervals
    y_ticks = np.arange(min_y, max_y + 0.5, 0.5)
    plt.yticks(y_ticks)
    
    # Customize plot
    plt.title(f"Convergence with Error Bands - {label}, Problem {problem_id}, Dimension {dimension} ({len(convergence_data)} runs)")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Best Function Value")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save or display
    if save_path:
        # Modify save path to indicate this is the error band plot
        error_band_path = str(save_path).replace('.png', '_error_bands.png')
        plt.savefig(error_band_path, dpi=300)
        print(f"Error band plot saved to {error_band_path}")
    else:
        plt.show()
    
    # Print summary statistics
    final_values = [data['best_so_far'][-1] for data in convergence_data]
    print(f"\nMean final value: {np.mean(final_values):.6e} ± {np.std(final_values)/np.sqrt(len(final_values)):.6e} (SE)")
    
    return plt

def create_comparison_plot(all_convergence_data, problem_id, dimension, save_path=None):
    """
    Create a plot comparing the convergence of different optimizers.
    
    Args:
        all_convergence_data: Dictionary mapping optimizer labels to their convergence data
        problem_id: Problem identifier
        dimension: Problem dimension
        save_path: Where to save the plot
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Use color cycle for different optimizers
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_convergence_data)))
    
    # For each optimizer
    for i, (label, data) in enumerate(all_convergence_data.items()):
        # Calculate mean performance
        max_length = max(len(run_data['best_so_far']) for run_data in data)
        mean_values = []
        
        for j in range(max_length):
            values_at_j = [run_data['best_so_far'][j] if j < len(run_data['best_so_far']) else run_data['best_so_far'][-1] 
                          for run_data in data]
            mean_values.append(np.mean(values_at_j))
        
        # Plot mean curve
        plt.plot(
            range(1, max_length + 1), 
            mean_values, 
            color=colors[i],
            linewidth=2.5,
            label=label
        )
    
    # Customize plot
    plt.title(f"Convergence Comparison - Problem {problem_id}, Dimension {dimension}")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Best Function Value")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    return plt

def create_comparison_plot_with_error_bands(all_convergence_data, problem_id, dimension, save_path=None):
    """
    Create a plot comparing the convergence of different optimizers with error bands.
    
    Args:
        all_convergence_data: Dictionary mapping optimizer labels to their convergence data
        problem_id: Problem identifier
        dimension: Problem dimension
        save_path: Where to save the plot
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Use color cycle for different optimizers
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_convergence_data)))
    
    # For each optimizer
    for i, (label, data) in enumerate(all_convergence_data.items()):
        # Calculate mean performance
        max_length = max(len(run_data['best_so_far']) for run_data in data)
        mean_values = []
        stderr_values = []
        
        for j in range(max_length):
            values_at_j = [run_data['best_so_far'][j] if j < len(run_data['best_so_far']) else run_data['best_so_far'][-1] 
                          for run_data in data]
            mean_values.append(np.mean(values_at_j))
            stderr_values.append(np.std(values_at_j) / np.sqrt(len(values_at_j)))
        
        # Plot mean curve
        plt.plot(
            range(1, max_length + 1), 
            mean_values, 
            color=colors[i],
            linewidth=2.0,
            label=label
        )
        
        # Calculate upper and lower bounds for error bands
        upper_bound = [mean + stderr for mean, stderr in zip(mean_values, stderr_values)]
        lower_bound = [mean - stderr for mean, stderr in zip(mean_values, stderr_values)]
        
        # Plot standard error bands
        plt.fill_between(
            range(1, max_length + 1),
            lower_bound,
            upper_bound,
            color=colors[i],
            alpha=0.2
        )
    
    # Customize plot
    plt.title(f"Convergence Comparison with Error Bands - Problem {problem_id}, Dimension {dimension}")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Best Function Value")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save or display
    if save_path:
        error_band_path = str(save_path).replace('.png', '_error_bands.png')
        plt.savefig(error_band_path, dpi=300)
        print(f"Comparison plot with error bands saved to {error_band_path}")
    else:
        plt.show()
    
    return plt

def main():
    """
    Run optimization with different surrogate models and create convergence plots.
    """
    # Define parameters
    problem_id = 1  # Just for identification
    dimension = 1
    n_runs = 10
    budget = 25
    n_doe = 5
    bounds = np.array([[-6.0, 6.0]])
    
    # Create plots directory
    plots_dir = Path("convergence_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Define the optimizers and acquisition functions to test
    optimizers = {
        "GP": "gp", 
        "TabPFN": "tabpfn"
    }
    
    acquisition_functions = [
        "expected_improvement"
    ]
    
    # Run optimizations and create plots
    for acq_func in acquisition_functions:
        print(f"\n{'-'*80}")
        print(f"Running optimizations with {acq_func}")
        print(f"{'-'*80}")
        
        # Dictionary to collect results for all optimizers with this acquisition function
        all_results = {}
        
        for label, optimizer_type in optimizers.items():
            print(f"\nRunning {label} optimizer")
            
            # Run optimization
            results = run_optimization(
                optimizer_type=optimizer_type,
                problem_id=problem_id,
                acquisition_function=acq_func,
                dimension=dimension,
                bounds=bounds,
                budget=budget,
                n_doe=n_doe,
                n_runs=n_runs
            )
            
            # Store results for this optimizer
            all_results[label] = results
            
            # Create standard convergence plot
            plot_convergence(
                results,
                label=label,
                problem_id=problem_id,
                dimension=dimension,
                save_path=plots_dir / f"{optimizer_type}_{acq_func}_convergence.png"
            )
            
            # Create error band plot
            plot_convergence_with_error_bands(
                results,
                label=label,
                problem_id=problem_id,
                dimension=dimension,
                save_path=plots_dir / f"{optimizer_type}_{acq_func}_convergence.png"
            )
        
        # Create comparison plots
        if len(all_results) > 1:
            # Regular comparison plot
            create_comparison_plot(
                all_results,
                problem_id=problem_id,
                dimension=dimension,
                save_path=plots_dir / f"comparison_{acq_func}.png"
            )
            
            # Comparison with error bands
            create_comparison_plot_with_error_bands(
                all_results,
                problem_id=problem_id,
                dimension=dimension,
                save_path=plots_dir / f"comparison_{acq_func}.png"
            )
    
    print("\nAll optimizations and plotting complete!")

if __name__ == "__main__":
    main()