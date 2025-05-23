r"""
This script runs Bayesian Optimization and creates convergence plots from the results.
"""

### -------------------------------------------------------------
### IMPORT LIBRARIES/REPOSITORIES
###---------------------------------------------------------------

# Algorithm import
from Algorithms import Vanilla_BO
from Algorithms.BayesianOptimization.TabPFN_BO.TabPFN_BO import TabPFN_BO

# Choose which algorithm to run: "vanilla" or "tabpfn"
ALGORITHM = "vanilla"  # Change this to "tabpfn" to run the TabPFN_BO algorithm
# Choose which acquisition function to use: "expected_improvement", "probability_of_improvement", "upper_confidence_bound", or "log_expected_improvement"
ACQUISITION_FUNCTION = "expected_improvement"

# Standard libraries
import os
from pathlib import Path
from numpy.linalg import norm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid rendering issues
import matplotlib.pyplot as plt
plt.style.use('default')  # Reset to default style
import pandas as pd
import traceback
import torch

# Check for GPU availability
device = "cuda"


# IOH Experimenter libraries
try:
    from ioh import get_problem
    import ioh.iohcpp.logger as logger_lib
    from ioh.iohcpp.logger import Analyzer
    from ioh.iohcpp.logger.trigger import Each, ON_IMPROVEMENT, ALWAYS
except ModuleNotFoundError as e:
    print(e.args)
except Exception as e:
    print(e.args)

### ---------------------------------------------------------------
### LOGGER SETUP
### ---------------------------------------------------------------

# Define experiment name
experiment_folder = f"bo-experiment-{ALGORITHM}"

# These are the triggers to set how to log data
triggers = [
    Each(1),  # Log after every evaluation (for detailed convergence curves)
    ON_IMPROVEMENT  # Log when there's an improvement
]

# Set algorithm name based on choice
algorithm_name = "Vanilla BO" if ALGORITHM == "vanilla" else "TabPFN BO"
algorithm_info = "Bo-Torch Implementation" if ALGORITHM == "vanilla" else "TabPFN-BoTorch Implementation"

logger = Analyzer(
    triggers=triggers,
    root=os.getcwd(),
    folder_name=experiment_folder,
    algorithm_name=algorithm_name,
    algorithm_info=algorithm_info,
    additional_properties=[logger_lib.property.RAWYBEST],
    store_positions=True
)

# Modify the run_optimization function to better handle this issue

def run_optimization(problem_id=1, dimension=2, budget=None, n_runs=5, instance=1, 
                   fit_mode="fit_with_cache", device="cuda", memory_saving_mode = False):
    """
    Run Bayesian Optimization on the specified problem multiple times.
    Results are saved to .dat files in the experiment folder.
    
    Args:
        problem_id: IOH problem ID
        dimension: Problem dimension
        budget: Evaluation budget (default: 50*dimension or 200, whichever is smaller)
        n_runs: Number of independent runs
        instance: Problem instance
        fit_mode: Mode for TabPFN fitting ("low_memory", "fit_preprocessors", or "fit_with_cache")
        device: Device to use for computation ("auto", "cpu", or "cuda")
    """
    # Create a separate logger for each run to avoid conflicts
    for run in range(1, n_runs+1):
        print(f"\nStarting run {run}/{n_runs} for problem {problem_id}, dimension {dimension}")
        
        try:
            # Set up problem instance
            problem = get_problem(problem_id, instance=run, dimension=dimension)
            
            # Set algorithm name based on global choice
            run_algorithm_name = "Vanilla BO" if ALGORITHM == "vanilla" else "TabPFN BO"
            run_algorithm_info = "Bo-Torch Implementation" if ALGORITHM == "vanilla" else "TabPFN-BoTorch Implementation"
            
            # Create a unique logger for this run to avoid potential conflicts
            run_logger = Analyzer(
                triggers=triggers,
                root=os.getcwd(),
                folder_name=f"{experiment_folder}/run_{run}",  # Separate subfolder
                algorithm_name=f"{run_algorithm_name} Run {run}",
                algorithm_info=run_algorithm_info,
                additional_properties=[logger_lib.property.RAWYBEST],
                store_positions=True
            )
            
            # Attach the run-specific logger
            problem.attach_logger(run_logger)
            
            # Calculate n_DoE based on dimension
            n_DoE = min(3*problem.meta_data.n_variables, 100)  # Cap n_DoE at 100 points
            
            # Set budget based on dimension if not provided, ensuring it's greater than n_DoE
            if budget is None:
                actual_budget = max(n_DoE + 50, min(200, 50*problem.meta_data.n_variables))
            else:
                # Make sure budget is at least n_DoE + 10 to allow for optimization iterations
                actual_budget = max(budget, n_DoE + 10)
                
            # Common parameters for both optimizers
            acquisition_function = ACQUISITION_FUNCTION  # Use the global acquisition function
            random_seed = 15+run  # Different seed for each run
            verbose = True
            DoE_parameters = {'criterion': "center", 'iterations': 1000}
            optimization_parameters = {
                'num_restarts': 10,  # Increase number of restart points
                'raw_samples': 512,  # Increase number of candidates
                'maxiter': 500,  # Allow more iterations for optimization
                'batch_limit': 50,  # Limit batch size for better conditioning
                'options': {'maxiter': 200}  # scipy.optimize options
            }
            
            # Set up the optimizer based on the selected algorithm
            if ALGORITHM == "vanilla":
                # Set up the Vanilla BO with more robust optimization settings
                optimizer = Vanilla_BO(
                    budget=actual_budget,
                    n_DoE=n_DoE,
                    acquisition_function=acquisition_function,
                    random_seed=random_seed,
                    maximisation=False,
                    verbose=verbose,
                    DoE_parameters=DoE_parameters,
                    optimization_parameters=optimization_parameters,
                    device=device  # Use the detected device
                )
            else:
                # Set up the TabPFN BO
                optimizer = TabPFN_BO(
                    budget=actual_budget,
                    n_DoE=n_DoE,
                    acquisition_function=acquisition_function,
                    random_seed=random_seed,
                    n_estimators=16,  # Number of TabPFN estimators
                    fit_mode=fit_mode,  # Use the provided fit_mode
                    device=device,  # Use the detected device
                    memory_saving_mode=memory_saving_mode,
                    maximisation=False,
                    verbose=verbose,
                    DoE_parameters=DoE_parameters,
                    optimization_parameters=optimization_parameters
                )
            
            # Watch the optimizer with the run-specific logger
            run_logger.watch(optimizer, "acquisition_function_name")
            
            # Run optimization
            optimizer(problem=problem)
            
            print(f"Run {run} completed successfully.")
            print(f"  - Final regret: {problem.state.current_best.y - problem.optimum.y:.6e}")
            print(f"  - Number of evaluations: {len(optimizer.f_evals)}")
            
            # Close the run-specific logger
            run_logger.close()
            
        except Exception as e:
            print(f"ERROR: Run {run} failed with error: {e}")
            import traceback
            traceback.print_exc()
            continue

def scientific_format(x, pos=None):
    """Format numbers in scientific notation: a×10^b format"""
    if x == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(x))))
    coef = x/10**exp
    return f'{coef:.1f}×10^{exp}'

def plot_convergence(convergence_data, problem_id, dimension, save_path=None, algorithm_name="Vanilla BO"):
    """
    Plot convergence curves from optimization runs.
    """
    # Clear any existing plots
    plt.clf()
    plt.close('all')
    
    # Sort convergence data by run number
    convergence_data = sorted(convergence_data, key=lambda x: x['run'])
    
    # Create new figure with specific DPI
    fig = plt.figure(figsize=(12, 7), dpi=100)
    ax = fig.add_subplot(111)
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Use color cycle for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(convergence_data))))
    
    # Plot individual runs
    for i, run_data in enumerate(convergence_data):
        ax.plot(
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
    ax.plot(
        range(1, max_length + 1), 
        mean_values, 
        'k-', 
        linewidth=2.5,
        label='Mean'
    )
    
    # Calculate reasonable y-axis limits
    all_values = []
    for data in convergence_data:
        all_values.extend(data['best_so_far'])
    
    y_min = min(all_values)
    y_max = max(all_values)
    
    # Add some padding in log space
    log_range = np.log10(y_max) - np.log10(y_min)
    y_min = 10 ** (np.log10(y_min) - 0.1 * log_range)
    y_max = 10 ** (np.log10(y_max) + 0.1 * log_range)
    
    # Set axis limits
    ax.set_ylim(y_min, y_max)
    
    # Customize plot
    ax.set_title(f"{algorithm_name} Convergence - Problem {problem_id}, Dimension {dimension} ({len(convergence_data)} runs)")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Best Function Value")
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend(loc='best')
    
    # Adjust layout
    fig.tight_layout(pad=2.0)
    
    # Save or display
    if save_path:
        fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.5)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)
    
    # Print summary statistics
    final_values = [data['best_so_far'][-1] for data in convergence_data]
    print(f"\nMean final value: {np.mean(final_values):.6e} ± {np.std(final_values):.6e}")
    
    return fig

def plot_convergence_with_error_bands(convergence_data, problem_id, dimension, save_path=None, algorithm_name="Vanilla BO"):
    """
    Plot convergence curves with mean and standard error bands.
    """
    # Clear any existing plots
    plt.clf()
    plt.close('all')
    
    # Sort convergence data by run number
    convergence_data = sorted(convergence_data, key=lambda x: x['run'])
    
    # Create new figure with specific DPI
    fig = plt.figure(figsize=(12, 7), dpi=100)
    ax = fig.add_subplot(111)
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
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
    ax.plot(
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
    ax.fill_between(
        x_values,
        lower_bound,
        upper_bound,
        color='blue',
        alpha=0.2,
        label='Standard Error'
    )
    
    # Calculate reasonable y-axis limits
    all_values = []
    for data in convergence_data:
        all_values.extend(data['best_so_far'])
    
    y_min = min(all_values)
    y_max = max(all_values)
    
    # Add some padding in log space
    log_range = np.log10(y_max) - np.log10(y_min)
    y_min = 10 ** (np.log10(y_min) - 0.1 * log_range)
    y_max = 10 ** (np.log10(y_max) + 0.1 * log_range)
    
    # Set axis limits
    ax.set_ylim(y_min, y_max)
    
    # Customize plot
    ax.set_title(f"{algorithm_name} Convergence with Error Bands - Problem {problem_id}, Dimension {dimension} ({len(convergence_data)} runs)")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Best Function Value")
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend(loc='best')
    
    # Adjust layout
    fig.tight_layout(pad=2.0)
    
    # Save or display
    if save_path:
        error_band_path = str(save_path).replace('.png', '_error_bands.png')
        fig.savefig(error_band_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.5)
        print(f"Error band plot saved to {error_band_path}")
    else:
        plt.show()
    
    plt.close(fig)
    
    # Print summary statistics
    final_values = [data['best_so_far'][-1] for data in convergence_data]
    print(f"\nMean final value: {np.mean(final_values):.6e} ± {np.std(final_values)/np.sqrt(len(final_values)):.6e} (SE)")
    
    return fig

def read_dat_file(file_path):
    """
    Read a .dat file from IOHprofiler and return a pandas DataFrame
    """
    df = pd.read_csv(file_path, delimiter=' ', skipinitialspace=True)
    return df

def process_dat_file(file_path):
    """
    Process a .dat file and return data in the format expected by plot_convergence
    """
    df = read_dat_file(file_path)
    
    # Extract run number from directory path
    run_number = int(str(file_path).split("run_")[-1].split("\\")[0].split("/")[0])
    
    # Calculate distance from optimum if position columns exist
    x_cols = [col for col in df.columns if col.startswith('x')]
    final_distance = np.linalg.norm(df[x_cols].iloc[-1].values) if len(x_cols) >= 2 else 0
    
    # Create the data structure for plotting
    run_data = {
        'run': run_number,
        'evals': df['evaluations'].tolist(),
        'best_so_far': df['raw_y_best'].tolist(),
        'final_distance': final_distance,
        'final_regret': df['raw_y_best'].iloc[-1]
    }
    
    return run_data

def plot_comparison(vanilla_data, tabpfn_data, problem_id, dimension, save_path=None):
    """
    Plot the mean convergence curves of both Vanilla BO and TabPFN BO on a single plot.
    
    Args:
        vanilla_data: Convergence data for Vanilla BO
        tabpfn_data: Convergence data for TabPFN BO
        problem_id: IOH problem ID
        dimension: Problem dimension
        save_path: Path to save the plot (optional)
    """
    # Clear any existing plots
    plt.clf()
    plt.close('all')
    
    # Create new figure with specific DPI
    fig = plt.figure(figsize=(12, 7), dpi=100)
    ax = fig.add_subplot(111)
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Calculate mean values for vanilla BO
    vanilla_max_length = max(len(data['best_so_far']) for data in vanilla_data)
    vanilla_mean_values = []
    
    for i in range(vanilla_max_length):
        values_at_i = [data['best_so_far'][i] if i < len(data['best_so_far']) else data['best_so_far'][-1] 
                      for data in vanilla_data]
        vanilla_mean_values.append(np.mean(values_at_i))
    
    # Calculate mean values for TabPFN BO
    tabpfn_max_length = max(len(data['best_so_far']) for data in tabpfn_data)
    tabpfn_mean_values = []
    
    for i in range(tabpfn_max_length):
        values_at_i = [data['best_so_far'][i] if i < len(data['best_so_far']) else data['best_so_far'][-1] 
                      for data in tabpfn_data]
        tabpfn_mean_values.append(np.mean(values_at_i))
    
    # Plot mean curves
    ax.plot(
        range(1, vanilla_max_length + 1), 
        vanilla_mean_values, 
        'b-', 
        linewidth=2.5,
        label='Vanilla BO Mean'
    )
    
    ax.plot(
        range(1, tabpfn_max_length + 1), 
        tabpfn_mean_values, 
        'r-', 
        linewidth=2.5,
        label='TabPFN BO Mean'
    )
    
    # Calculate reasonable y-axis limits
    all_values = []
    for data in vanilla_data + tabpfn_data:
        all_values.extend(data['best_so_far'])
    
    y_min = min(all_values)
    y_max = max(all_values)
    
    # Add some padding in log space
    log_range = np.log10(y_max) - np.log10(y_min)
    y_min = 10 ** (np.log10(y_min) - 0.1 * log_range)
    y_max = 10 ** (np.log10(y_max) + 0.1 * log_range)
    
    # Set axis limits
    ax.set_ylim(y_min, y_max)
    
    # Customize plot
    ax.set_title(f"Algorithm Comparison - Problem {problem_id}, Dimension {dimension}")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Best Function Value (log scale)")
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend(loc='best')
    
    # Adjust layout
    fig.tight_layout(pad=2.0)
    
    # Save or display
    if save_path:
        fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.5)
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)
    
    # Print summary statistics for both algorithms
    vanilla_final_values = [data['best_so_far'][-1] for data in vanilla_data]
    tabpfn_final_values = [data['best_so_far'][-1] for data in tabpfn_data]
    
    print("\nComparison of final values:")
    print(f"Vanilla BO mean: {np.mean(vanilla_final_values):.6e} ± {np.std(vanilla_final_values):.6e}")
    print(f"TabPFN BO mean: {np.mean(tabpfn_final_values):.6e} ± {np.std(tabpfn_final_values):.6e}")
    
    return fig

if __name__ == "__main__":
    try:
        # Parameters
        problem_id = 17
        dimension = 40
        n_runs = 5
        budget = 150  # Increased from 100 to allow for optimization iterations after DoE
        instance = 1
        fit_mode = "fit_with_cache"
        
        # To compare Vanilla BO vs TabPFN BO:
        # 1. Run with ALGORITHM = "vanilla" first
        # 2. Then change to ALGORITHM = "tabpfn" and run again
        # 3. To test different acquisition behaviors, change ACQUISITION_FUNCTION to one of:
        #    - "expected_improvement" (default)
        #    - "probability_of_improvement" 
        #    - "upper_confidence_bound"
        #    - "log_expected_improvement"
        
        print(f"Parameters set: problem_id={problem_id}, dimension={dimension}, n_runs={n_runs}")
        print(f"Using algorithm: {ALGORITHM} on device: {device}")
        print(f"Using acquisition function: {ACQUISITION_FUNCTION}")
        print(f"TabPFN fit mode: {fit_mode}")
        
        # Create plots directory if it doesn't exist
        plots_dir = Path("convergence_plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Run the optimization to generate data files
        print("\nRunning optimization...")
        run_optimization(
            problem_id=problem_id,
            dimension=dimension,
            n_runs=n_runs,
            budget=budget,
            instance=instance,
            device=device,  # Pass the detected device
            fit_mode=fit_mode  # Pass the fit mode
        )
        print("Optimization completed.")
        
        # Read the generated data files and create plots
        print("\nReading data files and creating plots...")
        base_dir = Path(experiment_folder)
        
        # Process data from all run directories
        plot_data = []
        for run in range(1, n_runs + 1):
            run_dir = base_dir / f"run_{run}"
            if not run_dir.exists():
                continue
                
            # Find data directories
            data_dirs = [item for item in run_dir.iterdir() 
                        if item.is_dir() and f"data_f{problem_id}" in item.name]
            
            if not data_dirs:
                continue
            
            # Find .dat files
            dat_files = []
            for data_dir in data_dirs:
                dat_files.extend(list(data_dir.glob(f"*_f{problem_id}_*DIM{dimension}*.dat")))
            
            if not dat_files:
                continue
            
            try:
                # Process the first .dat file found
                run_data = process_dat_file(dat_files[0])
                plot_data.append(run_data)
            except Exception as e:
                print(f"Error processing run {run}: {e}")
        
        if not plot_data:
            raise FileNotFoundError(f"No valid data files found for problem {problem_id}, dimension {dimension}")
        
        # Create and save standard convergence plots
        print("Creating standard convergence plot...")
        plot_convergence(
            plot_data,
            problem_id,
            dimension,
            save_path=plots_dir / f"{ALGORITHM}_problem_{problem_id}_dim_{dimension}.png",
            algorithm_name=algorithm_name
        )
        
        # Create and save error band plots
        print("Creating error band convergence plot...")
        plot_convergence_with_error_bands(
            plot_data,
            problem_id,
            dimension,
            save_path=plots_dir / f"{ALGORITHM}_problem_{problem_id}_dim_{dimension}.png",
            algorithm_name=algorithm_name
        )
        
        # If both algorithm data exists, create comparison plot
        vanilla_folder = "bo-experiment-vanilla"
        tabpfn_folder = "bo-experiment-tabpfn"
        
        # Check if data exists for both algorithms
        if Path(vanilla_folder).exists() and Path(tabpfn_folder).exists():
            print("\nCreating algorithm comparison plot...")
            
            # Process vanilla BO data
            vanilla_data = []
            for run in range(1, n_runs + 1):
                run_dir = Path(vanilla_folder) / f"run_{run}"
                if not run_dir.exists():
                    continue
                    
                data_dirs = [item for item in run_dir.iterdir() 
                            if item.is_dir() and f"data_f{problem_id}" in item.name]
                
                dat_files = []
                for data_dir in data_dirs:
                    dat_files.extend(list(data_dir.glob(f"*_f{problem_id}_*DIM{dimension}*.dat")))
                
                if dat_files:
                    try:
                        run_data = process_dat_file(dat_files[0])
                        vanilla_data.append(run_data)
                    except Exception as e:
                        print(f"Error processing Vanilla BO run {run}: {e}")
            
            # Process TabPFN BO data
            tabpfn_data = []
            for run in range(1, n_runs + 1):
                run_dir = Path(tabpfn_folder) / f"run_{run}"
                if not run_dir.exists():
                    continue
                    
                data_dirs = [item for item in run_dir.iterdir() 
                            if item.is_dir() and f"data_f{problem_id}" in item.name]
                
                dat_files = []
                for data_dir in data_dirs:
                    dat_files.extend(list(data_dir.glob(f"*_f{problem_id}_*DIM{dimension}*.dat")))
                
                if dat_files:
                    try:
                        run_data = process_dat_file(dat_files[0])
                        tabpfn_data.append(run_data)
                    except Exception as e:
                        print(f"Error processing TabPFN BO run {run}: {e}")
            
            # Create comparison plot if we have data from both algorithms
            if vanilla_data and tabpfn_data:
                plot_comparison(
                    vanilla_data,
                    tabpfn_data,
                    problem_id,
                    dimension,
                    save_path=plots_dir / f"comparison_problem_{problem_id}_dim_{dimension}.png"
                )
            else:
                print("Not enough data available for comparison plot.")
        
        print("\nOptimization and plotting complete!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()