r"""
This script runs Bayesian Optimization and creates convergence plots from the results.
"""

### -------------------------------------------------------------
### IMPORT LIBRARIES/REPOSITORIES
###---------------------------------------------------------------

# Algorithm import
from Algorithms import Vanilla_BO

# Standard libraries
import os
from pathlib import Path
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import traceback

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
experiment_folder = "bo-experiment"

# These are the triggers to set how to log data
triggers = [
    Each(1),  # Log after every evaluation (for detailed convergence curves)
    ON_IMPROVEMENT  # Log when there's an improvement
]

logger = Analyzer(
    triggers=triggers,
    root=os.getcwd(),
    folder_name=experiment_folder,
    algorithm_name="Vanilla BO",
    algorithm_info="Bo-Torch Implementation",
    additional_properties=[logger_lib.property.RAWYBEST],
    store_positions=True
)

# Modify the run_optimization function to better handle this issue

def run_optimization(problem_id=1, dimension=2, budget=None, n_runs=5):
    """
    Run Bayesian Optimization on the specified problem multiple times.
    Results are saved to .dat files in the experiment folder.
    """
    # Create a separate logger for each run to avoid conflicts
    for run in range(1, n_runs+1):
        print(f"\nStarting run {run}/{n_runs} for problem {problem_id}, dimension {dimension}")
        
        try:
            # Set up problem instance with different instance ID for each run
            problem = get_problem(problem_id, instance=run, dimension=dimension)
            
            # Create a unique logger for this run to avoid potential conflicts
            run_logger = Analyzer(
                triggers=triggers,
                root=os.getcwd(),
                folder_name=f"{experiment_folder}/run_{run}",  # Separate subfolder
                algorithm_name=f"Vanilla BO Run {run}",
                algorithm_info="Bo-Torch Implementation",
                additional_properties=[logger_lib.property.RAWYBEST],
                store_positions=True
            )
            
            # Attach the run-specific logger
            problem.attach_logger(run_logger)
            
            # Set budget based on dimension if not provided
            if budget is None:
                actual_budget = min(200, 50*problem.meta_data.n_variables)
            else:
                actual_budget = budget
                
            # Set up the Vanilla BO with a different random seed for each run
            optimizer = Vanilla_BO(
                budget=actual_budget,
                n_DoE=3*problem.meta_data.n_variables,
                acquisition_function="probability_of_improvement",
                random_seed=15+run,  # Different seed for each run
                maximisation=False,
                verbose=True,
                DoE_parameters={'criterion': "center", 'iterations': 1000}
            )
            
            # Watch the optimizer with the run-specific logger
            run_logger.watch(optimizer, "acquistion_function_name")
            
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

def plot_convergence(convergence_data, problem_id, dimension, save_path=None):
    """
    Plot convergence curves from optimization runs.
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
    plt.title(f"Convergence - Problem {problem_id}, Dimension {dimension} ({len(convergence_data)} runs)")
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

def plot_convergence_with_error_bands(convergence_data, problem_id, dimension, save_path=None):
    """
    Plot convergence curves with mean and standard error bands.
    
    Args:
        convergence_data: List of dictionaries containing run data
        problem_id: ID of the problem being optimized
        dimension: Dimension of the problem
        save_path: Path to save the plot (optional)
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
    plt.title(f"Convergence with Error Bands - Problem {problem_id}, Dimension {dimension} ({len(convergence_data)} runs)")
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

if __name__ == "__main__":
    try:
        # Parameters
        problem_id = 1  # Sphere function
        dimension = 2
        n_runs = 10
        
        print(f"Parameters set: problem_id={problem_id}, dimension={dimension}, n_runs={n_runs}")
        
        # Create plots directory if it doesn't exist
        plots_dir = Path("convergence_plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Run the optimization to generate data files
        print("\nRunning optimization...")
        run_optimization(
            problem_id=problem_id,
            dimension=dimension,
            n_runs=n_runs
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
            save_path=plots_dir / f"problem_{problem_id}_dim_{dimension}.png"
        )
        
        # Create and save error band plots
        print("Creating error band convergence plot...")
        plot_convergence_with_error_bands(
            plot_data,
            problem_id,
            dimension,
            save_path=plots_dir / f"problem_{problem_id}_dim_{dimension}.png"
        )
        
        print("\nOptimization and plotting complete!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()