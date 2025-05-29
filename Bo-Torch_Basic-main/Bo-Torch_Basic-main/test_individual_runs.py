r"""
Test script that runs function 21 in dimension 2 for instance 1.
Shows individual runs for both TabPFN and Vanilla BO algorithms.
5 runs each, with plots showing individual convergence curves.
"""

### -------------------------------------------------------------
### IMPORT LIBRARIES/REPOSITORIES
###---------------------------------------------------------------

# Algorithm import
from Algorithms import Vanilla_BO
from Algorithms.BayesianOptimization.TabPFN_BO.TabPFN_BO import TabPFN_BO

# Standard libraries
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid rendering issues
import matplotlib.pyplot as plt
plt.style.use('default')  # Reset to default style
import pandas as pd
import traceback
import torch
import json
from datetime import datetime
import time

# Check for GPU availability
device = "cpu"  # Force CPU usage

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
### TEST CONFIGURATION
### ---------------------------------------------------------------

# Test parameters
PROBLEM_ID = 21  # Gallagher's Gaussian 21-hi Peaks
DIMENSION = 2
INSTANCE = 1
N_RUNS = 5
BASE_SEED = 42

# Algorithms to compare
ALGORITHMS = ["vanilla", "tabpfn"]
ACQUISITION_FUNCTION = "expected_improvement"

# Budget configuration
def get_budget(dimension):
    """Calculate budget based on dimension"""
    return 10 * dimension + 50  # Budget = 10*D + 50

def get_n_doe(dimension):
    """Calculate number of initial design points based on dimension"""
    return min(2 * dimension, 20)  # Cap at 20 initial points

def process_dat_file(file_path):
    """
    Process a .dat file and return data in the format expected by plot_convergence.
    Same function as in OG_convergence.py.
    """
    df = pd.read_csv(file_path, delimiter=' ', skipinitialspace=True)
    
    # Extract run number from filename or path
    filename = file_path.stem
    parts = filename.split('_')
    
    # Try to extract run number from filename
    run_number = 1
    for part in parts:
        if part.startswith('rep'):
            try:
                run_number = int(part[3:])
                break
            except:
                pass
    
    # Create the data structure for plotting
    run_data = {
        'run': run_number,
        'evals': df['evaluations'].tolist(),
        'best_so_far': df['raw_y_best'].tolist(),
        'final_value': df['raw_y_best'].iloc[-1]
    }
    
    return run_data

### ---------------------------------------------------------------
### EXPERIMENT EXECUTION
### ---------------------------------------------------------------

def run_single_optimization(algorithm, problem_id, dimension, instance, run_number, 
                          output_dir, fit_mode="fit_with_cache"):
    """
    Run a single optimization experiment.
    """
    # Calculate seed for reproducibility
    seed = BASE_SEED + run_number
    
    # Set up problem
    problem = get_problem(problem_id, instance=instance, dimension=dimension)
    
    # Calculate budget and DoE
    budget = get_budget(dimension)
    n_DoE = get_n_doe(dimension)
    
    # Create unique logger for this run
    run_name = f"{algorithm}_f{problem_id}_dim{dimension}_inst{instance}_rep{run_number}"
    logger_dir = output_dir / run_name
    
    # Set up logger
    triggers = [Each(1), ON_IMPROVEMENT]
    run_logger = Analyzer(
        triggers=triggers,
        root=str(logger_dir.parent),
        folder_name=logger_dir.name,
        algorithm_name=f"{algorithm.title()} BO",
        algorithm_info=f"Function {problem_id}, Dim {dimension}, Instance {instance}, Run {run_number}",
        additional_properties=[logger_lib.property.RAWYBEST],
        store_positions=True
    )
    
    problem.attach_logger(run_logger)
    
    try:
        # Common parameters
        common_params = {
            'budget': budget,
            'n_DoE': n_DoE,
            'acquisition_function': ACQUISITION_FUNCTION,
            'random_seed': seed,
            'maximisation': False,
            'verbose': True,  # Show progress for individual runs
            'device': "cpu"
        }
        
        # Create optimizer based on algorithm
        if algorithm == "vanilla":
            optimizer = Vanilla_BO(**common_params)
        elif algorithm == "tabpfn":
            optimizer = TabPFN_BO(
                **common_params,
                n_estimators=8,  # Conservative for testing
                fit_mode=fit_mode
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Watch the optimizer
        run_logger.watch(optimizer, "acquisition_function_name")
        
        # Run optimization
        start_time = time.time()
        optimizer(problem=problem)
        end_time = time.time()
        
        # Collect results
        result = {
            'algorithm': algorithm,
            'problem_id': problem_id,
            'dimension': dimension,
            'instance': instance,
            'run': run_number,
            'seed': seed,
            'budget': budget,
            'n_DoE': n_DoE,
            'final_best': float(optimizer.current_best),
            'final_regret': float(problem.state.current_best.y - problem.optimum.y),
            'n_evaluations': len(optimizer.f_evals),
            'runtime_seconds': end_time - start_time,
            'success': True
        }
        
        print(f"    ✓ {run_name}: Final regret = {result['final_regret']:.6e} in {result['runtime_seconds']:.1f}s")
        
    except Exception as e:
        print(f"    ✗ {run_name}: FAILED - {str(e)}")
        result = {
            'algorithm': algorithm,
            'problem_id': problem_id,
            'dimension': dimension,
            'instance': instance,
            'run': run_number,
            'seed': seed,
            'success': False,
            'error': str(e)
        }
    
    finally:
        run_logger.close()
    
    return result

### ---------------------------------------------------------------
### PLOTTING FUNCTIONS
### ---------------------------------------------------------------

def plot_individual_runs_comparison(vanilla_data, tabpfn_data, problem_id, dimension, instance, save_path):
    """
    Plot individual runs for both algorithms with mean curves.
    Similar to OG_convergence.py plot_convergence but comparing two algorithms.
    """
    plt.clf()
    plt.close('all')
    
    # Create new figure with specific DPI
    fig = plt.figure(figsize=(14, 8), dpi=100)
    ax = fig.add_subplot(111)
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Colors for algorithms
    vanilla_color = 'blue'
    tabpfn_color = 'red'
    
    # Plot individual Vanilla BO runs
    if vanilla_data:
        for i, run_data in enumerate(vanilla_data):
            ax.plot(
                run_data['evals'], 
                run_data['best_so_far'], 
                color=vanilla_color,
                linewidth=1.5,
                alpha=0.7,
                label=f"Vanilla BO Run {run_data['run']}" if i < 3 else None  # Only label first 3 for clarity
            )
        
        # Calculate and plot Vanilla BO mean
        max_length = max(len(data['best_so_far']) for data in vanilla_data)
        vanilla_mean = []
        
        for j in range(max_length):
            values_at_j = [data['best_so_far'][j] if j < len(data['best_so_far']) 
                          else data['best_so_far'][-1] for data in vanilla_data]
            vanilla_mean.append(np.mean(values_at_j))
        
        # Plot Vanilla BO mean curve
        ax.plot(
            range(1, max_length + 1), 
            vanilla_mean, 
            color=vanilla_color,
            linewidth=3.0,
            linestyle='--',
            label=f'Vanilla BO Mean ({len(vanilla_data)} runs)'
        )
    
    # Plot individual TabPFN BO runs
    if tabpfn_data:
        for i, run_data in enumerate(tabpfn_data):
            ax.plot(
                run_data['evals'], 
                run_data['best_so_far'], 
                color=tabpfn_color,
                linewidth=1.5,
                alpha=0.7,
                label=f"TabPFN BO Run {run_data['run']}" if i < 3 else None  # Only label first 3 for clarity
            )
        
        # Calculate and plot TabPFN BO mean
        max_length = max(len(data['best_so_far']) for data in tabpfn_data)
        tabpfn_mean = []
        
        for j in range(max_length):
            values_at_j = [data['best_so_far'][j] if j < len(data['best_so_far']) 
                          else data['best_so_far'][-1] for data in tabpfn_data]
            tabpfn_mean.append(np.mean(values_at_j))
        
        # Plot TabPFN BO mean curve
        ax.plot(
            range(1, max_length + 1), 
            tabpfn_mean, 
            color=tabpfn_color,
            linewidth=3.0,
            linestyle='--',
            label=f'TabPFN BO Mean ({len(tabpfn_data)} runs)'
        )
    
    # Calculate reasonable y-axis limits from all data
    all_values = []
    if vanilla_data:
        for data in vanilla_data:
            all_values.extend(data['best_so_far'])
    if tabpfn_data:
        for data in tabpfn_data:
            all_values.extend(data['best_so_far'])
    
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        
        # Add some padding in log space
        log_range = np.log10(y_max) - np.log10(y_min)
        y_min = 10 ** (np.log10(y_min) - 0.1 * log_range)
        y_max = 10 ** (np.log10(y_max) + 0.1 * log_range)
        
        # Set axis limits
        ax.set_ylim(y_min, y_max)
    
    # Customize plot
    ax.set_title(f"Individual Runs Comparison - BBOB f{problem_id} (Gallagher), {dimension}D, Instance {instance}")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Best Function Value (log scale)")
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend(loc='best', fontsize=9)
    
    # Adjust layout
    fig.tight_layout(pad=2.0)
    
    # Save plot
    fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    
    # Print summary statistics
    print(f"\n📊 Final Performance Summary:")
    if vanilla_data:
        vanilla_finals = [data['final_value'] for data in vanilla_data]
        print(f"{'Vanilla BO':15}: {np.mean(vanilla_finals):.6e} ± {np.std(vanilla_finals):.6e}")
    if tabpfn_data:
        tabpfn_finals = [data['final_value'] for data in tabpfn_data]
        print(f"{'TabPFN BO':15}: {np.mean(tabpfn_finals):.6e} ± {np.std(tabpfn_finals):.6e}")
    
    print(f"📈 Individual runs plot saved: {save_path.name}")
    
    return fig

### ---------------------------------------------------------------
### MAIN TEST EXECUTION
### ---------------------------------------------------------------

def run_test():
    """Run the individual runs test"""
    
    # Create output directory
    output_dir = Path("individual_runs_test")
    output_dir.mkdir(exist_ok=True)
    
    budget = get_budget(DIMENSION)
    n_doe = get_n_doe(DIMENSION)
    
    print(f"🧪 Individual Runs Test")
    print(f"📋 Function: {PROBLEM_ID} (Gallagher's Gaussian 21-hi Peaks)")
    print(f"📋 Dimension: {DIMENSION}D | Instance: {INSTANCE} | Runs: {N_RUNS}")
    print(f"📋 Budget: {budget} | DoE: {n_doe}")
    print("=" * 60)
    
    # Store results for both algorithms
    all_results = {'vanilla': [], 'tabpfn': []}
    
    # Run experiments for each algorithm
    for algorithm in ALGORITHMS:
        print(f"\n🔍 Running {algorithm.upper()} BO:")
        
        algo_dir = output_dir / algorithm
        algo_dir.mkdir(exist_ok=True)
        
        for run in range(1, N_RUNS + 1):
            try:
                result = run_single_optimization(
                    algorithm=algorithm,
                    problem_id=PROBLEM_ID,
                    dimension=DIMENSION,
                    instance=INSTANCE,
                    run_number=run,
                    output_dir=algo_dir
                )
                all_results[algorithm].append(result)
                
            except Exception as e:
                print(f"    ✗ Error in {algorithm} run {run}: {e}")
                continue
    
    # Process .dat files and create plots
    print(f"\n📊 Processing results and creating plots...")
    
    # Read .dat files for both algorithms
    vanilla_data = []
    tabpfn_data = []
    
    # Process Vanilla BO .dat files
    vanilla_dir = output_dir / "vanilla"
    if vanilla_dir.exists():
        dat_files = list(vanilla_dir.rglob("*.dat"))
        for dat_file in dat_files:
            try:
                if f"_f{PROBLEM_ID}_" in dat_file.name:
                    run_data = process_dat_file(dat_file)
                    vanilla_data.append(run_data)
            except Exception as e:
                print(f"Error processing vanilla .dat file {dat_file}: {e}")
    
    # Process TabPFN BO .dat files
    tabpfn_dir = output_dir / "tabpfn"
    if tabpfn_dir.exists():
        dat_files = list(tabpfn_dir.rglob("*.dat"))
        for dat_file in dat_files:
            try:
                if f"_f{PROBLEM_ID}_" in dat_file.name:
                    run_data = process_dat_file(dat_file)
                    tabpfn_data.append(run_data)
            except Exception as e:
                print(f"Error processing tabpfn .dat file {dat_file}: {e}")
    
    # Sort data by run number
    vanilla_data = sorted(vanilla_data, key=lambda x: x['run'])
    tabpfn_data = sorted(tabpfn_data, key=lambda x: x['run'])
    
    # Create individual runs comparison plot
    if vanilla_data or tabpfn_data:
        plot_path = output_dir / f"individual_runs_f{PROBLEM_ID}_dim{DIMENSION}_inst{INSTANCE}.png"
        plot_individual_runs_comparison(
            vanilla_data=vanilla_data,
            tabpfn_data=tabpfn_data,
            problem_id=PROBLEM_ID,
            dimension=DIMENSION,
            instance=INSTANCE,
            save_path=plot_path
        )
    
    # Save experiment summary
    experiment_log = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "problem_id": PROBLEM_ID,
            "dimension": DIMENSION,
            "instance": INSTANCE,
            "n_runs": N_RUNS,
            "budget": budget,
            "n_doe": n_doe,
            "algorithms": ALGORITHMS
        },
        "results": all_results
    }
    
    log_file = output_dir / "experiment_log.json"
    with open(log_file, 'w') as f:
        json.dump(experiment_log, f, indent=2)
    
    print(f"\n🎉 Individual Runs Test Complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📄 Experiment log: {log_file}")

if __name__ == "__main__":
    try:
        # Start the test
        run_test()
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        traceback.print_exc() 