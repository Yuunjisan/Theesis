r"""
This script runs a comprehensive publication-quality benchmark comparing Bayesian Optimization algorithms
on BBOB functions across multiple dimensions, instances, and repetitions.
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
from numpy.linalg import norm
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
### BENCHMARK CONFIGURATION
### ---------------------------------------------------------------

# BBOB functions to test (can modify this list)
BBOB_FUNCTIONS = list(range(1, 25))  # Functions 1-24
DIMENSIONS = [2]  # Test dimensions
INSTANCES = [1, 2, 3]  # Problem instances
N_REPETITIONS = 5  # Number of runs with different seeds
BASE_SEED = 42  # Base seed for reproducibility

# Algorithms to compare
ALGORITHMS = ["vanilla", "tabpfn"]
ACQUISITION_FUNCTION = "expected_improvement"

# Budget configuration (adaptive based on dimension)
def get_budget(dimension):
    """Calculate budget based on dimension"""
    return 10 * dimension + 50  # Budget = 10*D + 50

def get_n_doe(dimension):
    """Calculate number of initial design points based on dimension"""
    return min(2 * dimension, 20)  # Cap at 20 initial points

### ---------------------------------------------------------------
### EXPERIMENT MANAGEMENT
### ---------------------------------------------------------------

class BenchmarkManager:
    """Manages the comprehensive benchmark experiment"""
    
    def __init__(self, base_output_dir="bbob_benchmark_results"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create dimension-based folder structure
        for dim in DIMENSIONS:
            dim_dir = self.base_output_dir / f"dim_{dim}"
            dim_dir.mkdir(exist_ok=True)
            
            # Create algorithm subdirectories
            for algorithm in ALGORITHMS:
                algo_dir = dim_dir / algorithm
                algo_dir.mkdir(exist_ok=True)
        
        # Create summary directory
        self.summary_dir = self.base_output_dir / "summary_plots"
        self.summary_dir.mkdir(exist_ok=True)
        
        # Initialize runtime tracking
        self.runtime_data = {
            'vanilla': [],
            'tabpfn': []
        }
        self.runtime_summary = {}
        
        # Initialize experiment log
        self.experiment_log = {
            "start_time": datetime.now().isoformat(),
            "configuration": {
                "bbob_functions": BBOB_FUNCTIONS,
                "dimensions": DIMENSIONS,
                "instances": INSTANCES,
                "n_repetitions": N_REPETITIONS,
                "algorithms": ALGORITHMS,
                "acquisition_function": ACQUISITION_FUNCTION
            },
            "results": {}
        }
        
        print(f"Benchmark results will be saved to: {self.base_output_dir}")
        print(f"Total experiments to run: {len(BBOB_FUNCTIONS) * len(DIMENSIONS) * len(INSTANCES) * N_REPETITIONS * len(ALGORITHMS)}")

    def add_runtime_result(self, result):
        """Add runtime result for analysis"""
        if result.get('success', False):
            algorithm = result['algorithm']
            if algorithm in self.runtime_data:
                self.runtime_data[algorithm].append({
                    'runtime': result['runtime_seconds'],
                    'problem_id': result['problem_id'],
                    'dimension': result['dimension'],
                    'instance': result['instance'],
                    'repetition': result['repetition'],
                    'final_regret': result['final_regret']
                })

    def calculate_runtime_summary(self):
        """Calculate runtime statistics for all algorithms"""
        self.runtime_summary = {}
        
        for algorithm, data in self.runtime_data.items():
            if data:
                runtimes = [d['runtime'] for d in data]
                self.runtime_summary[algorithm] = {
                    'total_runtime': sum(runtimes),
                    'mean_runtime': np.mean(runtimes),
                    'std_runtime': np.std(runtimes),
                    'median_runtime': np.median(runtimes),
                    'min_runtime': min(runtimes),
                    'max_runtime': max(runtimes),
                    'n_runs': len(runtimes),
                    'speedup_vs_baseline': None  # Will be calculated
                }
        
        # Calculate speedup (assume vanilla is baseline)
        if 'vanilla' in self.runtime_summary and 'tabpfn' in self.runtime_summary:
            vanilla_mean = self.runtime_summary['vanilla']['mean_runtime']
            tabpfn_mean = self.runtime_summary['tabpfn']['mean_runtime']
            
            self.runtime_summary['tabpfn']['speedup_vs_vanilla'] = vanilla_mean / tabpfn_mean
            self.runtime_summary['vanilla']['speedup_vs_vanilla'] = 1.0

    def print_runtime_summary(self):
        """Print comprehensive runtime comparison"""
        print("\n" + "="*80)
        print("🕐 RUNTIME ANALYSIS SUMMARY")
        print("="*80)
        
        if not self.runtime_summary:
            print("No runtime data available.")
            return
        
        # Table format
        print(f"{'Algorithm':<15} {'Total (s)':<12} {'Mean (s)':<12} {'Std (s)':<12} {'Speedup':<10}")
        print("-" * 70)
        
        for algorithm, stats in self.runtime_summary.items():
            speedup = stats.get('speedup_vs_vanilla', '-')
            speedup_str = f"{speedup:.2f}x" if isinstance(speedup, float) else speedup
            
            print(f"{algorithm.title():<15} "
                  f"{stats['total_runtime']:<12.1f} "
                  f"{stats['mean_runtime']:<12.3f} "
                  f"{stats['std_runtime']:<12.3f} "
                  f"{speedup_str:<10}")
        
        # Detailed statistics
        print(f"\nDetailed Statistics:")
        for algorithm, stats in self.runtime_summary.items():
            print(f"\n{algorithm.title()} BO:")
            print(f"  • Total runtime:   {stats['total_runtime']:.1f} seconds")
            print(f"  • Runs completed:  {stats['n_runs']}")
            print(f"  • Mean per run:    {stats['mean_runtime']:.3f} ± {stats['std_runtime']:.3f} seconds")
            print(f"  • Median per run:  {stats['median_runtime']:.3f} seconds")
            print(f"  • Range:           {stats['min_runtime']:.3f} - {stats['max_runtime']:.3f} seconds")
            
            if 'speedup_vs_vanilla' in stats and stats['speedup_vs_vanilla'] != 1.0:
                speedup = stats['speedup_vs_vanilla']
                if speedup > 1:
                    print(f"  • 🚀 {speedup:.2f}x FASTER than Vanilla GP")
                else:
                    print(f"  • 🐌 {1/speedup:.2f}x SLOWER than Vanilla GP")

def run_single_optimization(algorithm, problem_id, dimension, instance, repetition, 
                          output_dir, fit_mode="fit_with_cache"):
    """
    Run a single optimization experiment.
    
    Args:
        algorithm: "vanilla" or "tabpfn"
        problem_id: BBOB function ID (1-24)
        dimension: Problem dimension
        instance: Problem instance (1-3)
        repetition: Repetition number (1-N_REPETITIONS)
        output_dir: Directory to save results
        fit_mode: TabPFN fit mode
    
    Returns:
        dict: Results summary
    """
    # Calculate seed for reproducibility
    seed = BASE_SEED + repetition
    
    # Set up problem
    problem = get_problem(problem_id, instance=instance, dimension=dimension)
    
    # Calculate budget and DoE
    budget = get_budget(dimension)
    n_DoE = get_n_doe(dimension)
    
    # Create unique logger for this run
    run_name = f"{algorithm}_f{problem_id}_dim{dimension}_inst{instance}_rep{repetition}"
    logger_dir = output_dir / run_name
    
    # Set up logger
    triggers = [Each(1), ON_IMPROVEMENT]
    run_logger = Analyzer(
        triggers=triggers,
        root=str(logger_dir.parent),
        folder_name=logger_dir.name,
        algorithm_name=f"{algorithm.title()} BO",
        algorithm_info=f"Function {problem_id}, Dim {dimension}, Instance {instance}, Rep {repetition}",
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
            'verbose': False,  # Set to False for cleaner output during benchmark
            'device': "cpu"
        }
        
        # Create optimizer based on algorithm
        if algorithm == "vanilla":
            optimizer = Vanilla_BO(**common_params)
        elif algorithm == "tabpfn":
            optimizer = TabPFN_BO(
                **common_params,
                n_estimators=8,  # Conservative for large benchmark
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
            'repetition': repetition,
            'seed': seed,
            'budget': budget,
            'n_DoE': n_DoE,
            'final_best': float(optimizer.current_best),
            'final_regret': float(problem.state.current_best.y - problem.optimum.y),
            'n_evaluations': len(optimizer.f_evals),
            'runtime_seconds': end_time - start_time,
            'success': True
        }
        
        print(f"✓ {run_name}: Final regret = {result['final_regret']:.6e} in {result['runtime_seconds']:.1f}s")
        
    except Exception as e:
        print(f"✗ {run_name}: FAILED - {str(e)}")
        result = {
            'algorithm': algorithm,
            'problem_id': problem_id,
            'dimension': dimension,
            'instance': instance,
            'repetition': repetition,
            'seed': seed,
            'success': False,
            'error': str(e)
        }
    
    finally:
        run_logger.close()
    
    return result

def process_dat_file(file_path):
    """
    Process a .dat file and return data in the format expected by plot_convergence.
    Same function as in OG_convergence.py.
    """
    df = pd.read_csv(file_path, delimiter=' ', skipinitialspace=True)
    
    # Extract run number from directory or filename
    run_number = 1  # Default run number for benchmark
    
    # Create the data structure for plotting
    run_data = {
        'run': run_number,
        'evals': df['evaluations'].tolist(),
        'best_so_far': df['raw_y_best'].tolist(),
        'final_value': df['raw_y_best'].iloc[-1]
    }
    
    return run_data

def plot_function_comparison(vanilla_data, tabpfn_data, problem_id, dimension, save_path):
    """
    Plot comparison between algorithms for a specific function and dimension.
    Shows only mean convergence curves (no individual runs).
    Matches the exact format of plot_multi_algorithm_comparison from OG_convergence.py.
    """
    plt.clf()
    plt.close('all')
    
    # Create new figure with specific DPI (same as OG_convergence.py)
    fig = plt.figure(figsize=(12, 7), dpi=100)
    ax = fig.add_subplot(111)
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Define colors for different algorithms (same as OG_convergence.py)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Prepare algorithm data in same format as OG_convergence.py
    algorithm_data = {}
    if vanilla_data:
        algorithm_data["Vanilla BO"] = vanilla_data
    if tabpfn_data:
        algorithm_data["TabPFN BO"] = tabpfn_data
    
    # Plot each algorithm (same logic as OG_convergence.py)
    for i, (algorithm_name, convergence_data) in enumerate(algorithm_data.items()):
        # Calculate mean values (exact same as OG_convergence.py)
        max_length = max(len(data['best_so_far']) for data in convergence_data)
        mean_values = []
        
        for j in range(max_length):
            values_at_j = [data['best_so_far'][j] if j < len(data['best_so_far']) else data['best_so_far'][-1] 
                          for data in convergence_data]
            mean_values.append(np.mean(values_at_j))
        
        # Plot mean curve (exact same as OG_convergence.py)
        ax.plot(
            range(1, max_length + 1), 
            mean_values, 
            color=colors[i % len(colors)],
            linewidth=2.5,
            label=f'{algorithm_name} Mean ({len(convergence_data)} runs)'
        )
    
    # Calculate reasonable y-axis limits from all data (same as OG_convergence.py)
    all_values = []
    for convergence_data in algorithm_data.values():
        for data in convergence_data:
            all_values.extend(data['best_so_far'])
    
    y_min = min(all_values)
    y_max = max(all_values)
    
    # Add some padding in log space (same as OG_convergence.py)
    log_range = np.log10(y_max) - np.log10(y_min)
    y_min = 10 ** (np.log10(y_min) - 0.1 * log_range)
    y_max = 10 ** (np.log10(y_max) + 0.1 * log_range)
    
    # Set axis limits
    ax.set_ylim(y_min, y_max)
    
    # Customize plot (same as OG_convergence.py)
    ax.set_title(f"Multi-Algorithm Comparison - Problem {problem_id}, Dimension {dimension}")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Best Function Value (log scale)")
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend(loc='best')
    
    # Adjust layout (same as OG_convergence.py)
    fig.tight_layout(pad=2.0)
    
    # Save plot (same parameters as OG_convergence.py)
    fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    
    # Print summary statistics for all algorithms (same as OG_convergence.py)
    print("\nComparison of final values across algorithms:")
    for algorithm_name, convergence_data in algorithm_data.items():
        final_values = [data['best_so_far'][-1] for data in convergence_data]
        print(f"{algorithm_name:20}: {np.mean(final_values):.6e} ± {np.std(final_values):.6e}")
    
    print(f"    📈 Multi-algorithm comparison plot saved: {save_path.name}")
    
    return fig

def plot_runtime_comparison(benchmark_manager, save_dir):
    """Create runtime comparison plots"""
    runtime_data = benchmark_manager.runtime_data
    runtime_summary = benchmark_manager.runtime_summary
    
    if not runtime_data or len(runtime_data) < 2:
        print("Insufficient data for runtime comparison plots")
        return
    
    # Create runtime plots directory
    runtime_plots_dir = save_dir / "runtime_analysis"
    runtime_plots_dir.mkdir(exist_ok=True)
    
    # Cumulative runtime over experiments
    plt.figure(figsize=(12, 6), dpi=100)
    
    algorithms = list(runtime_data.keys())
    for algo in algorithms:
        runtimes = [d['runtime'] for d in runtime_data[algo]]
        cumulative_runtime = np.cumsum(runtimes)
        experiments = range(1, len(runtimes) + 1)
        
        plt.plot(experiments, cumulative_runtime, 
                label=f'{algo.title()} BO (Total: {cumulative_runtime[-1]:.1f}s)', 
                linewidth=2)
    
    plt.title('Cumulative Runtime Over Experiments', fontsize=14, fontweight='bold')
    plt.xlabel('Experiment Number', fontsize=12)
    plt.ylabel('Cumulative Runtime (seconds)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(runtime_plots_dir / "cumulative_runtime.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Runtime analysis plots saved to: {runtime_plots_dir}")
    
    return runtime_plots_dir

def save_runtime_report(benchmark_manager, save_path):
    """Save detailed runtime analysis report"""
    
    report = []
    report.append("="*80)
    report.append("TABPFN vs VANILLA GP - RUNTIME ANALYSIS REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    if not benchmark_manager.runtime_summary:
        report.append("No runtime data available.")
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        return
    
    # Summary table
    report.append("SUMMARY")
    report.append("-" * 40)
    for algorithm, stats in benchmark_manager.runtime_summary.items():
        report.append(f"{algorithm.title()} BO:")
        report.append(f"  Total Runtime:    {stats['total_runtime']:.1f} seconds")
        report.append(f"  Experiments:      {stats['n_runs']}")
        report.append(f"  Mean per run:     {stats['mean_runtime']:.3f} ± {stats['std_runtime']:.3f} seconds")
        
        if 'speedup_vs_vanilla' in stats:
            speedup = stats['speedup_vs_vanilla']
            if speedup != 1.0:
                report.append(f"  Speedup:          {speedup:.2f}x vs Vanilla")
        report.append("")
    
    # Performance analysis by dimension
    report.append("PERFORMANCE BY DIMENSION")
    report.append("-" * 40)
    
    for algo in benchmark_manager.runtime_data:
        report.append(f"{algo.title()} BO:")
        
        # Group by dimension
        dim_stats = {}
        for d in benchmark_manager.runtime_data[algo]:
            dim = d['dimension']
            if dim not in dim_stats:
                dim_stats[dim] = []
            dim_stats[dim].append(d['runtime'])
        
        for dim in sorted(dim_stats.keys()):
            runtimes = dim_stats[dim]
            report.append(f"  Dimension {dim}D: {np.mean(runtimes):.3f} ± {np.std(runtimes):.3f} seconds (n={len(runtimes)})")
        report.append("")
    
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"📄 Runtime report saved: {save_path}")

def run_benchmark():
    """Run the complete benchmark experiment"""
    
    benchmark_manager = BenchmarkManager()
    
    print(f"\n🚀 Starting BBOB Benchmark")
    print(f"Functions: {len(BBOB_FUNCTIONS)} | Dimensions: {DIMENSIONS}")
    print(f"Instances: {INSTANCES} | Repetitions: {N_REPETITIONS}")
    print(f"Algorithms: {ALGORITHMS}")
    print("=" * 80)
    
    total_experiments = len(BBOB_FUNCTIONS) * len(DIMENSIONS) * len(INSTANCES) * N_REPETITIONS * len(ALGORITHMS)
    completed_experiments = 0
    
    # Track start time for total benchmark
    total_start_time = time.time()
    
    # Run experiments for each dimension
    for dimension in DIMENSIONS:
        print(f"\n📊 Processing Dimension {dimension}D")
        print("-" * 40)
        
        dim_dir = benchmark_manager.base_output_dir / f"dim_{dimension}"
        
        # For each function
        for problem_id in BBOB_FUNCTIONS:
            print(f"\nFunction {problem_id:2d}/24 (Dim {dimension}D):")
            
            # For each instance
            for instance in INSTANCES:
                # For each repetition
                for repetition in range(1, N_REPETITIONS + 1):
                    # For each algorithm
                    for algorithm in ALGORITHMS:
                        try:
                            result = run_single_optimization(
                                algorithm=algorithm,
                                problem_id=problem_id,
                                dimension=dimension,
                                instance=instance,
                                repetition=repetition,
                                output_dir=dim_dir / algorithm
                            )
                            
                            completed_experiments += 1
                            
                            # Track runtime data
                            benchmark_manager.add_runtime_result(result)
                            
                        except Exception as e:
                            print(f"    ✗ Error in {algorithm}: {e}")
                            completed_experiments += 1
                    continue
            
            # Create comparison plot for this function and dimension using .dat files
            if True:  # Always try to create plots from .dat files
                # Process .dat files from both algorithms (same as OG_convergence.py)
                vanilla_data = []
                tabpfn_data = []
                
                # Read Vanilla BO .dat files
                vanilla_dir = dim_dir / "vanilla"
                if vanilla_dir.exists():
                    dat_files = list(vanilla_dir.rglob("*.dat"))
                    for dat_file in dat_files:
                        try:
                            if f"_f{problem_id}_" in dat_file.name:
                                run_data = process_dat_file(dat_file)
                                vanilla_data.append(run_data)
                        except Exception as e:
                            print(f"Error processing vanilla .dat file {dat_file}: {e}")
                
                # Read TabPFN BO .dat files  
                tabpfn_dir = dim_dir / "tabpfn"
                if tabpfn_dir.exists():
                    dat_files = list(tabpfn_dir.rglob("*.dat"))
                    for dat_file in dat_files:
                        try:
                            if f"_f{problem_id}_" in dat_file.name:
                                run_data = process_dat_file(dat_file)
                                tabpfn_data.append(run_data)
                        except Exception as e:
                            print(f"Error processing tabpfn .dat file {dat_file}: {e}")
                
                if vanilla_data or tabpfn_data:
                    plot_path = dim_dir / f"function_{problem_id:02d}_comparison.png"
                    plot_function_comparison(
                        vanilla_data=vanilla_data,
                        tabpfn_data=tabpfn_data,
                        problem_id=problem_id,
                        dimension=dimension,
                        save_path=plot_path
                    )
            
            # Progress update
            progress = (completed_experiments / total_experiments) * 100
            print(f"    Progress: {completed_experiments}/{total_experiments} ({progress:.1f}%)")
    
    # Calculate total benchmark time
    total_end_time = time.time()
    total_benchmark_time = total_end_time - total_start_time
    
    # Calculate runtime summary
    benchmark_manager.calculate_runtime_summary()
    benchmark_manager.print_runtime_summary()
    
    # Create runtime comparison plots
    plot_runtime_comparison(benchmark_manager, benchmark_manager.base_output_dir)
    
    # Save runtime report
    runtime_dir = benchmark_manager.base_output_dir / "runtime_analysis"
    runtime_dir.mkdir(exist_ok=True)
    save_runtime_report(benchmark_manager, runtime_dir / "runtime_report.txt")
    
    # Save experiment summary
    benchmark_manager.experiment_log["end_time"] = datetime.now().isoformat()
    benchmark_manager.experiment_log["total_experiments"] = total_experiments
    benchmark_manager.experiment_log["completed_experiments"] = completed_experiments
    benchmark_manager.experiment_log["total_benchmark_time_seconds"] = total_benchmark_time
    benchmark_manager.experiment_log["runtime_summary"] = benchmark_manager.runtime_summary
    
    log_file = benchmark_manager.base_output_dir / "experiment_log.json"
    with open(log_file, 'w') as f:
        json.dump(benchmark_manager.experiment_log, f, indent=2)
    
    print(f"\n🎉 Benchmark Complete!")
    print(f"Total benchmark time: {total_benchmark_time:.1f} seconds")
    print(f"Results saved to: {benchmark_manager.base_output_dir}")
    print(f"Experiment log: {log_file}")

if __name__ == "__main__":
    try:
        # Start the benchmark
        run_benchmark()
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {str(e)}")
        traceback.print_exc()