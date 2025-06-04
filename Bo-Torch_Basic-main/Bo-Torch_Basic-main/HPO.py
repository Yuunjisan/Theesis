import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import time
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import the TabPFN_BO class from the existing codebase
from Algorithms.BayesianOptimization.TabPFN_BO.TabPFN_BO import TabPFN_BO

# IOH Experimenter libraries (same as Convergence.py)
try:
    from ioh import get_problem
    import ioh.iohcpp.logger as logger_lib
    from ioh.iohcpp.logger import Analyzer
    from ioh.iohcpp.logger.trigger import Each, ON_IMPROVEMENT, ALWAYS
    print("✅ IOH library imported successfully")
except ModuleNotFoundError as e:
    print(f"❌ IOH library not found: {e}")
    print("Please install IOH: pip install ioh")
    raise
except Exception as e:
    print(f"❌ Error importing IOH: {e}")
    raise

class TabPFNHPO:
    """Hyperparameter Optimization for TabPFN using grid search
    
    Now using IOH BBOB functions (same as Convergence.py) and testing both n_estimators and temperature!
    """
    
    def __init__(self):
        # BBOB function IDs from Le Riche's study (same as the problem list you specified)
        self.bbob_functions = {
            'f1_sphere': 1,      # Smooth unimodal
            'f2_ellipsoid': 2,   # Ill-conditioned unimodal
            'f10_rotated_ellipsoid': 10,  # High conditioning and unimodal  
            'f15_rastrigin': 15, # Multimodal, adequate structure
            'f20_schwefel': 20,  # Multimodal, weak structure
            'f21_gallagher': 21, # Many local optima
            'f16_weierstrass': 16, # Highly rugged
            'f8_rosenbrock': 8   # Narrow valley
        }
        
        # HPO parameters - now testing both!
        self.ensemble_sizes = [8, 16, 32, 64, 128]  # n_estimators parameter
        self.temperatures = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]  # temperature parameter
        
        print("✅ TabPFN_BO supports both ensemble sizes and temperature testing!")
        print("✅ Using IOH BBOB functions (same as Convergence.py)")
        print(f"   Testing functions: {list(self.bbob_functions.keys())}")
        
        self.results = []
        self.best_configs = {}
        
    def evaluate_config(self, ensemble_size, temperature, func_name, func_id, dim, n_trials=3):
        """Evaluate a single configuration using TabPFN_BO with IOH BBOB functions"""
        scores = []
        
        for trial in range(n_trials):
            try:
                # Create IOH BBOB problem (same pattern as Convergence.py)
                problem = get_problem(func_id, instance=1, dimension=dim)
                
                # Set budget and DoE based on dimension (same as Convergence.py)
                budget = 10 * dim + 50  # Budget = 10*D + 50
                n_DoE = 3 * dim  # Initial design points
                
                # Create TabPFN_BO optimizer with current configuration
                optimizer = TabPFN_BO(
                    budget=budget,
                    n_DoE=n_DoE,
                    acquisition_function="expected_improvement",
                    random_seed=42 + trial,  # Different seed per trial
                    n_estimators=ensemble_size,  # Testing this parameter
                    temperature=temperature,     # Testing this parameter too!
                    fit_mode="fit_with_cache",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    maximisation=False,
                    verbose=False
                )
                
                # Run optimization (same pattern as Convergence.py)
                optimizer(problem=problem)
                
                # Get final performance and calculate regret (same as Convergence.py)
                final_best = optimizer.current_best
                true_optimum = problem.optimum.y  # IOH provides the true optimum
                regret = abs(final_best - true_optimum)
                
                scores.append({
                    'final_value': final_best,
                    'regret': regret,
                    'true_optimum': true_optimum,
                    'n_evaluations': len(optimizer.f_evals)
                })
                
            except Exception as e:
                print(f"    Error in trial {trial+1} for {func_name} (dim={dim}): {e}")
                scores.append({
                    'final_value': float('inf'),
                    'regret': float('inf'),
                    'true_optimum': 0.0,
                    'n_evaluations': 0
                })
        
        # Calculate average scores (use regret as primary metric - lower is better)
        valid_scores = [s for s in scores if np.isfinite(s['regret'])]
        
        if not valid_scores:
            return {
                'final_value': float('inf'),
                'regret': float('inf'),
                'regret_std': float('inf'),
                'true_optimum': 0.0,
                'n_evaluations': 0
            }
        
        avg_scores = {
            'final_value': np.mean([s['final_value'] for s in valid_scores]),
            'regret': np.mean([s['regret'] for s in valid_scores]),
            'regret_std': np.std([s['regret'] for s in valid_scores]),
            'true_optimum': np.mean([s['true_optimum'] for s in valid_scores]),
            'n_evaluations': np.mean([s['n_evaluations'] for s in valid_scores])
        }
        
        return avg_scores
    
    def run_grid_search_2d(self):
        """Run full grid search on 2D functions"""
        print("Starting 2D grid search...")
        print(f"Total configurations: {len(self.ensemble_sizes)} × {len(self.temperatures)} = {len(self.ensemble_sizes) * len(self.temperatures)}")
        
        total_configs = len(self.ensemble_sizes) * len(self.temperatures)
        config_count = 0
        
        for ensemble_size, temperature in product(self.ensemble_sizes, self.temperatures):
            config_count += 1
            print(f"\nConfiguration {config_count}/{total_configs}: Ensemble={ensemble_size}, Temperature={temperature}")
            
            config_results = {
                'ensemble_size': ensemble_size,
                'temperature': temperature,
                'dimension': 2,
                'functions': {}
            }
            
            total_score = 0
            function_count = 0
            
            for func_name, func_id in self.bbob_functions.items():
                print(f"  Evaluating {func_name} (BBOB f{func_id})...")
                start_time = time.time()
                
                scores = self.evaluate_config(ensemble_size, temperature, func_name, func_id, 2)
                config_results['functions'][func_name] = scores
                
                # Use negative regret for ranking (higher is better, since lower regret is better)
                total_score += -scores['regret']  # Negative because we want to minimize regret
                function_count += 1
                
                elapsed = time.time() - start_time
                print(f"    Regret: {scores['regret']:.6e} ± {scores['regret_std']:.6e} (vs optimum: {scores['true_optimum']:.3f}) ({elapsed:.1f}s)")
            
            config_results['avg_neg_regret'] = total_score / function_count
            self.results.append(config_results)
        
        # Sort by average negative regret (descending, so best regret first)
        self.results.sort(key=lambda x: x['avg_neg_regret'], reverse=True)
        
        print(f"\nTop 3 configurations from 2D:")
        for i, result in enumerate(self.results[:3]):
            avg_regret = -result['avg_neg_regret']  # Convert back to positive regret
            print(f"{i+1}. Ensemble={result['ensemble_size']}, Temperature={result['temperature']}, Avg Regret={avg_regret:.6e}")
    
    def run_higher_dimensions(self, dimensions, top_n=3):
        """Run top configurations on higher dimensions"""
        top_configs = self.results[:top_n]
        
        for dim in dimensions:
            print(f"\nEvaluating top {top_n} configurations on {dim}D...")
            
            dim_results = []
            
            for i, config in enumerate(top_configs):
                ensemble_size = config['ensemble_size']
                temperature = config['temperature']
                
                print(f"\nConfiguration {i+1}/{top_n}: Ensemble={ensemble_size}, Temperature={temperature}")
                
                config_results = {
                    'ensemble_size': ensemble_size,
                    'temperature': temperature,
                    'dimension': dim,
                    'functions': {}
                }
                
                total_score = 0
                function_count = 0
                
                for func_name, func_id in self.bbob_functions.items():
                    print(f"  Evaluating {func_name} (BBOB f{func_id})...")
                    start_time = time.time()
                    
                    scores = self.evaluate_config(ensemble_size, temperature, func_name, func_id, dim)
                    config_results['functions'][func_name] = scores
                    
                    total_score += -scores['regret']  # Negative regret for ranking
                    function_count += 1
                    
                    elapsed = time.time() - start_time
                    print(f"    Regret: {scores['regret']:.6e} ± {scores['regret_std']:.6e} (vs optimum: {scores['true_optimum']:.3f}) ({elapsed:.1f}s)")
                
                config_results['avg_neg_regret'] = total_score / function_count
                dim_results.append(config_results)
            
            # Sort by average negative regret for this dimension
            dim_results.sort(key=lambda x: x['avg_neg_regret'], reverse=True)
            self.best_configs[f'{dim}D'] = dim_results
            
            print(f"\nTop configurations for {dim}D:")
            for i, result in enumerate(dim_results):
                avg_regret = -result['avg_neg_regret']
                print(f"{i+1}. Ensemble={result['ensemble_size']}, Temperature={result['temperature']}, Avg Regret={avg_regret:.6e}")
    
    def save_results(self, filename='tabpfn_hpo_results.json'):
        """Save all results to JSON file"""
        results_data = {
            'full_2d_results': self.results,
            'best_configs_higher_dims': self.best_configs,
            'parameters': {
                'ensemble_sizes': self.ensemble_sizes,
                'temperatures': self.temperatures,
                'functions': list(self.bbob_functions.keys())
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def create_visualizations(self):
        """Create visualization plots for the results"""
        # Prepare data for heatmap
        heatmap_data = np.zeros((len(self.ensemble_sizes), len(self.temperatures)))
        
        for result in self.results:
            i = self.ensemble_sizes.index(result['ensemble_size'])
            j = self.temperatures.index(result['temperature'])
            heatmap_data[i, j] = -result['avg_neg_regret']
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, 
                   xticklabels=self.temperatures,
                   yticklabels=self.ensemble_sizes,
                   annot=True, fmt='.2e', cmap='viridis_r')  # Reverse colormap since lower regret is better
        plt.title('TabPFN HPO Results: Average Regret Across All 2D Functions\n(Lower is Better)')
        plt.xlabel('Temperature')
        plt.ylabel('Ensemble Size')
        plt.tight_layout()
        plt.savefig('tabpfn_hpo_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Performance comparison across functions
        plt.figure(figsize=(15, 10))
        
        # Get top 5 configurations
        top_configs = self.results[:5]
        
        function_names = list(self.bbob_functions.keys())
        x = np.arange(len(function_names))
        width = 0.15
        
        for i, config in enumerate(top_configs):
            regret_scores = [config['functions'][func]['regret'] for func in function_names]
            plt.bar(x + i*width, regret_scores, width, 
                   label=f"E={config['ensemble_size']}, T={config['temperature']}")
        
        plt.xlabel('BBOB Functions')
        plt.ylabel('Regret (log scale)')
        plt.yscale('log')  # Use log scale for regret
        plt.title('Top 5 TabPFN Configurations Performance Across BBOB Functions (2D)')
        plt.xticks(x + width*2, function_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('tabpfn_function_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """Print a comprehensive summary of results"""
        print("\n" + "="*80)
        print("TABPFN HPO STUDY SUMMARY")
        print("="*80)
        
        print(f"\nParameter ranges tested:")
        print(f"  Ensemble sizes: {self.ensemble_sizes}")
        print(f"  Temperatures: {self.temperatures}")
        print(f"  Total configurations: {len(self.ensemble_sizes) * len(self.temperatures)}")
        
        print(f"\nFunctions tested: {len(self.bbob_functions)}")
        for func_name in self.bbob_functions.keys():
            print(f"  - {func_name}")
        
        print(f"\nTop 5 configurations (2D):")
        for i, result in enumerate(self.results[:5]):
            avg_regret = -result['avg_neg_regret']  # Convert back to positive regret
            print(f"  {i+1}. Ensemble={result['ensemble_size']}, Temperature={result['temperature']}, Avg Regret={avg_regret:.6e}")
        
        if self.best_configs:
            for dim_key, configs in self.best_configs.items():
                print(f"\nTop configurations for {dim_key}:")
                for i, config in enumerate(configs[:3]):
                    avg_regret = -config['avg_neg_regret']
                    print(f"  {i+1}. Ensemble={config['ensemble_size']}, Temperature={config['temperature']}, Avg Regret={avg_regret:.6e}")

def main():
    """Main execution function"""
    print("TabPFN Hyperparameter Optimization Study")
    print("Using BBOB functions from Le Riche's comprehensive analysis [RP21]")
    print("Now using the actual TabPFN_BO class for realistic BO evaluation")
    print("-" * 60)
    
    # Initialize HPO study
    hpo = TabPFNHPO()
    
    # Run 2D grid search
    start_time = time.time()
    hpo.run_grid_search_2d()
    grid_search_time = time.time() - start_time
    
    print(f"\n2D grid search completed in {grid_search_time/60:.1f} minutes")
    
    # Run on higher dimensions with top 3 configs
    hpo.run_higher_dimensions([5, 20], top_n=3)
    
    # Run on 40D with top 2 from 5D and 20D
    if '5D' in hpo.best_configs and '20D' in hpo.best_configs:
        # Combine top 2 from both 5D and 20D
        combined_configs = hpo.best_configs['5D'][:2] + hpo.best_configs['20D'][:2]
        
        # Remove duplicates if any
        unique_configs = []
        seen = set()
        for config in combined_configs:
            key = (config['ensemble_size'], config['temperature'])
            if key not in seen:
                unique_configs.append(config)
                seen.add(key)
        
        print(f"\nEvaluating {len(unique_configs)} unique top configurations on 40D...")
        
        # Manually evaluate on 40D
        results_40d = []
        for i, config in enumerate(unique_configs):
            ensemble_size = config['ensemble_size']
            temperature = config['temperature']
            
            print(f"\nConfiguration {i+1}/{len(unique_configs)}: Ensemble={ensemble_size}, Temperature={temperature}")
            
            config_results = {
                'ensemble_size': ensemble_size,
                'temperature': temperature,
                'dimension': 40,
                'functions': {}
            }
            
            total_score = 0
            function_count = 0
            
            for func_name, func_id in hpo.bbob_functions.items():
                print(f"  Evaluating {func_name} (BBOB f{func_id})...")
                start_time = time.time()
                
                scores = hpo.evaluate_config(ensemble_size, temperature, func_name, func_id, 40)
                config_results['functions'][func_name] = scores
                
                total_score += -scores['regret']  # Negative regret for ranking
                function_count += 1
                
                elapsed = time.time() - start_time
                print(f"    Regret: {scores['regret']:.6e} ± {scores['regret_std']:.6e} (vs optimum: {scores['true_optimum']:.3f}) ({elapsed:.1f}s)")
            
            config_results['avg_neg_regret'] = total_score / function_count
            results_40d.append(config_results)
        
        results_40d.sort(key=lambda x: x['avg_neg_regret'], reverse=True)
        hpo.best_configs['40D'] = results_40d
    
    # Save results and create visualizations
    hpo.save_results()
    hpo.create_visualizations()
    hpo.print_summary()
    
    total_time = time.time() - start_time
    print(f"\nTotal study completed in {total_time/3600:.2f} hours")

if __name__ == "__main__":
    main()
