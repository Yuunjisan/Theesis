# Bayesian Optimization Research

This repository contains implementations and experiments related to Bayesian Optimization using BoTorch, PyTorch, and GPyTorch. It includes:

- Basic Bayesian Optimization implementation with BoTorch
- Experiments with BBOB benchmark functions
- TabPFN experiments for tabular data regression

## Setup Instructions

### Option 1: Using pip (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/thesis-bayesian-optimization.git
cd thesis-bayesian-optimization

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install the package and all dependencies
pip install -e .
```

### Option 2: Manual installation

```bash
# Clone the repository
git clone https://github.com/yourusername/thesis-bayesian-optimization.git
cd thesis-bayesian-optimization

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- `Algorithms/`: Implementation of optimization algorithms

  - `BayesianOptimization/`: BO-specific algorithms
    - `Vanilla_BO/`: Basic implementation of Bayesian Optimization
  - `AbstractAlgorithm.py`: Base class for all optimization algorithms

- `convergence_plots.py`: Utility for generating convergence plots
- `main.py`: Example usage and entry point
- `BBOB_Exploration.py`: Experiments with BBOB benchmark functions
- `TabPFNExplo.py`: Experiments with TabPFN for regression

## Usage

Basic usage example:

```python
from Algorithms import Vanilla_BO
from ioh import get_problem, ProblemClass

# Define problem
problem = get_problem(1, instance=1, dimension=2, problem_class=ProblemClass.BBOB)

# Configure optimizer
optimizer = Vanilla_BO(budget=100,
                      n_DoE=10,
                      acquisition_function="expected_improvement",
                      random_seed=42,
                      maximisation=False)

# Run optimization
optimizer(problem=problem)

print(f"Best solution: {problem.state.current_best.x}")
print(f"Best objective value: {problem.state.current_best.y}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
