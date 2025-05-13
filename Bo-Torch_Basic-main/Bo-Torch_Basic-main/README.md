# Basic Bayesian Optimization Methods with Bo-Torch 
This repository contains an archetype to build algorithms by using BO-Torch. In this moment this repository uses the `Bo-Torch`, `GPyTorch` and `PyTorch` as main ones to develop the forthcoming methods. Nevertheless, this library just includes a directory archetype and a main function to test the built algorithm with the BBOB problems called from IOH-Experimenter interface (see: https://iohprofiler.github.io/IOHexperimenter/)

# Libraries and dependencies

The implementation is in Python 3.10.12 and all the libraries used are listed in `requirements.txt`.

# Structure
- `main.py` -> An archetype file, which is an example on how to call an instance of one of the 24 BBOB problems by using IOH interface and use the Vanilla-BO Algorithm stored in the repository.
- _myexperiment_ -> An example of the generated files by a logger to assess the performance of your algorithm on a problem (namely from BBOB).
- _/Algorithms_ -> A folder which stores the Algorithms to be developed. This repository contains an `AbstractAlgorithm` class, which works as a basic set up of the Algorithm. As this works with IOH, then there are some bypasses whenever the optimization comes from the `BBOB`/`RealSingleObjective` instances from IOH in order to ease the definition of the properties of the Optimizer such as the dimensionality and the bounds. To call an instance of a new algorithm, you must include your new algorithm within the `__init__.py` file in the same level. 
-  _/Algorithms/Bayesian_Optimization_ -> In this folder you may save all the new Bayesian Based Optimizers. There's an `AbstractBayesianOptimizer` class to define an archetypical one. When you make a new BO algorithm, consider building upon this class as a parent class as this might ease your implementation. Additionally, generate a new folder per new variant of the algorithm you might create.

# Execution from source
## Dependencies to run from source

Running this code from source requires Python 3.10.14, and the libraries given in `requirements.txt` (Warning: preferably use a virtual environment for this specific project, to avoid breaking the dependencies of your other projects). In Ubuntu, installing the dependencies can be done using the following command:

```
pip install -r requirements.txt
```

Then, in order to have a glance on how to use it with any problem, see the `main.py` file to call the optimizer.
