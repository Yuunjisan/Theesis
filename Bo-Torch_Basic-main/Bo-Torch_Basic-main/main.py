__authors__ = ("Elena Raponi",
               "Ivan Olarte-Rodriguez")

r"""
This is a basic script to show how to call an algorithm from the Bayesian Algorithm Repository.
"""


### -------------------------------------------------------------
### IMPORT LIBRARIES/REPOSITORIES
###---------------------------------------------------------------

# In this case, all the algorithms are contained within the directory called "Algorithms"
from Algorithms import Vanilla_BO

import os
from pathlib import Path
from numpy.linalg import norm

# This part of the code is to call the IOH Experimenter Objects which are required to 
# define a Black-Box Optimization problem from the BBOB Bench and set up the logger (to save your results).

# NOTE: For more information on IOH, we suggest you to check the following repository:
#           - https://iohprofiler.github.io/IOHexperimenter/
#       In case you want another hands-on example, check the following Jupyter Notebook:
#           - https://github.com/IOHprofiler/IOHexperimenter/blob/master/example/tutorial.ipynb
try:
    from ioh import get_problem # This is a function to set up a problem from the BBOB
    import ioh.iohcpp.logger as logger_lib
    from ioh.iohcpp.logger import Analyzer # The Analyzer class of a logger.
    from ioh.iohcpp.logger.trigger import Each, ON_IMPROVEMENT, ALWAYS # These are triggers (define how often to log results).
except ModuleNotFoundError as e:
    print(e.args)
except Exception as e:
    print(e.args)




### ---------------------------------------------------------------
### LOGGER SETUP
### ---------------------------------------------------------------

# These are the triggers to set a form
# how to log your data
triggers = [
    Each(10), # Log after (10) evaluations
    ON_IMPROVEMENT # Log when there's an improvement
]

logger = Analyzer(
    triggers=triggers,
    root=os.getcwd(),                  # Store data in the current working directory
    folder_name="my-experiment",       # in a folder named: 'my-experiment'
    algorithm_name="Vanilla BO",    # meta-data for the algorithm used to generate these results
    algorithm_info="Bo-Torch Implementation", # Some meta-data about the algorithm used (for reference)
    additional_properties= [logger_lib.property.RAWYBEST
                            ], # Use this to log the best-so-far
    store_positions=True               # store x-variables in the logged files
)

# this automatically creates a folder 'my-experiment' in the current working directory
# if the folder already exists, it will given an additional number to make the name unique.

# In order to log data for a problem, we only have to attach it to a logger
problem = get_problem(1, # An integer denoting one of the 24 BBOB problem
                      instance=1, # An instance, meaning the optimum of the problem is changed via some transformations
                      dimension=2,  # The problem's dimension
                      )

problem.attach_logger(logger)


# Set up the Vanilla BO
optimizer = Vanilla_BO(budget=min(200,50*problem.meta_data.n_variables), # Define the budget
                       n_DoE=3*problem.meta_data.n_variables, # The number of points to sample (as part of the DoE)
                       acquisition_function="expected_improvement", # The name of an acquisition function 
                                                                    #(you may change between 'expected_improvement',
                                                                    # 'probability_of_improvement',
                                                                    # 'upper_confidence_bound')
                       random_seed=44,
                       maximisation = False, # BBOB is meant to minimize (This is relevant for your acquisition function!)
                       verbose=True, # Print the best result-so-far)
                       DoE_parameters = {'criterion':"center", # Use this to configure the initial DoE
                                         'iterations':1000}
                        )

logger.watch(optimizer,"acquistion_function_name")

# Run the optimization loop
optimizer(problem=problem)

# Compare the distance from optimum and regret of the optimizer at the end
print("The distance from optimum is: ", norm(problem.state.current_best.x-problem.optimum.x))
print("The regret is: ", problem.state.current_best.y - problem.optimum.y )


# Close the logger
logger.close()