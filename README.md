python files
------------
optimization.py: classes for doing the optimisation scheme
auxiliary.py: functions for reconstruction, plotting, and evaluating results

notebooks
---------
simulation.ipynb: notebook environment to conduct the simulations
summary.ipynb: a summary and discussion of the simulation results, intended to be a final report of the internship
reuse.ipynb: notebook environment to conduct the simulations using optimal parameters from previous simulations as starting parameters
    for another optimisation
pretrained.ipynb: notebook environment to conduct the simulations of maximizing the fidelity of a state which has been prepared under the
    circuit with optimal parameters
kerrtranslation.ipynb: notebook environment to conduct the simulations belonging to the ansatz of translating good Kerr learning 
    to not optimal phase learning
reinsert.ipynb: implementation of the reinsertion using the approximation of a cubic phase gate via the driven Kerr gate
distribution.ipynb: analysis of how initial gaussian errors propagate through a polynomial 
quadrature.ipynb: algebraic implementation to express the preparation circuit as polynomial of the initial quadratures

remarks
-------
some notebooks and files require own implementation of certain gates in the strawberryfields framework
