# Weighted Least Squares Including Correlation in Error

Code for performing the weighted Least Squares Including Correlation in Error
(WLS-ICE) algorithm when fitting a function to ensemble averages, implemented
in languages listed below, with example data with M=100 fractional Brownian motion
trajectories (generated with parameters as in paper "[Fitting a function to
time-dependent ensemble averaged data](https://www.nature.com/articles/s41598-018-24983-y)")
(or [arXiv](http://arxiv.org/abs/1805.03057)):

1. Python scripts that read in the data and apply the WLS-ICE algorithm to it.
   Run ./example.py ../data from python-folder, for a demonstration.

2. Octave/Matlab - code ported from Python. `f.m`, `df.m`, and `d2f.m`
   defines the analytical function to fit, its gradient and hessian,
   respectively. Run `example.m` file for example.

3. [Hy](https://github.com/hylang/hy/) Lisp, analogous to Python.

# Generating trajectories

We also provide the scripts (Python) used to generate trajectories for the
four example systems.


# License

All code is made available under GNU General Public License v3.
https://www.gnu.org/licenses/gpl-3.0.html
