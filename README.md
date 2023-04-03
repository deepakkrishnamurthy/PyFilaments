# PyFilaments: Simulate Active Filament Dynamics with Many-body Hydrodynamics in Python

<p align="center">
  <img src="https://github.com/deepakkrishnamurthy/PyFilaments/blob/master/examples/sample_videos/Wiggly_I_FINAL_AdobeExpress.gif" />
</p>

## About 

[PyFilaments](https://github.com/deepakkrishnamurthy/PyFilaments) is a Python library to simulate the dynamics of active filaments inspired by our work to understand the remarkable behaviour of the ciliate *Lacrymaria olor*. The filaments are built out of spherical particles/colloids which can be either active, i.e. capable of applying stresses on the surrounding fluid, or passive. The library uses the [PyStokes](https://gitlab.com/rajeshrinet/pystokes) library for calculating rigid body motions of the colloids and their flows due to many-body hydrodynamic interactions. The colloids are connected into filaments using connection forces due to non-linear springs and bending potentials, as well as near-field repulsive potentials. The library allows the active stresses to be varied dynamically to simulate ciliary reversals which are a hallmark of ciliate behaviour. 

### Rich behavioral space of active filaments under time-varying forcing
Using the PyFilaments library we have discovered a rich space of filament behaviors under time-varying forcing, including periodic and aperiodic dynamics. We further find that aperiodic behaviors are due to a transition to chaos.

#### Periodic dynamics
<p align="center">
  <img width="50%" src="https://github.com/deepakkrishnamurthy/PyFilaments/blob/master/examples/sample_videos/PeriodicDynamics_AdobeExpress.gif" alt = "Periodic dynamics" />
</p>

#### Aperiodic dynamics
<p align="center">
  <img width="50%" src="https://github.com/deepakkrishnamurthy/PyFilaments/blob/master/examples/sample_videos/AperiodicDynamics_AdobeExpress.gif" alt = "Aperiodic dynamics" />
</p>


## Installation

### From a checkout of this repo

#### Install a package manager (conda)
  Follow instruction at https://docs.conda.io/projects/conda/en/latest/user-guide/install/ based on your operating system
  
```bash
>> git clone https://github.com/deepakkrishnamurthy/PyFilaments.git
>> cd PyFilaments
```
#### Create a python virtual environment.
```bash
>> conda create -n pyfilaments python=3.6
>> conda activate pyfilaments
>> pip install -r requirements.txt

```
#### Install odespy for numerical integration

- Download odespy repo from https://github.com/rajeshrinet/odespy and follow install instructions.

#### Install pystokes

Navigate to the pystokes folder within the pyfilaments repo and then run.
```bash
>> cd pystokes
>> python setup.py install
```
#### Install pyforces

Navigate to the pyforces folder within the pyfilaments repo and then run.
```bash
>> cd pyforces
>> python setup.py install
```

#### Install fast filamnent subroutines

Navigate to filament folder: /PyFilament/pyfilament/filament.
Run the installation for Cythonized filament subroutines
```bash
>> cd pyfilament/filament
>> python setup.py install
```

#### Create PYTHONPATH variables
Since the code base is under active development. Add the root folder of the repo to your PYTHONPATH by following the instructions [here](https://bic-berkeley.github.io/psych-214-fall-2016/using_pythonpath.html)

For eg. on MacOS, this would mean
- Open terminal app
- Open the file ~/.bash_profile in your text editor – e.g. atom ~/.bash_profile;
- Add the following line to the end:
```bash
 export PYTHONPATH="/Users/my_user/PyFilaments"
```
- Save the file
- Close the terminal app


#### OSX only: 
If openMP is not detected while installing pystokes and pyforces. Then follow the next steps

We need to reinstall gcc with openMP for which we first need to set permissions
```bash
sudo chown -R $(whoami) $(brew --prefix)/*
```
Now install gcc
```bash
brew reinstall gcc --without-multilib
```

## References

- Krishnamurthy, Deepak, and Manu Prakash. "Emergent Programmable Behavior and Chaos in Dynamically Driven Active Filaments." bioRxiv (2022): 2022-06.

- Singh, Rajesh, and Ronojoy Adhikari. "Pystokes: Phoresis and Stokesian hydrodynamics in python." arXiv preprint arXiv:1910.00909 (2019).





