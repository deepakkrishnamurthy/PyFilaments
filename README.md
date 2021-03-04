# PyFilaments

[PyFilaments](https://github.com/deepakkrishnamurthy/PyFilaments) is a Python library to simulate the dynamics of active filaments. The filaments are built out of spherical particles with Dirichlet (velocity) or Neumann (stress) boundary conditions on their surfaces. The library uses the [PyStokes](https://gitlab.com/rajeshrinet/pystokes) library for calculating Rigid body motions of the spheres and their flows. Connection forces due to non-linear springs and bending potentials are applied to simulate the elasto-hydrodynamics of active filaments.  

---

## Installation

### Open terminal

### Install a package manager (conda)
  Follow instruction at https://docs.conda.io/projects/conda/en/latest/user-guide/install/ based on your operating system
  
### Create a python virtual environment.
```bash
conda create -n myenv python=3.6
```
### Enter the conda virtual environment

```bash
conda activate myenv 
```
### Install the required packages using:

```bash
pip install -r requirements.txt
```
### Install odespy for numerical integration

- Download odespy repo from https://github.com/rajeshrinet/odespy and follow install instructions.

### Install pystokes

Navigate to the pystokes folder within the pyfilaments repo and then run.
```bash
python setup.py install
```
### Install pyforces

Navigate to the pyforces folder within the pyfilaments repo and then run.
```bash
python setup.py install
```

## OSX only: 
If openMP is not detected while installing pystokes and pyforces. Then follow the next steps

We need to reinstall gcc with openMP for which we first need to set permissions
```bash
sudo chown -R $(whoami) $(brew --prefix)/*
```
Now install gcc
```bash
brew reinstall gcc --without-multilib
```







