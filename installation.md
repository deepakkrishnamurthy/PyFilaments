Pyfilaments installation
---
1. Open terminal
2. Install a package manager (conda)
  Follow instruction at https://docs.conda.io/projects/conda/en/latest/user-guide/install/ based on your operating system
3. Create a python virtual environment.

```bash
conda create -n myenv python=3.6
```
4. Enter the conda virtual environment

```bash
conda activate myenv 
```
5. Install the required packages using:

```bash
pip install -r requirements.txt
```
6. Install odespy for numerical integration

- Download odespy repo from https://github.com/rajeshrinet/odespy and follow install instructions.

7. Install pystokes

Navigate to the pystokes folder within the pyfilaments repo and then run.
```bash
python setup.py install
```
8. Install pyforces

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





