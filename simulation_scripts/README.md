# Simulation scripts

Use the jupyter-notebooks in this folder to setup and run active filament simulations. Both individual as well as parallelized parameter sweeps are supported.

Smulations allow thebuser to specify the filament parameters as well as spatial and temporal activity profiles.

Simulation data is stored in the form of Nt x N x n_dim dimensional arrays 
Nt: length of the time array and N is the number of colloids and n_dim is the number of spatial dimensions in the system (default: 3)

The simulation data is stored in .hdf5 format along with metadata. Metadata is alsos tored in human-readable .csv format as 'metadata.csv' file. 


