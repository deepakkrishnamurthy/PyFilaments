# Active Filament Processing

This folder contains Juyter notebooks for processing Active filament data.

The input to this pipeline is raw simulation data. Outputs are stored in-place as '.hdf5' or '.csv' formats in the 'Analysis' folder created within the simulation folder.

## Description of analysis files

1. Run_Batch_Analysis.ipynb:
    
    Run analysis on a batch of simulation data. These can include:
    - Detecting periodic vs aperiodic behavior


2. PCAmodes_batch_analysis.ipynb : 

    Use this notebook to calculate the following for a batch of simulations

    Results:
    - Filament shape modes
    - Mode amplitude of the filament for each shape mode vs time
    - PCA dimensionality estimate using participation ratio
    
2. InitialConditionSensitivity_batchAnalysis.ipynb

    This notebook is used to calculate sensitivity of the filament dynamics to small changes in initial conditions.
    
    Results:
    - Maximum Lyapunov exponents
    
3. Filament_ReturnMaps_processing.ipynb

    This notebook calculates and stores filament base-tip and tip angles that are used for calculating recurrence maps of the filament dynamics.