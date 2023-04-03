# Active Filament Processing

This folder contains Juyter notebooks for processing Active filament data.

The input to this pipeline is raw simulation data in .hdf5 format. Outputs are stored either in-place as '.hdf5' or '.csv' formats in the 'Analysis' folder created within the simulation folder, or in a separate 'processed-data' folder for further analysis and plotting.

## Data download

Sample simulation data can be downloaded from this [data repository](https://doi.org/10.6078/D12T52).

## Description of analysis files

1. Run_Batch_Analysis.ipynb:
    
    Run any analysis on a batch of simulation data. These can include:
    - Detecting periodic vs aperiodic behavior
    - Calculating search metrics of the filament tip

2. Classify Filament Dynamics
    
    Takes raw filament simulation data and classifies the dynamics into periodic, aperiodic or escape.
    If periodic it also detects the period of the dynamics

3. PCAmodes_batch_analysis.ipynb : 

    Use this notebook to calculate the following for a batch of simulations

    Results:
    - Filament shape modes
    - Mode amplitude of the filament for each shape mode vs time
    - PCA dimensionality estimate using participation ratio
    
4. InitialConditionSensitivity_batchAnalysis.ipynb

    This notebook is used to calculate sensitivity of the filament dynamics to small changes in initial conditions.
    
    Results:
    - Maximum Lyapunov exponents
    
5. Filament_ReturnMaps_processing.ipynb

    This notebook calculates and stores filament base-tip and tip angles that are used for calculating recurrence maps of the filament dynamics.

6. Attractor Dimension Estimate
    
    Uses the correlation dimension method to estimate the attractor dimension of the filament dynamics

## Examples

### Loading a pyfilaments dataset and plot the shape

```python

# Example 1: Load a pyfilaments simulation file and plot the filament shape
import os
import numpy as np
import matplotlib.pyplot as plt 

import pyfilaments.analysisutils as analysis


file = './PyFilaments/examples/sample_data/SimResults_Np_32_Shape_line at angle_kappa_hat_6_k_25_b0_2.1_F_0_S_0_D_1.333_activityTime_750_simType_point/SimResults_00_.hdf5'

filament = analysis.analysisTools(file = file)

# Basic analysis

# Plot the initial filament shape
plt.figure()
filament.plotFilament(filament.R[0,:])

# Plot the final filament shape
plt.figure()
filament.plotFilament(filament.R[-1,:])

```

### Load a pyfilaments dataset and plot search clouds

```python

# Example 2: Load a pyfilaments dataset and plot search clouds
import os
import numpy as np
import matplotlib.pyplot as plt 

import pyfilaments.analysisutils as analysis

file = './PyFilaments/examples/sample_data/SimResults_Np_32_Shape_line at angle_kappa_hat_6_k_25_b0_2.1_F_0_S_0_D_1.333_activityTime_750_simType_point/SimResults_00_.hdf5'

filament = analysis.analysisTools(file = file)

filament.filament_tip_coverage(save = False, overwrite = True)
filament.plot_unique_tip_locations(save = False)

```