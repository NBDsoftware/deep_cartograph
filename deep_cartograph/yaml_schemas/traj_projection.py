from pydantic import BaseModel
from typing import List, Union, Literal, Optional

class BiasArgs(BaseModel):
    
    # Common args for all bias methods
    
    # Temperature in Kelvin
    temperature: float = 300.0
    # Widths of the Gaussian hills (or initial width for opes)
    sigma: float = 0.05
    # The frequency for kernel depositions (how often the bias is updated)
    pace: int = 500
    # The lower bounds for the grid (lower value that will be explored in the CV, same for all components)
    grid_min: float = -1.0 
    # The upper bounds for the grid (upper value that will be explored in the CV, same for all components)
    grid_max: float = 1.0
    # The number of grid bins (number of points in the grid, same for all components)
    grid_bin: int = 300
    
    # Metadynamics specific args
    
    # Height of the Gaussian hills
    height: float = 1.0
    # Bias factor
    bias_factor: float = 10.0
    
    # Opes specific args
    
    # Barrier 
    barrier: float = 50.0
    # Observation steps (for opes_expanded)
    observation_steps: int = 100            # pace units
    # Compression threshold
    compression_threshold: float = 0.1
    
class Bias(BaseModel):
    
    # Name of the method
    method: Literal['wt_metadynamics', 'opes_metad', 'opes_metad_explore', 'opes_expanded'] = 'opes_metad'
    # Keyword arguments for the method
    args: BiasArgs = BiasArgs() 

class FesFigure(BaseModel):
      
    # Calculate the Free Energy Surface
    compute: bool = True
    # Save the calculated Free Energy Surface in .npy files (otherwise it just plots 1D or 2D FES)
    save: bool = True
    # Temperature in Kelvin
    temperature: int = 300
    # Bandwidth for the Kernel Density Estimation of the Free Energy Surface
    bandwidth: float = 0.05
    # Number of bins for the Kernel Density Estimation of the Free Energy Surface
    num_bins: int = 150
    # Maximum value for the Free Energy Surface (above which the value is set to NaN)
    max_fes: float = 30

class TrajProjection(BaseModel):
    
    # Plot the Projected Clustered Trajectory
    plot: bool = True
    # Number of bins for the Kernel Density Estimation of the Projected Clustered Trajectory
    num_bins: int = 100
    # Bandwidth for the Kernel Density Estimation of the Projected Clustered Trajectory
    bandwidth: float = 0.25
    # Transparency of the points in the Projected Clustered Trajectory
    alpha: float = 0.8
    # Colormap for the Projected Clustered Trajectory
    cmap: str = "turbo"
    # Size of the markers in the Projected Clustered Trajectory
    marker_size: int = 5

class Figures(BaseModel):
      
    # Settings for the Free Energy Surface calculation
    fes: FesFigure = FesFigure()
    # Settings for the projection of the trajectory onto the CV space
    traj_projection: TrajProjection = TrajProjection()
    # Bias method for the PLUMED input file (all Collective Variables)
    bias: Bias = Bias()
    

class TrajProjectionSchema(BaseModel):
    
    # Settings for additional figures
    figures: Figures = Figures()

