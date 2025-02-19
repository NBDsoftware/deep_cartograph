from pydantic import BaseModel
from typing import Union

class FilterSettings(BaseModel):
    
    # Compute Hartigan's dip test
    compute_diptest: bool = True
    # Compute entropy of the features
    compute_entropy: bool = False
    # Compute standard deviation of the features
    compute_std: bool = False
    # Hartigan's dip test significance level
    diptest_significance_level: float = 0.05
    # Entropy quantile to use for filtering (0 to skip filter)
    entropy_quantile: float = 0
    # Standard deviation quantile to use for filtering (0 to skip filter)
    std_quantile: float = 0

class SamplingSettings(BaseModel):
    
    # Number of samples to use for each feature
    num_samples:  Union[int, None] = None
    # Total number of samples per feature in the colvars file
    total_num_samples: Union[int, None] = None
    # Relaxation time of the system in number of samples
    relaxation_time: int = 1

class FilterFeaturesSchema(BaseModel):
        
    # Definition of filter settings
    filter_settings: FilterSettings = FilterSettings()
    # Definition of sampling settings
    sampling_settings: SamplingSettings = SamplingSettings()