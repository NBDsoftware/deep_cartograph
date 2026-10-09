"""
Schema for the configuration of the filter features tool.
"""
from pydantic import BaseModel
from typing import Union, Optional

class FilterSettings(BaseModel):
    """Validates the `filter_settings` section: which filters to apply and their thresholds."""

    # Distance threshold to distinguish local contacts (in Angstroms) (None to skip filter)
    local_distance_threshold: Optional[float] = None
    # Hartigan's dip test significance level (None to skip filter)
    diptest_significance_level: Optional[float] = 0.05
    # Entropy quantile to use for filtering (None to skip filter)
    entropy_quantile: Optional[float] = None
    # Standard deviation quantile to use for filtering (None to skip filter)
    std_quantile: Optional[float] = None

class SamplingSettings(BaseModel):
    """Validates the `sampling_settings` section (currently not used by the filtering code)."""
    
    # Number of samples to use for each feature
    num_samples:  Union[int, None] = None
    # Total number of samples per feature in the colvars file
    total_num_samples: Union[int, None] = None
    # Relaxation time of the system in number of samples
    relaxation_time: int = 1

class FilterFeaturesSchema(BaseModel):
    """Validates the `filter_features` section of the configuration."""
        
    # Definition of filter settings
    filter_settings: FilterSettings = FilterSettings()
    # Definition of sampling settings
    sampling_settings: SamplingSettings = SamplingSettings()