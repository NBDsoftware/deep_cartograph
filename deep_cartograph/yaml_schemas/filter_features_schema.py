from pydantic import BaseModel
from typing import Union

class FilterSettingsSchema(BaseModel):
    
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

class AminoSettingsSchema(BaseModel):
    
    # Run amino analysis
    run_amino: bool = False
    # Maximum number of independent features to find
    max_independent_features: int = 20
    # Minimum number of independent features to find
    min_independent_features: int = 5
    # Max number of features that will be analyzed simultaneously (all by default, must be greater or equal than max_ind_features)
    features_batch_size: Union[int, None] = None
    # Number of bins to use for the histograms
    num_bins: int = 50
    # Bandwidth to use for the kernel density estimation
    bandwidth: float = 0.1

class SamplingSettingsSchema(BaseModel):
    
    # Number of samples to use for each feature
    num_samples:  Union[int, None] = None
    # Total number of samples per feature in the colvars file
    total_num_samples: Union[int, None] = None
    # Relaxation time of the system in number of samples
    relaxation_time: int = 1

class FilterFeaturesSchema(BaseModel):
        
        # Definition of filter settings
        filter_settings: FilterSettingsSchema = FilterSettingsSchema()
        # Definition of amino settings
        amino_settings: AminoSettingsSchema = AminoSettingsSchema()
        # Definition of sampling settings
        sampling_settings: SamplingSettingsSchema = SamplingSettingsSchema()