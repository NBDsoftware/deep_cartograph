from deep_cartograph.tools.filter_features import filter_features
from deep_cartograph.modules.common import read_feature_constraints
import importlib.resources as resources
from deep_cartograph import tests
import shutil
import yaml
import os

# Find the path to the tests and data folders
tests_path = resources.files(tests)
data_path = os.path.join(tests_path, "data")

def get_config():
    yaml_content = """
    filter_settings:
        compute_diptest: True              
        compute_entropy: False             
        compute_std: False                
        diptest_significance_level: 0.05   
        entropy_quantile: 0                
        std_quantile: 0                        
    sampling_settings:
        relaxation_time: 1  
    """
    return yaml.safe_load(yaml_content)


def test_filter_features():
    
    print("Testing filter_features...")
    
    # Inputs and reference files
    colvars_path = os.path.join(data_path, "reference", "compute_features", "virtual_dihedrals.dat")
    reference_features_path = os.path.join(data_path, "reference", "filter_features", "filtered_virtual_dihedrals.txt")
    
    # Output files
    output_path = os.path.join(tests_path, "output_filter_features")
    
    # Remove output folder if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        
    # Call API
    output_features_path = filter_features(
        configuration = get_config(),
        colvars_paths = [colvars_path],
        output_folder = output_path)
    
    # Read the all reference filtered features into a list
    reference_features = read_feature_constraints(reference_features_path)
    
    # Read the all output filtered features into a list
    output_features = read_feature_constraints(output_features_path)
    
    # Compare them without considering the order
    reference_set = set(reference_features)
    output_set = set(output_features)    
    test_passed = reference_set == output_set
    assert test_passed
    
    # If the test passed, clean the output folder
    if test_passed:
        shutil.rmtree(output_path)