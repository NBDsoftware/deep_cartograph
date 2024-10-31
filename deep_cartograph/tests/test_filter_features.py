from deep_cartograph.tools.filter_features import filter_features
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
    amino_settings:
        run_amino: False           
        max_independent_ops: 20     
        min_independent_ops: 5     
        ops_batch_size: null       
        num_bins: 100               
        bandwidth: 0.1           
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
    output_path = os.path.join(tests_path, "output_compute_features")
    output_features_path = os.path.join(output_path, "filtered_features.txt")
    
    # Call API
    output_features = filter_features(
        configuration = get_config(),
        colvars_path = colvars_path,
        csv_summary = True,
        filtered_features_path = output_features_path,
        output_folder = output_path)
    
    # Read the all reference filtered features into a list
    with open(reference_features_path, 'r') as f:
        reference_features = f.readlines()
    
    # Remove the newline characters
    reference_features = [line.strip() for line in reference_features]
    
    # Compare the reference and output features
    assert output_features == reference_features
    
    # If the test passed, clean the output folder
    shutil.rmtree(output_path)