from pydantic import BaseModel

from deep_cartograph.yaml_schemas.compute_features import ComputeFeatures
from deep_cartograph.yaml_schemas.filter_features import FilterFeatures
from deep_cartograph.yaml_schemas.train_colvars import TrainColvars

class DeepCartograph(BaseModel):
    
    # Schema for the computation of features
    compute_features: ComputeFeatures = ComputeFeatures()

    # Schema for the filtering of features
    filter_features: FilterFeatures = FilterFeatures()

    # Schema for the training of the colvars file
    train_colvars: TrainColvars = TrainColvars()