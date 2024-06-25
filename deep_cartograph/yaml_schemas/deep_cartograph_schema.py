from pydantic import BaseModel

from deep_cartograph.yaml_schemas.compute_features_schema import ComputeFeaturesSchema
from deep_cartograph.yaml_schemas.filter_features_schema import FilterFeaturesSchema
from deep_cartograph.yaml_schemas.train_colvars_schema import TrainColvarsSchema

class DeepCartographSchema(BaseModel):
    
    # Schema for the computation of features
    compute_features: ComputeFeaturesSchema = ComputeFeaturesSchema()

    # Schema for the filtering of features
    filter_features: FilterFeaturesSchema = FilterFeaturesSchema()

    # Schema for the training of the colvars file
    train_colvars: TrainColvarsSchema = TrainColvarsSchema()