from pydantic import BaseModel

from deep_cartograph.yaml_schemas.analyze_geometry import AnalyzeGeometrySchema
from deep_cartograph.yaml_schemas.compute_features import ComputeFeaturesSchema
from deep_cartograph.yaml_schemas.filter_features import FilterFeaturesSchema
from deep_cartograph.yaml_schemas.train_colvars import TrainColvarsSchema

class DeepCartograph(BaseModel):
    
    # Schema for the geometric analysis
    analyze_geometry: AnalyzeGeometrySchema = AnalyzeGeometrySchema()
    
    # Schema for the computation of features
    compute_features: ComputeFeaturesSchema = ComputeFeaturesSchema()

    # Schema for the filtering of features
    filter_features: FilterFeaturesSchema = FilterFeaturesSchema()

    # Schema for the training of the colvars file
    train_colvars: TrainColvarsSchema = TrainColvarsSchema()