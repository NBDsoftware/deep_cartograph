from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict


class RMSSettings(BaseModel):
    
    # Title for the RMS calculation
    title: str
    
    # Selection of atoms to compute the RMS
    selection: str = "protein and name CA"
    
    # Selection of atoms to fit the trajectory before computing the RMS
    fit_selection: str = "protein and name CA"
    
class RMSDSettings(RMSSettings):
    
    # Title for the RMSD calculation
    title: str = "Protein Backbone RMSD"

class RMSFSettings(RMSSettings):
    
    # Title for the RMSF calculation
    title: str = "Protein Backbone RMSF"
    

class AnalysisList(BaseModel):
    
    RMSD: Dict[str, RMSDSettings] = {}
    
    RMSF: Dict[str, RMSFSettings] = {}
    
    
class AnalyzeGeometrySchema(BaseModel):
    
    analysis: AnalysisList = AnalysisList()
    
    dt_per_frame: float = 1.0