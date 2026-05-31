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

class dRMSDSettings(BaseModel):
    
    # Title for the dRMSD calculation
    title: str = "Protein Backbone dRMSD"
    
    # Selection of atoms to compute the dRMSD
    selection: str = "protein and name CA"
    
    # Stride for the selection of atoms. Include only every stride-th atom in the selection
    selection_stride: int = 5

class AnalysisList(BaseModel):
    
    RMSD: Dict[str, RMSDSettings] = {}
    
    RMSF: Dict[str, RMSFSettings] = {}
    
    dRMSD: Dict[str, dRMSDSettings] = {}
    
    
class AnalyzeGeometrySchema(BaseModel):
    
    analysis: AnalysisList = AnalysisList()
    
    dt_per_frame: float = 1.0
    
    run: bool = True