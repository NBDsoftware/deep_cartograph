"""
Schema for the configuration of the analyze geometry tool.
"""
from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict


class RMSSettings(BaseModel):
    """Common settings of an RMSD or RMSF analysis."""
    
    # Title for the RMS calculation
    title: str
    
    # Selection of atoms to compute the RMS
    selection: str = "protein and name CA"
    
    # Selection of atoms to fit the trajectory before computing the RMS
    fit_selection: str = "protein and name CA"
    
class RMSDSettings(RMSSettings):
    """Settings of one RMSD analysis."""
    
    # Title for the RMSD calculation
    title: str = "Protein Backbone RMSD"

class RMSFSettings(RMSSettings):
    """Settings of one RMSF analysis."""
    
    # Title for the RMSF calculation
    title: str = "Protein Backbone RMSF"

class dRMSDSettings(BaseModel):
    """Settings of one dRMSD analysis."""
    
    # Title for the dRMSD calculation
    title: str = "Protein Backbone dRMSD"
    
    # Selection of atoms to compute the dRMSD
    selection: str = "protein and name CA"
    
    # Stride for the selection of atoms. Include only every stride-th atom in the selection
    selection_stride: int = 5

class AnalysisList(BaseModel):
    """Validates the `analysis` section: the analyses to run, grouped by type."""
    
    # RMSD analyses to run, keyed by a name of your choice
    RMSD: Dict[str, RMSDSettings] = {}
    
    # RMSF analyses to run, keyed by a name of your choice
    RMSF: Dict[str, RMSFSettings] = {}
    
    # dRMSD analyses to run, keyed by a name of your choice
    dRMSD: Dict[str, dRMSDSettings] = {}
    
    
class AnalyzeGeometrySchema(BaseModel):
    """Validates the `analyze_geometry` section of the configuration."""
    
    # Analyses to run
    analysis: AnalysisList = AnalysisList()
    
    # Time between consecutive trajectory frames (in ps), used for the time axis of the plots
    dt_per_frame: float = 1.0
    
    # Whether to run this step or not
    run: bool = True