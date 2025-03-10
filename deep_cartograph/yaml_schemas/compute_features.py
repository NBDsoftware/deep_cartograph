from pydantic import BaseModel
from typing import Dict, List, Literal, Union

class DistanceGroup(BaseModel):

    # Selection of atoms to be included in the first selection of this group (MDAnalysis selection syntax)
    first_selection: str = "not name H*"
    # Selection of atoms to be included in the second selection of this group (MDAnalysis selection syntax)
    second_selection: str = "not name H*"
    # Stride for the first selection. Include only every first_stride-th atom in the selection
    first_stride: int = 1
    # Stride for the second selection. Include only every second_stride-th atom in the selection
    second_stride: int = 5
    # If True, skip distances involving atoms in consecutive/neighboring residues
    skip_neigh_residues: bool = False
    # If True, skip distances between atoms that are bonded
    skip_bonded_atoms: bool = True

class DihedralGroup(BaseModel):
    
    # Selection of atoms to be included in this group (MDAnalysis selection syntax)
    selection: str = "not name H*"
    # If True, encode the dihedral angle into the sin and cos of the angle (to obtain smooth features as 0 and 360 degrees correspond to the same angle)
    periodic_encoding: bool = True
    # Mode to search for the dihedrals.
    search_mode: Literal["virtual", "protein_backbone", "real"] = "real"

class DistanceToCenterGroup(BaseModel):

    # Selection of atoms to compute the distance to the geometric center to (MDAnalysis selection syntax)
    selection: str = "not name H*"
    # Selection of atoms to be included in geometric center calculation (MDAnalysis selection syntax)
    center_selection: str = "not name H*"

class Features(BaseModel):
    
    # Groups of distance features. Dictionary with the group name as key and the group schema as value
    distance_groups: Dict[str, DistanceGroup] = {}
    # Groups of dihedral features. Dictionary with the group name as key and the group schema as value
    dihedral_groups: Dict[str, DihedralGroup] = {}
    # Groups of distances to a geometric center. Dictionary with the group name as key and the group schema as value
    distance_to_center_groups: Dict[str, DistanceToCenterGroup] = {}

class PlumedSettings(BaseModel):

    # Time out for the plumed command to run (in seconds)
    timeout: int = 172800
    # Stride for the trajectory. Include only one every traj_stride-th frame in the trajectory
    traj_stride: int = 1
    # Definition of features to be included in the PLUMED input file
    features: Features = Features()

class PlumedEnvironment(BaseModel):

    # Path to the PLUMED binary
    bin_path: str = "plumed"
    # Path to the PLUMED kernel library
    kernel_path: Union[str, None] = None
    # List of commands to run before running the plumed command
    env_commands: List[str] = []

class ComputeFeaturesSchema(BaseModel):
    
    # Plumed settings
    plumed_settings: PlumedSettings = PlumedSettings()
    # Plumed environment
    plumed_environment: PlumedEnvironment = PlumedEnvironment()