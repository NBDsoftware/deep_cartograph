from pydantic import BaseModel
from typing import Literal

class TrajAugmentationSchema(BaseModel):
    
    # Number of frames in the augmented trajectory
    num_frames: int = 1000
    # Interpolation method to use
    interpolation_method: Literal['akima', 'pchip'] = 'pchip' # akima looks smoother but pchip avoids oscillations better - respects monotonicity
    # Selection of atoms to cosider for the new trajectory - MDAnalysis selection syntax
    atom_selection: str = "all"
    # Output trajectory format
    traj_format: Literal['xtc', 'dcd', 'nc', 'pdb'] = 'xtc'