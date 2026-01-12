from pydantic import BaseModel
from typing import Literal, Optional

class TrajAugmentationSchema(BaseModel):
    
    # Number of frames in the augmented trajectory
    num_frames: int = 1000
    # Wether to keep the original frames in the augmented trajectory or not
    keep_original_frames: bool = True
    # Interpolation method to use
    interpolation_method: Literal['akima', 'pchip'] = 'pchip' # akima looks smoother but pchip avoids oscillations better - respects monotonicity
    # Standard deviation of the gaussian noise added to the new trajectory frames (None means no noise is added)
    noise_std: Optional[float] = None
    # Selection of atoms to write in the new trajectory - MDAnalysis selection syntax
    atom_selection: str = "all"
    # Output trajectory format
    traj_format: Literal['xtc', 'dcd', 'nc', 'pdb'] = 'xtc'