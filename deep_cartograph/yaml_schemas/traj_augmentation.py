from pydantic import BaseModel
from typing import Literal, Optional

class TrajAugmentationSchema(BaseModel):
    
    # Number of frames in the interpolated trajectory (only used if interpolation_method is not None)
    num_frames: int = 1000
    # Whether to keep the original frames in the interpolated trajectory
    keep_original_frames: bool = False
    # Interpolation method to use
    # akima looks smoother but pchip avoids oscillations better - respects monotonicity
    interpolation_method: Optional[Literal['akima', 'pchip']] = 'pchip'
    # Standard deviation of the gaussian noise added to the new trajectory frames (None means no noise is added)
    noise_std: Optional[float] = None
    # Selection of atoms to write in the new trajectory - MDAnalysis selection syntax
    atom_selection: str = "all"
    # Output trajectory format
    traj_format: Literal['xtc', 'dcd', 'nc', 'pdb'] = 'xtc'
    # Wether to prepare the trajectory before augmentation (unwrapping and centering) - better to do it beforehand
    prepare_trajectory: bool = False