"""
PyTorch Lightning callbacks used during CV training: best-model checkpointing
after annealing, KL weight (beta) annealing for VAEs and delayed learning-rate reduction.
"""

# Import modules
import os
import logging
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Literal
import torch

# Set logger
logger = logging.getLogger(__name__)

class PostAnnealingCheckpoint(Callback):
    """
    Custom callback to save the best model based on validation loss,
    but only after the beta-annealing phase is complete.

    Only one checkpoint is kept: each new best model replaces the previous one.

    Parameters
    ----------
    monitor : str
        Name of the logged metric to monitor (lower is better), e.g. 'valid_loss'
    dirpath : str
        Folder where the checkpoint is saved. It is created if it doesn't exist.
    annealing_end_epoch : int
        Epoch from which monitoring starts
    """
    def __init__(self, monitor: str, dirpath: str, annealing_end_epoch: int):
        super().__init__()
        self.monitor = monitor
        self.dirpath = dirpath
        self.annealing_end_epoch = annealing_end_epoch
        self.best_score = torch.inf
        self.best_model_path = ""

        # Ensure the directory exists
        os.makedirs(self.dirpath, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Save a checkpoint if the monitored metric improved (only after the annealing phase)."""
        # Only start monitoring after the annealing phase
        if trainer.current_epoch < self.annealing_end_epoch:
            return

        # Get the metric to monitor
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        # Compare and save the best model checkpoint
        if current_score < self.best_score:
            self.best_score = current_score
            filename = f"best-post-anneal-epoch={trainer.current_epoch}.ckpt"
            # Erase previous best model if exists
            if os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
            self.best_model_path = os.path.join(self.dirpath, filename)
            trainer.save_checkpoint(self.best_model_path)
            logger.debug(f"\nSaved new best post-annealing model with {self.monitor}={current_score:.4f} at epoch {trainer.current_epoch}")
            
class KLAAnnealing(Callback):
    """
    Callback to anneal the KL divergence weight (beta).
    
    This callback changes the beta factor that weights the KL divergence term
    (the regularization of the latent space) during training of a VAE. It helps
    avoid posterior collapse, where the VAE lowers its loss mainly by shrinking the
    KL term and ignores the reconstruction, giving an uninformative latent space.

    Three types of annealing are implemented:

    - Linear annealing: linearly increases beta from start_beta to max_beta
      over n_epochs_anneal epochs, starting after start_epoch.
    - Sigmoid annealing: increases beta following a sigmoid (S-shaped) curve
      from start_beta to max_beta over n_epochs_anneal epochs, starting after start_epoch.
    - Cyclical annealing: repeats a ramp from start_beta to max_beta n_cycles times.

    After the annealing period, beta stays at max_beta (linear and cyclical). The
    LightningModule must have a 'beta' attribute; the value is also logged as 'beta'.

    Parameters
    ----------
    type : {'linear', 'sigmoid', 'cyclical'}, optional
        Type of annealing. Default is 'cyclical'.
    start_beta : float, optional
        The beta value before annealing starts. Default is 0.0.
    max_beta : float, optional
        The final (or maximum) beta value to reach. Default is 0.01.
    start_epoch : int, optional
        Annealing starts after this epoch. Default is 1000.
    n_cycles : int, optional
        For 'cyclical' type: the number of full cycles to perform. Default is 4.
    n_epochs_anneal : int, optional
        'linear' or 'sigmoid' types: the number of epochs to increase beta
        from start_beta to max_beta.
        'cyclical' type: total length of all cycles (divided by n_cycles to get the cycle length).
        Default is 1000.

    Raises
    ------
    ValueError
        If type is not valid, or if n_epochs_anneal < n_cycles for cyclical annealing
    """
    
    def __init__(self, 
                 type: Literal['linear', 'sigmoid', 'cyclical'] = 'cyclical', 
                 start_beta: float = 0.0,
                 max_beta: float = 0.01, 
                 start_epoch: int = 1000, 
                 n_cycles: int = 4,
                 n_epochs_anneal: int = 1000
        ):
        
        super().__init__()
        self.type = type
        self.start_beta = start_beta
        self.max_beta = max_beta
        self.start_epoch = start_epoch
        self.n_cycles = n_cycles
        self.n_epochs_anneal = n_epochs_anneal
        self.cycle_length = n_epochs_anneal // n_cycles
        
        if self.type not in ['linear', 'sigmoid', 'cyclical']:
            raise ValueError("Invalid type for KLAAnnealing. Must be 'linear' or 'cyclical'.")
    
        if self.type == 'cyclical':
            # n_epochs_anneal should be larger than n_cycles
            if n_epochs_anneal < n_cycles:
                raise ValueError("n_epochs_anneal must be greater than or equal to n_cycles for cyclical annealing.")
            
        print(f"KLAAnnealing initialized with type={self.type}, start_beta={self.start_beta}, "
              f"max_beta={self.max_beta})")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Compute beta for the current epoch and set it on the LightningModule."""

        # Default beta and current epoch
        beta = self.start_beta
        current_epoch = trainer.current_epoch 
        
        # Start annealing
        if current_epoch > self.start_epoch:
            
            annealing_epoch = current_epoch - self.start_epoch
            
            if self.type == 'linear':
                beta = self.linear_anneal(annealing_epoch, self.n_epochs_anneal)
            elif self.type == 'sigmoid':
                beta = self.sigmoid_anneal(annealing_epoch, self.n_epochs_anneal)
            elif self.type == 'cyclical':
                beta = self.cyclical_anneal(annealing_epoch, self.n_epochs_anneal)
        
            
        # Set beta in the LightningModule
        # Assumes your pytorch_lightning module has a beta attribute
        if not hasattr(pl_module, 'beta'):
            logger.warning("The LightningModule does not have a 'beta' attribute. "
                           "Please ensure it is defined to use KLAAnnealing.")
            return
        
        pl_module.beta = beta
        pl_module.log('beta', beta, on_step=False, on_epoch=True)
    
    def linear_anneal(self, epoch: int,
                      n_epochs_anneal: int
                      ) -> float:
        """
        Linearly increases beta from start_beta to max_beta over n_epochs_anneal epochs.
        
        Parameters
        ----------
        epoch : int
            The current epoch since the annealing started.
        
        n_epochs_anneal : int
            The total number of epochs over which to anneal beta.
        
        Returns
        -------
        float
            The annealed beta value.
        """
        if epoch >= n_epochs_anneal:
            return self.max_beta
        
        return self.start_beta + (self.max_beta - self.start_beta) * (epoch / n_epochs_anneal)
    
    def cyclical_anneal(self, epoch: int,
                      n_epochs_anneal: int
                      ) -> float:
        """
        Cyclical annealing of beta, cycling between start_beta and max_beta.

        Each cycle has a first half where beta increases linearly
        from start_beta to max_beta, and a second half where it remains at max_beta.
        After n_epochs_anneal epochs, beta stays at max_beta.

        Parameters
        ----------
        epoch : int
            The current epoch since the annealing started.
            
        n_epochs_anneal : int
            The total number of epochs over which to anneal beta.
        
        Returns
        -------
        float
            The annealed beta value.
        """
        
        if epoch >= n_epochs_anneal:
            return self.max_beta
        
        # Progress within the current cycle
        cycle_progress = epoch % self.cycle_length
        
        return self.linear_anneal(cycle_progress, self.cycle_length // 2)

    def sigmoid_anneal(self, epoch: int,
                        n_epochs_anneal: int,
                        ) -> float:
            """
            Sigmoid annealing of beta from start_beta to max_beta over n_epochs_anneal epochs.

            Beta is close to start_beta at the beginning, reaches the midpoint after half
            of n_epochs_anneal and gets close to max_beta at the end.

            Parameters
            ----------
            epoch : int
                The current epoch since the annealing started.
            
            n_epochs_anneal : int
                The total number of epochs over which to anneal beta.
            
            Returns
            -------
            float
                The annealed beta value.
            """
            
            import numpy as np
            
            # Value of sigmoid at start (eps) and at end (1-eps)
            eps = 1e-3         
            
            # Sigmoid parameters    
            midpoint = self.start_epoch + n_epochs_anneal // 2
            steepness = np.log(eps / (1-eps)) / (self.start_epoch - midpoint)
            epoch += self.start_epoch
            
            # Sigmoid function
            beta = self.start_beta + (self.max_beta - self.start_beta) / (1 + np.exp(-steepness * (epoch - midpoint)))
            
            return beta
            
class LROnPlateauManager(Callback):
    """
    Manages the ReduceLROnPlateau scheduler to start monitoring
    only after a specified epoch.

    ReduceLROnPlateau lowers the learning rate when the validation loss stops improving.
    This callback steps the scheduler with the logged 'valid_loss' at the end of each
    validation epoch, starting from start_epoch.

    Parameters
    ----------
    start_epoch : int
        Epoch from which the scheduler starts monitoring the validation loss
    """
    def __init__(self, start_epoch: int):
        super().__init__()
        self.start_epoch = start_epoch
        print(f"LROnPlateauManager initialized. Will start monitoring validation loss at epoch {self.start_epoch}.")

    def on_validation_epoch_end(self, trainer: L.Trainer, _):
        """Step any ReduceLROnPlateau scheduler with the validation loss (after start_epoch)."""
        if trainer.current_epoch < self.start_epoch:
            return

        # Get the validation loss from the trainer's metrics
        validation_loss = trainer.callback_metrics.get('valid_loss')
        if validation_loss is None:
            # Add a warning if the metric is not found after the start epoch
            if trainer.current_epoch == self.start_epoch:
                print(f"Warning: 'valid_loss' not found in callback_metrics. "
                      f"Ensure you are logging it via self.log('valid_loss', ...).")
            return

        lr_schedulers = trainer.lightning_module.lr_schedulers()

        if not isinstance(lr_schedulers, list):
            lr_schedulers = [lr_schedulers]

        for scheduler in lr_schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(validation_loss)
