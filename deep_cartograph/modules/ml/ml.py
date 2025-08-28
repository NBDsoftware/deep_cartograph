# Import modules
import logging
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Literal

# Set logger
logger = logging.getLogger(__name__)

class KLAAnnealing(Callback):
    """
    Callback to anneal the KL divergence weight (beta).
    
    This callback modifies the beta factor controlling the weight
    of the KL divergence, and thus regularization of the latent space,
    during training. Useful to avoid posterior collapse. A phenomena occuring 
    whenever the VAE learns to minimize the ELBO loss function just by decreasing
    the KL divergence term and ignoring the reconstruction term. Thus learning an
    uninformative latent space.
    
    Two types of annealing are implemented:
    - Linear annealing: linearly increases the beta value from start_beta to max_beta
        over a specified number of epochs, starting from a given epoch.
    - Sigmoid annealing: increases the beta value following a sigmoid curve
        from start_beta to max_beta over a specified number of epochs, starting from a given epoch
    - Cyclical annealing: cycles the beta value between start_beta and max_beta
        for a specified number of cycles, each with a given length. 
    
    Inputs
    ------
    
    type:
        Type of annealing ('linear', 'sigmoid' or 'cyclical'). Default is 'cyclical'.
        
    start_beta: 
        The initial beta value before annealing starts. Default is 0.0.
        
    max_beta: 
        The final (or maximum) beta value to reach. Default is 0.01.
        
    start_epoch: 
        The epoch at which to start annealing. Default is 1000.
        
    n_cycles:
        For 'cyclical' type: The number of full cycles to perform. Default is 4.
        
    n_epochs_anneal: 
        'linear' or 'sigmoid' types: The total number of epochs to increase beta 
        from start_beta to max_beta.
        'cyclical' type: will be divided by n_cycles to determine cycle length.
        Default is 1000.
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
        Cyclical annealing of beta, cycling between 0 and max_beta.
        
        The cycle length will include a first half where beta increases
        from 0 to max_beta, and a second half where it remains at max_beta.

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
    """
    def __init__(self, start_epoch: int):
        super().__init__()
        self.start_epoch = start_epoch
        print(f"LROnPlateauManager initialized. Will start monitoring validation loss at epoch {self.start_epoch}.")

    def on_validation_epoch_end(self, trainer: L.Trainer, _):
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
