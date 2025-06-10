# Import modules
import logging
import lightning

# Set logger
logger = logging.getLogger(__name__)

class KLAAnnealing(lightning.Callback):
    """
    Callback to anneal the KL divergence weight (beta).
    
    This callback linearly increases the beta value from 0 to max_beta
    over a specified number of epochs, starting from a given epoch.
    
    Inputs
    ------
    
    max_beta: 
        The final beta value to reach at the end of annealing.
    start_epoch: 
        The epoch at which to start annealing.
    n_epochs_anneal: 
        The number of epochs over which to anneal beta.
    """
    def __init__(self, max_beta=1.0, start_epoch=0, n_epochs_anneal=100):
        super().__init__()
        # The final beta value you want to reach
        self.max_beta = max_beta
        self.start_epoch = start_epoch
        self.n_epochs_anneal = n_epochs_anneal

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch < self.start_epoch:
            beta = 0.0
        elif current_epoch >= self.start_epoch + self.n_epochs_anneal:
            beta = self.max_beta
        else:
            # Linearly increase beta
            progress = (current_epoch - self.start_epoch) / self.n_epochs_anneal
            beta = self.max_beta * progress
            
        # Set beta in the LightningModule
        # Assumes your pytorch_lightning module has a beta attribute
        if not hasattr(pl_module, 'beta'):
            logger.warning("The LightningModule does not have a 'beta' attribute. "
                           "Please ensure it is defined to use KLAAnnealing.")
            return
        pl_module.beta = beta
        pl_module.log('beta', beta, on_step=False, on_epoch=True)