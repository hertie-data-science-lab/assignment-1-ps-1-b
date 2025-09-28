import numpy as np

def cosine_annealing(initial_lr, epoch, total_epochs, min_lr=0.0):
    """
    TODO: implement a cosine annealing learning rate scheduler.
    
    Args:
        initial_lr (float): Initial learning rate.
        epoch (int): Current epoch number.
        total_epochs (int): Total number of epochs.
        min_lr (float): Minimum learning rate to reach.
        
    Returns:
        float: Adjusted learning rate for the current epoch.
    """

    l_t = min_lr + ((initial_lr - min_lr) / 2) * (1 + np.cos(np.pi * epoch / total_epochs))

    return l_t