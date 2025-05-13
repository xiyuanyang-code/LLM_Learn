import torch
def save_model(model, filepath):
    """Save the trained model to the specified filepath."""
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    """Load a pre-trained model from the specified filepath."""
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

def set_seed(seed):
    """Set the random seed for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)