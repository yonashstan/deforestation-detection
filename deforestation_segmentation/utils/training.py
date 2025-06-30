"""Training utilities."""

import math
import torch

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_pct: float = 0.1
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler with warmup.
    
    Args:
        optimizer: The optimizer to schedule
        total_steps: Total number of training steps
        warmup_pct: Percentage of steps to use for warmup
        
    Returns:
        Learning rate scheduler with linear warmup and cosine decay
    """
    warmup_steps = int(total_steps * warmup_pct)
    
    def lr_lambda(step: int):
        # Linear warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Cosine decay
        return 0.5 * (1.0 + math.cos(math.pi * float(step - warmup_steps) / float(total_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) 