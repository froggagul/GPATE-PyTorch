import torch
import torch.nn as nn
from typing import Optional

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def _forward_with_no_condition(self, x) -> torch.Tensor:
        raise NotImplementedError("forward with no condition must be implemented")

    def _forward_with_condition(self, x, y) -> torch.Tensor:
        raise NotImplementedError("forward with condition must be implemented")
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if y is None:
            return self._forward_with_no_condition(x)
        else:
            return self._forward_with_condition(x, y)

