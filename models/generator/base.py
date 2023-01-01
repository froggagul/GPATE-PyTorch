import torch
import torch.nn as nn
from typing import Optional

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

    def _forward_with_no_condition(self, z) -> torch.Tensor:
        raise NotImplementedError("forward with no condition must be implemented")

    def _forward_with_condition(self, z, y) -> torch.Tensor:
        raise NotImplementedError("forward with condition must be implemented")
    
    def forward(self, z: torch.Tensor, y: Optional[torch.Tensor] = None):
        if y is None:
            return self._forward_with_no_condition(z)
        else:
            return self._forward_with_condition(z, y)

