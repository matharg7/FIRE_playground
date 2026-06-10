import torch
import torch.nn as nn
from torch import autograd
from typing import Tuple, Any

import deepspeed

from sparsimony.parametrization.fake_sparsity import FakeSparsity



class dfsb(FakeSparsity):
    
    def __name__(self):
        return "dfsb"

    def __init__(self, mask):
        super().__init__(mask)
        self.register_full_backward_hook(_grad_bhook)

    def forward(self, x):
        return x

   

    
def _grad_bhook(self, grad_input, grad_output): 
    if len(grad_input)>1:
        raise ValueError("long input")
    return (grad_output[0]*self.mask,)


class DFSB_RigL(FakeSparsity):
    def __name__(self):
        return "DFSB_RigL"

    def __init__(self, mask):
        super().__init__(mask)
        self.register_full_backward_hook(_accumulate_grad_bhook)
        self.register_buffer(
            "dense_grad", torch.zeros(self.mask.shape, device=mask.device)
        )
        # Default true in case we need to update mask on step 1
        self.accumulate = False

    def forward(self, x):
        # Dense forward pass
        return x


def _accumulate_grad_bhook(self, grad_input, grad_output):
    if len(grad_input) > 1:
        raise ValueError("long input")
    
    # Store dense gradient (unmasked)
    if self.accumulate:
        self.dense_grad += grad_output[0]
    else:
        self.dense_grad = torch.zeros(self.mask.shape, device=self.mask.device)
        
    # Python backward pass (masked gradient)
    return (grad_output[0] * self.mask,)


    