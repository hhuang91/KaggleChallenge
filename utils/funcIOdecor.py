# -*- coding: utf-8 -*-
"""
Decorator functions that handles input and output
conversion between np.array and torch.tensor

@author: H. Huang
"""
import torch

def tensorIn_TensorOut(inputFunc):
    """
    decorator for functions that operate on tensors and are expected to return tensors
    """
    def outputFunc(*args,**kwargs):
        newArgs = [x if torch.is_tensor(x) else torch.tensor(x) for x in args]
        newKwArgs = {key:(x if torch.is_tensor(x) else torch.tensor(x)) for (key,x) in kwargs.items()}
        output = inputFunc(*newArgs,**newKwArgs)
        if isinstance(output, tuple):
            newOutput = [x if torch.is_tensor(x) else torch.tensor(x) for x in output]
        else:
            newOutput = output if torch.is_tensor(output) else torch.tensor(output)
        return newOutput
    return outputFunc


