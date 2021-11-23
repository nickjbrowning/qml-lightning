import torch.nn as nn


class Representation(nn.Module):
    
    def __init__(self):
        pass
    
    def get_representation_torch(self):
        raise NotImplementedError("Abstract method only.")
    
    def get_representation(self):
        raise NotImplementedError("Abstract method only.")
    
    def get_representation_and_derivative(self):
        raise NotImplementedError("Abstract method only.")
    
    def forward(self):
        raise NotImplementedError("Abstract method only.")

