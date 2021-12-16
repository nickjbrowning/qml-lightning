'''
Created on 11 Dec 2021

@author: Nicholas J. Browning
'''
import torch
from qml_lightning.cuda import utils_gpu


class MulInPlaceByConst(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, f):
        
        utils_gpu.MulInPlaceByConstCUDA(X, f)
        
        ctx.f = f
        
        return X

    @staticmethod
    def backward(ctx, gradX):
        
        utils_gpu.MulInPlaceByConstCUDA(gradX, ctx.f)
        
        return gradX, None
    

class CosFeatures(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, indexes, bias, normalisation):
        
        ctx.normalisation = normalisation
        ctx.save_for_backward(X, indexes, bias)
        
        return utils_gpu.CosFeaturesCUDA(X, indexes, bias, normalisation)

    @staticmethod
    def backward(ctx, gradX):
        
        print ("gradX shape: ", gradX.shape)
        X, indexes, bias = ctx.saved_tensors
        
        return utils_gpu.DerivativeCosFeaturesCUDA(X, indexes, bias, ctx.normalisation, gradX), None, None, None

