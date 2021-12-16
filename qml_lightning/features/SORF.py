'''
Created on 1 Apr 2021

@author: Nicholas J. Browning
@contact: nickjbrowning@gmail.com

@copyright: 

'''
import torch
import numpy as np
from qml_lightning.cuda import sorf_gpu


class SORFTransformCuda2(torch.autograd.Function):
    ''' 
        Wrapper for forward/backward hadamard transforms for pytorch autograd support.
        
    '''

    @staticmethod
    def forward(ctx, u, d, coeff_normalisation):

        ctx.save_for_backward(d)
        ctx.coeff_normalisation = coeff_normalisation
        
        return sorf_gpu.hadamard_transform_gpu2(u, d, coeff_normalisation)

    @staticmethod
    def backward(ctx, grad):

        d = ctx.saved_tensors[0]
        
        # print ("--grad--")
        
        # print (grad.shape)
        # print (grad)
        
        # d_new = torch.ones_like(d)
        
        # grads = d * sorf_gpu.hadamard_transform_gpu2(grad, d_new, ctx.coeff_normalisation)
        
        grads = sorf_gpu.hadamard_transform_backwards_gpu2(grad, d, ctx.coeff_normalisation) 
        # print ("transform: ")
        # print (grads.shape)
        # print (grads)
        
        return grads, None, None
    

class SORFTransformCuda3(torch.autograd.Function):
    ''' 
        Wrapper for forward/backward hadamard transforms for pytorch autograd support.
        
    '''

    @staticmethod
    def forward(ctx, u, d, coeff_normalisation):

        ctx.save_for_backward(d)
        ctx.coeff_normalisation = coeff_normalisation
        
        # print (u.shape)
        
        return sorf_gpu.hadamard_transform_gpu3(u, d, coeff_normalisation)

    @staticmethod
    def backward(ctx, grad):

        d = ctx.saved_tensors[0]
        
#         print ("--SORFTransformCuda3 input grad--", grad.shape)
#     
#         print (grad)
#         
#         # d_new = torch.ones_like(d)
#         
#         # grads = d * sorf_gpu.hadamard_transform_gpu2(grad, d_new, ctx.coeff_normalisation)
#         
        grads = sorf_gpu.hadamard_transform_backwards_gpu3(grad, d, ctx.coeff_normalisation) 
#         # print ("transform: ")
#         # print (grads.shape)
#         # print (grads)
#         
#         print ("--SORFTransformCuda3 output grad--", grad.shape)
#     
#         print (grads)
#         
        return grads, None, None
    

class SORFTransformCuda(torch.autograd.Function):
    ''' 
        Wrapper for forward/backward hadamard transforms for pytorch autograd support.
        
    '''
    
    @staticmethod
    def forward(ctx, u):

        return sorf_gpu.hadamard_transform_gpu(u)

    @staticmethod
    def backward(ctx, grad):

        # print ("--grad--")
        
        # print (grad.shape)
        # print (grad)
        
        grads = sorf_gpu.hadamard_transform_gpu(grad)
        
        # print ("transform: ")
        # print (grads.shape)
        # print (grads)
        
        return grads

      
class CosFeatures(torch.autograd.Function):
    ''' 
        Wrapper for forward/backward structured orthogonal random feature transforms for pytorch autograd support.
        
    '''
    
    @staticmethod
    def forward(ctx, coeffs, b, nmol, batch_indexes):
        
        ctx.save_for_backward(coeffs, b, batch_indexes)
        ctx.nmol = nmol
        
        features = sorf_gpu.CosFeaturesCUDA(coeffs, b, nmol, batch_indexes)
        
        # print ("features:", features.shape)
        return features

    @staticmethod
    def backward(ctx, grad):

        coeffs, b, batch_indexes = ctx.saved_tensors
        
        # print ("inp grad:", grad.shape)
        # print (grad)

        # print ("--CosFeatures grad--", grad.shape)
    
        # print (grad)
        
        grads = sorf_gpu.CosDerivativeFeaturesCUDA(grad, coeffs, b, ctx.nmol, batch_indexes)
        
        # print (grads.shape)
        # print ("outp grad:", grads.shape)
        # print (grads)
        
        # print ("--CosFeatures output grad--", grads.shape)
    
        # print (grads)
        
        return grads, None, None, None

        
class SORFFeatures(torch.autograd.Function):
    ''' 
        Wrapper for forward/backward structured orthogonal random feature transforms for pytorch autograd support.
        
    '''
    
    @staticmethod
    def forward(ctx, sub, D, b, coeff_normalisation, nmol, batch_indexes):
        
        ctx.save_for_backward(sub, D, b, batch_indexes)
        ctx.coeff_normalisation = coeff_normalisation
        ctx.nmol = nmol
        
        features = sorf_gpu.SORFFeaturesCUDA(sub, D, b, coeff_normalisation, nmol, batch_indexes)
        
        # print ("features:", features.shape)
        return features

    @staticmethod
    def backward(ctx, grad):

        # grad is -alphas, repeated nmol times
        
        sub, D, b, batch_indexes = ctx.saved_tensors
        
        # print ("inp grad:", grad.shape)
        # print (grad)
        
        # print (sub.shape)
        # print (grad.shape)
        # print (D.shape)
        # print (b.shape)
        grads = sorf_gpu.SORFFeaturesBackwardsCUDA(grad, sub, D, b, ctx.coeff_normalisation, ctx.nmol, batch_indexes)
        
        # print (grads.shape)
        # print ("outp grad:", grads.shape)
        # print (grads)
        # print ("feature grads:", grads)
        
        return grads, None, None, None, None, None


def get_SORF_diagonals(elements, ntransforms, nfeatures, npcas):
    
    Dmat = {}
    
    for e  in elements:
        D = np.random.uniform(-1, 1, (ntransforms, np.int(np.float(nfeatures) / npcas), npcas))
        D[D > 0.0] = 1.0
        D[D < 0.0] = -1.0
        
        Dmat[e] = torch.from_numpy(D).float().cuda()
        
    return Dmat


def get_bias(elements, nfeatures):
    
    b = {}
    
    for e  in elements:
        v = np.random.uniform(0.0, 1.0, [nfeatures]) * 2.0 * np.pi
        b[e] = torch.from_numpy(v).float().cuda()
        
    return b


def get_SORF_coefficients(input_rep, nfeatures, diagonals, normalization, print_timings=False):
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    coeffs = normalization * sorf_gpu.sorf_matrix_gpu(input_rep, diagonals, nfeatures)
    
    end.record()
    torch.cuda.synchronize()
    
    if (print_timings):
        print("SORF coefficients time: ", start.elapsed_time(end), "ms")
    
    return coeffs


def get_features(coeffs, bias, batch_indexes, batch_num, print_timings=False):
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    features = sorf_gpu.molecular_featurisation_gpu(coeffs, bias, batch_indexes, batch_num)
    end.record()
    torch.cuda.synchronize()
    
    if (print_timings):
        print("features time: ", start.elapsed_time(end), "ms")
    
    return features


def get_feature_derivatives(coeffs, bias, diagonals, input_grad, batch_indexes, batch_num, normalization, print_timings=False):
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    feature_derivs = normalization * sorf_gpu.molecular_featurisation_derivative_gpu(coeffs, bias, diagonals, input_grad, batch_indexes, batch_num)
    end.record()
    torch.cuda.synchronize()
    
    if (print_timings):
        print("feature derivatives time: ", start.elapsed_time(end), "ms")
        
    return feature_derivs
