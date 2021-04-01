'''
Created on 1 Apr 2021

@author: Nicholas J. Browning
'''
import numpy as np
from qml_lightning.cuda import sorf_gpu
import torch


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
    
    coeffs = normalization * sorf_gpu.sorf_matrix_gpu(input_rep, diagonals, nfeatures)
    
    end.record()
    torch.cuda.synchronize()
    
    if (print_timings):
        print("SORF coefficients time: ", start.elapsed_time(end), "ms")
    
    return coeffs


def get_features(coeffs, bias, batch_indexes, batch_num, print_timings=False):
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    features = sorf_gpu.molecular_featurisation_gpu(coeffs, bias, batch_indexes, batch_num)
    
    end.record()
    torch.cuda.synchronize()
    
    if (print_timings):
        print("features time: ", start.elapsed_time(end), "ms")
    
    return features


def get_feature_derivatives(coeffs, bias, diagonals, input_grad, batch_indexes, batch_num, normalization, print_timings=False):
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    feature_derivs = normalization * sorf_gpu.molecular_featurisation_derivative_gpu(coeffs, bias, diagonals, input_grad, batch_indexes, batch_num)
    
    end.record()
    torch.cuda.synchronize()
    
    if (print_timings):
        print("feature derivatives time: ", start.elapsed_time(end), "ms")
        
    return feature_derivs
