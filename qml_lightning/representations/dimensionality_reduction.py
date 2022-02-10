'''
Created on 1 Apr 2021

@author: Nicholas J. Browning
@contact: nickjbrowning@gmail.com

@copyright: 

'''

import torch


def get_reductors(gto, charges, npcas, species):
    
    reductors = {}
    
    for e in species:

        indexes = charges == e

        sub = gto[indexes]
        
        if (sub.shape[0] == 0):
            continue
        
        perm = torch.randperm(sub.size(0))
        idx = perm[:512]

        choice_input = sub[idx]
    
        eigvecs, eigvals, vh = torch.linalg.svd(choice_input.T, full_matrices=False, compute_uv=True)
    
        cev = 100 - (torch.sum(eigvals) - torch.sum(eigvals[:npcas])) / torch.sum(eigvals) * 100
    
        reductor = eigvecs[:,:npcas]
        size_from = reductor.shape[0]
        size_to = reductor.shape[1]
    
        print (f"{size_from} -> {size_to}  Cumulative Explained Feature Variance = {cev:6.2f} %%")
        
        reductors[e] = reductor
    
    return reductors


def project_representation(X, reductor):
   
    '''
    
    projects the representation from shape: 
    nsamples x repsize 
    to 
    nsamples x npcas
    
    '''
    
    return torch.matmul(X, reductor)


def project_derivative(dX, reductor):
    '''
    
    projects the representation derivative from shape:
    
    nsamples x natoms x 3 x repsize 
    to 
    nsamples x natoms x 3 x npcas
    
    '''

    return torch.einsum('jmnk, kl->jmnl', dX, reductor)
