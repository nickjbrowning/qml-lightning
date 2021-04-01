'''
Created on 1 Apr 2021

@author: Nicholas J. Browning
'''

import torch


def get_reductors(X, charges, npcas, elements):
    
    '''
    
    '''
    
    reductors = {}
    
    for e in elements:
        
        indexes = charges == e
    
        batch_indexes = torch.where(indexes)[0].type(torch.int)
    
        sub = X[indexes]
        
        perm = torch.randperm(sub.size(0))
        idx = perm[:500]

        choice_input = sub[idx]

        eigvecs, eigvals, vh = torch.linalg.svd(choice_input.T, full_matrices=False, compute_uv=True)
    
        cev = 100 - (torch.sum(eigvals) - torch.sum(eigvals[:npcas])) / torch.sum(eigvals) * 100
    
        reductor = eigvecs[:,:npcas]
        size_from = reductor.shape[0]
        size_to = reductor.shape[1]
    
        print (f"{size_from} -> {size_to}  Cumulative Explained Feature Variance = {cev:6.2f} %%")
        
        reductors[e] = reductor

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
