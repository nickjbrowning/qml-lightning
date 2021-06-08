'''
Created on 1 Apr 2021

@author: Nicholas J. Browning
@contact: nickjbrowning@gmail.com

@copyright: 

'''

import torch

   
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
