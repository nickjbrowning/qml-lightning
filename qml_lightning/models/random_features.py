'''
Created on 3 May 2021

@author: Nicholas J. Browning
'''
import torch
import torch.nn as nn
import numpy as np

from qml_lightning.features.SORF import get_SORF_diagonals, get_bias, SORFTransformCuda
from qml_lightning.representations.dimensionality_reduction import  project_representation


class StructuredOrthgonalRandomFeatures(nn.Module):
    
    def __init__(self, fingerprint, species=np.array([1, 6, 7, 8]), ntransforms=1, sigma=16.0, llambda=1e-10, nfeatures=8192, npcas=256, nbatch=4):
        
        super(StructuredOrthgonalRandomFeatures, self).__init__()
        
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
    
        self.species = species
        self.npcas = npcas
        self.ntransforms = ntransforms
        self.sigma = sigma
        self.llambda = llambda
        self.nfeatures = nfeatures
        self.nbatch = nbatch
        
        self.bias = None
        self.reductors = None
        self.Dmat = None
        
        self.feature_normalisation = np.sqrt(2.0 / nfeatures)
        self.coeff_normalisation = np.sqrt(npcas) / sigma
        
        self.nstacks = int(float(nfeatures) / npcas)
        
        self.fingerprint = fingerprint 

    def get_reductors(self, X, Z, print_summary=True):
    
        reductors = {}
        
        with torch.no_grad():
            
            for e in self.species:
                
                inputs = []
                
                for coordinates, charges in zip(X, Z):
                    
                    xyz = coordinates.unsqueeze(0)
                    q = charges.unsqueeze(0)
                    
                    rep = self.fingerprint.forward(xyz, q.int())
                    
                    indexes = q == e
                    
                    sub = rep[indexes]
                    
                    if (sub.shape[0] == 0):
                        continue
                    
                    inputs.append(sub)
                
                if (len(inputs) == 0):
                    continue
                
                mat = torch.cat(inputs)
                
                perm = torch.randperm(mat.size(0))
                idx = perm[:512]
        
                choice_input = mat[idx]
    
                eigvecs, eigvals, vh = torch.linalg.svd(choice_input.T, full_matrices=False, compute_uv=True)
            
                cev = 100 - (torch.sum(eigvals) - torch.sum(eigvals[:self.npcas])) / torch.sum(eigvals) * 100
            
                reductor = eigvecs[:,:self.npcas]
                size_from = reductor.shape[0]
                size_to = reductor.shape[1]
                
                if (print_summary):
                    print (f"{size_from} -> {size_to}  Cumulative Explained Feature Variance = {cev:6.2f} %%")
                
                reductors[e] = reductor
        
        return reductors
        
    def train(self, coordinates, charges, energies):
        '''
        coordinates, List of Tensors: [size(*, 3)]
        charges, List of Tensors: [size(*)]
        energies, Tensor: size(ntrain)
        
        where * indicates a specific number of atoms for each Tensor in the list.
        
        '''
        
        with torch.no_grad():
            
            if (self.reductors is None):
                self.reductors = self.get_reductors(coordinates, charges)
                
            if (self.bias is None):
                self.bias = get_bias(self.species, self.nfeatures)
                
            if (self.Dmat is None):
                self.Dmat = get_SORF_diagonals(self.species, self.ntransforms, self.nfeatures, self.npcas)
            
            ntrain = len(coordinates)
          
            Ztrain = torch.zeros(ntrain, self.nfeatures, device=self.device, dtype=torch.float64)
        
            for i, (X, Z) in enumerate(zip(coordinates, charges)): 
            
                X = X.unsqueeze(0)
                Z = Z.unsqueeze(0)
                
                rep = self.fingerprint.forward(X, Z.int())
                
                for e in self.species:
                    
                    indexes = Z.int() == e
    
                    sub = rep[indexes]
                    
                    if (sub.shape[0] == 0):
                        continue
                    
                    sub = project_representation(sub, self.reductors[e])
                    
                    sub = sub.repeat(1, self.nstacks).reshape(sub.shape[0], self.nstacks, self.npcas)
                    
                    coeffs = self.coeff_normalisation * SORFTransformCuda.apply(self.Dmat[e] * sub)
                    coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1] * coeffs.shape[2])
                    
                    Ztrain[i,:] += self.feature_normalisation * torch.sum(torch.cos(coeffs + self.bias[e]).double(), dim=0)
            
            ZtrainY = torch.matmul(Ztrain.T, energies.double()[:, None])
    
            ZTZ = torch.matmul(Ztrain.T, Ztrain)
            
            ZTZ[torch.eye(self.nfeatures).bool()] += self.llambda
            
            self.alpha = torch.solve(ZtrainY, ZTZ).solution[:, 0]
        
    def forward(self, coordinates, charges):
        
        '''computes the out-of-sample predictions'''
        
        ntest = len(coordinates)
        
        Ztest = torch.zeros(ntest, self.nfeatures, device=self.device, dtype=torch.float64)

        for i, (X, Z) in enumerate(zip(coordinates, charges)): 
            
            X = X.unsqueeze(0)
            Z = Z.unsqueeze(0)
            
            rep = self.fingerprint.forward(X, Z.int())
            
            for e in self.species:
                
                indexes = Z.int() == e

                sub = rep[indexes]
                
                if (sub.shape[0] == 0):
                    continue
                
                sub = project_representation(sub, self.reductors[e])
                
                sub = sub.repeat(1, self.nstacks).reshape(sub.shape[0], self.nstacks, self.npcas)
                
                coeffs = self.coeff_normalisation * SORFTransformCuda.apply(self.Dmat[e] * sub)
                coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1] * coeffs.shape[2])
                
                Ztest[i,:] += self.feature_normalisation * torch.sum(torch.cos(coeffs + self.bias[e]).double(), dim=0)
        
        return torch.matmul(Ztest, self.alpha)
