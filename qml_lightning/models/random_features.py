'''
Created on 3 May 2021

@author: Nicholas J. Browning
'''
import torch
import numpy as np

from qml_lightning.cuda.rff_gpu import get_rff, get_rff_derivatives

from qml_lightning.models.kernel import BaseKernel

from qml_lightning.representations.dimensionality_reduction import project_representation, project_derivative


class RandomFourrierFeaturesModel(BaseKernel):
    
    def __init__(self, rep=None, elements=np.array([1, 6, 7, 8]), sigma=2.0, llambda=1e-10,
                 nfeatures=8192, npcas=128, nbatch=1024):
        
        super(RandomFourrierFeaturesModel, self).__init__(rep, elements, sigma, llambda)
        
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')

        self.nfeatures = nfeatures
        self.npcas = npcas
        self.nbatch = nbatch

        self.species = torch.from_numpy(elements).float().cuda()
        
        self.sample_elemental_basis()
        
        self.alpha = None
    
    def sample_elemental_basis(self):
    
        D = self.nfeatures
        d = self.npcas
    
        W = dict()
        b = dict()
        
        for e in self.elements:
            
            W[e] = torch.from_numpy(np.random.normal(0.0, 1.0, [d, D]) / (2 * self.sigma ** 2)).cuda()
            b[e] = torch.from_numpy(np.random.uniform(0.0, 1.0, [D]) * 2.0 * np.pi).cuda()
            
        self.W = W
        self.b = b
        
    def calculate_features(self, rep, element, indexes, feature_matrix, grad=None, derivative_matrix=None):
        get_rff(rep, self.W[element], self.b[element], indexes, feature_matrix)
        
        if (derivative_matrix is not None and grad is not None):
            get_rff_derivatives(rep, grad, self.W[element], self.b[element], indexes, derivative_matrix)
            
    def get_finite_difference_features(self, rep, X, Z, elements, atomIDs, molIDs, natom_counts, dx=0.001):
        device = X.device
        nbatch = X.shape[0]
        max_natoms = X.shape[1]
        
        if (self.reductors is None):
            print ("Reductors not initialised.")
            exit()
        
        feature_derivative = torch.zeros(nbatch, max_natoms, 3, self.nfeatures, dtype=torch.float64, device=device)

        for j in range(max_natoms):

            for x in range (3):
                
                X_copy = X.clone()
                
                X_copy[:, j, x] += dx
                
                gto_plus = rep.get_representation(X_copy, Z, atomIDs, molIDs, natom_counts)
                
                features_plus = torch.zeros(nbatch, self.nfeatures, dtype=torch.float64, device=device)
                
                for e in elements:
                    indexes = Z == e
                    
                    batch_indexes = torch.where(indexes)[0].type(torch.int)
                    
                    sub = gto_plus[indexes]
      
                    sub = project_representation(sub, self.reductors[e])
                
                    self.calculate_features(sub, e, batch_indexes, features_plus)
                  
                X_copy[:, j, x] -= 2 * dx
                
                gto_minus = rep.get_representation(X_copy, Z, atomIDs, molIDs, natom_counts)
                
                features_minus = torch.zeros(nbatch, self.nfeatures, dtype=torch.float64, device=device)
                
                for e in elements:
                    indexes = Z == e
                    
                    batch_indexes = torch.where(indexes)[0].type(torch.int)
                    
                    sub = gto_minus[indexes]
                    
                    sub = project_representation(sub, self.reductors[e])
                    
                    self.calculate_features(sub, e, batch_indexes, features_minus)
                    
                feature_derivative[:, j, x,:] = (features_plus - features_minus) / (2 * dx)
                
        return feature_derivative
        
    def save_model(self, file_name="model.yaml"):
        pass
    
    def load_model(self, file_name="model.yaml"):
        pass
    
