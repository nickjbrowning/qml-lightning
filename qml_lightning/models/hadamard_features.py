'''
Created on 3 May 2021

@author: Nicholas J. Browning
'''
import torch
import numpy as np
from tqdm import tqdm

from qml_lightning.features.SORF import get_SORF_diagonals, get_bias, SORFTransformCuda

from qml_lightning.cuda.sorf_gpu import compute_hadamard_features, sorf_matrix_gpu, compute_partial_feature_derivatives, compute_molecular_featurization_derivative
from qml_lightning.models.kernel import BaseKernel
from qml_lightning.representations.dimensionality_reduction import project_representation, project_derivative


class HadamardFeaturesModel(BaseKernel):
    
    def __init__(self, rep=None, elements=np.array([1, 6, 7, 8]), ntransforms=1, sigma=3.0, llambda=1e-11,
                 nfeatures=8192, npcas=128, nbatch=64):
        
        super(HadamardFeaturesModel, self).__init__(rep, elements, sigma, llambda)
        
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
        
        self.ntransforms = ntransforms
  
        self.nfeatures = nfeatures
        self.nbatch = nbatch

        self.species = torch.from_numpy(elements).float().cuda()
        
        self.feature_normalisation = np.sqrt(2.0 / nfeatures)
        
        self.npcas = npcas
        
        self.nstacks = int(float(nfeatures) / npcas)

        self.Dmat = get_SORF_diagonals(elements, ntransforms, nfeatures, npcas)
        self.bk = get_bias(elements, nfeatures)
        
        self.alpha = None
    
    def calculate_features(self, representation, element, indexes, feature_matrix, grad=None, derivative_features=None):
        coeff_normalisation = np.sqrt(representation.shape[1]) / self.sigma

        coeffs = coeff_normalisation * sorf_matrix_gpu(representation, self.Dmat[element], self.nfeatures)

        compute_hadamard_features(coeffs, self.bk[element], indexes, feature_matrix)
        
        if (derivative_features is not None and grad is not None):
            cos_derivs = torch.zeros(coeffs.shape, device=coeffs.device, dtype=torch.float64)
            compute_partial_feature_derivatives(coeffs, self.bk[element], cos_derivs)

            compute_molecular_featurization_derivative(cos_derivs, coeff_normalisation, self.Dmat[element], grad, indexes, derivative_features)
    
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
    
    def predict_torch(self, X, Z, max_natoms, cells=None, forces=False, print_info=True):
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        
        coeff_normalisation = np.sqrt(self.npcas) / self.sigma
        
        Dmat_new = {}
        
        for e in self.elements:
            Dmat_new[e] = self.Dmat[e].reshape(self.ntransforms, int(self.nfeatures / self.npcas), self.npcas)
        
        if (self.alpha is None):
            print ("Error: must train the model first by calling train()!")
            exit()
        
        predict_energies = torch.zeros(len(X), device=self.device, dtype=torch.float64)
        predict_forces = torch.zeros(len(X), max_natoms, 3, device=self.device, dtype=torch.float64)

        for i in tqdm(range(0, len(X), self.nbatch)) if print_info else range(0, len(X), self.nbatch):
            
            coordinates = X[i:i + self.nbatch]
            charges = Z[i:i + self.nbatch]
            
            zbatch = len(coordinates)
            
            if (cells is not None):
                zcells = cells[i:i + self.nbatch]
            else: zcells = None
                
            data = self.format_data(coordinates, charges, cells=zcells)
            
            coordinates = data['coordinates']
            charges = data['charges']
            atomIDs = data['atomIDs']
            molIDs = data['molIDs']
            natom_counts = data['natom_counts']
            zcells = data['cells']
            
            if (forces):
                coordinates.requires_grad = True
                
            torch_rep = self.rep.get_representation_torch(coordinates, charges, atomIDs, molIDs, natom_counts, zcells)
            
            Ztest = torch.zeros(zbatch, self.nfeatures, device=torch.device('cuda'), dtype=torch.float32)
            
            for e in self.elements:
                indexes = charges == e
                
                batch_indexes = torch.where(indexes)[0].type(torch.int)
                
                sub = torch_rep[indexes]
                
                if (sub.shape[0] == 0): continue
                
                sub = project_representation(sub, self.reductors[e])
                
                sub = sub.repeat(1, int(self.nfeatures / self.npcas)).reshape(sub.shape[0], int(self.nfeatures / self.npcas), self.npcas)
                
                coeffs = coeff_normalisation
                
                for j in range(self.ntransforms):
                    coeffs *= SORFTransformCuda.apply(Dmat_new[e][None, j] * sub)
               
                coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1] * coeffs.shape[2])
 
                Ztest.index_add_(0, batch_indexes, self.feature_normalisation * torch.cos(coeffs + self.bk[e]))
        
            total_energies = torch.matmul(Ztest.double(), self.alpha)
            
            predict_energies[i:i + self.nbatch] = total_energies
            
            if (forces):
                forces_cuda, = torch.autograd.grad(-total_energies.sum(), coordinates)

                for j in range(forces_cuda.shape[0]):
                    predict_forces[i + j,:natom_counts[j]] = forces_cuda[j,:natom_counts[j]]
  
        end.record()
        torch.cuda.synchronize()
        
        if (print_info):
            print("prediction for", len(X), "molecules time: ", start.elapsed_time(end), "ms")
        
        if (forces):
            return (predict_energies, predict_forces)
        else:
            return predict_energies
        
    def save_model(self, file_name="model.yaml"):
        pass
    
    def load_model(self, file_name="model.yaml"):
        pass
    
