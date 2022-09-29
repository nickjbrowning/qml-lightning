'''
Created on 3 May 2021

@author: Nicholas J. Browning
'''
import torch
import numpy as np
from tqdm import tqdm
from qml_lightning.features.SORF import SORFTransformCuda, CosFeatures
from qml_lightning.cuda.sorf_gpu import compute_hadamard_features, sorf_matrix_gpu, compute_partial_feature_derivatives, compute_molecular_featurization_derivative
from qml_lightning.models.kernel import BaseKernel
from qml_lightning.representations.dimensionality_reduction import project_representation


class HadamardFeaturesModel(BaseKernel, torch.nn.Module):
    
    def __init__(self, rep=None, elements=np.array([1, 6, 7, 8]), ntransforms=2, sigma=3.0, llambda=1e-11, npcas=128,
                 nbatch_train=64, nbatch_test=64, nstacks=32):
        
        super(HadamardFeaturesModel, self).__init__(rep, npcas, elements, sigma, llambda)

        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
        
        self.ntransforms = torch.tensor(ntransforms)
        
        self.nstacks = nstacks

        self.nbatch_train = nbatch_train
        self.nbatch_test = nbatch_test
        
        self.species = torch.from_numpy(elements).float().cuda()

        self.npcas = npcas
        
        self.Dmat = torch.zeros((len(elements), ntransforms, nstacks, npcas), dtype=torch.float32, device='cuda')
        self.get_SORF_diagonals(elements, ntransforms, nstacks, npcas)
        
        self.bias = torch.zeros((len(elements), self.nfeatures()), dtype=torch.float32, device='cuda')
        self.get_bias(elements, nstacks * npcas)
        
        self.is_trained = False
        self.alpha = torch.zeros(self.nfeatures(), device=self.device, dtype=torch.float)
    
    def get_SORF_diagonals(self, elements, ntransforms, nstacks, npcas):
    
        for e  in elements:
            element_idx = self.element2index[e]
            
            D = np.random.uniform(-1, 1, (ntransforms, nstacks, npcas))
            D[D > 0.0] = 1.0
            D[D < 0.0] = -1.0
            
            self.Dmat[element_idx] = torch.from_numpy(D).float().cuda()
    
    def get_bias(self, elements, nfeatures):
        
        for e  in elements:
            element_idx = self.element2index[e]
            
            v = np.random.uniform(0.0, 1.0, [nfeatures]) * 2.0 * np.pi
            self.bias[element_idx] = torch.from_numpy(v).float().cuda()

    def nfeatures(self):
        return self.npcas * self.nstacks

    def calculate_features(self, representation, element, indexes, feature_matrix, grad=None, derivative_features=None):
        
        coeff_normalisation = np.sqrt(representation.shape[1]) / self.sigma

        element_idx = self.element2index[element]
        
        coeffs = coeff_normalisation * sorf_matrix_gpu(representation, self.Dmat[element_idx])

        compute_hadamard_features(coeffs, self.bias[element_idx], indexes, feature_matrix)
        
        if (derivative_features is not None and grad is not None):
            cos_derivs = torch.zeros(coeffs.shape, device=coeffs.device, dtype=torch.float64)
            compute_partial_feature_derivatives(coeffs, self.bias[element_idx], cos_derivs)

            compute_molecular_featurization_derivative(cos_derivs, coeff_normalisation, self.Dmat[element_idx], grad, indexes, derivative_features)
    
    def forward(self, X, Z, atomIDs, molIDs, atom_counts,
                cell=torch.empty(0, 3, 3, device=torch.device('cuda')), inv_cell=torch.empty(0, 3, 3, device=torch.device('cuda'))):

        coeff_normalisation = torch.sqrt(torch.tensor(self.npcas)) / self.sigma
        
        torch_rep = self.rep.forward(X, Z, atomIDs, molIDs, atom_counts, cell, inv_cell)
    
        Ztest = torch.zeros(X.shape[0], self.nfeatures(), device=torch.device('cuda'), dtype=torch.float32)
         
        for e in self.elements:
              
            indexes = Z.int() == e.item()
            
            element_idx = self.element2index[e]
            
            batch_indexes = torch.where(indexes)[0].type(torch.int)
              
            sub = torch_rep[indexes]
             
            if (sub.shape[0] == 0): continue
                 
            sub = project_representation(sub, self.reductors[element_idx])
            
            coeffs = torch.ops.qml_lightning_sorf.get_SORF_coefficients(sub, self.Dmat[element_idx], coeff_normalisation, self.ntransforms)
               
            coeffs = coeffs.view(coeffs.shape[0], coeffs.shape[1] * coeffs.shape[2])
            
            features = torch.ops.qml_lightning_sorf.get_cos_features(coeffs, self.bias[element_idx], torch.tensor(X.shape[0]), batch_indexes)

            Ztest += features

        total_energies = torch.matmul(Ztest, self.alpha.float())
         
        return total_energies
    
    def predict(self, X, Z, max_natoms, cells=None, inv_cells=None, forces=True, print_info=True, profiler=False):
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        with torch.autograd.profiler.profile(enabled=profiler, use_cuda=True, with_stack=True) as prof:
            
            start.record()
            
            predict_energies = torch.zeros(len(X), device=self.device, dtype=torch.float32)
            predict_forces = torch.zeros(len(X), max_natoms, 3, device=self.device, dtype=torch.float32)
            
            for i in tqdm(range(0, len(X), self.nbatch_test)) if print_info else range(0, len(X), self.nbatch_test):
                
                coordinates = X[i:i + self.nbatch_test]
                charges = Z[i:i + self.nbatch_test]

                if (cells is not None):
                    zcells = cells[i:i + self.nbatch_test]
                    z_invcells = inv_cells[i:i + self.nbatch_test]
                else: 
                    zcells = None 
                    z_invcells = None
                    
                data = self.format_data(coordinates, charges, cells=zcells, inv_cells=z_invcells)
                
                coordinates = data['coordinates']
                charges = data['charges']
                atomIDs = data['atomIDs']
                molIDs = data['molIDs']
                natom_counts = data['natom_counts']
                zcells = data['cells']
                z_invcells = data['inv_cells']

                result = self.predict_opt(coordinates, charges, atomIDs, molIDs, natom_counts, zcells, z_invcells, forces=forces, print_info=True, profiler=False)
                
                if (forces):
                    predict_energies[i:i + self.nbatch_test] = result[0]
                    
                    forces_cuda = result[1]
    
                    for j in range(forces_cuda.shape[0]):
                        predict_forces[i + j,:natom_counts[j]] = forces_cuda[j,:natom_counts[j]]
                else:
                    predict_energies[i:i + self.nbatch_test] = result
      
            end.record()
            torch.cuda.synchronize()
        
        if (profiler):
            print(prof.key_averages(group_by_stack_n=30).table(sort_by='self_cuda_time_total', row_limit=30))
            
        if (print_info):
            print("prediction for", len(X), "molecules time: ", start.elapsed_time(end), "ms")
        
        if (forces):
            return (predict_energies, predict_forces)
        else:
            return predict_energies
        
    def predict_opt(self, coordinates, charges, atomIDs, molIDs, natom_counts, cells=torch.empty(0, 3, 3, device='cuda'), inv_cells=torch.empty(0, 3, 3, device='cuda'),
                    forces=True, print_info=True, profiler=False):
        
        with torch.autograd.profiler.profile(enabled=profiler, use_cuda=True, with_stack=True) as prof:
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        
            start.record()
            
            coeff_normalisation = np.sqrt(self.npcas) / self.sigma

            if (self.is_trained is False):
                print ("Error: must train the model first by calling train()!")
                exit()

            if (forces):
                coordinates.requires_grad = True
      
            torch_rep = self.rep.forward(coordinates, charges, atomIDs, molIDs, natom_counts, cells, inv_cells)
    
            Ztest = torch.zeros(coordinates.shape[0], self.nfeatures(), device=torch.device('cuda'), dtype=torch.float32)
            
            for e in self.elements:
                
                element_idx = self.element2index[e]
                
                indexes = charges.int() == e
                 
                batch_indexes = torch.where(indexes)[0].type(torch.int)
                 
                sub = torch_rep[indexes]
                
                if (sub.shape[0] == 0): continue
                    
                sub = project_representation(sub, self.reductors[element_idx])
            
                coeffs = SORFTransformCuda.apply(sub, self.Dmat[element_idx], coeff_normalisation, self.ntransforms)
                  
                coeffs = coeffs.view(coeffs.shape[0], coeffs.shape[1] * coeffs.shape[2])
                 
                Ztest += CosFeatures.apply(coeffs, self.bias[element_idx], coordinates.shape[0], batch_indexes)
   
            total_energies = torch.matmul(Ztest, self.alpha.float())
            
            if (forces):
                forces_torch, = torch.autograd.grad(-total_energies.sum(), coordinates)
            
            end.record()
            torch.cuda.synchronize()

        if (print_info):
            print("prediction for", coordinates.shape[0], "molecules time: ", start.elapsed_time(end), "ms")
        
        if (forces):
            result = total_energies, forces_torch

        else:
            result = total_energies
    
        if (profiler):
            result = (result,) + (prof,)
            
        return result
    
    def save_jit_model(self, file_name='model.pt'):
        script = torch.jit.script(self)
        script.save(file_name)
        
