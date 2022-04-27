'''
Created on 3 May 2021

@author: Nicholas J. Browning
'''
import torch
import numpy as np
from tqdm import tqdm
from qml_lightning.features.SORF import get_SORF_diagonals, get_bias, SORFTransformCuda, CosFeatures
import time
from qml_lightning.cuda.sorf_gpu import compute_hadamard_features, sorf_matrix_gpu, compute_partial_feature_derivatives, compute_molecular_featurization_derivative
from qml_lightning.models.kernel import BaseKernel
from qml_lightning.representations.dimensionality_reduction import project_representation, project_derivative


class HadamardFeaturesModel(BaseKernel):
    
    def __init__(self, rep=None, elements=np.array([1, 6, 7, 8]), ntransforms=1, sigma=3.0, llambda=1e-11, npcas=128,
                 nbatch_train=64, nbatch_test=64, nstacks=32):
        
        super(HadamardFeaturesModel, self).__init__(rep, elements, sigma, llambda)
        
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
        
        self.ntransforms = ntransforms
        
        self.nstacks = nstacks

        self.nbatch_train = nbatch_train
        self.nbatch_test = nbatch_test
        
        self.species = torch.from_numpy(elements).float().cuda()

        self.npcas = npcas

        self.Dmat = get_SORF_diagonals(elements, ntransforms, nstacks, npcas)
        self.bk = get_bias(elements, nstacks * npcas)
        
        self.is_trained = False
        self.alpha = torch.zeros(self.nfeatures(), device=self.device, dtype=torch.float)
    
    def nfeatures(self):
        return self.npcas * self.nstacks

    def calculate_features(self, representation, element, indexes, feature_matrix, grad=None, derivative_features=None):
        coeff_normalisation = np.sqrt(representation.shape[1]) / self.sigma

        coeffs = coeff_normalisation * sorf_matrix_gpu(representation, self.Dmat[element])

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
        
        feature_derivative = torch.zeros(nbatch, max_natoms, 3, self.nfeatures(), dtype=torch.float64, device=device)

        for j in range(max_natoms):

            for x in range (3):
                
                X_copy = X.clone()
                
                X_copy[:, j, x] += dx
                
                gto_plus = rep.get_representation(X_copy, Z, atomIDs, molIDs, natom_counts)
                
                features_plus = torch.zeros(nbatch, self.nfeatures(), dtype=torch.float64, device=device)
                
                for e in elements:
                    indexes = Z == e
                    
                    batch_indexes = torch.where(indexes)[0].type(torch.int)
                    
                    sub = gto_plus[indexes]
      
                    sub = project_representation(sub, self.reductors[e])
                
                    self.calculate_features(sub, e, batch_indexes, features_plus)
                  
                X_copy[:, j, x] -= 2 * dx
                
                gto_minus = rep.get_representation(X_copy, Z, atomIDs, molIDs, natom_counts)
                
                features_minus = torch.zeros(nbatch, self.nfeatures(), dtype=torch.float64, device=device)
                
                for e in elements:
                    indexes = Z == e
                    
                    batch_indexes = torch.where(indexes)[0].type(torch.int)
                    
                    sub = gto_minus[indexes]
                    
                    sub = project_representation(sub, self.reductors[e])
                    
                    self.calculate_features(sub, e, batch_indexes, features_minus)
                    
                feature_derivative[:, j, x,:] = (features_plus - features_minus) / (2 * dx)
                
        return feature_derivative
    
    def SGD(self, X, Z, E, cells=None, inv_cells=None, lr=0.01, niter=100000):

        coeff_normalisation = np.sqrt(self.npcas) / self.sigma
   
        data = self.format_data(X, Z, E=E, cells=cells, inv_cells=inv_cells)
        
        coordinates = data['coordinates']
        charges = data['charges']
        atomIDs = data['atomIDs']
        molIDs = data['molIDs']
        natom_counts = data['natom_counts']
        zcells = data['cells']
        zinv_cells = data['inv_cells']
        energies = data['energies']
        
        torch_rep = self.rep.forward(coordinates, charges, atomIDs, molIDs, natom_counts, zcells, zinv_cells)
        
        permutation = torch.randperm(coordinates.shape[0])
        
        for i in range(0, niter):
            for j in range(0, coordinates.shape[0], self.nbatch_train):
            
                indices = permutation[j:j + self.nbatch_train]
        
                sub_rep = torch_rep[indices]

                Ztrain = torch.zeros(sub_rep.shape[0], self.nfeatures(), device=torch.device('cuda'), dtype=torch.float32)
            
                for e in self.elements:
                    indexes = charges[indices] == e
                    
                    batch_indexes = torch.where(indexes)[0].type(torch.int)
                    
                    sub = sub_rep[indexes]
                    
                    if (sub.shape[0] == 0): continue
             
                    sub = project_representation(sub, self.reductors[e])
        
                    coeffs = SORFTransformCuda.apply(sub, self.Dmat[e], coeff_normalisation, self.ntransforms)
                 
                    coeffs = coeffs.view(coeffs.shape[0], coeffs.shape[1] * coeffs.shape[2])
       
                    Ztrain += CosFeatures.apply(coeffs, self.bk[e], sub_rep.shape[0], batch_indexes)
            
                batch_predictions = torch.matmul(Ztrain, self.alpha)
                
                self.alpha -= lr * (1.0 / sub_rep.shape[0]) * torch.matmul(batch_predictions - energies[indices].float(), Ztrain)
            
            print ("iteration: ", i, torch.mean(torch.abs(batch_predictions - energies[indices].float())))
            print ("alpha:", self.alpha)
    
    def predict(self, X, Z, max_natoms, cells=None, inv_cells=None, forces=True, print_info=True, use_backward=True, profiler=False):

        if (not use_backward):
            return self.predict_cuda(X, Z, max_natoms, cells, forces, print_info)
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        with torch.autograd.profiler.profile(enabled=profiler, use_cuda=True, with_stack=True) as prof:
            
            start.record()

            if (self.is_trained is False):
                print ("Error: must train the model first by calling train()!")
                exit()
            
            predict_energies = torch.zeros(len(X), device=self.device, dtype=torch.float64)
            predict_forces = torch.zeros(len(X), max_natoms, 3, device=self.device, dtype=torch.float64)
            
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

                result = self.predict_opt(coordinates, charges, atomIDs, molIDs, natom_counts, zcells, z_invcells, forces=forces, print_info=False, profiler=False)
                
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
            start1 = torch.cuda.Event(enable_timing=True)
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

            start1.record()
            
            for e in self.elements:
                 
                indexes = charges.int() == e
                 
                batch_indexes = torch.where(indexes)[0].type(torch.int)
                 
                sub = torch_rep[indexes]
                
                if (sub.shape[0] == 0): continue
                    
                sub = project_representation(sub, self.reductors[e])
            
                coeffs = SORFTransformCuda.apply(sub, self.Dmat[e], coeff_normalisation, self.ntransforms)
                  
                coeffs = coeffs.view(coeffs.shape[0], coeffs.shape[1] * coeffs.shape[2])
                 
                Ztest += CosFeatures.apply(coeffs, self.bk[e], coordinates.shape[0], batch_indexes)
   
            total_energies = torch.matmul(Ztest, self.alpha.float())
            
            if (forces):
                forces_torch, = torch.autograd.grad(-total_energies.sum(), coordinates)
            
            end.record()
            torch.cuda.synchronize()
            
        # if (profiler):
            # <FunctionEventAvg key=cudaEventCreateWithFlags self_cpu_time=6.000us cpu_time=1.500us  self_cuda_time=0.000us cuda_time=0.000us input_shapes= cpu_memory_usage=0 cuda_memory_usage=0>
            # print(prof.key_averages()[0])
            # print(prof.key_averages(group_by_stack_n=8).table(sort_by='self_cuda_time_total', row_limit=15)['Name'])
            
        if (print_info):
            print("prediction for", coordinates.shape[0], "molecules time: ", start.elapsed_time(end), "ms")
        
        if (forces):
            result = (total_energies, forces_torch)

        else:
            result = (total_energies,)
    
        if (profiler):
            result = result + (prof,)
            
        return result

    def save_model(self, file_name="model"):
  
        data = {'elements': self.elements,
                'ntransforms': self.ntransforms,
                'nfeatures': self.nfeatures,
                'nbatch_train': self.nbatch_train,
                'nbatch_test': self.nbatch_test,
                'npcas': self.npcas,
                'nstacks': self.nstacks,
                'is_trained': self.is_trained,
                '_subtract_self_energies':self._subtract_self_energies,
                'sigma': self.sigma,
                'llambda': self.llambda,
                'alpha': self.alpha.cpu().numpy()
                }

        if (self._subtract_self_energies):
            data['self_energies'] = self.self_energy.cpu().numpy()
            
        for e in self.elements:
            data[f'dmat_{e}'] = self.Dmat[e].cpu().numpy()
            data[f'bk_{e}'] = self.bk[e].cpu().numpy()
            data[f'reductors_{e}'] = self.reductors[e].cpu().numpy()
            
        np.save(file_name, data)
    
    def load_model(self, file_name="model"):
        
        data = np.load(file_name  if ".npy" in file_name else file_name + ".npy", allow_pickle=True)[()]
        
        self.elements = data['elements']
        self.species = torch.from_numpy(self.elements).float().cuda()
        self.sigma = data['sigma']
        self.llambda = data['llambda']
        
        self.ntransforms = data['ntransforms']
        self.nfeatures = data['nfeatures']
        self.nbatch_train = data['nbatch_train']
        self.nbatch_test = data['nbatch_test']
        
        self.npcas = data['npcas']
        self.nstacks = data['nstacks']
        
        self.is_trained = data['is_trained']
        self.alpha = torch.from_numpy(data['alpha']).double().cuda()
        
        self._subtract_self_energies = data['_subtract_self_energies']

        if (self._subtract_self_energies):
            self.self_energy = torch.from_numpy(data['self_energies']).float()
            
        self.Dmat = {}
        self.bk = {}
        self.reductors = {}
        
        for e in self.elements:
            self.Dmat[e] = torch.from_numpy(data[f'dmat_{e}']).cuda()
            self.bk[e] = torch.from_numpy(data[f'bk_{e}']).cuda()
            self.reductors[e] = torch.from_numpy(data[f'reductors_{e}']).cuda()
        
