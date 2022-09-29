'''
Created on 3 May 2021

@author: Nicholas J. Browning
'''
import torch
import numpy as np
from tqdm import tqdm

from qml_lightning.cuda.rff_gpu import get_rff, get_rff_derivatives

from qml_lightning.models.kernel import BaseKernel

from qml_lightning.representations.dimensionality_reduction import project_representation


class RandomFourrierFeaturesModel(BaseKernel):
    
    # def __init__(self, rep, npcas, elements, sigma, llambda):
        
    def __init__(self, rep=None, elements=np.array([1, 6, 7, 8]), sigma=2.0, llambda=1e-10,
                 nstacks=8192, npcas=128, nbatch_train=64, nbatch_test=64):

        super(RandomFourrierFeaturesModel, self).__init__(rep, npcas, elements, sigma, llambda)
        
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
        
        self.npcas = npcas
        self.nbatch_train = nbatch_train
        self.nbatch_test = nbatch_test
        
        self._nfeatures = nstacks * self.npcas

        self.species = torch.from_numpy(elements).float().cuda()
        
        self.sample_elemental_basis()
        
        self.feature_normalisation = np.sqrt(2.0 / self._nfeatures)
        
        self.alpha = None
    
    def sample_elemental_basis(self):
    
        D = self.nfeatures()
        d = self.npcas
    
        self.W = torch.zeros((len(self.elements), d, D), dtype=torch.float32, device='cuda')
        self.b = torch.zeros((len(self.elements), D), dtype=torch.float32, device='cuda')
        
        for e in self.elements:
            
            idx = self.element2index[e]
            
            self.W[idx] = torch.from_numpy(np.random.normal(0.0, 1.0, [d, D]) / (2 * self.sigma ** 2)).cuda()
            self.b[idx] = torch.from_numpy(np.random.uniform(0.0, 1.0, [D]) * 2.0 * np.pi).cuda()
    
    def nfeatures(self):
        return self._nfeatures
    
    def calculate_features(self, rep, element, indexes, feature_matrix, grad=None, derivative_matrix=None):
        
        idx = self.element2index[element]
        
        get_rff(rep, self.W[idx].double() , self.b[idx].double() , indexes, feature_matrix)
        
        if (derivative_matrix is not None and grad is not None):
            get_rff_derivatives(rep, grad, self.W[idx], self.b[idx], indexes, derivative_matrix)
    
    def predict_opt(self, coordinates, charges, atomIDs, molIDs, natom_counts, zcells, zinv_cells,
                    forces=True, print_info=True, profiler=False):
        
        with torch.autograd.profiler.profile(enabled=profiler, use_cuda=True, with_stack=True) as prof:
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        
            start.record()

            if (self.is_trained is False):
                print ("Error: must train the model first by calling train()!")
                exit()

            if (forces):
                coordinates.requires_grad = True
   
            torch_rep = self.rep.forward(coordinates, charges, atomIDs, molIDs, natom_counts, zcells, zinv_cells)
    
            Ztest = torch.zeros(coordinates.shape[0], self.nfeatures(), device=torch.device('cuda'), dtype=torch.float32)
            
            for e in self.elements:
                
                idx = self.element2index[e]
                
                indexes = charges.int() == e
                 
                batch_indexes = torch.where(indexes)[0].type(torch.int)
                 
                sub = torch_rep[indexes]
                
                if (sub.shape[0] == 0): continue
                    
                sub = project_representation(sub, self.reductors[idx])
 
                Ztest.index_add_(0, batch_indexes, self.feature_normalisation * torch.cos(torch.matmul(sub , self.W[idx]) + self.b[idx][None,:]))
   
            total_energies = torch.matmul(Ztest, self.alpha.float())
            
            if (forces):
                forces_torch, = torch.autograd.grad(-total_energies.sum(), coordinates)
            
            end.record()
            torch.cuda.synchronize()

        if (print_info):
            print("prediction for", coordinates.shape[0], "molecules time: ", start.elapsed_time(end), "ms")
        
        if (forces):
            result = (total_energies, forces_torch)

        else:
            result = total_energies
    
        if (profiler):
            result = result + (prof.key_averages(),)
            
        return result
    
    def predict(self, X, Z, max_natoms, cells=None, inv_cells=None, forces=True, print_info=True, profiler=False):
        
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
                
                zbatch = len(coordinates)
                
                if (cells is not None):
                    zcells = cells[i:i + self.nbatch_test]
                else: zcells = None
                    
                data = self.format_data(coordinates, charges, cells=zcells, inv_cells=inv_cells)
                
                coordinates = data['coordinates']
                charges = data['charges']
                atomIDs = data['atomIDs']
                molIDs = data['molIDs']
                natom_counts = data['natom_counts']
                zcells = data['cells']
                zinv_cells = data['inv_cells']
                
                result = self.predict_opt(coordinates, charges, atomIDs, molIDs, natom_counts, zcells, zinv_cells, forces=forces, print_info=True, profiler=False)
                
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
            # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
            
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
