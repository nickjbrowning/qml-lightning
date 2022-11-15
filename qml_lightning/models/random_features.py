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


class RandomFourrierFeaturesModel(BaseKernel, torch.nn.Module):
    
        
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
            get_rff_derivatives(rep, grad, self.W[idx].double(), self.b[idx].double(), indexes, derivative_matrix)
    
    def forward(self, X, Z, atomIDs, molIDs, atom_counts,
                cell=torch.empty(0, 3, 3, device=torch.device('cuda')), inv_cell=torch.empty(0, 3, 3, device=torch.device('cuda'))):

        assert X.device == torch.device('cuda:0'), "X tensor must reside on GPU"
        assert Z.device == torch.device('cuda:0'), "Z tensor must reside on GPU"
        assert atomIDs.device == torch.device('cuda:0'), "atomIDs tensor must reside on GPU"
        assert molIDs.device == torch.device('cuda:0'), "molIDs tensor must reside on GPU"
        assert atom_counts.device == torch.device('cuda:0'), "atom_counts tensor must reside on GPU"
        assert cell.device == torch.device('cuda:0'), "cell tensor must reside on GPU"
        assert inv_cell.device == torch.device('cuda:0'), "inv_cell tensor must reside on GPU"
    
        torch_rep = self.rep.forward(X, Z, atomIDs, molIDs, atom_counts, cell, inv_cell)
    
        Ztest = torch.zeros(X.shape[0], self.nfeatures(), device=torch.device('cuda'), dtype=torch.float32)
            
        for e in self.elements:
           
            indexes = Z.int() == e.item()
            
            element_idx = self.element2index[e]
            
            batch_indexes = torch.where(indexes)[0].type(torch.int)
              
            sub = torch_rep[indexes]
            
            if (sub.shape[0] == 0): continue
                
            sub = project_representation(sub, self.reductors[element_idx])
            
            Ztest.index_add_(0, batch_indexes, self.feature_normalisation * torch.cos(torch.matmul(sub , self.W[element_idx]) + self.b[element_idx][None,:]))
   
        total_energies = torch.matmul(Ztest, self.alpha.float())
            
        return total_energies
    
    def predict_opt(self, coordinates: torch.Tensor, charges: torch.Tensor, atomIDs: torch.Tensor, molIDs: torch.Tensor,
                     natom_counts: torch.Tensor, zcells: torch.Tensor, zinv_cells: torch.Tensor,
                    forces=True, print_info=False, profiler=False) -> torch.Tensor:
        
        '''
            Barebones prediction method which doesn't require format_data to be called. Also includes profiling and some basic information printing.
            
            all inputs must be torch.Tensor residing on the GPU
            

        '''
        
        assert coordinates.device == torch.device('cuda:0'), "coordinates tensor must reside on GPU"
        assert charges.device == torch.device('cuda:0'), "charges tensor must reside on GPU"
        assert atomIDs.device == torch.device('cuda:0'), "atomIDs tensor must reside on GPU"
        assert molIDs.device == torch.device('cuda:0'), "molIDs tensor must reside on GPU"
        assert natom_counts.device == torch.device('cuda:0'), "natom_counts tensor must reside on GPU"
        assert zcells.device == torch.device('cuda:0'), "zcells tensor must reside on GPU"
        assert zinv_cells.device == torch.device('cuda:0'), "zinv_cells tensor must reside on GPU"
        
        with torch.autograd.profiler.profile(enabled=profiler, use_cuda=True, with_stack=True) as prof:
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        
            start.record()

            if (self.is_trained is False):
                print ("Error: must train the model first by calling train()!")
                exit()

            if (forces):
                coordinates.requires_grad = True
                    
            total_energies = self.forward(coordinates, charges, atomIDs, molIDs, natom_counts, zcells, zinv_cells)
        
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
    
    def predict(self, X, Z, max_natoms, cells=None, inv_cells=None, forces=True, print_info=False, profiler=False):
        
        '''
        max_natoms: maximum number of atoms for any molecule in the batch
        X: list of numpy coordinate matrices of shape [natoms, 3] * nbatch
        Z: list of numpy charge vectors of shape  [natoms] * nbatch
        cells: list of numpy 3x3 matrices of shape [3,3] * nbatch
        inv_cells:  list of numpy 3x3 matrices of shape [3,3] * nbatch
        
        format_data will convert the above to (zero-padded where relevant) torch tensors.
        
        '''
        
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
                
                result = self.predict_opt(coordinates, charges, atomIDs, molIDs, natom_counts, zcells, zinv_cells, forces=forces, print_info=print_info, profiler=False)
                
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
        
    def save_jit_model(self, file_name='model_rff.pt'):
        script = torch.jit.script(self)
        script.save(file_name)
