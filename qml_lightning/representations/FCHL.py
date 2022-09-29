'''
Created on 1 Apr 2021

@author: Nicholas J. Browning
@contact: nickjbrowning@gmail.com

@copyright: 

'''
import torch
from qml_lightning.cuda import pairlist_gpu
from qml_lightning.cuda import fchl_gpu
import numpy as np


class FCHLCuda(torch.nn.Module):

    def __init__(self, species=np.array([1, 6, 7, 8]), rcut=6.0, nRs2=24, nRs3=20,
                 eta2=0.32, eta3=2.7, two_body_decay=1.8, three_body_weight=13.4, three_body_decay=0.57):
        
        super(FCHLCuda, self).__init__()
        
        self.species = torch.from_numpy(species).float().cuda()
        self.nspecies = len(species)
    
        self.rcut = torch.tensor(rcut)
        
        self.nRs2 = torch.tensor(nRs2)
        self.nRs3 = torch.tensor(nRs3)
        
        self.eta2 = torch.tensor(eta2)
        self.eta3 = torch.tensor(eta3)
        
        self.two_body_decay = torch.tensor(two_body_decay)
        self.three_body_weight = np.sqrt(eta3 / np.pi) * torch.tensor(three_body_weight)
        
        self.three_body_decay = torch.tensor(three_body_decay)
        
        self.Rs2 = torch.linspace(0.0, self.rcut, nRs2 + 1)[1:].cuda()
        self.Rs3 = torch.linspace(0.0, self.rcut, nRs3 + 1)[1:].cuda()

        self.fp_size = self.nspecies * nRs2 + (self.nspecies * (self.nspecies + 1)) * nRs3
        
        self.pi = torch.acos(torch.zeros(1)).cuda() * 2
        
        self.device = torch.device('cuda')
    
    def get_repsize(self):
        return self.fp_size
    
    def get_neighbourlist(self, X:torch.Tensor, Z: torch.Tensor, max_neighbours, atom_counts: torch.Tensor,
                           cell=torch.empty(0, 3, 3, device=torch.device('cuda')), inv_cell=torch.empty(0, 3, 3, device=torch.device('cuda'))):
        
        return pairlist_gpu.get_neighbour_list_gpu(X, atom_counts, max_neighbours, self.rcut.item(),
                                                            cell, inv_cell)
    
    def get_neighbours(self, X:torch.Tensor, Z: torch.Tensor, atom_counts: torch.Tensor,
                           cell=torch.empty(0, 3, 3, device=torch.device('cuda')), inv_cell=torch.empty(0, 3, 3, device=torch.device('cuda'))):
        
        return pairlist_gpu.get_num_neighbours_gpu(X, atom_counts, self.rcut.item(),
                                                         cell , inv_cell)
    
    def get_element_types(self, X:torch.Tensor, Z: torch.Tensor, atom_counts: torch.Tensor):
        return fchl_gpu.get_element_types_gpu(X, Z, atom_counts, self.species) 
    
    def get_representation(self, X:torch.Tensor, Z: torch.Tensor, atomIDs: torch.Tensor, molIDs: torch.Tensor, atom_counts: torch.Tensor,
                           cell=torch.empty(0, 3, 3, device=torch.device('cuda')), inv_cell=torch.empty(0, 3, 3, device=torch.device('cuda')), nneighbours=None, neighbourlist=None):
        
        if (nneighbours is None and neighbourlist is None):
            nneighbours = self.get_neighbours(X, Z, atom_counts, cell, inv_cell)
            max_neighbours = nneighbours.max().item()
            neighbourlist = self.get_neighbourlist(X, Z, max_neighbours, atom_counts, cell, inv_cell)
        
        element_types = self.get_element_types(X, Z, atom_counts)
        
        output = fchl_gpu.get_fchl_representation(X, Z, self.species, element_types, cell, inv_cell, atomIDs, molIDs, neighbourlist, nneighbours,
                            self.Rs2, self.Rs3, self.eta2, self.eta3, self.two_body_decay, self.three_body_weight, self.three_body_decay,
                            self.rcut.item())
        
        return output
    
    def get_representation_and_derivative(self, X:torch.Tensor, Z: torch.Tensor, atomIDs: torch.Tensor, molIDs: torch.Tensor, atom_counts: torch.Tensor,
                                                                     cell=torch.empty(0, 3, 3, device=torch.device('cuda')),
                                                                     inv_cell=torch.empty(0, 3, 3, device=torch.device('cuda')), nneighbours=None, neighbourlist=None):
        
        if (nneighbours is None and neighbourlist is None):
            nneighbours = self.get_neighbours(X, Z, atom_counts, cell, inv_cell)
            max_neighbours = nneighbours.max().item()
            neighbourlist = self.get_neighbourlist(X, Z, max_neighbours, atom_counts, cell, inv_cell)
        
        element_types = self.get_element_types(X, Z, atom_counts)
        
        output = fchl_gpu.get_fchl_and_derivative(X, Z, self.species, element_types, cell, inv_cell, atomIDs, molIDs, neighbourlist, nneighbours,
                               self.Rs2, self.Rs3, self.eta2, self.eta3, self.two_body_decay, self.three_body_weight, self.three_body_decay,
                               self.rcut.item(), True)
         
        return output[0], output[1]
    
    def rep_deriv_fd(self, X, Z, atomIDs, molIDs, natom_counts,
                     cells=torch.empty(0, 3, 3, device=torch.device('cuda')),
                     inv_cells=torch.empty(0, 3, 3, device=torch.device('cuda')), dx=0.005):
    
        rep_derivative_fd = torch.zeros(X.shape[0], X.shape[1], X.shape[1], 3, self.fp_size, dtype=torch.float64, device=X.device)
        
        nneighbours = self.get_neighbours(X, Z, natom_counts, cells, inv_cells)
        max_neighbours = nneighbours.max().item()
        neighbourlist = self.get_neighbourlist(X, Z, max_neighbours, natom_counts, cells, inv_cells)
        
        for i in range(X.shape[1]):
        
            for x in range (3):
                
                X_copy = X.clone()
                
                X_copy[:, i, x] += dx
                
                rep_plus = self.get_representation(X_copy, Z, atomIDs, molIDs, natom_counts, cells, inv_cells, nneighbours, neighbourlist)
      
                X_copy[:, i, x] -= 2.0 * dx
                
                rep_minus = self.get_representation(X_copy, Z, atomIDs, molIDs, natom_counts, cells, inv_cells, nneighbours, neighbourlist)
                
                rep_derivative_fd[:,:, i, x,:] = (rep_plus - rep_minus) / (2.0 * dx)
                
        return rep_derivative_fd
    
    def forward(self, X, Z, atomIDs, molIDs, atom_counts,
                cell=torch.empty(0, 3, 3, device=torch.device('cuda')), inv_cell=torch.empty(0, 3, 3, device=torch.device('cuda'))):

        return torch.ops.qml_lightning_fchl.fchl_forwards(X, Z, self.species, atomIDs, molIDs, atom_counts, cell, inv_cell,
                                                          self.Rs2, self.Rs3, self.eta2, self.eta3, self.two_body_decay,
                                                          self.three_body_weight, self.three_body_decay, self.rcut)

