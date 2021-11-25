'''
Created on 1 Apr 2021

@author: Nicholas J. Browning
@contact: nickjbrowning@gmail.com

@copyright: 

'''
import torch
from qml_lightning.cuda import pairlist_gpu
from qml_lightning.cuda import fchl_gpu, egto_gpu
import numpy as np
from qml_lightning.representations.Representation import Representation


class FCHLCuda(Representation):

    def __init__(self, species=np.array([1, 6, 7, 8]), low_cutoff=0.0, high_cutoff=8.0, nRs2=24, nRs3=20,
                 eta2=0.32, eta3=2.7, two_body_decay=1.8, three_body_weight=13.4, three_body_decay=0.57):
        
        super(FCHLCuda, self).__init__()
        
        self.species = torch.from_numpy(species).float().cuda()
        self.nspecies = len(species)
        
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        
        self.nRs2 = nRs2
        self.nRs3 = nRs3
        
        self.eta2 = eta2
        self.eta3 = eta3
        
        self.two_body_decay = two_body_decay
        self.three_body_weight = np.sqrt(eta3 / np.pi) * three_body_weight
        
        self.three_body_decay = three_body_decay
        
        self.Rs2 = torch.linspace(0.0, self.high_cutoff, nRs2 + 1)[1:].cuda()
        self.Rs3 = torch.linspace(0.0, self.high_cutoff, nRs3 + 1)[1:].cuda()

        self.fp_size = self.nspecies * nRs2 + (self.nspecies * (self.nspecies + 1)) * nRs3
        
        self.pi = torch.acos(torch.zeros(1)).cuda() * 2
        
        self.device = torch.device('cuda')
    
    def get_representation(self, X:torch.Tensor, Z: torch.Tensor, atomIDs: torch.Tensor, molIDs: torch.Tensor, atom_counts: torch.Tensor,
                           cell=torch.empty(0, 3, 3, device=torch.device('cuda'))):
        
        inv_cell = torch.empty(0, 3, 3, device=torch.device('cuda'))
        
        if (cell.shape[0] > 0):
            inv_cell = torch.inverse(cell)
            
        nneighbours = pairlist_gpu.get_num_neighbours_gpu(X, atom_counts, self.high_cutoff,
                                                         cell , inv_cell)
      
        max_neighbours = nneighbours.max().item()
     
        neighbourlist = pairlist_gpu.get_neighbour_list_gpu(X, atom_counts, max_neighbours, self.high_cutoff,
                                                            cell, inv_cell)
         
        element_types = egto_gpu.get_element_types_gpu(X, Z, atom_counts, self.species) 

        output = fchl_gpu.get_fchl(X, Z, self.species, element_types, atomIDs, molIDs, neighbourlist, nneighbours,
                               self.Rs2, self.Rs3, self.eta2, self.eta3, self.two_body_decay, self.three_body_weight, self.three_body_decay,
                               self.high_cutoff, False)
        
        return output[0]
    
    def get_representation_and_derivative(self, X:torch.Tensor, Z: torch.Tensor, atomIDs: torch.Tensor, molIDs: torch.Tensor, atom_counts: torch.Tensor,
                                                                     cell=torch.empty(0, 3, 3, device=torch.device('cuda'))):
        
        inv_cell = torch.empty(0, 3, 3, device=torch.device('cuda'))
        
        if (cell.shape[0] > 0):
            inv_cell = torch.inverse(cell)

        nneighbours = pairlist_gpu.get_num_neighbours_gpu(X, atom_counts, self.high_cutoff,
                                                          cell, inv_cell)
        
        max_neighbours = nneighbours.max().item()
     
        neighbourlist = pairlist_gpu.get_neighbour_list_gpu(X, atom_counts, max_neighbours, self.high_cutoff,
                                                            cell, inv_cell)
        
        element_types = egto_gpu.get_element_types_gpu(X, Z, atom_counts, self.species) 
        
        output = fchl_gpu.get_fchl(X, Z, self.species, element_types, atomIDs, molIDs, neighbourlist, nneighbours,
                               self.Rs2, self.Rs3, self.eta2, self.eta3, self.two_body_decay, self.three_body_weight, self.three_body_decay,
                               self.high_cutoff, True)
         
        return output[0], output[1]
    
    def rep_deriv_fd(self, X, Z, atomIDs, molIDs, natom_counts, dx=0.005):
    
        rep_derivative_fd = torch.zeros(X.shape[0], X.shape[1], X.shape[1], 3, self.fp_size, dtype=torch.float64, device=X.device)
        
        for i in range(X.shape[1]):
        
            for x in range (3):
                
                X_copy = X.clone()
                
                X_copy[:, i, x] += dx
                
                gto_plus = self.get_representation(X_copy, Z, atomIDs, molIDs, natom_counts)
      
                X_copy[:, i, x] -= 2.0 * dx
                
                gto_minus = self.get_representation(X_copy, Z, atomIDs, molIDs, natom_counts)
                
                rep_derivative_fd[:,:, i, x,:] = (gto_plus - gto_minus) / (2.0 * dx)
                
        return rep_derivative_fd
    
    def get_representation_torch(self, X:torch.Tensor, Z: torch.Tensor, atomIDs: torch.Tensor, molIDs: torch.Tensor, atom_counts: torch.Tensor, cell=torch.empty(0, 3, 3, device=torch.device('cuda'))):
        return self.forward(X, Z, atom_counts, cell)
    
    def get_representation_derivative_torch(self, X:torch.Tensor, Z: torch.Tensor, atomIDs: torch.Tensor, molIDs: torch.Tensor, atom_counts: torch.Tensor, cell=torch.empty(0, 3, 3, device=torch.device('cuda'))):
        
        X.requires_grad = True
        
        gto = self.forward(X, Z, atom_counts, cell)

        derivative = torch.zeros(X.shape[0], X.shape[1], X.shape[1], 3, gto.shape[2], device=torch.device('cuda'))
        
        for i in range (gto.shape[0]):
            for j in range(gto.shape[1]):
                for k in range(gto.shape[2]):
                    grad, = torch.autograd.grad(gto[i, j, k], X, retain_graph=True, allow_unused=True)
              
                    derivative[:, j,:,:, k] = grad
    
        return derivative
        
    def forward(self, coordinates, nuclear_charges, natom_counts, cell=torch.empty(0, 3, 3, device=torch.device('cuda'))):
        raise Exception("Not Implemented!")
        return None

