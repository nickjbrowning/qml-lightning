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


class FCHLFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, non_grad_parameters):
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        
        Z, species, atomIDs, molIDs, atom_counts, cell, \
                Rs2, Rs3, eta2, eta3, two_body_decay, three_body_weight, three_body_decay, rcut = non_grad_parameters

        inv_cell = torch.empty(0, 3, 3, device=torch.device('cuda'))
        
        if (cell.shape[0] > 0):
            inv_cell = torch.inverse(cell)
            
        nneighbours = pairlist_gpu.get_num_neighbours_gpu(X, atom_counts, rcut,
                                                         cell , inv_cell)
      
        max_neighbours = nneighbours.max().item()
     
        neighbourlist = pairlist_gpu.get_neighbour_list_gpu(X, atom_counts, max_neighbours, rcut,
                                                            cell, inv_cell)
         
        element_types = egto_gpu.get_element_types_gpu(X, Z, atom_counts, species) 
        
        ctx.save_for_backward(X, Z, species, atomIDs, molIDs, element_types, neighbourlist, nneighbours)
        
        ctx.Rs2 = Rs2
        ctx.Rs3 = Rs3
        ctx.eta2 = eta2
        ctx.eta3 = eta3
        ctx.two_body_decay = two_body_decay
        ctx.three_body_weight = three_body_weight
        ctx.three_body_decay = three_body_decay 
        ctx.rcut = rcut
        
        output = fchl_gpu.get_fchl_representation(X, Z, species, element_types, atomIDs, molIDs, neighbourlist, nneighbours,
                               Rs2, Rs3, eta2, eta3, two_body_decay, three_body_weight, three_body_decay,
                               rcut)
        
        end.record()
        torch.cuda.synchronize()
        
        # print ("forwards time: ", start.elapsed_time(end))
        
        return output

    @staticmethod
    def backward(ctx, gradX):

        X, Z, species, atomIDs, molIDs, element_types, neighbourlist, nneighbours = ctx.saved_tensors
        
        grad_out = fchl_gpu.fchl_backwards(X, Z, species, element_types, atomIDs, molIDs, neighbourlist, nneighbours,
                               ctx.Rs2, ctx.Rs3, ctx.eta2, ctx.eta3, ctx.two_body_decay, ctx.three_body_weight, ctx.three_body_decay,
                               ctx.rcut, gradX)
        
        # grad = fchl_gpu.get_fchl_derivative(X, Z, species, element_types, atomIDs, molIDs, neighbourlist, nneighbours,
        #                       ctx.Rs2, ctx.Rs3, ctx.eta2, ctx.eta3, ctx.two_body_decay, ctx.three_body_weight, ctx.three_body_decay,
        #                       ctx.rcut)
        #            a       b        c    d     e
        # output : nbatch, natoms, natoms, 3, repsize
        #            a       b                   e
        # gradX :  nbatch, natoms,      1, 1, repsize
        # grad_out = torch.einsum('abcde,abe->acd', grad, gradX)
        
        # print ("backwards time: ", start.elapsed_time(end))
        
        return grad_out, None
    

class FCHLCuda(torch.nn.Module):

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

        output = fchl_gpu.get_fchl_representation(X, Z, self.species, element_types, atomIDs, molIDs, neighbourlist, nneighbours,
                               self.Rs2, self.Rs3, self.eta2, self.eta3, self.two_body_decay, self.three_body_weight, self.three_body_decay,
                               self.high_cutoff)
 
        return output
    
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
        
        output = fchl_gpu.get_fchl_and_derivative(X, Z, self.species, element_types, atomIDs, molIDs, neighbourlist, nneighbours,
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

    def forward(self, X, Z, atomIDs, molIDs, atom_counts, cell):
        return FCHLFunction.apply(X, (Z, self.species, atomIDs, molIDs, atom_counts, cell,
                self.Rs2, self.Rs3, self.eta2, self.eta3, self.two_body_decay, self.three_body_weight, self.three_body_decay, self.high_cutoff))
    
