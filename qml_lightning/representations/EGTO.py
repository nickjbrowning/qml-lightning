'''
Created on 1 Apr 2021

@author: Nicholas J. Browning
@contact: nickjbrowning@gmail.com

@copyright: 

'''
import torch
from qml_lightning.cuda import pairlist_gpu
from qml_lightning.cuda import egto_gpu
import numpy as np
from qml_lightning.representations.Representation import Representation


class EGTOCuda(Representation):

    def __init__(self, species=np.array([1, 6, 7, 8]), low_cutoff=0.0, high_cutoff=6.0, ngaussians=24, eta=1.2, lmax=3,
                 lchannel_weights=1.0, inv_factors=1.0, rswitch=4.5, cutoff_function="cosine", distribution="gaussian"):
        
        super(EGTOCuda, self).__init__()
        
        self.species = torch.from_numpy(species).float().cuda()
        self.nspecies = len(species)
        
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.ngaussians = ngaussians
        self.eta = eta
        self.lmax = lmax
        
        if isinstance(lchannel_weights, list) or isinstance(lchannel_weights, np.ndarray):
            self.lchannel_weights = torch.Tensor(lchannel_weights).cuda()
        else:
            # assume scalar
            self.lchannel_weights = torch.zeros(lmax + 1).cuda()
            self.lchannel_weights[:] = lchannel_weights
            
        self.init_inv_factors(inv_factors)
        
        self.generate_angular_numbers()
        
        # eta = (0.5 / ((1.0 - np.exp(-cutoff)) / K)) ** 2 
        
        if (distribution == "expexp"):
            self.offset = torch.linspace(np.exp(-0.4), np.exp(-self.high_cutoff), ngaussians).cuda()
        else:
            self.offset = torch.linspace(0.0, self.high_cutoff, ngaussians + 1)[1:].cuda()

        mbody_list = torch.zeros(species.shape[0], species.shape[0], dtype=torch.int32)
        
        count = 0
        
        for i in range(species.shape[0]):
            mbody_list[i][i] = count
            count += 1
            
        for i in range(species.shape[0]):
            for j in range(i + 1, species.shape[0]):
                mbody_list[i][j] = count
                mbody_list[j][i] = count
                count += 1
            
        self.mbody_list = mbody_list.cuda()
        
        element_to_id = torch.zeros(max(self.species.int()) + 1, dtype=torch.long, device=torch.device('cuda'))
        
        for i, el in enumerate(self.species.int()):
            element_to_id[el] = i
            
        self.element_to_id = element_to_id
        
        element_combinations = []
        
        for i in range(self.nspecies):
            for j in range(i + 1, self.nspecies):
                element_combinations.append([int(self.species[i]), int(self.species[j])])
        
        self.element_combinations = torch.cuda.LongTensor(element_combinations)
        
        self.nmbody = self.element_combinations.shape[0] + self.nspecies
        
        self.fp_size = ngaussians * (lmax + 1) * self.nmbody
        
        self.pi = torch.acos(torch.zeros(1)).cuda() * 2
        
        self.device = torch.device('cuda')
        
        self.rswitch = rswitch
        
        if (cutoff_function == "cosine"):
            self.cut_func = 0
        elif (cutoff_function == "switch"):
            self.cut_func = 1
        else: self.cut_func = 0
            
        if (distribution == "gaussian"):
            self.dist_func = 0
        elif (distribution == "lognormal"):
            self.dist_func = 1
        elif(distribution == "expexp"):
            self.dist_func = 2
        else: self.dist_func = 0

    def init_inv_factors(self, factors):
        
        if isinstance(factors, list) or isinstance(factors, np.ndarray):
            self.inv_factors = torch.Tensor(factors).cuda()
        else:
            # assume scalar
            self.inv_factors = torch.zeros(self.lmax + 1).cuda()
            self.inv_factors[:] = factors
            
        inv_factors = []
        for i in range(self.lmax + 1):
            for k in range (i + 1):
                for m in range(i - k + 1):
                    inv_factors.append(self.inv_factors[i])
                    
        self.inv_factors_torch = torch.Tensor(inv_factors).cuda()
        
    def generate_angular_numbers(self):
        angular_components = []
        angular_weights = []
        angular_indexes = []
        
        for i in range(self.lmax + 1):
            for k in range (i + 1):
                for m in range(i - k + 1):
                    n = i - k - m
                    angular_components.append([n, m, k])
                    angular_weights.append(np.math.factorial(i) / (np.math.factorial(n) * np.math.factorial(m) * np.math.factorial(k)))
                    angular_indexes.append(i)
                    
        angular_components = torch.FloatTensor(angular_components)
        angular_weights = torch.FloatTensor(angular_weights)
        angular_indexes = torch.IntTensor(angular_indexes)
        
        self.orbital_components = angular_components.cuda()
        self.orbital_weights = angular_weights.cuda()
        self.orbital_indexes = angular_indexes.cuda()
    
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

        output = egto_gpu.get_egto(X, Z, self.species, element_types, atomIDs, molIDs, neighbourlist, nneighbours, self.mbody_list,
                               self.orbital_components, self.orbital_weights, self.orbital_indexes, self.offset, self.lchannel_weights, self.inv_factors, self.eta, self.lmax, self.high_cutoff, self.rswitch,
                               cell, inv_cell, self.cut_func, self.dist_func, False)
        
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
        
        output = egto_gpu.get_egto(X, Z, self.species, element_types, atomIDs, molIDs, neighbourlist, nneighbours, self.mbody_list,
        self.orbital_components, self.orbital_weights, self.orbital_indexes, self.offset, self.lchannel_weights, self.inv_factors, self.eta, self.lmax, self.high_cutoff, self.rswitch,
        cell, inv_cell, self.cut_func, self.dist_func, True)
        
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

        n_batch, max_natoms, _ = coordinates.shape
        
        inv_cell = torch.empty(0, 3, 3, device=torch.device('cuda'))
            
        num_neighbours = pairlist_gpu.get_num_neighbours_gpu(coordinates, natom_counts, self.high_cutoff,
                                                          cell, inv_cell)
        
        max_neighbours = num_neighbours.max().item()
        
        neighbours = pairlist_gpu.get_neighbour_list_gpu(coordinates, natom_counts, max_neighbours, self.high_cutoff,
                                                            cell, inv_cell)

        pairlist_mask = (neighbours != -1)

        # hack to get rid of the -1's - picks a valid index for a given atom and fills -1 values with that. pairlist_mask stores
        # the "real" atom indexes
        pairlist_gpu.safe_fill_gpu(neighbours)
        
        idx_m = torch.arange(coordinates.shape[0], dtype=torch.long)[:, None, None]
        
        local_atoms = coordinates[idx_m, neighbours.long()]

        nbh_coords = (coordinates[:,:, None,:] - local_atoms)
        
        # TODO apply PBC here if cell is defined
        if (cell.shape[0] > 0):
            inv_cell = torch.inverse(cell)
            
        distances = torch.linalg.norm(nbh_coords, dim=3)
        
        # mask for the "dummy" atoms introduced when padding the neighbourlist to n_max_neighbours
        parlist_maskval = torch.ones_like(neighbours)
        pairlist_coeffs = parlist_maskval * pairlist_mask
     
        centered_distances = torch.pow(distances[..., None] - self.offset, 2)

        neighbor_numbers = nuclear_charges[idx_m, neighbours[:,:,:].long()]
  
        cutoffs = 0.5 * (torch.cos(distances * self.pi / self.high_cutoff) + 1.0)
       
        radial_basis = torch.sqrt(self.eta / self.pi) * torch.exp(-self.eta * centered_distances) * cutoffs[..., None] * pairlist_coeffs[..., None]
   
        inv_scaling = torch.pow(1.0 / distances[..., None], self.inv_factors_torch + self.orbital_indexes) * pairlist_coeffs[..., None]
      
        angular_terms = inv_scaling * torch.pow(nbh_coords[..., None, 0] , self.orbital_components[:, 0 ]) * \
                torch.pow(nbh_coords[..., None, 1] , self.orbital_components[:, 1 ]) * \
                torch.pow(nbh_coords[..., None, 2] , self.orbital_components[:, 2 ])
        
        fingerprint = torch.zeros(n_batch, max_natoms, self.lmax + 1, self.nmbody, self.ngaussians, dtype=radial_basis.dtype, device=self.device)
        
        # first construct the single-species three-body terms, e.g X-HH, X-CC...
        for i in range(self.nspecies):

            elemental_fingerprint = torch.zeros(n_batch, max_natoms, self.lmax + 1, self.ngaussians, dtype=radial_basis.dtype, device=self.device)
    
            mask = (neighbor_numbers[..., None] == self.species[i]).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)

            filtered_radial = coeffs[..., None] * radial_basis 

            test = angular_terms[... , None] * filtered_radial[..., None,:] 
        
            test = torch.sum(test, dim=2)

            orbitals = self.orbital_weights[None, None,:, None] * torch.pow(test, 2)
          
            elemental_fingerprint.index_add_(2, self.orbital_indexes, orbitals)
            
            fingerprint[:,:,:, i,:] = (self.lchannel_weights[None, None,:, None] * elemental_fingerprint)
        
        # now construct the two-species three-body terms, e.g X-CH, X-CN, while negating out the single-species term
        for i in range(self.element_combinations.shape[0]):
    
            elemental_fingerprint = torch.zeros(n_batch, max_natoms, self.lmax + 1, self.ngaussians, dtype=radial_basis.dtype, device=self.device)
            
            mbody = self.element_combinations[i]
            
            mask = (neighbor_numbers[..., None] == mbody).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)
            
            masked_radial = coeffs[..., None] * radial_basis
        
            test = angular_terms[... , None] * masked_radial[..., None,:]
            
            test = torch.sum(test, dim=2)
        
            orbitals = self.orbital_weights[None, None,:, None] * torch.pow(test, 2)
            
            single_species_id = self.element_to_id[mbody]
            
            elemental_fingerprint.index_add_(2, self.orbital_indexes, orbitals)
            
            fingerprint[:,:,:, self.nspecies + i,:] = (self.lchannel_weights[None, None,:, None] * elemental_fingerprint) - (fingerprint[:,:,:, single_species_id[0],:] + \
                                                                                fingerprint[:,:,:, single_species_id[1],:])
        # zero out any invalid atom_ids
        for i in range(n_batch):
            fingerprint[i, natom_counts[i]:,:,:,:] = 0.0
            
        return fingerprint.reshape(n_batch, max_natoms, self.fp_size)
    

class EGTOCuda_ver2(Representation):

    def __init__(self, species=np.array([1, 6, 7, 8]), low_cutoff=0.0, high_cutoff=6.0, ngaussians=24,
                 eta=torch.linspace(3.0, 1.0, 24, device=torch.device('cuda')), lmax=3,
                 lchannel_weights=1.0, inv_factors=torch.linspace(3.0, 1.0, 24, device=torch.device('cuda')),
                 rswitch=4.5, cutoff_function="cosine", distribution="gaussian"):
        
        super(EGTOCuda_ver2, self).__init__()
        
        self.species = torch.from_numpy(species).float().cuda()
        self.nspecies = len(species)
        
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.ngaussians = ngaussians
        self.eta = eta
        self.lmax = lmax
        
        if isinstance(lchannel_weights, list) or isinstance(lchannel_weights, np.ndarray):
            self.lchannel_weights = torch.Tensor(lchannel_weights).cuda()
        else:
            # assume scalar
            self.lchannel_weights = torch.zeros(lmax + 1).cuda()
            self.lchannel_weights[:] = lchannel_weights
        
        self.inv_factors = inv_factors
        
        self.generate_angular_numbers()
        
        # eta = (0.5 / ((1.0 - np.exp(-cutoff)) / K)) ** 2 
        
        if (distribution == "expexp"):
            self.offset = torch.linspace(np.exp(-0.4), np.exp(-self.high_cutoff), ngaussians).cuda()
        else:
            self.offset = torch.linspace(0.0, self.high_cutoff, ngaussians + 1)[1:].cuda()
        
        print (self.offset)
        mbody_list = torch.zeros(species.shape[0], species.shape[0], dtype=torch.int32)
        
        count = 0
        
        for i in range(species.shape[0]):
            mbody_list[i][i] = count
            count += 1
            
        for i in range(species.shape[0]):
            for j in range(i + 1, species.shape[0]):
                mbody_list[i][j] = count
                mbody_list[j][i] = count
                count += 1
            
        self.mbody_list = mbody_list.cuda()
        
        element_to_id = torch.zeros(max(self.species.int()) + 1, dtype=torch.long, device=torch.device('cuda'))
        
        for i, el in enumerate(self.species.int()):
            element_to_id[el] = i
            
        self.element_to_id = element_to_id
        
        element_combinations = []
        
        for i in range(self.nspecies):
            for j in range(i + 1, self.nspecies):
                element_combinations.append([int(self.species[i]), int(self.species[j])])
        
        self.element_combinations = torch.cuda.LongTensor(element_combinations)
        
        self.nmbody = self.element_combinations.shape[0] + self.nspecies
        
        self.fp_size = ngaussians * (lmax + 1) * self.nmbody
        
        self.pi = torch.acos(torch.zeros(1)).cuda() * 2
        
        self.device = torch.device('cuda')
        
        self.rswitch = rswitch
        
        if (cutoff_function == "cosine"):
            self.cut_func = 0
        elif (cutoff_function == "switch"):
            self.cut_func = 1
        else: self.cut_func = 0
            
        if (distribution == "gaussian"):
            self.dist_func = 0
        elif (distribution == "lognormal"):
            self.dist_func = 1
        elif(distribution == "expexp"):
            self.dist_func = 2
        else: self.dist_func = 0
        
    def generate_angular_numbers(self):
        angular_components = []
        angular_weights = []
        angular_indexes = []
        
        for i in range(self.lmax + 1):
            for k in range (i + 1):
                for m in range(i - k + 1):
                    n = i - k - m
                    angular_components.append([n, m, k])
                    angular_weights.append(np.math.factorial(i) / (np.math.factorial(n) * np.math.factorial(m) * np.math.factorial(k)))
                    angular_indexes.append(i)
                    
        angular_components = torch.FloatTensor(angular_components)
        angular_weights = torch.FloatTensor(angular_weights)
        angular_indexes = torch.IntTensor(angular_indexes)
        
        self.orbital_components = angular_components.cuda()
        self.orbital_weights = angular_weights.cuda()
        self.orbital_indexes = angular_indexes.cuda()
    
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

        output = egto_gpu.get_egto_ver2(X, Z, self.species, element_types, atomIDs, molIDs, neighbourlist, nneighbours, self.mbody_list,
                               self.orbital_components, self.orbital_weights, self.orbital_indexes, self.offset, self.lchannel_weights, self.inv_factors, self.eta, self.lmax, self.high_cutoff, self.rswitch,
                               cell, inv_cell, self.cut_func, self.dist_func, False)
        
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
        
        output = egto_gpu.get_egto_ver2(X, Z, self.species, element_types, atomIDs, molIDs, neighbourlist, nneighbours, self.mbody_list,
        self.orbital_components, self.orbital_weights, self.orbital_indexes, self.offset, self.lchannel_weights, self.inv_factors, self.eta, self.lmax, self.high_cutoff, self.rswitch,
        cell, inv_cell, self.cut_func, self.dist_func, True)
        
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

        n_batch, max_natoms, _ = coordinates.shape
        
        inv_cell = torch.empty(0, 3, 3, device=torch.device('cuda'))
            
        num_neighbours = pairlist_gpu.get_num_neighbours_gpu(coordinates, natom_counts, self.high_cutoff,
                                                          cell, inv_cell)
        
        max_neighbours = num_neighbours.max().item()
        
        neighbours = pairlist_gpu.get_neighbour_list_gpu(coordinates, natom_counts, max_neighbours, self.high_cutoff,
                                                            cell, inv_cell)

        pairlist_mask = (neighbours != -1)

        # hack to get rid of the -1's - picks a valid index for a given atom and fills -1 values with that. pairlist_mask stores
        # the "real" atom indexes
        pairlist_gpu.safe_fill_gpu(neighbours)
        
        idx_m = torch.arange(coordinates.shape[0], dtype=torch.long)[:, None, None]
        
        local_atoms = coordinates[idx_m, neighbours.long()]

        nbh_coords = (coordinates[:,:, None,:] - local_atoms)
        
        # TODO apply PBC here if cell is defined
        if (cell.shape[0] > 0):
            inv_cell = torch.inverse(cell)
            
        distances = torch.linalg.norm(nbh_coords, dim=3)
        
        # mask for the "dummy" atoms introduced when padding the neighbourlist to n_max_neighbours
        parlist_maskval = torch.ones_like(neighbours)
        pairlist_coeffs = parlist_maskval * pairlist_mask
     
        centered_distances = torch.pow(distances[..., None] - self.offset, 2)

        neighbor_numbers = nuclear_charges[idx_m, neighbours[:,:,:].long()]
  
        cutoffs = 0.5 * (torch.cos(distances * self.pi / self.high_cutoff) + 1.0)
       
        radial_basis = torch.sqrt(self.eta / self.pi) * torch.exp(-self.eta * centered_distances) * cutoffs[..., None] * pairlist_coeffs[..., None]
   
        inv_scaling = torch.pow(1.0 / distances[..., None], self.inv_factors + self.orbital_indexes) * pairlist_coeffs[..., None]
      
        angular_terms = inv_scaling * torch.pow(nbh_coords[..., None, 0] , self.orbital_components[:, 0 ]) * \
                torch.pow(nbh_coords[..., None, 1] , self.orbital_components[:, 1 ]) * \
                torch.pow(nbh_coords[..., None, 2] , self.orbital_components[:, 2 ])
        
        fingerprint = torch.zeros(n_batch, max_natoms, self.lmax + 1, self.nmbody, self.ngaussians, dtype=radial_basis.dtype, device=self.device)
        
        # first construct the single-species three-body terms, e.g X-HH, X-CC...
        for i in range(self.nspecies):

            elemental_fingerprint = torch.zeros(n_batch, max_natoms, self.lmax + 1, self.ngaussians, dtype=radial_basis.dtype, device=self.device)
    
            mask = (neighbor_numbers[..., None] == self.species[i]).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)

            filtered_radial = coeffs[..., None] * radial_basis 

            test = angular_terms[... , None] * filtered_radial[..., None,:] 
        
            test = torch.sum(test, dim=2)

            orbitals = self.orbital_weights[None, None,:, None] * torch.pow(test, 2)
          
            elemental_fingerprint.index_add_(2, self.orbital_indexes, orbitals)
            
            fingerprint[:,:,:, i,:] = (self.lchannel_weights[None, None,:, None] * elemental_fingerprint)
        
        # now construct the two-species three-body terms, e.g X-CH, X-CN, while negating out the single-species term
        for i in range(self.element_combinations.shape[0]):
    
            elemental_fingerprint = torch.zeros(n_batch, max_natoms, self.lmax + 1, self.ngaussians, dtype=radial_basis.dtype, device=self.device)
            
            mbody = self.element_combinations[i]
            
            mask = (neighbor_numbers[..., None] == mbody).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)
            
            masked_radial = coeffs[..., None] * radial_basis
        
            test = angular_terms[... , None] * masked_radial[..., None,:]
            
            test = torch.sum(test, dim=2)
        
            orbitals = self.orbital_weights[None, None,:, None] * torch.pow(test, 2)
            
            single_species_id = self.element_to_id[mbody]
            
            elemental_fingerprint.index_add_(2, self.orbital_indexes, orbitals)
            
            fingerprint[:,:,:, self.nspecies + i,:] = (self.lchannel_weights[None, None,:, None] * elemental_fingerprint) - (fingerprint[:,:,:, single_species_id[0],:] + \
                                                                                fingerprint[:,:,:, single_species_id[1],:])
        # zero out any invalid atom_ids
        for i in range(n_batch):
            fingerprint[i, natom_counts[i]:,:,:,:] = 0.0
            
        return fingerprint.reshape(n_batch, max_natoms, self.fp_size)
    

class EGTOCuda_ver3(Representation):

    def __init__(self, species=np.array([1, 6, 7, 8]), low_cutoff=0.0, high_cutoff=6.0, ngaussians=24,
                 eta=1.2, lmax=3, lchannel_weights=1.0, inv_factor=1.2,
                 rswitch=4.5, cutoff_function="cosine", distribution="gaussian"):
        
        super(EGTOCuda_ver3, self).__init__()
        
        self.species = torch.from_numpy(species).float().cuda()
        self.nspecies = len(species)
        
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.ngaussians = ngaussians
        self.eta = eta
        self.lmax = lmax
        
        if isinstance(lchannel_weights, list) or isinstance(lchannel_weights, np.ndarray):
            self.lchannel_weights = torch.Tensor(lchannel_weights).cuda()
        else:
            # assume scalar
            self.lchannel_weights = torch.zeros(lmax + 1).cuda()
            self.lchannel_weights[:] = lchannel_weights
        
        self.inv_factor = inv_factor
        
        self.generate_angular_numbers()
        
        # eta = (0.5 / ((1.0 - np.exp(-cutoff)) / K)) ** 2 
        
        if (distribution == "expexp"):
            self.offset = torch.linspace(np.exp(-0.4), np.exp(-self.high_cutoff), ngaussians).cuda()
        else:
            self.offset = torch.linspace(0.0, self.high_cutoff, ngaussians + 1)[1:].cuda()
        
        print (self.offset)
        mbody_list = torch.zeros(species.shape[0], species.shape[0], dtype=torch.int32)
        
        count = 0
        
        for i in range(species.shape[0]):
            mbody_list[i][i] = count
            count += 1
            
        for i in range(species.shape[0]):
            for j in range(i + 1, species.shape[0]):
                mbody_list[i][j] = count
                mbody_list[j][i] = count
                count += 1
            
        self.mbody_list = mbody_list.cuda()
        
        element_to_id = torch.zeros(max(self.species.int()) + 1, dtype=torch.long, device=torch.device('cuda'))
        
        for i, el in enumerate(self.species.int()):
            element_to_id[el] = i
            
        self.element_to_id = element_to_id
        
        element_combinations = []
        
        for i in range(self.nspecies):
            for j in range(i + 1, self.nspecies):
                element_combinations.append([int(self.species[i]), int(self.species[j])])
        
        self.element_combinations = torch.cuda.LongTensor(element_combinations)
        
        self.nmbody = self.element_combinations.shape[0] + self.nspecies
        
        self.fp_size = ngaussians * (lmax + 1) * self.nmbody
        
        self.pi = torch.acos(torch.zeros(1)).cuda() * 2
        
        self.device = torch.device('cuda')
        
        self.rswitch = rswitch
        
        if (cutoff_function == "cosine"):
            self.cut_func = 0
        elif (cutoff_function == "switch"):
            self.cut_func = 1
        else: self.cut_func = 0
            
        if (distribution == "gaussian"):
            self.dist_func = 0
        elif (distribution == "lognormal"):
            self.dist_func = 1
        elif(distribution == "expexp"):
            self.dist_func = 2
        else: self.dist_func = 0
        
    def generate_angular_numbers(self):
        angular_components = []
        angular_weights = []
        angular_indexes = []
        
        for i in range(self.lmax + 1):
            for k in range (i + 1):
                for m in range(i - k + 1):
                    n = i - k - m
                    angular_components.append([n, m, k])
                    angular_weights.append(np.math.factorial(i) / (np.math.factorial(n) * np.math.factorial(m) * np.math.factorial(k)))
                    angular_indexes.append(i)
                    
        angular_components = torch.FloatTensor(angular_components)
        angular_weights = torch.FloatTensor(angular_weights)
        angular_indexes = torch.IntTensor(angular_indexes)
        
        self.orbital_components = angular_components.cuda()
        self.orbital_weights = angular_weights.cuda()
        self.orbital_indexes = angular_indexes.cuda()
    
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

        output = egto_gpu.get_egto_ver3(X, Z, self.species, element_types, atomIDs, molIDs, neighbourlist, nneighbours, self.mbody_list,
                               self.orbital_components, self.orbital_weights, self.orbital_indexes, self.offset, self.lchannel_weights, self.inv_factor, self.eta, self.lmax, self.high_cutoff, self.rswitch,
                               cell, inv_cell, self.cut_func, self.dist_func, False)
        
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
        
        output = egto_gpu.get_egto_ver3(X, Z, self.species, element_types, atomIDs, molIDs, neighbourlist, nneighbours, self.mbody_list,
        self.orbital_components, self.orbital_weights, self.orbital_indexes, self.offset, self.lchannel_weights, self.inv_factor, self.eta, self.lmax, self.high_cutoff, self.rswitch,
        cell, inv_cell, self.cut_func, self.dist_func, True)
        
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

        n_batch, max_natoms, _ = coordinates.shape
        
        inv_cell = torch.empty(0, 3, 3, device=torch.device('cuda'))
            
        num_neighbours = pairlist_gpu.get_num_neighbours_gpu(coordinates, natom_counts, self.high_cutoff,
                                                          cell, inv_cell)
        
        max_neighbours = num_neighbours.max().item()
        
        neighbours = pairlist_gpu.get_neighbour_list_gpu(coordinates, natom_counts, max_neighbours, self.high_cutoff,
                                                            cell, inv_cell)

        pairlist_mask = (neighbours != -1)

        # hack to get rid of the -1's - picks a valid index for a given atom and fills -1 values with that. pairlist_mask stores
        # the "real" atom indexes
        pairlist_gpu.safe_fill_gpu(neighbours)
        
        idx_m = torch.arange(coordinates.shape[0], dtype=torch.long)[:, None, None]
        
        local_atoms = coordinates[idx_m, neighbours.long()]

        nbh_coords = (coordinates[:,:, None,:] - local_atoms)
        
        # TODO apply PBC here if cell is defined
        if (cell.shape[0] > 0):
            inv_cell = torch.inverse(cell)
            
        distances = torch.linalg.norm(nbh_coords, dim=3)
        
        # mask for the "dummy" atoms introduced when padding the neighbourlist to n_max_neighbours
        parlist_maskval = torch.ones_like(neighbours)
        pairlist_coeffs = parlist_maskval * pairlist_mask
     
        centered_distances = torch.pow(distances[..., None] - self.offset, 2)

        neighbor_numbers = nuclear_charges[idx_m, neighbours[:,:,:].long()]
  
        cutoffs = 0.5 * (torch.cos(distances * self.pi / self.high_cutoff) + 1.0)
       
        radial_basis = torch.sqrt(self.eta / self.pi) * torch.exp(-self.eta * centered_distances) * cutoffs[..., None] * pairlist_coeffs[..., None]
   
        inv_scaling = torch.pow(1.0 / distances[..., None], self.inv_factors + self.orbital_indexes) * pairlist_coeffs[..., None]
      
        angular_terms = inv_scaling * torch.pow(nbh_coords[..., None, 0] , self.orbital_components[:, 0 ]) * \
                torch.pow(nbh_coords[..., None, 1] , self.orbital_components[:, 1 ]) * \
                torch.pow(nbh_coords[..., None, 2] , self.orbital_components[:, 2 ])
        
        fingerprint = torch.zeros(n_batch, max_natoms, self.lmax + 1, self.nmbody, self.ngaussians, dtype=radial_basis.dtype, device=self.device)
        
        # first construct the single-species three-body terms, e.g X-HH, X-CC...
        for i in range(self.nspecies):

            elemental_fingerprint = torch.zeros(n_batch, max_natoms, self.lmax + 1, self.ngaussians, dtype=radial_basis.dtype, device=self.device)
    
            mask = (neighbor_numbers[..., None] == self.species[i]).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)

            filtered_radial = coeffs[..., None] * radial_basis 

            test = angular_terms[... , None] * filtered_radial[..., None,:] 
        
            test = torch.sum(test, dim=2)

            orbitals = self.orbital_weights[None, None,:, None] * torch.pow(test, 2)
          
            elemental_fingerprint.index_add_(2, self.orbital_indexes, orbitals)
            
            fingerprint[:,:,:, i,:] = (self.lchannel_weights[None, None,:, None] * elemental_fingerprint)
        
        # now construct the two-species three-body terms, e.g X-CH, X-CN, while negating out the single-species term
        for i in range(self.element_combinations.shape[0]):
    
            elemental_fingerprint = torch.zeros(n_batch, max_natoms, self.lmax + 1, self.ngaussians, dtype=radial_basis.dtype, device=self.device)
            
            mbody = self.element_combinations[i]
            
            mask = (neighbor_numbers[..., None] == mbody).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)
            
            masked_radial = coeffs[..., None] * radial_basis
        
            test = angular_terms[... , None] * masked_radial[..., None,:]
            
            test = torch.sum(test, dim=2)
        
            orbitals = self.orbital_weights[None, None,:, None] * torch.pow(test, 2)
            
            single_species_id = self.element_to_id[mbody]
            
            elemental_fingerprint.index_add_(2, self.orbital_indexes, orbitals)
            
            fingerprint[:,:,:, self.nspecies + i,:] = (self.lchannel_weights[None, None,:, None] * elemental_fingerprint) - (fingerprint[:,:,:, single_species_id[0],:] + \
                                                                                fingerprint[:,:,:, single_species_id[1],:])
        # zero out any invalid atom_ids
        for i in range(n_batch):
            fingerprint[i, natom_counts[i]:,:,:,:] = 0.0
            
        return fingerprint.reshape(n_batch, max_natoms, self.fp_size)
    

class EGTOCuda_ver4(Representation):

    def __init__(self, element_vectors, species=np.array([1, 6, 7, 8]), low_cutoff=0.0, high_cutoff=6.0, ngaussians=24,
                 eta=1.2, lmax=3, lchannel_weights=1.0, inv_factor=1.2,
                 rswitch=4.5, cutoff_function="cosine", distribution="gaussian"):
        
        super(EGTOCuda_ver4, self).__init__()
        
        self.species = torch.from_numpy(species).float().cuda()
        self.nspecies = len(species)
        
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.ngaussians = ngaussians
        self.eta = eta
        self.lmax = lmax
        
        self.element_vectors = element_vectors
        
        if isinstance(lchannel_weights, list) or isinstance(lchannel_weights, np.ndarray):
            self.lchannel_weights = torch.Tensor(lchannel_weights).cuda()
        else:
            # assume scalar
            self.lchannel_weights = torch.zeros(lmax + 1).cuda()
            self.lchannel_weights[:] = lchannel_weights
        
        self.inv_factor = inv_factor
        
        self.generate_angular_numbers()
        
        # eta = (0.5 / ((1.0 - np.exp(-cutoff)) / K)) ** 2 
        
        if (distribution == "expexp"):
            self.offset = torch.linspace(np.exp(-0.4), np.exp(-self.high_cutoff), ngaussians).cuda()
        else:
            self.offset = torch.linspace(0.0, self.high_cutoff, ngaussians + 1)[1:].cuda()
        
        print (self.offset)
        mbody_list = torch.zeros(species.shape[0], species.shape[0], dtype=torch.int32)
        
        count = 0
        
        for i in range(species.shape[0]):
            mbody_list[i][i] = count
            count += 1
            
        for i in range(species.shape[0]):
            for j in range(i + 1, species.shape[0]):
                mbody_list[i][j] = count
                mbody_list[j][i] = count
                count += 1
            
        self.mbody_list = mbody_list.cuda()
        
        element_to_id = torch.zeros(max(self.species.int()) + 1, dtype=torch.long, device=torch.device('cuda'))
        
        for i, el in enumerate(self.species.int()):
            element_to_id[el] = i
            
        self.element_to_id = element_to_id
        
        element_combinations = []
        
        for i in range(self.nspecies):
            for j in range(i + 1, self.nspecies):
                element_combinations.append([int(self.species[i]), int(self.species[j])])
        
        self.element_combinations = torch.cuda.LongTensor(element_combinations)
        
        self.nmbody = self.element_combinations.shape[0] + self.nspecies
        
        self.fp_size = ngaussians * (lmax + 1) * self.nmbody
        
        self.pi = torch.acos(torch.zeros(1)).cuda() * 2
        
        self.device = torch.device('cuda')
        
        self.rswitch = rswitch
        
        if (cutoff_function == "cosine"):
            self.cut_func = 0
        elif (cutoff_function == "switch"):
            self.cut_func = 1
        else: self.cut_func = 0
            
        if (distribution == "gaussian"):
            self.dist_func = 0
        elif (distribution == "lognormal"):
            self.dist_func = 1
        elif(distribution == "expexp"):
            self.dist_func = 2
        else: self.dist_func = 0
        
    def generate_angular_numbers(self):
        angular_components = []
        angular_weights = []
        angular_indexes = []
        
        for i in range(self.lmax + 1):
            for k in range (i + 1):
                for m in range(i - k + 1):
                    n = i - k - m
                    angular_components.append([n, m, k])
                    angular_weights.append(np.math.factorial(i) / (np.math.factorial(n) * np.math.factorial(m) * np.math.factorial(k)))
                    angular_indexes.append(i)
                    
        angular_components = torch.FloatTensor(angular_components)
        angular_weights = torch.FloatTensor(angular_weights)
        angular_indexes = torch.IntTensor(angular_indexes)
        
        self.orbital_components = angular_components.cuda()
        self.orbital_weights = angular_weights.cuda()
        self.orbital_indexes = angular_indexes.cuda()
    
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

        output = egto_gpu.get_egto_ver4(X, Z, self.species, element_types, self.element_vectors, atomIDs, molIDs, neighbourlist, nneighbours, self.mbody_list,
                               self.orbital_components, self.orbital_weights, self.orbital_indexes, self.offset, self.lchannel_weights, self.inv_factor, self.eta, self.lmax, self.high_cutoff, self.rswitch,
                               cell, inv_cell, self.cut_func, self.dist_func, False)
        
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
        
        output = egto_gpu.get_egto_ver4(X, Z, self.species, element_types, self.element_vectors, atomIDs, molIDs, neighbourlist, nneighbours, self.mbody_list,
        self.orbital_components, self.orbital_weights, self.orbital_indexes, self.offset, self.lchannel_weights, self.inv_factor, self.eta, self.lmax, self.high_cutoff, self.rswitch,
        cell, inv_cell, self.cut_func, self.dist_func, True)
        
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

        n_batch, max_natoms, _ = coordinates.shape
        
        inv_cell = torch.empty(0, 3, 3, device=torch.device('cuda'))
            
        num_neighbours = pairlist_gpu.get_num_neighbours_gpu(coordinates, natom_counts, self.high_cutoff,
                                                          cell, inv_cell)
        
        max_neighbours = num_neighbours.max().item()
        
        neighbours = pairlist_gpu.get_neighbour_list_gpu(coordinates, natom_counts, max_neighbours, self.high_cutoff,
                                                            cell, inv_cell)

        pairlist_mask = (neighbours != -1)

        # hack to get rid of the -1's - picks a valid index for a given atom and fills -1 values with that. pairlist_mask stores
        # the "real" atom indexes
        pairlist_gpu.safe_fill_gpu(neighbours)
        
        idx_m = torch.arange(coordinates.shape[0], dtype=torch.long)[:, None, None]
        
        local_atoms = coordinates[idx_m, neighbours.long()]

        nbh_coords = (coordinates[:,:, None,:] - local_atoms)
        
        # TODO apply PBC here if cell is defined
        if (cell.shape[0] > 0):
            inv_cell = torch.inverse(cell)
            
        distances = torch.linalg.norm(nbh_coords, dim=3)
        
        # mask for the "dummy" atoms introduced when padding the neighbourlist to n_max_neighbours
        parlist_maskval = torch.ones_like(neighbours)
        pairlist_coeffs = parlist_maskval * pairlist_mask
     
        centered_distances = torch.pow(distances[..., None] - self.offset, 2)

        neighbor_numbers = nuclear_charges[idx_m, neighbours[:,:,:].long()]
  
        cutoffs = 0.5 * (torch.cos(distances * self.pi / self.high_cutoff) + 1.0)
       
        radial_basis = torch.sqrt(self.eta / self.pi) * torch.exp(-self.eta * centered_distances) * cutoffs[..., None] * pairlist_coeffs[..., None]
   
        inv_scaling = torch.pow(1.0 / distances[..., None], self.inv_factors + self.orbital_indexes) * pairlist_coeffs[..., None]
      
        angular_terms = inv_scaling * torch.pow(nbh_coords[..., None, 0] , self.orbital_components[:, 0 ]) * \
                torch.pow(nbh_coords[..., None, 1] , self.orbital_components[:, 1 ]) * \
                torch.pow(nbh_coords[..., None, 2] , self.orbital_components[:, 2 ])
        
        fingerprint = torch.zeros(n_batch, max_natoms, self.lmax + 1, self.nmbody, self.ngaussians, dtype=radial_basis.dtype, device=self.device)
        
        # first construct the single-species three-body terms, e.g X-HH, X-CC...
        for i in range(self.nspecies):

            elemental_fingerprint = torch.zeros(n_batch, max_natoms, self.lmax + 1, self.ngaussians, dtype=radial_basis.dtype, device=self.device)
    
            mask = (neighbor_numbers[..., None] == self.species[i]).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)

            filtered_radial = coeffs[..., None] * radial_basis 

            test = angular_terms[... , None] * filtered_radial[..., None,:] 
        
            test = torch.sum(test, dim=2)

            orbitals = self.orbital_weights[None, None,:, None] * torch.pow(test, 2)
          
            elemental_fingerprint.index_add_(2, self.orbital_indexes, orbitals)
            
            fingerprint[:,:,:, i,:] = (self.lchannel_weights[None, None,:, None] * elemental_fingerprint)
        
        # now construct the two-species three-body terms, e.g X-CH, X-CN, while negating out the single-species term
        for i in range(self.element_combinations.shape[0]):
    
            elemental_fingerprint = torch.zeros(n_batch, max_natoms, self.lmax + 1, self.ngaussians, dtype=radial_basis.dtype, device=self.device)
            
            mbody = self.element_combinations[i]
            
            mask = (neighbor_numbers[..., None] == mbody).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)
            
            masked_radial = coeffs[..., None] * radial_basis
        
            test = angular_terms[... , None] * masked_radial[..., None,:]
            
            test = torch.sum(test, dim=2)
        
            orbitals = self.orbital_weights[None, None,:, None] * torch.pow(test, 2)
            
            single_species_id = self.element_to_id[mbody]
            
            elemental_fingerprint.index_add_(2, self.orbital_indexes, orbitals)
            
            fingerprint[:,:,:, self.nspecies + i,:] = (self.lchannel_weights[None, None,:, None] * elemental_fingerprint) - (fingerprint[:,:,:, single_species_id[0],:] + \
                                                                                fingerprint[:,:,:, single_species_id[1],:])
        # zero out any invalid atom_ids
        for i in range(n_batch):
            fingerprint[i, natom_counts[i]:,:,:,:] = 0.0
            
        return fingerprint.reshape(n_batch, max_natoms, self.fp_size)

