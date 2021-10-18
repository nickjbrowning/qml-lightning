'''
Created on 1 Apr 2021

@author: Nicholas J. Browning
@contact: nickjbrowning@gmail.com

@copyright: 

'''
import torch
from qml_lightning.cuda import pairlist_gpu
from qml_lightning.cuda import egto_gpu2
import numpy as np
import torch.nn as nn


def get_element_types(coordinates: torch.Tensor, charges: torch.Tensor, species: torch.Tensor):
    return egto_gpu2.get_element_types_gpu(coordinates, charges, species)


def generate_angular_numbers(lmax):
        angular_components = []
        angular_weights = []
        angular_indexes = []
        
        for i in range(lmax + 1):
            for k in range (i + 1):
                for m in range(i - k + 1):
                    n = i - k - m
                    angular_components.append([n, m, k])
                    angular_weights.append(np.math.factorial(i) / (np.math.factorial(n) * np.math.factorial(m) * np.math.factorial(k)))
                    angular_indexes.append(i)
                    
        angular_components = torch.FloatTensor(angular_components)
        angular_weights = torch.FloatTensor(angular_weights)
        angular_indexes = torch.IntTensor(angular_indexes)
        
        return angular_components, angular_weights, angular_indexes


def get_egto(X: torch.Tensor, Z: torch.Tensor, atomIDs: torch.Tensor, molIDs: torch.Tensor, atom_counts: torch.Tensor,
             species=torch.Tensor([1, 6, 7, 8, 9]).float().cuda(), ngaussians=20, eta=2.3, lmax=2, rcut=6.0, gradients=False):
    
    orbital_components, orbital_weights, orbital_indexes = generate_angular_numbers(lmax)
    orbital_components = orbital_components.cuda()
    orbital_weights = orbital_weights.cuda()
    orbital_indexes = orbital_indexes.cuda()
    
    offset = torch.linspace(0.0, rcut, ngaussians + 1)[1:].cuda()
    
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
            
    mbody_list = mbody_list.cuda()
    
    nneighbours = pairlist_gpu.get_num_neighbours_gpu2(X, atom_counts, rcut)
        
    max_neighbours = nneighbours.max().item()
 
    neighbourlist = pairlist_gpu.get_neighbour_list_gpu2(X, atom_counts, max_neighbours, rcut)
    
    print (neighbourlist)
    
    element_types = egto_gpu2.get_element_types_gpu2(X, Z, atom_counts, species) 
    
    print (element_types)
    
    output = egto_gpu2.get_egto(X, Z, species, element_types,
    atomIDs, molIDs, neighbourlist, nneighbours, mbody_list,
    orbital_components, orbital_weights, orbital_indexes, offset, eta, lmax, rcut,
    gradients)
    
    if (gradients):
        return output[0], output[1] 
        
    else:
        return output[0]


class Representation():
    
    def __init__(self):
        pass
    
    def get_representation(self):
        raise NotImplementedError("Abstract method only.")
    
    def get_representation_and_derivative(self):
        raise NotImplementedError("Abstract method only.")


class EGTOCuda(Representation):

    def __init__(self, species=np.array([1, 6, 7, 8]), low_cutoff=0.0, high_cutoff=6.0, ngaussians=20, eta=2.5, lmax=2, lchannel_weights=[1.0, 1.0, 1.0], inv_factors=[2.0, 2.0, 2.0]):
        
        super(EGTOCuda, self).__init__()
        
        self.species = torch.from_numpy(species).float().cuda()
        
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.ngaussians = ngaussians
        self.eta = eta
        self.lmax = lmax
        self.lchannel_weights = torch.Tensor(lchannel_weights).cuda()
        self.inv_factors = torch.Tensor(inv_factors).cuda()
        
        self.generate_angular_numbers()
        
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
         
        element_types = egto_gpu2.get_element_types_gpu2(X, Z, atom_counts, self.species) 

        output = egto_gpu2.get_egto(X, Z, self.species, element_types, atomIDs, molIDs, neighbourlist, nneighbours, self.mbody_list,
        self.orbital_components, self.orbital_weights, self.orbital_indexes, self.offset, self.lchannel_weights, self.inv_factors, self.eta, self.lmax, self.high_cutoff,
        cell, inv_cell, False)
        
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
        
        element_types = egto_gpu2.get_element_types_gpu2(X, Z, atom_counts, self.species) 
        
        output = egto_gpu2.get_egto(X, Z, self.species, element_types, atomIDs, molIDs, neighbourlist, nneighbours, self.mbody_list,
        self.orbital_components, self.orbital_weights, self.orbital_indexes, self.offset, self.lchannel_weights, self.inv_factors, self.eta, self.lmax, self.high_cutoff,
        cell, inv_cell, True)
        
        return output[0], output[1]
    
    def rep_deriv_fd(self, X, Z, repsize, atomIDs, molIDs, natom_counts, dx=0.005):
    
        rep_derivative_fd = torch.zeros(X.shape[0], X.shape[1], X.shape[1], 3, repsize, dtype=torch.float64, device=X.device)
        
        for i in range(X.shape[1]):
        
            for x in range (3):
                
                X_copy = X.clone()
                
                X_copy[:, i, x] += dx
                
                gto_plus = self.get_representation(X_copy, Z, atomIDs, molIDs, natom_counts)
      
                X_copy[:, i, x] -= 2 * dx
                
                gto_minus = self.get_representation(X_copy, Z, atomIDs, molIDs, natom_counts)
                
                rep_derivative_fd[:,:, i, x,:] = (gto_plus - gto_minus) / (2 * dx)
                
        return rep_derivative_fd
    
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
    
    
class ElementalGTO(nn.Module):
    
    ''' implementation of Elemental GTO for pytorch autograd support. At the moment only the pairlist is implemented directly as a cuda kernel,
        but might move more of this to cuda when I have time.
    '''

    def __init__(self, species=[1, 6, 7, 8], low_cutoff=0.0, high_cutoff=6.0, n_gaussians=20, eta=2.0, lmax=2,
                 lweights=[1.0, 1.0, 1.0], inv_factors=[2.0, 2.0, 2.0], device=torch.device('cuda')):
        
        super(ElementalGTO, self).__init__()
        self.device = device
        
        self.high_cutoff = high_cutoff
        self.n_gaussians = n_gaussians
        self.n_species = len(species)
        
        self.species = species
        
        self.Lmax = lmax
        self.lweights = torch.Tensor(lweights).cuda()
        
        offset = torch.linspace(low_cutoff, high_cutoff, n_gaussians + 1, device=device)[1:]

        # inv_scaling = torch.full([n_gaussians], 2.0, device=device)
        
        element_to_id = torch.zeros(max(species) + 1, dtype=torch.long, device=device)
        
        for i, el in enumerate(species):
            element_to_id[el] = i
            
        self.element_to_id = element_to_id

        self.generate_angular_numbers()
        
        element_combinations = []
        
        for i in range(self.n_species):
            for j in range(i + 1, self.n_species):
                element_combinations.append([int(self.species[i]), int(self.species[j])])
        
        self.element_combinations = torch.cuda.LongTensor(element_combinations)
        
        self.n_mbody = self.element_combinations.shape[0] + self.n_species
        
        self.fp_size = n_gaussians * (lmax + 1) * self.n_mbody
        
        self.register_buffer("fingerprint_size", torch.FloatTensor(self.fp_size))
        
        self.init_inv_factors(inv_factors)
                    
        self.offsets = offset
        self.width = eta
            
        self.pi = torch.acos(torch.zeros(1)).cuda() * 2
    
    def init_inv_factors(self, factors):
        
        inv_factors = []
        for i in range(self.Lmax + 1):
            for k in range (i + 1):
                for m in range(i - k + 1):
                    inv_factors.append(factors[i])
                    
        self.inv_factors = torch.Tensor(inv_factors).cuda()
        
    def generate_angular_numbers(self):
        angular_components = []
        angular_weights = []
        angular_indexes = []
        
        for i in range(self.Lmax + 1):
            for k in range (i + 1):
                for m in range(i - k + 1):
                    n = i - k - m
                    angular_components.append([n, m, k])
                    angular_weights.append(np.math.factorial(i) / (np.math.factorial(n) * np.math.factorial(m) * np.math.factorial(k)))
                    angular_indexes.append(i)
                    
        self.angular_components = torch.cuda.FloatTensor(angular_components)
        self.angular_weights = torch.cuda.FloatTensor(angular_weights)
        self.angular_indexes = torch.cuda.LongTensor(angular_indexes)

    def __len__(self):
        return self.fp_size
    
    def get_representation(self, X:torch.Tensor, Z: torch.Tensor, atomIDs: torch.Tensor, molIDs: torch.Tensor, atom_counts: torch.Tensor):
        return self.forward(X, Z, atom_counts)
    
    def forward(self, coordinates, nuclear_charges, natom_counts):

        n_batch, max_natoms, _ = coordinates.shape
        
        num_neighbours = pairlist_gpu.get_num_neighbours_gpu(coordinates, natom_counts, self.high_cutoff)
        
        max_neighbours = num_neighbours.max().item()
     
        neighbours = pairlist_gpu.get_neighbour_list_gpu(coordinates, natom_counts, max_neighbours, self.high_cutoff)

        pairlist_mask = (neighbours != -1)

        # get rid of the -1's - picks a valid index for a given atom and fills -1 values with that. pairlist_mask stores
        # the "real" atom indexes
        
        pairlist_gpu.safe_fill_gpu(neighbours)
        
        idx_m = torch.arange(coordinates.shape[0], dtype=torch.long)[:, None, None]
        
        local_atoms = coordinates[idx_m, neighbours.long()]

        nbh_coords = (coordinates[:,:, None,:] - local_atoms)

        distances = torch.linalg.norm(nbh_coords, dim=3)
        
        # mask for the "dummy" atoms introduced when padding the neighbourlist to n_max_neighbours
        parlist_maskval = torch.ones_like(neighbours)
        pairlist_coeffs = parlist_maskval * pairlist_mask
     
        centered_distances = torch.pow(distances[..., None] - self.offsets, 2)

        neighbor_numbers = nuclear_charges[idx_m, neighbours[:,:,:].long()]
  
        cutoffs = 0.5 * (torch.cos(distances * self.pi / self.high_cutoff) + 1.0)
       
        radial_basis = torch.sqrt(self.width / self.pi) * torch.exp(-self.width * centered_distances) * cutoffs[..., None] * pairlist_coeffs[..., None]
   
        inv_scaling = torch.pow(1.0 / distances[..., None], self.inv_factors + self.angular_indexes) * pairlist_coeffs[..., None]
      
        angular_terms = inv_scaling * torch.pow(nbh_coords[..., None, 0] , self.angular_components[:, 0 ]) * \
                torch.pow(nbh_coords[..., None, 1] , self.angular_components[:, 1 ]) * \
                torch.pow(nbh_coords[..., None, 2] , self.angular_components[:, 2 ])
        
        fingerprint = torch.zeros(n_batch, max_natoms, self.Lmax + 1, self.n_mbody, self.n_gaussians, dtype=radial_basis.dtype, device=self.device)
        
        # first construct the single-species three-body terms, e.g X-HH, X-CC...
        for i in range(self.n_species):

            elemental_fingerprint = torch.zeros(n_batch, max_natoms, self.Lmax + 1, self.n_gaussians, dtype=radial_basis.dtype, device=self.device)
    
            mask = (neighbor_numbers[..., None] == self.species[i]).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)

            filtered_radial = coeffs[..., None] * radial_basis 

            test = angular_terms[... , None] * filtered_radial[..., None,:] 
        
            test = torch.sum(test, dim=2)

            orbitals = self.angular_weights[None, None,:, None] * torch.pow(test, 2)
          
            elemental_fingerprint.index_add_(2, self.angular_indexes, orbitals)
            
            fingerprint[:,:,:, i,:] = (self.lweights[None, None,:, None] * elemental_fingerprint)
        
        # now construct the two-species three-body terms, e.g X-CH, X-CN, while negating out the single-species term
        for i in range(self.element_combinations.shape[0]):
    
            elemental_fingerprint = torch.zeros(n_batch, max_natoms, self.Lmax + 1, self.n_gaussians, dtype=radial_basis.dtype, device=self.device)
            
            mbody = self.element_combinations[i]
            
            mask = (neighbor_numbers[..., None] == mbody).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)
            
            masked_radial = coeffs[..., None] * radial_basis
        
            test = angular_terms[... , None] * masked_radial[..., None,:]
            
            test = torch.sum(test, dim=2)
        
            orbitals = self.angular_weights[None, None,:, None] * torch.pow(test, 2)
            
            single_species_id = self.element_to_id[mbody]
            
            elemental_fingerprint.index_add_(2, self.angular_indexes, orbitals)
            
            fingerprint[:,:,:, self.n_species + i,:] = (self.lweights[None, None,:, None] * elemental_fingerprint) - (fingerprint[:,:,:, single_species_id[0],:] + \
                                                                                fingerprint[:,:,:, single_species_id[1],:])
        # zero out any invalid atom_ids
        for i in range(n_batch):
            fingerprint[i, natom_counts[i]:,:,:,:] = 0.0
            
        return fingerprint.reshape(n_batch, max_natoms, self.fp_size)

    
class ElementalGTOLogNormal(nn.Module):
    
    ''' implementation of Elemental GTO for pytorch autograd support. At the moment only the pairlist is implemented directly as a cuda kernel,
        but might move more of this to cuda when I have time.
    '''

    def __init__(self, species=[1, 6, 7, 8], low_cutoff=0.0, high_cutoff=6.0, n_gaussians=20, rswitch=1.0, w=2.0, lmax=2,
                lweights=[1.0, 1.0, 1.0], inv_factors=[2.0, 2.0, 2.0], device=torch.device('cuda'), trainable_basis=False):
        
        super(ElementalGTOLogNormal, self).__init__()
        self.device = device
        
        self.high_cutoff = high_cutoff
        self.n_gaussians = n_gaussians
        self.n_species = len(species)
        
        self.species = species
        
        self.Lmax = lmax
        self.lweights = torch.Tensor(lweights).cuda()
        
        offset = torch.linspace(low_cutoff, high_cutoff, n_gaussians + 1, device=device)[1:]
        widths = torch.full([n_gaussians], w, device=device)
        # inv_scaling = torch.full([n_gaussians], 2.0, device=device)
        
        element_to_id = torch.zeros(max(species) + 1, dtype=torch.long, device=device)
        
        for i, el in enumerate(species):
            element_to_id[el] = i
            
        self.element_to_id = element_to_id

        self.generate_angular_numbers()
        
        self.rswitch = rswitch
        
        element_combinations = []
        
        for i in range(self.n_species):
            for j in range(i + 1, self.n_species):
                element_combinations.append([int(self.species[i]), int(self.species[j])])
        
        self.element_combinations = torch.cuda.LongTensor(element_combinations)
        
        self.n_mbody = self.element_combinations.shape[0] + self.n_species
        
        self.fp_size = n_gaussians * (lmax + 1) * self.n_mbody
        
        self.register_buffer("fingerprint_size", torch.FloatTensor(self.fp_size))
                    
        self.init_inv_factors(inv_factors)
    
        self.offsets = offset
        self.width = w
            
        self.pi = torch.acos(torch.zeros(1)).cuda() * 2
        
        self.sqrtpi = torch.sqrt(self.pi)
        
    def init_inv_factors(self, factors):
        
        inv_factors = []
        for i in range(self.Lmax + 1):
            for k in range (i + 1):
                for m in range(i - k + 1):
                    inv_factors.append(factors[i])
                    
        self.inv_factors = torch.Tensor(inv_factors).cuda()
        
    def generate_angular_numbers(self):
        angular_components = []
        angular_weights = []
        angular_indexes = []
        
        for i in range(self.Lmax + 1):
            for k in range (i + 1):
                for m in range(i - k + 1):
                    n = i - k - m
                    angular_components.append([n, m, k])
                    angular_weights.append(np.math.factorial(i) / (np.math.factorial(n) * np.math.factorial(m) * np.math.factorial(k)))
                    angular_indexes.append(i)
                    
        self.angular_components = torch.cuda.FloatTensor(angular_components)
        self.angular_weights = torch.cuda.FloatTensor(angular_weights)
        self.angular_indexes = torch.cuda.LongTensor(angular_indexes)

    def __len__(self):
        return self.fp_size
    
    def get_representation(self, X:torch.Tensor, Z: torch.Tensor, atomIDs: torch.Tensor, molIDs: torch.Tensor, atom_counts: torch.Tensor):
        return self.forward(X, Z, atom_counts)
    
    def forward(self, coordinates, nuclear_charges, natom_counts):

        n_batch, max_natoms, _ = coordinates.shape
        
        num_neighbours = pairlist_gpu.get_num_neighbours_gpu(coordinates, natom_counts, self.high_cutoff)
        
        max_neighbours = num_neighbours.max().item()
     
        neighbours = pairlist_gpu.get_neighbour_list_gpu(coordinates, natom_counts, max_neighbours, self.high_cutoff)

        pairlist_mask = (neighbours != -1)

        # get rid of the -1's - picks a valid index for a given atom and fills -1 values with that. pairlist_mask stores
        # the "real" atom indexes
        
        pairlist_gpu.safe_fill_gpu(neighbours)
        
        idx_m = torch.arange(coordinates.shape[0], dtype=torch.long)[:, None, None]
        
        local_atoms = coordinates[idx_m, neighbours.long()]

        nbh_coords = (coordinates[:,:, None,:] - local_atoms)

        distances = torch.linalg.norm(nbh_coords, dim=3)
        
        # mask for the "dummy" atoms introduced when padding the neighbourlist to n_max_neighbours
        parlist_maskval = torch.ones_like(neighbours)
        pairlist_coeffs = parlist_maskval * pairlist_mask
  
        neighbor_numbers = nuclear_charges[idx_m, neighbours[:,:,:].long()]
  
        cutoffs = 0.5 * (torch.cos(distances * self.pi / self.high_cutoff) + 1.0)
        
        hyperws = self.width
        
        distances2 = torch.pow(distances, 2)
        
        sigma2 = torch.log(1.0 + (hyperws / distances2))
        
        mu = torch.log(distances / torch.sqrt(1.0 + (hyperws / distances2)))
        
        centered_distances = torch.pow(torch.log(self.offsets) - mu[..., None], 2)
        
        radial_basis = (1.0 / (self.offsets * torch.sqrt(sigma2[..., None]) * self.sqrtpi)) \
        * torch.exp(-centered_distances / (2.0 * sigma2[..., None])) * cutoffs[..., None] * pairlist_coeffs[..., None]

        inv_scaling = torch.pow(1.0 / distances[..., None], self.inv_factors + self.angular_indexes) * pairlist_coeffs[..., None]

        angular_terms = inv_scaling * torch.pow(nbh_coords[..., None, 0] , self.angular_components[:, 0 ]) * \
                torch.pow(nbh_coords[..., None, 1] , self.angular_components[:, 1 ]) * \
                torch.pow(nbh_coords[..., None, 2] , self.angular_components[:, 2 ])

        fingerprint = torch.zeros(n_batch, max_natoms, self.Lmax + 1, self.n_mbody, self.n_gaussians, dtype=radial_basis.dtype, device=self.device)
        
        # first construct the single-species three-body terms, e.g X-HH, X-CC...
        for i in range(self.n_species):

            elemental_fingerprint = torch.zeros(n_batch, max_natoms, self.Lmax + 1, self.n_gaussians, dtype=radial_basis.dtype, device=self.device)
    
            mask = (neighbor_numbers[..., None] == self.species[i]).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)

            filtered_radial = coeffs[..., None] * radial_basis 

            test = angular_terms[... , None] * filtered_radial[..., None,:] 
        
            test = torch.sum(test, dim=2)

            orbitals = self.angular_weights[None, None,:, None] * torch.pow(test, 2)
          
            elemental_fingerprint.index_add_(2, self.angular_indexes, orbitals)
            
            fingerprint[:,:,:, i,:] = self.lweights[None, None,:, None] * elemental_fingerprint
        
        # now construct the two-species three-body terms, e.g X-CH, X-CN, while negating out the single-species term
        for i in range(self.element_combinations.shape[0]):
    
            elemental_fingerprint = torch.zeros(n_batch, max_natoms, self.Lmax + 1, self.n_gaussians, dtype=radial_basis.dtype, device=self.device)
            
            mbody = self.element_combinations[i]
            
            mask = (neighbor_numbers[..., None] == mbody).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)
            
            masked_radial = coeffs[..., None] * radial_basis
        
            test = angular_terms[... , None] * masked_radial[..., None,:]
            
            test = torch.sum(test, dim=2)
        
            orbitals = self.angular_weights[None, None,:, None] * torch.pow(test, 2)
            
            single_species_id = self.element_to_id[mbody]
            
            elemental_fingerprint.index_add_(2, self.angular_indexes, orbitals)
            
            fingerprint[:,:,:, self.n_species + i,:] = (self.lweights[None, None,:, None] * elemental_fingerprint) - (fingerprint[:,:,:, single_species_id[0],:] + \
                                                                                fingerprint[:,:,:, single_species_id[1],:])
        # zero out any invalid atom_ids
        for i in range(n_batch):
            fingerprint[i, natom_counts[i]:,:,:,:] = 0.0
            
        return fingerprint.reshape(n_batch, max_natoms, self.fp_size)

    
class ElementalGTOLogNormalSkinCutoff(nn.Module):
    
    ''' implementation of Elemental GTO for pytorch autograd support. At the moment only the pairlist is implemented directly as a cuda kernel,
        but might move more of this to cuda when I have time.
    '''

    def __init__(self, species=[1, 6, 7, 8], low_cutoff=0.0, high_cutoff=6.0, n_gaussians=20, rswitch=1.0, w=2.0, lmax=2,
                  lweights=[1.0, 1.0, 1.0], inv_factors=[2.0, 2.0, 2.0], device=torch.device('cuda')):
        
        super(ElementalGTOLogNormalSkinCutoff, self).__init__()
        self.device = device
        
        self.high_cutoff = high_cutoff
        self.n_gaussians = n_gaussians
        self.n_species = len(species)
        
        self.species = species
        
        self.Lmax = lmax
        self.lweights = torch.Tensor(lweights).cuda()
        
        offset = torch.linspace(low_cutoff, high_cutoff, n_gaussians + 1, device=device)[1:]
   
        # inv_scaling = torch.full([n_gaussians], 2.0, device=device)
        
        element_to_id = torch.zeros(max(species) + 1, dtype=torch.long, device=device)
        
        for i, el in enumerate(species):
            element_to_id[el] = i
            
        self.element_to_id = element_to_id

        self.generate_angular_numbers()
        
        self.rswitch = rswitch
        
        element_combinations = []
        
        for i in range(self.n_species):
            for j in range(i + 1, self.n_species):
                element_combinations.append([int(self.species[i]), int(self.species[j])])
        
        self.element_combinations = torch.cuda.LongTensor(element_combinations)
        
        self.n_mbody = self.element_combinations.shape[0] + self.n_species
        
        self.fp_size = n_gaussians * (lmax + 1) * self.n_mbody
        
        self.register_buffer("fingerprint_size", torch.FloatTensor(self.fp_size))
        
        self.init_inv_factors(inv_factors)
        
        for i in range(self.Lmax + 1):
            for k in range (i + 1):
                for m in range(i - k + 1):
                    self.inv_factors.append(inv_factors[i])
        
        self.offsets = offset
        self.width = w
            
        self.pi = torch.acos(torch.zeros(1)).cuda() * 2
        
        self.sqrtpi = torch.sqrt(self.pi)
    
    def init_inv_factors(self, factors):
        
        inv_factors = []
        for i in range(self.Lmax + 1):
            for k in range (i + 1):
                for m in range(i - k + 1):
                    inv_factors.append(factors[i])
                    
        self.inv_factors = torch.Tensor(inv_factors).cuda()
                    
    def generate_angular_numbers(self):
        angular_components = []
        angular_weights = []
        angular_indexes = []
        
        for i in range(self.Lmax + 1):
            for k in range (i + 1):
                for m in range(i - k + 1):
                    n = i - k - m
                    angular_components.append([n, m, k])
                    angular_weights.append(np.math.factorial(i) / (np.math.factorial(n) * np.math.factorial(m) * np.math.factorial(k)))
                    angular_indexes.append(i)
                    
        self.angular_components = torch.cuda.FloatTensor(angular_components)
        self.angular_weights = torch.cuda.FloatTensor(angular_weights)
        self.angular_indexes = torch.cuda.LongTensor(angular_indexes)

    def __len__(self):
        return self.fp_size
    
    def get_representation(self, X:torch.Tensor, Z: torch.Tensor, atomIDs: torch.Tensor, molIDs: torch.Tensor, atom_counts: torch.Tensor):
        return self.forward(X, Z, atom_counts)
    
    def forward(self, coordinates, nuclear_charges, natom_counts):

        n_batch, max_natoms, _ = coordinates.shape
        
        num_neighbours = pairlist_gpu.get_num_neighbours_gpu(coordinates, natom_counts, self.high_cutoff)
        
        max_neighbours = num_neighbours.max().item()
     
        neighbours = pairlist_gpu.get_neighbour_list_gpu(coordinates, natom_counts, max_neighbours, self.high_cutoff)

        pairlist_mask = (neighbours != -1)

        # get rid of the -1's - picks a valid index for a given atom and fills -1 values with that. pairlist_mask stores
        # the "real" atom indexes
        
        pairlist_gpu.safe_fill_gpu(neighbours)
        
        idx_m = torch.arange(coordinates.shape[0], dtype=torch.long)[:, None, None]
        
        local_atoms = coordinates[idx_m, neighbours.long()]

        nbh_coords = (coordinates[:,:, None,:] - local_atoms)

        distances = torch.linalg.norm(nbh_coords, dim=3)
        
        # mask for the "dummy" atoms introduced when padding the neighbourlist to n_max_neighbours
        parlist_maskval = torch.ones_like(neighbours)
        pairlist_coeffs = parlist_maskval * pairlist_mask

        neighbor_numbers = nuclear_charges[idx_m, neighbours[:,:,:].long()]

        dswitch = (distances - self.rswitch) / (self.high_cutoff - self.rswitch)
        
        cutoffs = 1.0 - 6 * torch.pow(dswitch, 5) + 15.0 * torch.pow(dswitch, 4) - 10 * torch.pow(dswitch, 3)

        hyperws = self.width
        
        distances2 = torch.pow(distances, 2)
        
        sigma2 = torch.log(1.0 + (hyperws / distances2))
        
        mu = torch.log(distances / torch.sqrt(1.0 + (hyperws / distances2)))
        
        centered_distances = torch.pow(torch.log(self.offsets) - mu[..., None], 2)
        
        radial_basis = (1.0 / (self.offsets * torch.sqrt(sigma2[..., None]) * self.sqrtpi)) \
        * torch.exp(-centered_distances / (2.0 * sigma2[..., None])) * cutoffs[..., None] * pairlist_coeffs[..., None]

        inv_scaling = torch.pow(1.0 / distances[..., None], self.inv_factors + self.angular_indexes) * pairlist_coeffs[..., None]

        angular_terms = inv_scaling * torch.pow(nbh_coords[..., None, 0] , self.angular_components[:, 0 ]) * \
                torch.pow(nbh_coords[..., None, 1] , self.angular_components[:, 1 ]) * \
                torch.pow(nbh_coords[..., None, 2] , self.angular_components[:, 2 ])

        fingerprint = torch.zeros(n_batch, max_natoms, self.Lmax + 1, self.n_mbody, self.n_gaussians, dtype=radial_basis.dtype, device=self.device)
        
        # first construct the single-species three-body terms, e.g X-HH, X-CC...
        for i in range(self.n_species):

            elemental_fingerprint = torch.zeros(n_batch, max_natoms, self.Lmax + 1, self.n_gaussians, dtype=radial_basis.dtype, device=self.device)
    
            mask = (neighbor_numbers[..., None] == self.species[i]).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)

            filtered_radial = coeffs[..., None] * radial_basis 

            test = angular_terms[... , None] * filtered_radial[..., None,:] 
        
            test = torch.sum(test, dim=2)

            orbitals = self.angular_weights[None, None,:, None] * torch.pow(test, 2)
          
            elemental_fingerprint.index_add_(2, self.angular_indexes, orbitals)

            fingerprint[:,:,:, i,:] = self.lweights[None, None,:, None] * elemental_fingerprint
        
        # now construct the two-species three-body terms, e.g X-CH, X-CN, while negating out the single-species term
        for i in range(self.element_combinations.shape[0]):
    
            elemental_fingerprint = torch.zeros(n_batch, max_natoms, self.Lmax + 1, self.n_gaussians, dtype=radial_basis.dtype, device=self.device)
            
            mbody = self.element_combinations[i]
            
            mask = (neighbor_numbers[..., None] == mbody).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)
            
            masked_radial = coeffs[..., None] * radial_basis
        
            test = angular_terms[... , None] * masked_radial[..., None,:]
            
            test = torch.sum(test, dim=2)
        
            orbitals = self.angular_weights[None, None,:, None] * torch.pow(test, 2)
            
            single_species_id = self.element_to_id[mbody]
            
            elemental_fingerprint.index_add_(2, self.angular_indexes, orbitals)
            
            fingerprint[:,:,:, self.n_species + i,:] = (self.lweights[None, None,:, None] * elemental_fingerprint) - (fingerprint[:,:,:, single_species_id[0],:] + \
                                                                                fingerprint[:,:,:, single_species_id[1],:])
        # zero out any invalid atom_ids
        for i in range(n_batch):
            fingerprint[i, natom_counts[i]:,:,:,:] = 0.0
            
        return fingerprint.reshape(n_batch, max_natoms, self.fp_size)
