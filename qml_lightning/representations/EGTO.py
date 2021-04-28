'''
Created on 1 Apr 2021

@author: Nicholas J. Browning
@contact: nickjbrowning@gmail.com

@copyright: 

'''

from qml_lightning.cuda import egto_gpu
from qml_lightning.cuda import pairlist_gpu

import torch
import numpy as np
import torch.nn as nn


def get_element_types(coordinates: torch.Tensor, charges: torch.Tensor, species: torch.Tensor):
    return egto_gpu.get_element_types_gpu(coordinates, charges, species)


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


def get_elemental_gto(coordinates: torch.Tensor, charges: torch.Tensor, species: torch.Tensor, ngaussians: int, eta: float, lmax: int, rcut: float, gradients=False,
                                 print_timings=False):
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    offset = torch.linspace(0.0, rcut, ngaussians + 1)[1:]

    orbital_components, orbital_weights, orbital_indexes = generate_angular_numbers(lmax)
    
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
    
    offset = offset.cuda()
    mbody_list = mbody_list.cuda()
    orbital_components = orbital_components.cuda()
    orbital_weights = orbital_weights.cuda()
    orbital_indexes = orbital_indexes.cuda()
    
    coordinates = coordinates.cuda()
    charges = charges.cuda()
    species = species.cuda()
    
    start.record()
    element_types = get_element_types(coordinates, charges, species)
    end.record()
    torch.cuda.synchronize()

    if (gradients):
        start.record()
        output = egto_gpu.elemental_gto_gpu_shared(coordinates, charges, species, element_types, mbody_list,
                                        orbital_components, orbital_weights, orbital_indexes, offset, eta, lmax, rcut, gradients)
        rep, grad = output[0], output[1]
        end.record()
        torch.cuda.synchronize()
        
        if (print_timings):
            print ("representation + grad time: ", start.elapsed_time(end))
        return rep, grad
    else: 
        start.record()
        output = egto_gpu.elemental_gto_gpu_shared(coordinates, charges, species, element_types, mbody_list,
                                        orbital_components, orbital_weights, orbital_indexes, offset, eta, lmax, rcut, gradients)
        end.record()
        torch.cuda.synchronize()
        
        if (print_timings):
            print ("representation time: ", start.elapsed_time(end))
            
        return output[0]


def get_nbh_coords(coords, cutoff):
    n_frames, n_atoms, _ = coords.shape
    nbh_coords = coords[..., None,:] - coords[:, None, ...]
    
    nbh_coords = nbh_coords.type(torch.cuda.FloatTensor)
    
    mask = torch.eye(n_atoms, dtype=torch.bool, device=coords.device)

    distances = nbh_coords.norm(dim=3).float()
    
    nbh_coords = nbh_coords[:, ~mask].reshape(n_frames, n_atoms, n_atoms - 1, 3)

    distances = distances[:, ~mask].reshape(n_frames, n_atoms, n_atoms - 1)

    neighbors, neighbor_mask = get_neighbours(distances, cutoff)
    
    return nbh_coords, distances, neighbors, neighbor_mask


def get_neighbours(distances, cutoff):
    n_frames, n_atoms, n_neighbors = distances.shape

    # Create a simple neighbor list of shape [n_frames, n_atoms, n_neighbors]
    # in which every bead sees each other but themselves.
    # First, create a matrix that contains all indices.
    neighbors = torch.arange(n_atoms).repeat(*(n_frames, n_atoms, 1))

    # To remove the self interaction of n_atoms, an inverted identity matrix
    # is used to exclude the respective indices in the neighbor list.
    neighbors = neighbors[:, ~torch.eye(n_atoms, dtype=torch.bool)].reshape(
        n_frames,
        n_atoms,
        n_neighbors)

    if cutoff is not None:
        # Create an index mask for neighbors that are inside the cutoff
        neighbor_mask = distances < cutoff
        neighbor_mask = neighbor_mask.type(torch.float32)
    else:
        neighbor_mask = torch.ones((n_frames, n_atoms, n_neighbors), dtype=torch.float32)

    return neighbors, neighbor_mask


class ElementalGTO(nn.Module):
    
    ''' implementation of Elemental GTO for pytorch autograd support. At the moment only the pairlist is implemented directly as a cuda kernel,
        but might move more of this to cuda when I have time.
    '''

    def __init__(self, species=[1, 6, 7, 8], low_cutoff=0.0, high_cutoff=6.0, n_gaussians=20, eta=2.3, Lmax=2, device=torch.device('cpu')):
        
        super(ElementalGTO, self).__init__()
        self.device = device
        
        self.high_cutoff = high_cutoff
        self.n_gaussians = n_gaussians
        self.n_species = len(species)
        
        self.species = species
        
        self.Lmax = Lmax
        
        offset = torch.linspace(low_cutoff, high_cutoff, n_gaussians + 1, device=device)[1:]
        
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
        
        self.fp_size = n_gaussians * (Lmax + 1) * self.n_mbody
        
        self.register_buffer("fingerprint_size", torch.FloatTensor(self.fp_size))

        self.offsets = offset
        self.width = eta
            
        self.pi = torch.acos(torch.zeros(1)).cuda() * 2
    
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
    
    def forward(self, coordinates, nuclear_charges):
        
        n_batch, n_atoms, _ = coordinates.shape

        num_neighbours = pairlist_gpu.get_num_neighbours_gpu(coordinates, self.high_cutoff)
       
        max_neighbours = torch.max(num_neighbours)

        neighbours = pairlist_gpu.get_neighbour_list_gpu(coordinates, max_neighbours.item(), self.high_cutoff)
        
        pairlist_mask = (neighbours != -1)

        # get rid of the -1's
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
       
        inv_scaling = torch.pow(1.0 / distances[..., None], 2.0 + self.angular_indexes) * pairlist_coeffs[..., None]
        
        angular_terms = inv_scaling * torch.pow(nbh_coords[..., None, 0] , self.angular_components[:, 0 ]) * \
                torch.pow(nbh_coords[..., None, 1] , self.angular_components[:, 1 ]) * \
                torch.pow(nbh_coords[..., None, 2] , self.angular_components[:, 2 ])

        fingerprint = torch.zeros(n_batch, n_atoms, self.Lmax + 1, self.n_mbody, self.n_gaussians, dtype=radial_basis.dtype, device=self.device)
        
        # first construct the single-species three-body terms, e.g X-HH, X-CC...
        for i in range(self.n_species):

            elemental_fingerprint = torch.zeros(n_batch, n_atoms, self.Lmax + 1, self.n_gaussians, dtype=radial_basis.dtype, device=self.device)
    
            mask = (neighbor_numbers[..., None] == self.species[i]).any(-1)
            
            maskval = torch.ones_like(neighbor_numbers[..., None])
            
            mask_ = mask.unsqueeze(-1).expand(neighbours[..., None].size())
            
            coeffs = torch.squeeze(maskval * mask_, dim=-1)

            filtered_radial = coeffs[..., None] * radial_basis 
        
            test = angular_terms[... , None] * filtered_radial[..., None,:] 
        
            test = torch.sum(test, dim=2)

            orbitals = self.angular_weights[None, None,:, None] * torch.pow(test, 2)
          
            elemental_fingerprint.index_add_(2, self.angular_indexes, orbitals)
            
            fingerprint[:,:,:, i,:] = elemental_fingerprint
        
        # now construct the two-species three-body terms, e.g X-CH, X-CN, while negating out the single-species term
        for i in range(self.element_combinations.shape[0]):
    
            elemental_fingerprint = torch.zeros(n_batch, n_atoms, self.Lmax + 1, self.n_gaussians, dtype=radial_basis.dtype, device=self.device)
            
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
            
            fingerprint[:,:,:, self.n_species + i,:] = elemental_fingerprint - (fingerprint[:,:,:, single_species_id[0],:] + \
                                                                                fingerprint[:,:,:, single_species_id[1],:])
       
        return fingerprint.reshape(n_batch, n_atoms, self.fp_size)
