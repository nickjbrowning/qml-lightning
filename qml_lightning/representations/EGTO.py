'''
Created on 1 Apr 2021

@author: Nicholas J. Browning
'''

from qml_lightning.cuda import egto_gpu
import torch
import numpy as np


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
