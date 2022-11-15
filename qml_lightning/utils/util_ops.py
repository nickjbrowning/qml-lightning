'''
Created on 11 Dec 2021

@author: Nicholas J. Browning
'''
import torch
from qml_lightning.cuda import utils_gpu


def calculate_self_energy(Q, E, elements):
        
        '''
        computes the per-atom contribution from Q to the property E
        '''
        
        nmol = len(Q)
        
        natom_counts = torch.zeros(nmol, dtype=torch.int)
        
        for i in range(nmol):
            natom_counts[i] = Q[i].shape[0]
            
        max_atoms = natom_counts.max().item()
        
        paddedZ = torch.zeros(nmol, max_atoms, dtype=torch.int)
        
        X = torch.zeros(nmol, len(elements), dtype=torch.float64)
        
        for i in range(nmol):
            paddedZ[i,:natom_counts[i]] = torch.from_numpy(Q[i])

        for i, e in enumerate(elements):
            indexes = paddedZ == e
            
            numbers = torch.sum(indexes, dim=1)
            
            X[:, i] = numbers
        
        XTX = torch.matmul(X.T, X)
            
        energies = torch.DoubleTensor(E)

        beta = torch.linalg.lstsq(XTX, torch.matmul(X.T, energies[:, None])).solution[:, 0]
        
        del XTX
        
        species = elements.long()
        
        self_energy = torch.zeros(torch.max(species) + 1, dtype=torch.float64)
    
        self_energy[species] = beta
        
        return self_energy
        

def format_data(X, Q, E=None, F=None, cells=None, inv_cells=None, same_mol=False):
    
        '''
        assumes input lists of type list(ndarrays), e.g for X: [(5, 3), (3,3), (21, 3), ...] 
        and converts them to fixed-size Torch Tensor of shape [zbatch, max_atoms, ...], e.g  [zbatch, 21, ...]
        
        also outputs natom counts, atomIDs and molIDs necessary for the CUDA implementation
        
        same_mol is a more efficient version for lists containing different configurations of the same molecule
        
        '''
        
        data_dict = {}
        
        # pad a dimension as this is what the GPU code expects...
        
        if (len(X.shape) == 2):
            X = X[None, ...]
            
        if (len(Q.shape) == 1):
            Q = Q[None, ...]
            
        if (F is not None and len(F.shape) == 2):
            F = F[None, ...]
            
        if (cells is not None and len(cells.shape) == 2):
            cells = cells[None, ...]
            
        if (inv_cells is not None and len(inv_cells.shape) == 2):
            inv_cells = inv_cells[None, ...]
            
        zbatch = len(X)
        
        natom_counts = torch.zeros(zbatch, dtype=torch.int32, device='cuda')
        
        if (same_mol):
            
            coordinates = torch.tensor(X).float().cuda()
            charges = torch.tensor(Q).float().cuda()
    
            natom_counts.fill_(coordinates.shape[1])
            
            atomIDs = torch.arange(coordinates.shape[1], device='cuda', dtype=torch.int32).repeat(zbatch)
            
            molIDs = torch.arange(zbatch, device='cuda', dtype=torch.int32)[:, None].repeat(1, coordinates.shape[1]).flatten()
            
            data_dict['coordinates'] = coordinates
            data_dict['charges'] = charges
            data_dict['natom_counts'] = natom_counts
            data_dict['atomIDs'] = atomIDs
            data_dict['molIDs'] = molIDs
            
            if (E is not None):
                all_energies = torch.tensor(E).double().cuda()
                data_dict['energies'] = all_energies
                
            if (F is not None):
                all_forces = torch.tensor(F).float().cuda()
                data_dict['forces'] = all_forces
                
            if (cells is not None):
                all_cells = torch.tensor(cells).float().cuda()
                all_inv_cells = torch.tensor(inv_cells).float().cuda()
                
                data_dict['cells'] = all_cells
                data_dict['inv_cells'] = all_inv_cells
                
            else:
                data_dict['cells'] = torch.empty((0, 3, 3), device='cuda')
                data_dict['inv_cells'] = torch.empty((0, 3, 3), device='cuda')
            
        else:
    
            natom_counts = torch.zeros(zbatch, dtype=torch.int32)
                
            for j in range(zbatch):
                
                coordinates = X[j]
        
                natom_counts[j] = coordinates.shape[0]
                
            max_atoms = natom_counts.max().item()
             
            all_coordinates = torch.zeros(zbatch, max_atoms, 3, dtype=torch.float32)
            all_charges = torch.zeros(zbatch, max_atoms, dtype=torch.float32)
            
            if (E is not None):
                all_energies = torch.DoubleTensor(E)
                
            if (F is not None):
                all_forces = torch.zeros(zbatch, max_atoms, 3, dtype=torch.float64)
                
            if (cells is not None):
                all_cells = torch.zeros(zbatch, 3, 3, dtype=torch.float32)
                all_inv_cells = torch.zeros(zbatch, 3, 3, dtype=torch.float32)  
            
            molIDs = torch.Tensor([])
            atomIDs = torch.Tensor([])
                
            for j in range(zbatch):
                
                charges = torch.from_numpy(Q[j]).float()
                coordinates = torch.from_numpy(X[j]).float()
                
                natoms = natom_counts[j]
                
                all_charges[j,:natoms] = charges
                all_coordinates[j,:natoms,:] = coordinates
                
                molID = torch.empty(natoms, dtype=torch.int32).fill_(j)
                atomID = torch.arange(0, natoms)
                
                molIDs = torch.cat((molIDs, molID), dim=0)
                atomIDs = torch.cat((atomIDs, atomID), dim=0)
                    
                if (F is not None):
                    forces = torch.from_numpy(F[j]).double()
                        
                    all_forces[j,:natoms,:] = forces
                    
                if (cells is not None):
                    cell = torch.from_numpy(cells[j]).float()
                    inv_cell = torch.from_numpy(inv_cells[j]).float()
                    all_cells[j] = cell
                    all_inv_cells[j] = inv_cell
      
            all_coordinates = all_coordinates.cuda()
            all_charges = all_charges.cuda()
            natom_counts = natom_counts.cuda()
            atomIDs = atomIDs.int().cuda()
            molIDs = molIDs.int().cuda()
            
            data_dict['coordinates'] = all_coordinates
            data_dict['charges'] = all_charges
            data_dict['natom_counts'] = natom_counts
            data_dict['atomIDs'] = atomIDs
            data_dict['molIDs'] = molIDs
            
            data_dict['energies'] = None
            data_dict['forces'] = None
            data_dict['cells'] = torch.empty((0, 3, 3), device='cuda')
            data_dict['inv_cells'] = torch.empty((0, 3, 3), device='cuda')
                 
            if (E is not None):
                all_energies = all_energies.cuda()
                data_dict['energies'] = all_energies
                
            if (F is not None):
                all_forces = all_forces.cuda()
                data_dict['forces'] = all_forces
                
            if (cells is not None):
                all_cells = all_cells.cuda()
                all_inv_cells = all_inv_cells.cuda()
                data_dict['cells'] = all_cells
                data_dict['inv_cells'] = all_inv_cells
                
        return data_dict


class CosFeatures(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, indexes, bias, normalisation):
        
        ctx.normalisation = normalisation
        ctx.save_for_backward(X, indexes, bias)
        
        return utils_gpu.CosFeaturesCUDA(X, indexes, bias, normalisation)

    @staticmethod
    def backward(ctx, gradX):
        
        print ("gradX shape: ", gradX.shape)
        X, indexes, bias = ctx.saved_tensors
        
        return utils_gpu.DerivativeCosFeaturesCUDA(X, indexes, bias, ctx.normalisation, gradX), None, None, None

