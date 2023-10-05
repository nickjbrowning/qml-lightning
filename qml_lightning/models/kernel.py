'''
Created on 22 Jun 2021

@author: Nicholas J. Browning
'''
import torch

import numpy as np
from qml_lightning.representations.dimensionality_reduction import project_representation, project_derivative
from tqdm import tqdm
from qml_lightning.cuda.utils_gpu import matmul_and_reduce
#from qml_lightning.torchscript import setup


class BaseKernel():
 
    def __init__(self, rep, npcas, elements, sigma, llambda):
        
        super(BaseKernel, self).__init__()
        
        self.npcas = npcas
        self.self_energy = None
        
        self.elements = torch.tensor(elements)
        
        self.rep = rep
        
        self.reductors = torch.zeros(len(elements), rep.get_repsize(), npcas, dtype=torch.float32, device='cuda')
        self.element2index = torch.zeros(max(elements) + 1, dtype=torch.long)
        
        for i, e in enumerate(self.elements):
            self.element2index[int(e)] = i
        
        self._convert_from_hartree_to_kcal = False
        
        self.hartree2kcalmol = 627.5
        
        self._subtract_self_energies = False
        
        self.sigma = sigma
        self.llambda = llambda
    
    def calculate_features(self, rep, element, indexes, feature_matrix, grad=None, derivative_matrix=None):
        raise NotImplementedError("Abstract method only.")
        
    def build_Z_components(self, X, Q, E=None, F=None, cells=None, inv_cells=None, print_info=True, cpu_solve=False, ntiles=1, use_specialized_matmul=False):
        
        '''
        X: list of coordinates : numpy arrays of shape [natom_i, 3]
        Z: list of charges: numpy arrays of shape [natom_i]
        
        cpu_solve: set to True to store Z^TZ on the CPU, and solve on the CPU
        ntiles: > 1 sets the Z^TZ matrix to be constructed in (self.nfeatures()/ntiles) features per batch.
         
        print_info: Boolean defining whether or not to print timings + progress bar
        
        returns: Z^TZ, ZY
        
        '''
        
        if (self.reductors is None):
            print("ERROR: Must call model.get_reductors() first to initialize the projection matrices.")
            exit()
            
        if (E is None and F is None):
            print("ERROR: must have either E, F or both as input to train().")
            exit()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        ZTZ = torch.zeros(self.nfeatures(), self.nfeatures(), device=torch.device('cpu') if cpu_solve else self.device, dtype=torch.float64)
            
        ZtrainY = torch.zeros(self.nfeatures(), 1, device=self.device, dtype=torch.float64)
        
        start.record()
        
        nsub_features = int(np.ceil(self.nfeatures() / ntiles))

        for tile in range(0, ntiles):
            
            if (ntiles > 1):
                start_tile = tile * nsub_features
                end_tile = min(nsub_features * (tile + 1), self.nfeatures())
                
                tile_size = end_tile - start_tile
                
                if (print_info):
                    print ("tile:", tile + 1 , "of", ntiles, "tiles, tile size:", tile_size)
                
                ZTZ_tile = torch.zeros(tile_size, self.nfeatures(), device=self.device, dtype=torch.float64)
             
            for i in tqdm(range(0, len(X), self.nbatch_train)) if print_info else (range(0, len(X), self.nbatch_train)):
                
                coordinates = X[i:i + self.nbatch_train]
                charges = Q[i:i + self.nbatch_train]
                
                energies = E[i:i + self.nbatch_train] if E is not None else None
                forces = F[i:i + self.nbatch_train] if F is not None else None
                zcells = cells[i:i + self.nbatch_train]  if cells is not None else None
                zinv_cells = inv_cells[i:i + self.nbatch_train]  if cells is not None else None
                
                zbatch = len(coordinates)
                
                data = self.format_data(coordinates, charges, E=energies, F=forces, cells=zcells, inv_cells=zinv_cells)
                
                coordinates = data['coordinates']
                charges = data['charges']
                atomIDs = data['atomIDs']
                molIDs = data['molIDs']
                natom_counts = data['natom_counts']
                zcells = data['cells']
                zinv_cells = data['inv_cells']
                
                energies = data['energies']
                forces = data['forces']
                
                if (E is not None and F is None):
                    targets = energies[:, None] 
                elif (E is None and F is not None):
                    # zero out energy targets so it has the correct dimensions for matmul
                    targets = torch.cat((torch.zeros((zbatch, 1), device=self.device), forces.flatten()[:, None]), dim=0)
                else:
                    targets = torch.cat((energies[:, None], forces.flatten()[:, None]), dim=0)
                
                max_natoms = natom_counts.max().item()
          
                if (F is None):
                    representation = self.rep.get_representation(coordinates, charges, atomIDs, molIDs, natom_counts, zcells, zinv_cells)
                else:
                    representation, rep_derivative = self.rep.get_representation_and_derivative(coordinates, charges, atomIDs, molIDs,
                                                                                     natom_counts, zcells, zinv_cells)
                
                Ztrain = torch.zeros(zbatch, self.nfeatures(), device=self.device, dtype=torch.float64)
                
                Gtrain_derivative = None
                
                if (F is not None):
                    Gtrain_derivative = torch.zeros(zbatch, max_natoms, 3, self.nfeatures(), device=self.device, dtype=torch.float64)
                    
                for e in self.elements:
                
                    indexes = charges.int() == e.item()
                    
                    element_idx = self.element2index[e]
                    
                    batch_indexes = torch.where(indexes)[0].type(torch.int)
                    
                    sub = representation[indexes]
                    
                    if (sub.shape[0] == 0):
                        continue
                    
                    sub = project_representation(sub, self.reductors[element_idx])
         
                    sub_grad = None
                    
                    if (F is not None):
                        sub_grad = rep_derivative[indexes]
                        sub_grad = project_derivative(sub_grad, self.reductors[element_idx])
         
                    self.calculate_features(sub, e, batch_indexes, Ztrain, sub_grad, Gtrain_derivative)
                
                if (E is None):
                    Ztrain.fill_(0)  # hack to set all energy features to 0, such that they do not contribute to Z.T Z
                
                if (F is not None):
                    Gtrain_derivative = Gtrain_derivative.reshape(zbatch * max_natoms * 3, self.nfeatures())
                    
                    Ztrain = torch.cat((Ztrain, Gtrain_derivative), dim=0)
                
                if (ntiles > 1):
                    
                    sub = Ztrain[:, start_tile:end_tile]
                    
                    if (use_specialized_matmul):
                        matmul_and_reduce(sub.float().T, Ztrain.float(), ZTZ_tile)
                    else:
                        ZTZ_tile += torch.matmul(sub.T, Ztrain)
                    
                    ZtrainY[ start_tile:end_tile,:] += torch.matmul(sub.T, targets)
                    
                else:
                    if (cpu_solve):
                        ZTZ += torch.matmul(Ztrain.T, Ztrain).cpu()
                    else:
                        if (use_specialized_matmul):
                            matmul_and_reduce(Ztrain.float().T, Ztrain.float(), ZTZ)
                        else:
                            ZTZ += torch.matmul(Ztrain.T, Ztrain)
                        
                    ZtrainY += torch.matmul(Ztrain.T, targets)
    
                del Ztrain
                del representation
                del sub
                
                if (F is not None):
                    del Gtrain_derivative
                    del rep_derivative
                    del sub_grad
                    
                torch.cuda.empty_cache()
            
            if (ntiles > 1):
                if (cpu_solve):
                    ZTZ[start_tile:end_tile,:] += ZTZ_tile.cpu()
                else:
                    ZTZ[start_tile:end_tile,:] += ZTZ_tile
                
        end.record()
        torch.cuda.synchronize()
        
        if (print_info):
            print("ZTZ time: ", start.elapsed_time(end), "ms")
            
        return ZTZ, ZtrainY
     
    def train(self, X, Q, E=None, F=None, cells=None, inv_cells=None, print_info=True, cpu_solve=False, ntiles=1, use_specialized_matmul=False):
        
        ZTZ, ZtrainY = self.build_Z_components(X, Q, E, F, cells, inv_cells, print_info, cpu_solve, ntiles, use_specialized_matmul=use_specialized_matmul)
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        for i in range(self.nfeatures()):
            ZTZ[i, i] += self.llambda
        
        start.record()
        
        if (cpu_solve):
            self.alpha = torch.linalg.solve(ZTZ, ZtrainY.cpu())[:, 0].cuda()
        else:
            self.alpha = torch.linalg.solve(ZTZ, ZtrainY)[:, 0]
            
        end.record()
        torch.cuda.synchronize()
        
        if (print_info):
            print("coefficients time: ", start.elapsed_time(end), "ms")
        
        del ZtrainY
        del ZTZ
        
        self.is_trained = True
        
        torch.cuda.empty_cache()
    
    def hyperparam_opt_nested_cv(self, X, Q, E, F=None, cells=None, inv_cells=None,
                                 sigmas=np.linspace(1.5, 12.5, 10), lambdas=np.logspace(-11, -5, 9), kfolds=5,
                                 print_info=True, use_rmse=False):
        
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=kfolds, shuffle=False)
        
        errors = []
        grid_locs = []
        
        for i, sigma in enumerate(sigmas):
            for j, llambda in enumerate(lambdas):
                
                fold_error = 0.0
                
                for k, (train_index, test_index) in enumerate(kf.split(X)):
                    Xtrain = [X[z] for z in train_index]
                    Ztrain = [Q[z] for z in train_index]
                    Etrain = [E[z] for z in train_index]
                        
                    Xtest = [X[z] for z in test_index]
                    Ztest = [Q[z] for z in test_index]
                    Etest = [E[z] for z in test_index]
                    
                    Ftrain = None
                    Ftest = None
                    
                    if (F is not None):
                        Ftrain = [F[z] for z in train_index]
                        Ftest = [F[z] for z in test_index]
                    
                    self.sigma = sigma
                    self.llambda = llambda
      
                    self.train(Xtrain, Ztrain, Etrain, Ftrain, cells, inv_cells, print_info=False)
  
                    data = self.format_data(Xtest, Ztest, Etest, Ftest, cells=cells, inv_cells=inv_cells)
                    
                    test_energies = data['energies']
                    test_forces = data['forces']
                    max_natoms = data['natom_counts'].max().item()
                    zcells = data['cells']
                    zinv_cells = data['inv_cells']
                    
                    if (F is not None):
                        energy_predictions, force_predictions = self.predict(Xtest, Ztest, max_natoms,
                                                                             cells=cells, inv_cells=inv_cells,
                                                                             forces=True, print_info=False)
                    else:
                        energy_predictions = self.predict(Xtest, Ztest, max_natoms, cells=cells, inv_cells=inv_cells,
                                                          forces=False, print_info=False)
                    
                    if (use_rmse):
                        energy_error = torch.sqrt(torch.mean(torch.pow(energy_predictions - test_energies, 2)))
                    else:
                        energy_error = torch.mean(torch.abs(energy_predictions - test_energies))
                    
                    if (F is not None):
                        
                        if (use_rmse):
                            force_error = torch.sqrt(torch.mean(torch.pow(force_predictions - test_forces, 2)))
                        else:
                            force_error = torch.mean(torch.abs(force_predictions - test_forces))
                            
                        if (print_info):
                            print ("Fold:" , k, sigma, llambda, energy_error, force_error)
                        fold_error += 0.5 * (energy_error + force_error)
                        
                    else:
                        if (print_info):
                            print ("Fold:" , k, sigma, llambda, energy_error)
                        fold_error += energy_error
                    
                fold_error /= kfolds
                    
                errors.append(fold_error.item())
                grid_locs.append([i, j])

        errors = np.array(errors)
        
        if (print_info):
            print ("errors: ", errors)
            print (grid_locs)
        
        idxs = np.array(grid_locs)
        
        min_idx = np.argmin(errors)
        
        if (print_info):
            print ("min_idx:", min_idx)
        
        best_sigma = sigmas[idxs[min_idx, 0]]
        best_llambda = lambdas[idxs[min_idx, 1]]
        
        if (print_info):
            print ("Best error: ", errors[min_idx], "sigma = ", best_sigma, "lambda = ", best_llambda)
        
        return best_sigma, best_llambda

    def predict_cuda(self, X, Q, max_natoms, cells=None, inv_cells=None, forces=False, print_info=True):
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        if (self.is_trained is False):
            print ("Error: must train the model first by calling train()!")
            exit()
        
        predict_energies = torch.zeros(len(X), device=self.device, dtype=torch.float64)
        predict_forces = torch.zeros(len(X), max_natoms, 3, device=self.device, dtype=torch.float64)
        
        start.record()
        
        for i in tqdm(range(0, len(X), self.nbatch_test)) if print_info else range(0, len(X), self.nbatch_test):
            
            coordinates = X[i:i + self.nbatch_test]
            charges = Q[i:i + self.nbatch_test]
            
            zbatch = len(coordinates)
            
            if (cells is not None):
                zcells = cells[i:i + self.nbatch_test]
                zinv_cells = inv_cells[i:i + self.nbatch_test]
            else: 
                zcells = None
                zinv_cells = None
                
            data = self.format_data(coordinates, charges, cells=zcells, inv_cells=zinv_cells)
            
            coordinates = data['coordinates']
            charges = data['charges']
            atomIDs = data['atomIDs']
            molIDs = data['molIDs']
            natom_counts = data['natom_counts']
            zcells = data['cells']
            zinv_cells = data['inv_cells']
            
            batch_max_natoms = natom_counts.max().item()
            
            if (forces is True):
                gto, gto_derivative = self.rep.get_representation_and_derivative(coordinates, charges, atomIDs, molIDs, natom_counts, zcells, zinv_cells)
            else:
                gto = self.rep.get_representation(coordinates, charges, atomIDs, molIDs, natom_counts, zcells, zinv_cells)
                
            Ztest = torch.zeros(zbatch, self.nfeatures(), device=self.device, dtype=torch.float64)
            
            Gtest_derivative = None
            
            if (forces is True):
                Gtest_derivative = torch.zeros(zbatch, max_natoms, 3, self.nfeatures(), device=self.device, dtype=torch.float64)
                
            for e in self.elements:
                
                idx = self.element2index[e]
                
                indexes = charges.int() == e
                
                batch_indexes = torch.where(indexes)[0].type(torch.int)
                
                sub = gto[indexes]
                
                if (sub.shape[0] == 0):
                    continue
                
                sub = project_representation(sub, self.reductors[idx])
                
                sub_grad = None
                
                if (forces is True):
                    sub_grad = gto_derivative[indexes]
                    sub_grad = project_derivative(sub_grad, self.reductors[idx])

                self.calculate_features(sub, e, batch_indexes, Ztest, sub_grad, Gtest_derivative)
                
            predict_energies[i:i + self.nbatch_test] = torch.matmul(Ztest, self.alpha)
            
            if (forces is True):
                Gtest_derivative = Gtest_derivative.reshape(zbatch * max_natoms * 3, self.nfeatures())
                predict_forces[i:i + self.nbatch_test] = torch.matmul(Gtest_derivative, self.alpha).reshape(zbatch, max_natoms, 3)
        
        end.record()
        torch.cuda.synchronize()
        
        if (print_info):
            print("prediction for", len(X), "molecules time: ", start.elapsed_time(end), "ms")
        
        if (forces is True):
            return (predict_energies, predict_forces)
        else:
            return predict_energies
    
    def nfeatures(self):
        raise NotImplementedError("Abstract method only.")
    
    def predict(self, X, Q, max_natoms, cells=None, inv_cells=None, forces=False, print_info=True, use_backward=True):
        raise NotImplementedError("Abstract method only.")
            
    def get_reductors(self, X, Q, cells=None, inv_cells=None, npcas=128, npca_choice=256, nsamples=4096, print_info=True):
        
        '''
        npcas: length of low-dimension projection
        npca_choice: for each batch, select at most this many atomic representations to be added to the SVD matrix
        nsamples: maximum total number of selected atomic representations
        '''
        
        index_set = {}
 
        for e in self.elements:
            
            element_idx = self.element2index[e]
            
            idxs = []
            
            for i in range (len(Q)):
                if (e.item() in Q[i]):
                    idxs.append(i)
                    continue
            
            idxs = np.array(idxs)

            index_set[e] = idxs
            
            subsample_indexes = np.random.choice(index_set[e], size=np.min([len(index_set[e]), 1024]))
            
            subsample_coordinates = [X[i] for i in subsample_indexes]
            subsample_charges = [Q[i] for i in subsample_indexes]
            
            inputs = []
            
            nselected = 0
            
            for i in range(0, len(subsample_coordinates), self.nbatch_train):
                
                # only collect nsample representations to compute the SVD
                if (nselected > nsamples):
                    break
                
                coordinates = subsample_coordinates[i:i + self.nbatch_train]
                charges = subsample_charges[i:i + self.nbatch_train]
                
                data = self.format_data(coordinates, charges, cells, inv_cells)
                
                coords = data['coordinates']
                qs = data['charges']
                atomIDs = data['atomIDs']
                molIDs = data['molIDs']
                natom_counts = data['natom_counts']
                zcells = data['cells']
                zinv_cells = data['inv_cells']
                
                indexes = qs == e.item()
                
                rep = self.rep.get_representation(coords, qs, atomIDs, molIDs, natom_counts, zcells, zinv_cells)
                
                sub = rep[indexes]
                
                if (sub.shape[0] == 0):
                    continue
                
                perm = torch.randperm(sub.size(0))
                idx = perm[:npca_choice]
        
                choice_input = sub[idx]
                
                nselected += choice_input.shape[0]
                
                inputs.append(choice_input)
            
            if (len(inputs) == 0):
                continue
            
            mat = torch.cat(inputs)
            
            eigvecs, eigvals, vh = torch.linalg.svd(mat.T, full_matrices=False)

            cev = 100 - (torch.sum(eigvals) - torch.sum(eigvals[:npcas])) / torch.sum(eigvals) * 100
            
            reductor = eigvecs[:,:npcas]
            size_from = reductor.shape[0]
            size_to = reductor.shape[1]
            
            torch.cuda.synchronize()
            
            if (print_info):
                print (f"element {e}: {size_from} -> {size_to}  Cumulative Explained Feature Variance = {cev:6.2f} %")
                
            if (reductor.shape[1] < npcas):
                print (f"ERROR - not enough atomic environments in input dataset for element {e}. Either increase npca_choice or provide more structures.")
                exit()
            
            self.reductors[element_idx] = reductor
    
    def set_subtract_self_energies(self, subtract):
        self._subtract_self_energies = subtract
    
    def subtract_self_energies(self):
        return self._subtract_self_energies
    
    def set_convert_hartree2kcal(self, convert):
        self._convert_from_hartree_to_kcal = convert
        
    def convert_hartree2kcal(self):
        return self._convert_from_hartree_to_kcal
    
    def calculate_self_energy(self, Q, E):
        
        '''
        computes the per-atom contribution from Q towards the property E
        '''
        nmol = len(Q)
        
        natom_counts = torch.zeros(nmol, dtype=torch.int)
        
        for i in range(nmol):
            natom_counts[i] = Q[i].shape[0]
            
        max_atoms = natom_counts.max().item()
        
        paddedZ = torch.zeros(nmol, max_atoms, dtype=torch.int)
        
        X = torch.zeros(nmol, len(self.elements), dtype=torch.float64)
        
        for i in range(nmol):
            paddedZ[i,:natom_counts[i]] = torch.from_numpy(Q[i])

        for i, e in enumerate(self.elements):
            indexes = paddedZ == e
            
            numbers = torch.sum(indexes, dim=1)
            
            X[:, i] = numbers
        
        XTX = torch.matmul(X.T, X)
            
        energies = torch.DoubleTensor(E)
        
        if (self.convert_hartree2kcal()):
            energies *= self.hartree2kcalmol
        
        beta = torch.linalg.lstsq(XTX, torch.matmul(X.T, energies[:, None])).solution[:, 0]
        
        del XTX
        
        species = self.elements.long()
        
        self.self_energy = torch.zeros(torch.max(species) + 1, dtype=torch.float64)
    
        self.self_energy[species] = beta
        
        print ("Self Energies: ", self.self_energy)
        
    def format_data(self, X, Q, E=None, F=None, cells=None, inv_cells=None):
    
        '''
        assumes input lists of type list(ndarrays), e.g for X: [(5, 3), (3,3), (21, 3), ...] 
        and converts them to fixed-size Torch Tensor of shape [zbatch, max_atoms, ...], e.g  [zbatch, 21, ...]
        
        also outputs natom counts, atomIDs and molIDs necessary for the CUDA implementation
        
        '''
        
        if (self.subtract_self_energies() and self.self_energy is None):
            print("ERROR: must call model.calculate_self_energy first - this computes atomic contributes to the potential energy.")
            exit()
        
        data_dict = {}
  
        zbatch = len(X)

        natom_counts = torch.zeros(zbatch, dtype=torch.int32)
            
        for j in range(zbatch):
            
            coordinates = X[j]
            charges = Q[j]
    
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
            
            if (E is not None):
             
                energy = all_energies[j]
                
                if (self.convert_hartree2kcal()):
                    energy *= self.hartree2kcalmol
                
                if (self.subtract_self_energies()):
                    
                    self_interaction = self.self_energy[charges.long()].sum(axis=0)
                        
                    energy = energy - self_interaction
                
                all_energies[j] = energy
                
            if (F is not None):
                forces = torch.from_numpy(F[j]).double()
                
                if (self.convert_hartree2kcal()):
                    forces *= self.hartree2kcalmol
                    
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
