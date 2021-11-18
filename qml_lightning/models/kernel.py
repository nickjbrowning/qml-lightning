'''
Created on 22 Jun 2021

@author: Nicholas J. Browning
'''
import torch
import numpy as np
from qml_lightning.representations.dimensionality_reduction import project_representation, project_derivative
from tqdm import tqdm


class BaseKernel(object):
 
    def __init__(self, rep, elements, sigma, llambda):
        
        self.self_energy = None
        
        self.elements = elements
        
        self.rep = rep
        
        self.reductors = None
        
        self._convert_from_hartree_to_kcal = False
        
        self.hartree2kcalmol = 627.503
        
        self._subtract_self_energies = False
        
        self.sigma = sigma
        self.llambda = llambda
    
    def calculate_features(self, rep, element, indexes, feature_matrix, grad=None, derivative_matrix=None):
        raise NotImplementedError("Abstract method only.")
    
    def train_conjugate_gradient(self, X, Z, E, F=None, niters=1000, preconditioner=None):
        
        x = torch.zeros(self.nfeatures, 1, device=self.device, dtype=torch.float64)
        r = torch.zeros(self.nfeatures, 1, device=self.device, dtype=torch.float64)
        
        if (preconditioner is None):
            preconditioner = torch.eye(self.nfeatures, device=self.device, dtype=torch.float64)
            
        for i in range(0, len(X), self.nbatch):
            
            coordinates = X[i:i + self.nbatch]
            charges = Z[i:i + self.nbatch]
            energies = E[i:i + self.nbatch]
            
            if (F is not None):
                forces = F[i:i + self.nbatch]
            else:
                forces = None
                
            zbatch = len(coordinates)
            
            data = self.format_data(coordinates, charges, energies, forces)
            
            coordinates = data['coordinates']
            charges = data['charges']
            atomIDs = data['atomIDs']
            molIDs = data['molIDs']
            natom_counts = data['natom_counts']
            
            energies = data['energies']
            forces = data['forces']
            
            max_natoms = natom_counts.max().item()
            
            if (F is None):
                gto = self.rep.get_representation(coordinates, charges, atomIDs, molIDs, natom_counts)
            else:
                gto, gto_derivative = self.rep.get_representation_and_derivative(coordinates, charges, atomIDs, molIDs, natom_counts)
            
            Ztrain = torch.zeros(zbatch, self.nfeatures, device=self.device, dtype=torch.float64)
            
            Gtrain_derivative = None
            
            if (F is not None):
                Gtrain_derivative = torch.zeros(zbatch, max_natoms, 3, self.nfeatures, device=self.device, dtype=torch.float64)
                
            for e in self.elements:
            
                indexes = charges.int() == e
                
                batch_indexes = torch.where(indexes)[0].type(torch.int)
                
                sub = gto[indexes]
                
                if (sub.shape[0] == 0):
                    continue
                
                sub = project_representation(sub, self.reductors[e])
     
                sub_grad = None
                
                if (F is not None):
                    sub_grad = gto_derivative[indexes]
                    sub_grad = project_derivative(sub_grad, self.reductors[e])
     
                self.calculate_features(sub, e, batch_indexes, Ztrain, sub_grad, Gtrain_derivative)
            
            Y = energies[:, None]
            
            if (F is not None):
                Gtrain_derivative = Gtrain_derivative.reshape(zbatch * max_natoms * 3, self.nfeatures)
                
                Ztrain = torch.cat((Ztrain, Gtrain_derivative), dim=0)
                
                Y = torch.cat((Y, forces.flatten()[:, None]))
                
            r += torch.matmul(Ztrain.T, Y)

        rhat = torch.matmul(preconditioner, r)
        p = torch.matmul(preconditioner, r)
        
        rsold = torch.matmul(r.T, rhat)
    
        for j in range(niters):
            
            Ap = torch.zeros(self.nfeatures, 1, device=self.device, dtype=torch.float64)
            
            for i in range(0, len(X), self.nbatch):
            
                coordinates = X[i:i + self.nbatch]
                charges = Z[i:i + self.nbatch]
                energies = E[i:i + self.nbatch]
                
                if (F is not None):
                    forces = F[i:i + self.nbatch]
                else:
                    forces = None
                    
                zbatch = len(coordinates)
                
                data = self.format_data(coordinates, charges, energies, forces)
                
                coordinates = data['coordinates']
                charges = data['charges']
                atomIDs = data['atomIDs']
                molIDs = data['molIDs']
                natom_counts = data['natom_counts']
                
                energies = data['energies']
                forces = data['forces']
                
                max_natoms = natom_counts.max().item()
                
                if (F is None):
                    gto = self.rep.get_representation(coordinates, charges, atomIDs, molIDs, natom_counts)
                else:
                    gto, gto_derivative = self.rep.get_representation_and_derivative(coordinates, charges, atomIDs, molIDs, natom_counts)
                
                Ztrain = torch.zeros(zbatch, self.nfeatures, device=self.device, dtype=torch.float64)
                
                Gtrain_derivative = None
                
                if (F is not None):
                    Gtrain_derivative = torch.zeros(zbatch, max_natoms, 3, self.nfeatures, device=self.device, dtype=torch.float64)
                    
                for e in self.elements:
                
                    indexes = charges.int() == e
                    
                    batch_indexes = torch.where(indexes)[0].type(torch.int)
                    
                    sub = gto[indexes]
                    
                    if (sub.shape[0] == 0):
                        continue
                    
                    sub = project_representation(sub, self.reductors[e])
         
                    sub_grad = None
                    
                    if (F is not None):
                        sub_grad = gto_derivative[indexes]
                        sub_grad = project_derivative(sub_grad, self.reductors[e])
         
                    self.calculate_features(sub, e, batch_indexes, Ztrain, sub_grad, Gtrain_derivative)
                
                if (F is not None):
                    Gtrain_derivative = Gtrain_derivative.reshape(zbatch * max_natoms * 3, self.nfeatures)
                    
                    Ztrain = torch.cat((Ztrain, Gtrain_derivative), dim=0)
                
                Ap += torch.matmul(Ztrain.T, torch.matmul(Ztrain, p))
              
            a = rsold / torch.matmul(p.T, Ap)
        
            x = x + a * p
            r = r - a * Ap
            
            rhat_new = torch.matmul(preconditioner, r)
            
            rsnew = torch.matmul(r.T, rhat_new)
            
            p = rhat_new + (rsnew / rsold) * p
            
            print (j, rsnew)
            
            rsold = rsnew
            
        self.alpha = x[:, 0]
    
    def train(self, X, Z, E, F=None, cells=None, print_info=True):
            
        if (self.reductors is None):
            print("ERROR: Must call model.get_reductors() first to initialize the projection matrices.")
            exit()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        startg = torch.cuda.Event(enable_timing=True)
        endg = torch.cuda.Event(enable_timing=True)
        
        ZTZ = torch.zeros(self.nfeatures, self.nfeatures, device=self.device, dtype=torch.float64)
        ZtrainY = torch.zeros(self.nfeatures, 1, device=self.device, dtype=torch.float64)

        start.record()
        for i in tqdm(range(0, len(X), self.nbatch)) if print_info else (range(0, len(X), self.nbatch)):
            
            coordinates = X[i:i + self.nbatch]
            charges = Z[i:i + self.nbatch]
            energies = E[i:i + self.nbatch]
            
            if (F is not None):
                forces = F[i:i + self.nbatch]
            else:
                forces = None
                
            if (cells is not None):
                zcells = cells[i:i + self.nbatch] 
            else: zcells = None
                
            zbatch = len(coordinates)
            
            data = self.format_data(coordinates, charges, E=energies, F=forces, cells=zcells)
            
            coordinates = data['coordinates']
            charges = data['charges']
            atomIDs = data['atomIDs']
            molIDs = data['molIDs']
            natom_counts = data['natom_counts']
            zcells = data['cells']
            
            energies = data['energies']
            forces = data['forces']
            
            targets = energies[:, None]
            
            max_natoms = natom_counts.max().item()
            
            startg.record()
            if (F is None):
                gto = self.rep.get_representation(coordinates, charges, atomIDs, molIDs, natom_counts, zcells)
            else:
                gto, gto_derivative = self.rep.get_representation_and_derivative(coordinates, charges, atomIDs, molIDs, natom_counts, zcells)
            
            Ztrain = torch.zeros(zbatch, self.nfeatures, device=self.device, dtype=torch.float64)
            
            Gtrain_derivative = None
            
            if (F is not None):
                Gtrain_derivative = torch.zeros(zbatch, max_natoms, 3, self.nfeatures, device=self.device, dtype=torch.float64)
                
            for e in self.elements:
            
                indexes = charges.int() == e
                
                batch_indexes = torch.where(indexes)[0].type(torch.int)
                
                sub = gto[indexes]
                
                if (sub.shape[0] == 0):
                    continue
                
                sub = project_representation(sub, self.reductors[e])
     
                sub_grad = None
                
                if (F is not None):
                    sub_grad = gto_derivative[indexes]
                    sub_grad = project_derivative(sub_grad, self.reductors[e])
     
                self.calculate_features(sub, e, batch_indexes, Ztrain, sub_grad, Gtrain_derivative)
            
            if (F is not None):
                Gtrain_derivative = Gtrain_derivative.reshape(zbatch * max_natoms * 3, self.nfeatures)
                
                Ztrain = torch.cat((Ztrain, Gtrain_derivative), dim=0)
                
                targets = torch.cat((targets, forces.flatten()[:, None]), dim=0)
            
            endg.record()
            torch.cuda.synchronize()
            # print("Features and derivatives time: ", startg.elapsed_time(endg), "ms")
            
            ZtrainY += torch.matmul(Ztrain.T, targets)
            
            startg.record()
            ZTZ += torch.matmul(Ztrain.T, Ztrain)
            endg.record()
            torch.cuda.synchronize()
            # print("sub ZTZ time: ", startg.elapsed_time(endg), "ms")
            
            del Ztrain
            del gto
            del sub
            
            if (F is not None):
                del Gtrain_derivative
                del gto_derivative
                del sub_grad
                
            torch.cuda.empty_cache()
        
        end.record()
        torch.cuda.synchronize()
            
        # maxval = torch.max(torch.diagonal(ZTZ))
        
        for i in range(self.nfeatures):
            ZTZ[i, i] += self.llambda
        
        if (print_info):
            print (ZTZ)
            print("ZTZ time: ", start.elapsed_time(end), "ms")
        
        start.record()
        self.alpha = torch.linalg.solve(ZTZ, ZtrainY)[:, 0]
        end.record()
        torch.cuda.synchronize()
        if (print_info):
            print("coefficients time: ", start.elapsed_time(end), "ms")
        
        del ZtrainY
        del ZTZ
        torch.cuda.empty_cache()
    
    def hyperparam_opt_nested_cv(self, X, Z, E, F=None, sigmas=np.linspace(1.5, 12.5, 10), lambdas=np.logspace(-11, -5, 9), kfolds=5, print_info=True):
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=kfolds, shuffle=False)
        
        errors = []
        grid_locs = []
        
        for i, sigma in enumerate(sigmas):
            for j, llambda in enumerate(lambdas):
                
                fold_mae = 0.0
                
                for k, (train_index, test_index) in enumerate(kf.split(X)):
                    Xtrain = [X[i] for i in train_index]
                    Ztrain = [Z[i] for i in train_index]
                    Etrain = [E[i] for i in train_index]
                        
                    Xtest = [X[i] for i in test_index]
                    Ztest = [Z[i] for i in test_index]
                    Etest = [E[i] for i in test_index]
                    
                    Ftrain = None
                    Ftest = None
                    
                    if (F is not None):
                        Ftrain = [F[i] for i in train_index]
                        Ftest = [F[i] for i in test_index]
                    
                    self.sigma = sigma
                    self.llambda = llambda
                    
                    self.train(Xtrain, Ztrain, Etrain, Ftrain, print_info=False)
                    
                    data = self.format_data(Xtest, Ztest, Etest, Ftest)
                    
                    test_energies = data['energies']
                    test_forces = data['forces']
                    max_natoms = data['natom_counts'].max().item()
                    
                    if (F is not None):
                        energy_predictions, force_predictions = self.predict_cuda(Xtest, Ztest, max_natoms, forces=True, print_info=False)
                    else:
                        energy_predictions = self.predict_cuda(Xtest, Ztest, max_natoms, forces=False, print_info=False)
                        
                    EMAE = torch.mean(torch.abs(energy_predictions - test_energies))
                    
                    if (F is not None):
                        FMAE = torch.mean(torch.abs(force_predictions - test_forces))
                        if (print_info):
                            print ("Fold:" , k, sigma, llambda, EMAE, FMAE)
                        fold_mae += 0.5 * (EMAE + FMAE)
                        
                    else:
                        if (print_info):
                            print ("Fold:" , k, sigma, llambda, EMAE)
                        fold_mae += EMAE
                    
                fold_mae /= kfolds
                    
                errors.append(fold_mae.item())
                grid_locs.append([i, j])

        errors = np.array(errors)
        idxs = np.array(grid_locs)
        
        min_idx = np.argmin(errors)
        
        best_sigma = sigmas[idxs[min_idx, 0]]
        best_llambda = lambdas[idxs[min_idx, 1]]
        
        if (print_info):
            print ("Best MAE: ", errors[min_idx], "sigma = ", best_sigma, "lambda = ", best_llambda)
        
        return best_sigma, best_llambda
        
    def predict_cuda(self, X, Z, max_natoms, cells=None, forces=False, print_info=True):
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        if (self.alpha is None):
            print ("Error: must train the model first by calling train()!")
            exit()
        
        predict_energies = torch.zeros(len(X), device=self.device, dtype=torch.float64)
        predict_forces = torch.zeros(len(X), max_natoms, 3, device=self.device, dtype=torch.float64)
        
        start.record()
        
        for i in tqdm(range(0, len(X), self.nbatch)) if print_info else range(0, len(X), self.nbatch):
            
            coordinates = X[i:i + self.nbatch]
            charges = Z[i:i + self.nbatch]
            
            zbatch = len(coordinates)
            
            if (cells is not None):
                zcells = cells[i:i + self.nbatch]
            else: zcells = None
                
            data = self.format_data(coordinates, charges, cells=zcells)
            
            coordinates = data['coordinates']
            charges = data['charges']
            atomIDs = data['atomIDs']
            molIDs = data['molIDs']
            natom_counts = data['natom_counts']
            zcells = data['cells']
            
            batch_max_natoms = natom_counts.max().item()
            
            if (forces is True):
                gto, gto_derivative = self.rep.get_representation_and_derivative(coordinates, charges, atomIDs, molIDs, natom_counts, zcells)
            else:
                gto = self.rep.get_representation(coordinates, charges, atomIDs, molIDs, natom_counts, zcells)
                
            Ztest = torch.zeros(zbatch, self.nfeatures, device=self.device, dtype=torch.float64)
            
            Gtest_derivative = None
            
            if (forces is True):
                Gtest_derivative = torch.zeros(zbatch, max_natoms, 3, self.nfeatures, device=self.device, dtype=torch.float64)
                
            for e in self.elements:
            
                indexes = charges.int() == e
                
                batch_indexes = torch.where(indexes)[0].type(torch.int)
                
                sub = gto[indexes]
                
                if (sub.shape[0] == 0):
                    continue
                
                sub = project_representation(sub, self.reductors[e])
                
                sub_grad = None
                
                if (forces is True):
                    sub_grad = gto_derivative[indexes]
                    sub_grad = project_derivative(sub_grad, self.reductors[e])

                self.calculate_features(sub, e, batch_indexes, Ztest, sub_grad, Gtest_derivative)
                
            predict_energies[i:i + self.nbatch] = torch.matmul(Ztest, self.alpha)
            
            if (forces is True):
                Gtest_derivative = Gtest_derivative.reshape(zbatch * max_natoms * 3, self.nfeatures)
                predict_forces[i:i + self.nbatch] = torch.matmul(Gtest_derivative, self.alpha).reshape(zbatch, max_natoms, 3)
        
        end.record()
        torch.cuda.synchronize()
        
        if (print_info):
            print("prediction for", len(X), "molecules time: ", start.elapsed_time(end), "ms")
        
        if (forces is True):
            return (predict_energies, predict_forces)
        else:
            return predict_energies
    
    def predict_torch(self):
        raise NotImplementedError("Abstract method only.")
            
    def get_reductors(self, X, Z, cells=None, npcas=128, npca_choice=256, nsamples=4096, print_info=True):
        
        '''
        npcas: length of low-dimension projection
        npca_choice: for each batch, select at most this many atomic representations to be added to the SVD matrix
        nsamples: (minimum) total number of selected atomic representations
        '''
        
        self.reductors = {}
        
        index_set = {}
 
        for e in self.elements:
            
            idxs = []
            
            for i in range (len(Z)):
                if (e in Z[i]):
                    idxs.append(i)
                    continue
                
            idxs = np.array(idxs)
            
            index_set[e] = idxs
            
            if (print_info):
                print ("element:", e, "index_set:", index_set[e])
            subsample_indexes = np.random.choice(index_set[e], size=np.min([len(index_set[e]), 1024]))
            
            subsample_coordinates = [X[i] for i in subsample_indexes]
            subsample_charges = [Z[i] for i in subsample_indexes]
            
            inputs = []
            
            nselected = 0
            
            for i in range(0, len(subsample_coordinates), self.nbatch):
                
                # only collect nsample representations to compute the SVD
                if (nselected > nsamples):
                    break
                
                coordinates = subsample_coordinates[i:i + self.nbatch]
                charges = subsample_charges[i:i + self.nbatch]
                
                data = self.format_data(coordinates, charges)
                
                coords = data['coordinates']
                qs = data['charges']
                atomIDs = data['atomIDs']
                molIDs = data['molIDs']
                natom_counts = data['natom_counts']
                
                indexes = qs == e
                
                gto = self.rep.get_representation(coords, qs, atomIDs, molIDs, natom_counts)
         
                sub = gto[indexes]
            
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
            
            if (print_info):
                print ("SVD Matrix Size: ", mat.shape)
            
            eigvecs, eigvals, vh = torch.linalg.svd(mat.T, full_matrices=False)
        
            cev = 100 - (torch.sum(eigvals) - torch.sum(eigvals[:npcas])) / torch.sum(eigvals) * 100
        
            reductor = eigvecs[:,:npcas]
            size_from = reductor.shape[0]
            size_to = reductor.shape[1]
            
            del vh, mat
            
            torch.cuda.synchronize()
            
            if (print_info):
                print (f"{size_from} -> {size_to}  Cumulative Explained Feature Variance = {cev:6.2f} %%")
            
            self.reductors[e] = reductor
    
    def set_subtract_self_energies(self, subtract):
        self._subtract_self_energies = subtract
    
    def subtract_self_energies(self):
        return self._subtract_self_energies
    
    def set_convert_hartree2kcal(self, convert):
        self._convert_from_hartree_to_kcal = convert
        
    def convert_hartree2kcal(self):
        return self._convert_from_hartree_to_kcal
    
    def calculate_self_energy(self, Z, E):
        
        nmol = len(Z)
        
        natom_counts = torch.zeros(nmol, dtype=torch.int)
        
        for i in range(nmol):
            natom_counts[i] = Z[i].shape[0]
            
        max_atoms = natom_counts.max().item()
        
        paddedZ = torch.zeros(nmol, max_atoms, dtype=torch.int)
        
        X = torch.zeros(nmol, len(self.elements), dtype=torch.float64)
        
        for i in range(nmol):
            paddedZ[i,:natom_counts[i]] = torch.from_numpy(Z[i])

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
        
        species = torch.from_numpy(self.elements).long()
        
        self.self_energy = torch.zeros(torch.max(species) + 1, dtype=torch.float64)
    
        self.self_energy[species] = beta
        
        print ("DEBUG Self Energies: ", self.self_energy)
        
    def format_data(self, X, Z, E=None, F=None, cells=None):
    
        '''
        assumes input lists of type list(ndarrays), e.g for X: [(5, 3), (3,3), (21, 3), ...] 
        and converts them to fixed-size Torch Tensor of shape [zbatch, max_atoms, ...], e.g  [zbatch, 21, ...]
        
        also outputs natom counts, atomIDs and molIDs necessary for the CUDA implementation
        
        '''
        
        if (self.subtract_self_energies() and self.self_energy is None):
            print("ERROR: must call model.calculate_self_energy first - this computes atomic contributes to the potential energy.")
            exit()
        
        data_dict = {}
        
        # self_energy = torch.Tensor([0., -0.500273, 0., 0., 0., 0., -37.845355, -54.583861, -75.064579, -99.718730]).double()
        
        zbatch = len(X)
        
        natom_counts = torch.zeros(zbatch, dtype=torch.int32)
            
        for j in range(zbatch):
            
            coordinates = X[j]
            charges = Z[j]
    
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
        
        molIDs = torch.Tensor([])
        atomIDs = torch.Tensor([])
            
        for j in range(zbatch):
            
            charges = torch.from_numpy(Z[j]).float()
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
                all_cells[j] = cell
  
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
        data_dict['cells'] = torch.empty(0, 3, 3, device=torch.device('cuda'))
                
        if (E is not None):
            all_energies = all_energies.cuda()
            data_dict['energies'] = all_energies
            
        if (F is not None):
            all_forces = all_forces.cuda()
            data_dict['forces'] = all_forces
            
        if (cells is not None):
            all_cells = all_cells.cuda()
            data_dict['cells'] = all_cells
            
        return data_dict
