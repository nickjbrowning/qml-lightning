'''
Created on 3 May 2021

@author: Nicholas J. Browning
'''
import torch
import numpy as np

from qml_lightning.features.SORF import get_SORF_diagonals, get_bias, get_SORF_coefficients, get_features, get_feature_derivatives, SORFTransformCuda
from qml_lightning.representations.dimensionality_reduction import project_representation, project_derivative

from qml_lightning.representations.EGTO import get_egto
from qml_lightning.representations.EGTO import ElementalGTO


class RandomFeaturesModel():
    
    def __init__(self, elements=np.array([1, 6, 7, 8]), ntransforms=1, sigma=16.0, llambda=1e-10, nfeatures=8192, npcas=256, nbatch=1024,
                 npca_choice=1024, ngaussians=20, eta=2.3, lmax=2, rcut=6.0):
        
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')

        self.npcas = npcas
        self.ntransforms = ntransforms
        self.sigma = sigma
        self.llambda = llambda
        self.nfeatures = nfeatures
        self.nbatch = nbatch
        self.npca_choice = npca_choice
        
        self.elements = elements
        self.species = torch.from_numpy(elements).float().cuda()
        
        self.ngaussians = ngaussians
        self.eta = eta
        self.lmax = lmax
        self.rcut = rcut
        
        self.feature_normalisation = np.sqrt(2.0 / nfeatures)
        self.coeff_normalisation = np.sqrt(npcas) / sigma
        
        self.nstacks = int(float(nfeatures) / npcas)

        self.Dmat = get_SORF_diagonals(elements, ntransforms, nfeatures, npcas)
        self.bk = get_bias(elements, nfeatures)
        
        self.reductors = None
        self._reductors_initialised = False
        
        self.alpha = None
        
        self.fingerprint = ElementalGTO(species=elements, low_cutoff=0.0, high_cutoff=rcut, n_gaussians=ngaussians, eta=eta, Lmax=lmax, device=self.device)
        
        self.self_energy = None
        
        self.convert_from_hartree_to_kcal = True

    def format_data(self, X, Z, E=None, F=None):
    
        '''
        assumes input lists of type list(ndarrays), e.g for X: [(5, 3), (3,3), (21, 3), ...] 
        and converts them to fixed-size Torch Tensor of shape [zbatch, max_atoms, ...], e.g  [zbatch, 21, ...]
        
        also outputs natom counts, atomIDs and molIDs necessary for the CUDA implementation
        
        '''
        
        if (self.self_energy is None):
            print("ERROR: must call model.calculate_self_energy first - this computes atomic contributes to the potential energy.")
            exit()
        
        data_dict = {}
        
        # self_energy = torch.Tensor([0., -0.500273, 0., 0., 0., 0., -37.845355, -54.583861, -75.064579, -99.718730]).double()
        hartree2kcalmol = 627.5095
        
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
        
        molIDs = torch.Tensor([])
        atomIDs = torch.Tensor([])
            
        for j in range(zbatch):
            
            charges = torch.from_numpy(Z[j]).float()
            coordinates = torch.from_numpy(X[j]).float()
     
            natoms = natom_counts[j]
            
            all_charges[j,:natoms] = charges
            all_coordinates[j,:natoms,:] = coordinates
            
            if (F is not None):
                forces = torch.from_numpy(F[j]).double()
                
                if (self.convert_from_hartree_to_kcal):
                    forces *= hartree2kcalmol
                    
                all_forces[j,:natoms,:] = forces
            
            molID = torch.empty(natoms, dtype=torch.int32).fill_(j)
            atomID = torch.arange(0, natoms)
            
            molIDs = torch.cat((molIDs, molID), dim=0)
            atomIDs = torch.cat((atomIDs, atomID), dim=0)
            
            if (E is not None):
             
                energy = all_energies[j]
                    
                self_interaction = self.self_energy[charges.long()].sum(axis=0) * hartree2kcalmol
                
                if (self.convert_from_hartree_to_kcal):
                    energy *= hartree2kcalmol
                    
                energy = energy - self_interaction
                
                all_energies[j] = energy
  
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
        
        if (E is not None):
            all_energies = all_energies.cuda()
            data_dict['energies'] = all_energies
            
        if (F is not None):
            all_forces = all_forces.cuda()
            data_dict['forces'] = all_forces
            
        return data_dict

    def get_reductors(self, X, Z, print_info=True):
    
        reductors = {}
        
        for e in self.elements:
            
            inputs = []
            
            for i in range(0, len(X), self.nbatch):
                
                coordinates = X[i:i + self.nbatch]
                charges = Z[i:i + self.nbatch]
                
                data = self.format_data(coordinates, charges)
                
                charges = data['charges']
                coordinates = data['coordinates']
                atomIDs = data['atomIDs']
                molIDs = data['molIDs']
                natom_counts = data['natom_counts']
            
                gto = get_egto(coordinates, charges, atomIDs, molIDs, natom_counts,
                               self.species, self.ngaussians, self.eta, self.lmax, self.rcut, False)
            
                indexes = charges == e
    
                sub = gto[indexes]
            
                if (sub.shape[0] == 0):
                    continue
                
                perm = torch.randperm(sub.size(0))
                idx = perm[:self.npca_choice]
        
                choice_input = sub[idx]
                
                inputs.append(choice_input)
            
            if (len(inputs) == 0):
                continue
            
            mat = torch.cat(inputs)
    
            eigvecs, eigvals, vh = torch.linalg.svd(mat.T, full_matrices=False, compute_uv=True)
        
            cev = 100 - (torch.sum(eigvals) - torch.sum(eigvals[:self.npcas])) / torch.sum(eigvals) * 100
        
            reductor = eigvecs[:,:self.npcas]
            size_from = reductor.shape[0]
            size_to = reductor.shape[1]
            
            if (print_info):
                print (f"{size_from} -> {size_to}  Cumulative Explained Feature Variance = {cev:6.2f} %%")
            
            reductors[e] = reductor
        
        self._reductors_initialised = True
        
        self.reductors = reductors
        
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
        
        beta = torch.lstsq(torch.matmul(X.T, energies[:, None]), XTX).solution[:, 0]
        
        print ("Self Energies: ", beta)
        
        species = torch.from_numpy(self.elements).long()
        
        self.self_energy = torch.zeros(torch.max(species) + 1, dtype=torch.float64)
        
        self.self_energy[species] = beta

    def hyperparam_opt_nested_cv(self, X, Z, E, sigmas, lambdas, kfolds):
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
                    
                    data = self.format_data(Xtest, Ztest, Etest)
                    
                    test_energies = data['energies']
                    max_natoms = data['natom_counts'].max().item()
                    
                    self.sigma = sigma
                    self.coeff_normalisation = np.sqrt(self.npcas) / sigma
                    self.llambda = llambda
                    
                    self.train(Xtrain, Ztrain, Etrain, print_info=False)
                    
                    energy_predictions = self.predict_cuda(Xtest, Ztest, max_natoms, forces=False, print_info=False)
   
                    EMAE = torch.mean(torch.abs(energy_predictions - test_energies))
      
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
        
        print ("Best Kfold MAE: ", errors[min_idx], "sigma = ", best_sigma, "lambda = ", best_llambda)
    
    def mc_optimize_parameters(self, Xtrain, Ztrain, Etrain, Xval, Zval, Eval, niter=500):
        from tqdm import tqdm
        from copy import deepcopy

        def sample_discrete(x):
            if x <= 1:
                return 2
            else:
                return int(x + np.random.choice([-1, 1]))
        
        def sample_continuous(x):
        
            dx = 0.05
            return x * np.random.normal(1.0, 0.15)

        params = [ 20.0, 1e-10, 20, 2.0]
        
        samplers = [sample_continuous, sample_continuous, sample_discrete, sample_continuous]
        
        data = self.format_data(Xval, Zval, Eval)
    
        test_energies = data['energies']
        max_natoms = data['natom_counts'].max().item()
        
        self.sigma = params[0]
        self.llambda = params[1]
        self.ngaussians = params[2]
        self.eta = params[3]
            
        self.get_reductors(Xtrain, Ztrain, print_info=False)
        self.train(Xtrain, Ztrain, Etrain, print_info=False)
    
        energy_predictions = self.predict_cuda(Xval, Zval, max_natoms, forces=False, print_info=False)

        old_cost = torch.mean(torch.abs(energy_predictions - test_energies))
    
        for i in tqdm(range(niter)):
            
            idx = np.random.choice(len(params))

            sampler = samplers[idx]
            param = params[idx]
        
            new_params = deepcopy(params)
        
            new_params[idx] = sampler(param)
            
            self.sigma = params[0]
            self.llambda = params[1]
            self.ngaussians = params[2]
            self.eta = params[3]
            
            self.get_reductors(Xtrain, Ztrain, print_info=False)
            
            self.train(Xtrain, Ztrain, Etrain, print_info=False)
    
            energy_predictions = self.predict_cuda(Xval, Zval, max_natoms, forces=False, print_info=False)
   
            EMAE = torch.mean(torch.abs(energy_predictions - test_energies))
            
            if (old_cost == -1 or EMAE < old_cost):
                params = new_params
                old_cost = EMAE
                print ("iteration", i, "new params accepted:", params, EMAE)
        
    def train(self, X, Z, E, F=None, print_info=True):
            
        if (self._reductors_initialised == False):
            print("ERROR: Must call model.get_reductors() first to initialize the projection matrices.")
            exit()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        ZTZ = torch.zeros(self.nfeatures, self.nfeatures, device=self.device, dtype=torch.float64)
        ZtrainY = torch.zeros(self.nfeatures, 1, device=self.device, dtype=torch.float64)
        
        if (F is not None):
            GtrainY = torch.zeros(self.nfeatures, 1, device=self.device, dtype=torch.float64)
        
        start.record()
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
                gto = get_egto(coordinates, charges, atomIDs, molIDs, natom_counts,
                               self.species, self.ngaussians, self.eta, self.lmax, self.rcut, False)
            else:
                gto, gto_derivative = get_egto(coordinates, charges, atomIDs, molIDs, natom_counts,
                               self.species, self.ngaussians, self.eta, self.lmax, self.rcut, True)
            
            Ztrain = torch.zeros(zbatch, self.nfeatures, device=self.device, dtype=torch.float64)
            
            if (F is not None):
                Gtrain_derivative = torch.zeros(zbatch, max_natoms, 3, self.nfeatures, device=self.device, dtype=torch.float64)
                
            for e in self.elements:
            
                indexes = charges.int() == e
                
                batch_indexes = torch.where(indexes)[0].type(torch.int)
                
                sub = gto[indexes]
                
                if (sub.shape[0] == 0):
                    continue
                
                sub = project_representation(sub, self.reductors[e])
                
                coeffs = get_SORF_coefficients(sub, self.nfeatures, self.Dmat[e], self.coeff_normalisation)
                    
                Ztrain += get_features(coeffs, self.bk[e], batch_indexes, zbatch).double()
                
                if (F is not None):
                    sub_grad = gto_derivative[indexes]
                    sub_grad = project_derivative(sub_grad, self.reductors[e])
                    Gtrain_derivative -= get_feature_derivatives(coeffs, self.bk[e], self.Dmat[e], sub_grad, batch_indexes, zbatch, self.coeff_normalisation).double()
            
            ZtrainY += torch.matmul(Ztrain.T, energies[:, None])
            
            ZTZ += torch.matmul(Ztrain.T, Ztrain)
            
            if (F is not None):
                Gtrain_derivative = Gtrain_derivative.reshape(zbatch * max_natoms * 3, self.nfeatures)
                GtrainY += torch.matmul(Gtrain_derivative.T, forces.flatten()[:, None])
                ZTZ += torch.matmul(Gtrain_derivative.T, Gtrain_derivative)
        
        end.record()
        torch.cuda.synchronize()
        if (print_info):
            print("ZTZ time: ", start.elapsed_time(end), "ms")
        
        ZTZ[torch.eye(self.nfeatures).bool()] += self.llambda
        
        Y = ZtrainY
        
        if (F is not None):
            Y = Y + GtrainY
        
        start.record()
        self.alpha = torch.solve(Y, ZTZ).solution[:, 0]
        end.record()
        torch.cuda.synchronize()
        if (print_info):
            print("coefficients time: ", start.elapsed_time(end), "ms")
        
    def predict_torch(self, X, Z, forces=False, print_info=True):
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        if (self.alpha is None):
            print ("Error: must train the model first by calling train()!")
            exit()

        energy_predictions = []
        
        if (forces):
            force_predictions = []
        
        start.record()
        
        for i in range(len(X)):
            
            coordinates = torch.from_numpy(X[i]).float().cuda()
            charges = torch.from_numpy(Z[i]).int().cuda()
            
            coordinates = coordinates.unsqueeze(0)
            charges = charges.unsqueeze(0)
            
            coordinates.requires_grad = True
            
            rep = self.fingerprint.forward(coordinates, charges.int())
        
            Ztest = torch.zeros(1, self.nfeatures, device=self.device, dtype=torch.float64)
            
            for e in self.elements:
          
                indexes = charges.int() == e
                
                batch_indexes = torch.where(indexes)[0].type(torch.int)
                
                sub = rep[indexes]
                
                if (sub.shape[0] == 0):
                    continue
                
                sub = project_representation(sub, self.reductors[e])
                
                sub = sub.repeat(1, self.nstacks).reshape(sub.shape[0], self.nstacks, self.npcas)
                
                coeffs = self.coeff_normalisation * SORFTransformCuda.apply(self.Dmat[e] * sub)
                coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1] * coeffs.shape[2])
                
                Ztest.index_add_(0, batch_indexes, self.feature_normalisation * torch.cos(coeffs + self.bk[e]).double())
        
            energy = torch.matmul(Ztest, self.alpha)
                
            energy_predictions.append(energy)
            
            if (forces):
                neggrad, = torch.autograd.grad(-energy.sum(), coordinates, retain_graph=True)
                force_predictions.append(neggrad)
            
        end.record()
        torch.cuda.synchronize()
        
        if (print_info):
            print("prediction for", len(X), "molecules time: ", start.elapsed_time(end), "ms")
        
        if (forces):
            return energy_predictions, force_predictions
        else:
            return energy_predictions
        
    def predict_cuda(self, X, Z, max_natoms, forces=False, print_info=True):
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        if (self.alpha is None):
            print ("Error: must train the model first by calling train()!")
            exit()
        
        predict_energies = torch.zeros(len(X), device=self.device, dtype=torch.float64)
        predict_forces = torch.zeros(len(X), max_natoms, 3, device=self.device, dtype=torch.float64)
        
        start.record()
        
        for i in range(0, len(X), self.nbatch):
            
            coordinates = X[i:i + self.nbatch]
            charges = Z[i:i + self.nbatch]

            zbatch = len(coordinates)
            
            data = self.format_data(coordinates, charges)
            
            coordinates = data['coordinates']
            charges = data['charges']
            atomIDs = data['atomIDs']
            molIDs = data['molIDs']
            natom_counts = data['natom_counts']
            
            batch_max_natoms = natom_counts.max().item()
            
            if (forces is False):
                gto = get_egto(coordinates, charges, atomIDs, molIDs, natom_counts,
                               self.species, self.ngaussians, self.eta, self.lmax, self.rcut, False)
            else:
                gto, gto_derivative = get_egto(coordinates, charges, atomIDs, molIDs, natom_counts,
                               self.species, self.ngaussians, self.eta, self.lmax, self.rcut, True)
                
            Ztest = torch.zeros(zbatch, self.nfeatures, device=self.device, dtype=torch.float64)
            
            if (forces is True):
                Gtest_derivative = torch.zeros(zbatch, max_natoms, 3, self.nfeatures, device=self.device, dtype=torch.float64)
                
            for e in self.elements:
            
                indexes = charges.int() == e
                
                batch_indexes = torch.where(indexes)[0].type(torch.int)
                
                sub = gto[indexes]
                
                if (sub.shape[0] == 0):
                    continue
                
                sub = project_representation(sub, self.reductors[e])
                
                coeffs = get_SORF_coefficients(sub, self.nfeatures, self.Dmat[e], self.coeff_normalisation)
                    
                Ztest += get_features(coeffs, self.bk[e], batch_indexes, zbatch).double()
                
                if (forces is True):
                    sub_grad = gto_derivative[indexes]
                    sub_grad = project_derivative(sub_grad, self.reductors[e])
                    Gtest_derivative[:,:batch_max_natoms,:,:] -= get_feature_derivatives(coeffs, self.bk[e], self.Dmat[e], sub_grad, batch_indexes, zbatch, self.coeff_normalisation).double()
            
            predict_energies[i:i + self.nbatch] = torch.matmul(Ztest, self.alpha)
            
            if (forces is True):
                Gtrain_derivative = Gtest_derivative.reshape(zbatch * max_natoms * 3, self.nfeatures)
                predict_forces[i:i + self.nbatch] = torch.matmul(Gtrain_derivative, self.alpha).reshape(zbatch, max_natoms, 3)
        
        end.record()
        torch.cuda.synchronize()
        if (print_info):
            print("prediction for", len(X), "molecules time: ", start.elapsed_time(end), "ms")
        
        if (forces):
            return predict_energies, predict_forces
        else:
            return predict_energies
    
