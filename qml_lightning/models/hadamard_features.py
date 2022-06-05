'''
Created on 3 May 2021

@author: Nicholas J. Browning
'''
import torch
import numpy as np
from tqdm import tqdm
from qml_lightning.features.SORF import get_SORF_diagonals, get_bias, SORFTransformCuda, CosFeatures
import time
from qml_lightning.cuda.sorf_gpu import compute_hadamard_features, sorf_matrix_gpu, compute_partial_feature_derivatives, compute_molecular_featurization_derivative
from qml_lightning.models.kernel import BaseKernel
from qml_lightning.representations.dimensionality_reduction import project_representation, project_derivative
from qml_lightning.cuda.utils_gpu import matmul_and_reduce


class HadamardFeaturesModel(BaseKernel):
    
    def __init__(self, rep=None, elements=np.array([1, 6, 7, 8]), ntransforms=1, sigma=3.0, llambda=1e-11, npcas=128,
                 nbatch_train=64, nbatch_test=64, nstacks=32):
        
        super(HadamardFeaturesModel, self).__init__(rep, elements, sigma, llambda)
        
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
        
        self.ntransforms = ntransforms
        
        self.nstacks = nstacks

        self.nbatch_train = nbatch_train
        self.nbatch_test = nbatch_test
        
        self.species = torch.from_numpy(elements).float().cuda()

        self.npcas = npcas

        self.Dmat = get_SORF_diagonals(elements, ntransforms, nstacks, npcas)
        self.bk = get_bias(elements, nstacks * npcas)
        
        self.is_trained = False
        self.alpha = torch.zeros(self.nfeatures(), device=self.device, dtype=torch.float)
    
    def nfeatures(self):
        return self.npcas * self.nstacks

    def calculate_features(self, representation, element, indexes, feature_matrix, grad=None, derivative_features=None):
        coeff_normalisation = np.sqrt(representation.shape[1]) / self.sigma

        coeffs = coeff_normalisation * sorf_matrix_gpu(representation, self.Dmat[element])

        compute_hadamard_features(coeffs, self.bk[element], indexes, feature_matrix)
        
        if (derivative_features is not None and grad is not None):
            cos_derivs = torch.zeros(coeffs.shape, device=coeffs.device, dtype=torch.float64)
            compute_partial_feature_derivatives(coeffs, self.bk[element], cos_derivs)

            compute_molecular_featurization_derivative(cos_derivs, coeff_normalisation, self.Dmat[element], grad, indexes, derivative_features)
    
    def get_finite_difference_features(self, rep, X, Z, elements, atomIDs, molIDs, natom_counts, dx=0.001):
        device = X.device
        nbatch = X.shape[0]
        max_natoms = X.shape[1]
        
        if (self.reductors is None):
            print ("Reductors not initialised.")
            exit()
        
        feature_derivative = torch.zeros(nbatch, max_natoms, 3, self.nfeatures(), dtype=torch.float64, device=device)

        for j in range(max_natoms):

            for x in range (3):
                
                X_copy = X.clone()
                
                X_copy[:, j, x] += dx
                
                gto_plus = rep.get_representation(X_copy, Z, atomIDs, molIDs, natom_counts)
                
                features_plus = torch.zeros(nbatch, self.nfeatures(), dtype=torch.float64, device=device)
                
                for e in elements:
                    indexes = Z == e
                    
                    batch_indexes = torch.where(indexes)[0].type(torch.int)
                    
                    sub = gto_plus[indexes]
      
                    sub = project_representation(sub, self.reductors[e])
                
                    self.calculate_features(sub, e, batch_indexes, features_plus)
                  
                X_copy[:, j, x] -= 2 * dx
                
                gto_minus = rep.get_representation(X_copy, Z, atomIDs, molIDs, natom_counts)
                
                features_minus = torch.zeros(nbatch, self.nfeatures(), dtype=torch.float64, device=device)
                
                for e in elements:
                    indexes = Z == e
                    
                    batch_indexes = torch.where(indexes)[0].type(torch.int)
                    
                    sub = gto_minus[indexes]
                    
                    sub = project_representation(sub, self.reductors[e])
                    
                    self.calculate_features(sub, e, batch_indexes, features_minus)
                    
                feature_derivative[:, j, x,:] = (features_plus - features_minus) / (2 * dx)
                
        return feature_derivative
    
    def SGD(self, X, Z, E, cells=None, inv_cells=None, lr=0.01, niter=100000):

        coeff_normalisation = np.sqrt(self.npcas) / self.sigma
   
        data = self.format_data(X, Z, E=E, cells=cells, inv_cells=inv_cells)
        
        coordinates = data['coordinates']
        charges = data['charges']
        atomIDs = data['atomIDs']
        molIDs = data['molIDs']
        natom_counts = data['natom_counts']
        zcells = data['cells']
        zinv_cells = data['inv_cells']
        energies = data['energies']
        
        torch_rep = self.rep.forward(coordinates, charges, atomIDs, molIDs, natom_counts, zcells, zinv_cells)
        
        permutation = torch.randperm(coordinates.shape[0])
        
        for i in range(0, niter):
            for j in range(0, coordinates.shape[0], self.nbatch_train):
            
                indices = permutation[j:j + self.nbatch_train]
        
                sub_rep = torch_rep[indices]

                Ztrain = torch.zeros(sub_rep.shape[0], self.nfeatures(), device=torch.device('cuda'), dtype=torch.float32)
            
                for e in self.elements:
                    indexes = charges[indices] == e
                    
                    batch_indexes = torch.where(indexes)[0].type(torch.int)
                    
                    sub = sub_rep[indexes]
                    
                    if (sub.shape[0] == 0): continue
             
                    sub = project_representation(sub, self.reductors[e])
        
                    coeffs = SORFTransformCuda.apply(sub, self.Dmat[e], coeff_normalisation, self.ntransforms)
                 
                    coeffs = coeffs.view(coeffs.shape[0], coeffs.shape[1] * coeffs.shape[2])
       
                    Ztrain += CosFeatures.apply(coeffs, self.bk[e], sub_rep.shape[0], batch_indexes)
            
                batch_predictions = torch.matmul(Ztrain, self.alpha)
                
                self.alpha -= lr * (1.0 / sub_rep.shape[0]) * torch.matmul(batch_predictions - energies[indices].float(), Ztrain)
            
            print ("iteration: ", i, torch.mean(torch.abs(batch_predictions - energies[indices].float())))
            print ("alpha:", self.alpha)
    
    def predict(self, X, Z, max_natoms, cells=None, inv_cells=None, forces=True, print_info=True, use_backward=True, profiler=False):

        if (not use_backward):
            return self.predict_cuda(X, Z, max_natoms, cells, forces, print_info)
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        with torch.autograd.profiler.profile(enabled=profiler, use_cuda=True, with_stack=True) as prof:
            
            start.record()

            if (self.is_trained is False):
                print ("Error: must train the model first by calling train()!")
                exit()
            
            predict_energies = torch.zeros(len(X), device=self.device, dtype=torch.float64)
            predict_forces = torch.zeros(len(X), max_natoms, 3, device=self.device, dtype=torch.float64)
            
            for i in tqdm(range(0, len(X), self.nbatch_test)) if print_info else range(0, len(X), self.nbatch_test):
                
                coordinates = X[i:i + self.nbatch_test]
                charges = Z[i:i + self.nbatch_test]

                if (cells is not None):
                    zcells = cells[i:i + self.nbatch_test]
                    z_invcells = inv_cells[i:i + self.nbatch_test]
                else: 
                    zcells = None 
                    z_invcells = None
                    
                data = self.format_data(coordinates, charges, cells=zcells, inv_cells=z_invcells)
                
                coordinates = data['coordinates']
                charges = data['charges']
                atomIDs = data['atomIDs']
                molIDs = data['molIDs']
                natom_counts = data['natom_counts']
                zcells = data['cells']
                z_invcells = data['inv_cells']

                result = self.predict_opt(coordinates, charges, atomIDs, molIDs, natom_counts, zcells, z_invcells, forces=forces, print_info=False, profiler=False)
                
                if (forces):
                    predict_energies[i:i + self.nbatch_test] = result[0]
                    
                    forces_cuda = result[1]
    
                    for j in range(forces_cuda.shape[0]):
                        predict_forces[i + j,:natom_counts[j]] = forces_cuda[j,:natom_counts[j]]
                else:
                    predict_energies[i:i + self.nbatch_test] = result
      
            end.record()
            torch.cuda.synchronize()
        
        if (profiler):
            print(prof.key_averages(group_by_stack_n=30).table(sort_by='self_cuda_time_total', row_limit=30))
            
        if (print_info):
            print("prediction for", len(X), "molecules time: ", start.elapsed_time(end), "ms")
        
        if (forces):
            return (predict_energies, predict_forces)
        else:
            return predict_energies
        
    def predict_opt(self, coordinates, charges, atomIDs, molIDs, natom_counts, cells=torch.empty(0, 3, 3, device='cuda'), inv_cells=torch.empty(0, 3, 3, device='cuda'),
                    forces=True, print_info=True, profiler=False):
        
        with torch.autograd.profiler.profile(enabled=profiler, use_cuda=True, with_stack=True) as prof:
            
            start = torch.cuda.Event(enable_timing=True)
            start1 = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        
            start.record()
            
            coeff_normalisation = np.sqrt(self.npcas) / self.sigma

            if (self.is_trained is False):
                print ("Error: must train the model first by calling train()!")
                exit()

            if (forces):
                coordinates.requires_grad = True
      
            torch_rep = self.rep.forward(coordinates, charges, atomIDs, molIDs, natom_counts, cells, inv_cells)
    
            Ztest = torch.zeros(coordinates.shape[0], self.nfeatures(), device=torch.device('cuda'), dtype=torch.float32)

            start1.record()
            
            for e in self.elements:
                 
                indexes = charges.int() == e
                 
                batch_indexes = torch.where(indexes)[0].type(torch.int)
                 
                sub = torch_rep[indexes]
                
                if (sub.shape[0] == 0): continue
                    
                sub = project_representation(sub, self.reductors[e])
            
                coeffs = SORFTransformCuda.apply(sub, self.Dmat[e], coeff_normalisation, self.ntransforms)
                  
                coeffs = coeffs.view(coeffs.shape[0], coeffs.shape[1] * coeffs.shape[2])
                 
                Ztest += CosFeatures.apply(coeffs, self.bk[e], coordinates.shape[0], batch_indexes)
   
            total_energies = torch.matmul(Ztest, self.alpha.float())
            
            if (forces):
                forces_torch, = torch.autograd.grad(-total_energies.sum(), coordinates)
            
            end.record()
            torch.cuda.synchronize()
            
        # if (profiler):
            # <FunctionEventAvg key=cudaEventCreateWithFlags self_cpu_time=6.000us cpu_time=1.500us  self_cuda_time=0.000us cuda_time=0.000us input_shapes= cpu_memory_usage=0 cuda_memory_usage=0>
            # print(prof.key_averages()[0])
            # print(prof.key_averages(group_by_stack_n=8).table(sort_by='self_cuda_time_total', row_limit=15)['Name'])
            
        if (print_info):
            print("prediction for", coordinates.shape[0], "molecules time: ", start.elapsed_time(end), "ms")
        
        if (forces):
            result = total_energies, forces_torch

        else:
            result = total_energies
    
        if (profiler):
            result = (result,) + (prof,)
            
        return result

    def save_model(self, file_name="model"):
  
        data = {'elements': self.elements,
                'ntransforms': self.ntransforms,
                'nfeatures': self.nfeatures,
                'nbatch_train': self.nbatch_train,
                'nbatch_test': self.nbatch_test,
                'npcas': self.npcas,
                'nstacks': self.nstacks,
                'is_trained': self.is_trained,
                '_subtract_self_energies':self._subtract_self_energies,
                'sigma': self.sigma,
                'llambda': self.llambda,
                'alpha': self.alpha.cpu().numpy()
                }

        if (self._subtract_self_energies):
            data['self_energies'] = self.self_energy.cpu().numpy()
            
        for e in self.elements:
            data[f'dmat_{e}'] = self.Dmat[e].cpu().numpy()
            data[f'bk_{e}'] = self.bk[e].cpu().numpy()
            data[f'reductors_{e}'] = self.reductors[e].cpu().numpy()
            
        np.save(file_name, data)
    
    def load_model(self, file_name="model"):
        
        data = np.load(file_name  if ".npy" in file_name else file_name + ".npy", allow_pickle=True)[()]
        
        self.elements = data['elements']
        self.species = torch.from_numpy(self.elements).float().cuda()
        self.sigma = data['sigma']
        self.llambda = data['llambda']
        
        self.ntransforms = data['ntransforms']
        self.nfeatures = data['nfeatures']
        self.nbatch_train = data['nbatch_train']
        self.nbatch_test = data['nbatch_test']
        
        self.npcas = data['npcas']
        self.nstacks = data['nstacks']
        
        self.is_trained = data['is_trained']
        self.alpha = torch.from_numpy(data['alpha']).double().cuda()
        
        self._subtract_self_energies = data['_subtract_self_energies']

        if (self._subtract_self_energies):
            self.self_energy = torch.from_numpy(data['self_energies']).float()
            
        self.Dmat = {}
        self.bk = {}
        self.reductors = {}
        
        for e in self.elements:
            self.Dmat[e] = torch.from_numpy(data[f'dmat_{e}']).cuda()
            self.bk[e] = torch.from_numpy(data[f'bk_{e}']).cuda()
            self.reductors[e] = torch.from_numpy(data[f'reductors_{e}']).cuda()
            

class PartitionedSORFModel(BaseKernel):
    
    def __init__(self, rep=None, elements=np.array([1, 6, 7, 8]), ntransforms=1, sigma=3.0, llambda=1e-11,
                 npcas_radial=32, npcas_angular=96,
                 nbatch_train=64, nbatch_test=64, nstacks_radial=8, nstacks_angular=32):
        
        super(PartitionedSORFModel, self).__init__(rep, elements, sigma, llambda)
        
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
        
        self.ntransforms = ntransforms
        
        self.nstacks_radial = nstacks_radial
        self.nstacks_angular = nstacks_angular
        
        self.nRs2 = rep.nRs2
        self.nRs3 = rep.nRs3

        self.nbatch_train = nbatch_train
        self.nbatch_test = nbatch_test
        
        self.species = torch.from_numpy(elements).float().cuda()
        self.nspecies = self.species.shape[0]
        
        self.nradial = self.nspecies * self.nRs2
        self.nangular = (self.nspecies * (self.nspecies + 1)) * self.nRs3
        
        self.npcas_radial = npcas_radial
        self.npcas_angular = npcas_angular
        
        self.Dmat_radial = get_SORF_diagonals(elements, ntransforms, nstacks_radial, npcas_radial)
        
        self.b_radial = get_bias(elements, nstacks_radial * npcas_radial)
        
        self.Dmat_angular = get_SORF_diagonals(elements, ntransforms, nstacks_angular, npcas_angular)
        
        self.b_angular = get_bias(elements, nstacks_angular * npcas_angular)
        
        self.is_trained = False
        self.alpha = torch.zeros(self.nfeatures(), device=self.device, dtype=torch.float)
    
    def get_reductors(self, X, Q, cells=None, inv_cells=None, radial_npcas=32, angular_npcas=96,
                      npca_choice=256, nsamples=4096, print_info=True):
        
        '''
        npcas: length of low-dimension projection
        npca_choice: for each batch, select at most this many atomic representations to be added to the SVD matrix
        nsamples: maximum total number of selected atomic representations
        '''
        
        self.radial_reductors = {}
        self.angular_reductors = {}
        
        index_set = {}
 
        for e in self.elements:
            
            idxs = []
            
            for i in range (len(Q)):
                if (e in Q[i]):
                    idxs.append(i)
                    continue
                
            idxs = np.array(idxs)
            
            index_set[e] = idxs
            
            subsample_indexes = np.random.choice(index_set[e], size=np.min([len(index_set[e]), 1024]))
            
            subsample_coordinates = [X[i] for i in subsample_indexes]
            subsample_charges = [Q[i] for i in subsample_indexes]
            
            inputs_radial = []
            inputs_angular = []
            
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
                
                indexes = qs == e
                
                gto = self.rep.get_representation(coords, qs, atomIDs, molIDs, natom_counts, zcells, zinv_cells)
                
                sub = gto[indexes]
                
                sub = gto[indexes]
                
                if (sub.shape[0] == 0):
                    continue
                
                sub_radial = sub[:,:self.nradial]
                sub_angular = sub [:, self.nradial:]
                
                perm = torch.randperm(sub.size(0))
                idx = perm[:npca_choice]
        
                choice_input_radial = sub_radial[idx]
                choice_input_angular = sub_angular[idx]
                
                nselected += choice_input_radial.shape[0]
                
                inputs_radial.append(choice_input_radial)
                inputs_angular.append(choice_input_angular)
            
            if (len(inputs_radial) == 0):
                continue
            
            mat_radial = torch.cat(inputs_radial)
            mat_angular = torch.cat(inputs_angular)
            
            eigvecs, eigvals, vh = torch.linalg.svd(mat_radial.T, full_matrices=False)
        
            cev = 100 - (torch.sum(eigvals) - torch.sum(eigvals[:radial_npcas])) / torch.sum(eigvals) * 100
        
            reductor_radial = eigvecs[:,:radial_npcas]
            
            size_from = reductor_radial.shape[0]
            size_to = reductor_radial.shape[1]
            
            print ("--- Radial Reduction ---")
            if (print_info):
                print (f"element {e}: {size_from} -> {size_to}  Cumulative Explained Feature Variance = {cev:6.2f} %")
                
            if (reductor_radial.shape[1] < radial_npcas):
                print (f"ERROR - not enough atomic environments in input dataset for element {e}. Either increase npca_choice or provide more structures.")
                exit()
            
            print ("--- Angular Reduction ---")
            
            eigvecs, eigvals, vh = torch.linalg.svd(mat_angular.T, full_matrices=False)
        
            cev = 100 - (torch.sum(eigvals) - torch.sum(eigvals[:angular_npcas])) / torch.sum(eigvals) * 100
            
            reductor_angular = eigvecs[:,:angular_npcas]
            
            size_from = reductor_angular.shape[0]
            size_to = reductor_angular.shape[1]
            
            if (print_info):
                print (f"element {e}: {size_from} -> {size_to}  Cumulative Explained Feature Variance = {cev:6.2f} %")
                
            if (reductor_angular.shape[1] < angular_npcas):
                print (f"ERROR - not enough atomic environments in input dataset for element {e}. Either increase npca_choice or provide more structures.")
                exit()
            
            torch.cuda.synchronize()
            
            self.radial_reductors[e] = reductor_radial
            self.angular_reductors[e] = reductor_angular

    def nfeatures(self):
        return self.nstacks_radial * self.npcas_radial + self.nstacks_angular * self.npcas_angular 
        
    def build_Z_components(self, X, Q, E=None, F=None, cells=None, inv_cells=None, print_info=True, cpu_solve=False, ntiles=1, use_specialized_matmul=False):
        
        '''
        X: list of coordinates : numpy arrays of shape [natom_i, 3]
        Z: list of charges: numpy arrays of shape [natom_i]
        
        cpu_solve: set to True to store Z^TZ on the CPU, and solve on the CPU
        ntiles: > 1 sets the Z^TZ matrix to be constructed in (self.nfeatures()/ntiles) features per batch.
         
        print_info: Boolean defining whether or not to print timings + progress bar
        
        returns: Z^TZ, ZY
        
        '''
        
        if (self.radial_reductors is None):
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
                    gto = self.rep.get_representation(coordinates, charges, atomIDs, molIDs, natom_counts, zcells, zinv_cells)
                else:
                    gto, gto_derivative = self.rep.get_representation_and_derivative(coordinates, charges, atomIDs, molIDs,
                                                                                     natom_counts, zcells, zinv_cells)

                nfeatures_radial = self.nstacks_radial * self.npcas_radial
                nfeatures_angular = self.nstacks_angular * self.npcas_angular
                
                Ztrain_radial = torch.zeros(zbatch, nfeatures_radial, device=self.device, dtype=torch.float64)
                Ztrain_angular = torch.zeros(zbatch, nfeatures_angular, device=self.device, dtype=torch.float64)
                
                Gtrain_derivative = None
                
                if (F is not None):
                    Gtrain_derivative_radial = torch.zeros(zbatch, max_natoms, 3, nfeatures_radial, device=self.device, dtype=torch.float64)
                    Gtrain_derivative_angular = torch.zeros(zbatch, max_natoms, 3, nfeatures_angular, device=self.device, dtype=torch.float64)
                    
                for e in self.elements:
                
                    indexes = charges.int() == e
                    
                    batch_indexes = torch.where(indexes)[0].type(torch.int)
                    
                    sub = gto[indexes]
                    
                    if (sub.shape[0] == 0):
                        continue
                    
                    sub_radial = sub[:,:self.nradial]
                    sub_angular = sub[:, self.nradial:]
                    
                    sub_radial = project_representation(sub_radial, self.radial_reductors[e])
                    sub_angular = project_representation(sub_angular, self.angular_reductors[e])
                    
                    if (F is not None):
                        sub_grad = gto_derivative[indexes]

                        sub_grad_radial = sub_grad[...,:self.nradial]
                        sub_grad_angular = sub_grad[..., self.nradial:]
                        
                        sub_grad_radial = project_derivative(sub_grad[:,:self.nradial], self.radial_reductors[e])
                        sub_grad_angular = project_derivative(sub_grad[:, self.nradial:], self.angular_reductors[e])
         
                        self.calculate_features_radial(sub_radial, e, batch_indexes, Ztrain_radial, sub_grad_radial, Gtrain_derivative_radial)
                        self.calculate_features_angular(sub_angular, e, batch_indexes, Ztrain_angular, sub_grad_angular, Gtrain_derivative_angular)
                    else:
                        self.calculate_features_radial(sub_radial, e, batch_indexes, Ztrain_radial)
                        self.calculate_features_angular(sub_angular, e, batch_indexes, Ztrain_angular)
                        
                if (E is None):
                    Ztrain_radial.fill_(0)  # hack to set all energy features to 0, such that they do not contribute to Z.T Z
                    Ztrain_angular.fill_(0)  # hack to set all energy features to 0, such that they do not contribute to Z.T Z
                
                Ztrain = torch.cat((Ztrain_radial, Ztrain_angular), dim=1)
                
                if (F is not None):
                    Gtrain_derivative_radial = Gtrain_derivative_radial.reshape(zbatch * max_natoms * 3, nfeatures_radial)
                    Gtrain_derivative_angular = Gtrain_derivative_angular.reshape(zbatch * max_natoms * 3, nfeatures_angular)
                    
                    Ztrain = torch.cat((Ztrain, torch.cat((Gtrain_derivative_radial, Gtrain_derivative_angular), dim=1)), dim=0)
                
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
                del gto
                del sub
                
                if (F is not None):
                    del Gtrain_derivative
                    del gto_derivative
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
    
    def calculate_features_radial(self, radial_rep, element, indexes, feature_matrix, grad=None, derivative_features=None):
     
        coeff_normalisation_radial = np.sqrt(radial_rep.shape[1]) / self.sigma

        coeffs_radial = coeff_normalisation_radial * sorf_matrix_gpu(radial_rep, self.Dmat_radial[element])

        compute_hadamard_features(coeffs_radial, self.b_radial[element], indexes, feature_matrix)
        
        if (derivative_features is not None and grad is not None):
            cos_derivs = torch.zeros(coeffs_radial.shape, device=coeffs_radial.device, dtype=torch.float64)
            compute_partial_feature_derivatives(coeffs_radial, self.b_radial[element], cos_derivs)

            compute_molecular_featurization_derivative(cos_derivs, coeff_normalisation_radial, self.Dmat_radial[element], grad, indexes, derivative_features)
    
    def calculate_features_angular(self, angular_rep, element, indexes, feature_matrix, grad=None, derivative_features=None):
    
        coeff_normalisation_angular = np.sqrt(angular_rep.shape[1]) / self.sigma

        coeffs_angular = coeff_normalisation_angular * sorf_matrix_gpu(angular_rep, self.Dmat_angular[element])

        compute_hadamard_features(coeffs_angular, self.b_angular[element], indexes, feature_matrix)
        
        if (derivative_features is not None and grad is not None):
            cos_derivs = torch.zeros(coeffs_angular.shape, device=coeffs_angular.device, dtype=torch.float64)
            compute_partial_feature_derivatives(coeffs_angular, self.b_angular[element], cos_derivs)

            compute_molecular_featurization_derivative(cos_derivs, coeff_normalisation_angular, self.Dmat_angular[element], grad, indexes, derivative_features)
    
    def predict(self, X, Z, max_natoms, cells=None, inv_cells=None, forces=True, print_info=True, use_backward=True, profiler=False):

        if (not use_backward):
            return self.predict_cuda(X, Z, max_natoms, cells, forces, print_info)
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        with torch.autograd.profiler.profile(enabled=profiler, use_cuda=True, with_stack=True) as prof:
            
            start.record()

            if (self.is_trained is False):
                print ("Error: must train the model first by calling train()!")
                exit()
            
            predict_energies = torch.zeros(len(X), device=self.device, dtype=torch.float64)
            predict_forces = torch.zeros(len(X), max_natoms, 3, device=self.device, dtype=torch.float64)
            
            for i in tqdm(range(0, len(X), self.nbatch_test)) if print_info else range(0, len(X), self.nbatch_test):
                
                coordinates = X[i:i + self.nbatch_test]
                charges = Z[i:i + self.nbatch_test]

                if (cells is not None):
                    zcells = cells[i:i + self.nbatch_test]
                    z_invcells = inv_cells[i:i + self.nbatch_test]
                else: 
                    zcells = None 
                    z_invcells = None
                    
                data = self.format_data(coordinates, charges, cells=zcells, inv_cells=z_invcells)
                
                coordinates = data['coordinates']
                charges = data['charges']
                atomIDs = data['atomIDs']
                molIDs = data['molIDs']
                natom_counts = data['natom_counts']
                zcells = data['cells']
                z_invcells = data['inv_cells']

                result = self.predict_opt(coordinates, charges, atomIDs, molIDs, natom_counts, zcells, z_invcells, forces=forces, print_info=False, profiler=False)
                
                if (forces):
                    predict_energies[i:i + self.nbatch_test] = result[0]
                    
                    forces_cuda = result[1]
    
                    for j in range(forces_cuda.shape[0]):
                        predict_forces[i + j,:natom_counts[j]] = forces_cuda[j,:natom_counts[j]]
                else:
                    predict_energies[i:i + self.nbatch_test] = result
      
            end.record()
            torch.cuda.synchronize()
        
        if (profiler):
            print(prof.key_averages(group_by_stack_n=30).table(sort_by='self_cuda_time_total', row_limit=30))
            
        if (print_info):
            print("prediction for", len(X), "molecules time: ", start.elapsed_time(end), "ms")
        
        if (forces):
            return (predict_energies, predict_forces)
        else:
            return predict_energies
        
    def predict_opt(self, coordinates, charges, atomIDs, molIDs, natom_counts, cells=torch.empty(0, 3, 3, device='cuda'), inv_cells=torch.empty(0, 3, 3, device='cuda'),
                    forces=True, print_info=True, profiler=False):
        
        with torch.autograd.profiler.profile(enabled=profiler, use_cuda=True, with_stack=True) as prof:
            
            start = torch.cuda.Event(enable_timing=True)
            start1 = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        
            start.record()
            
            cn_radial = np.sqrt(self.npcas_radial) / self.sigma
            cn_angular = np.sqrt(self.npcas_angular) / self.sigma
            
            if (self.is_trained is False):
                print ("Error: must train the model first by calling train()!")
                exit()

            if (forces):
                coordinates.requires_grad = True
      
            torch_rep = self.rep.forward(coordinates, charges, atomIDs, molIDs, natom_counts, cells, inv_cells)
            
            nfeatures_radial = self.nstacks_radial * self.npcas_radial
            nfeatures_angular = self.nstacks_angular * self.npcas_angular
                
            Ztest_radial = torch.zeros(coordinates.shape[0], nfeatures_radial, device=torch.device('cuda'), dtype=torch.float32)
            Ztest_angular = torch.zeros(coordinates.shape[0], nfeatures_angular, device=torch.device('cuda'), dtype=torch.float32)

            start1.record()
            
            for e in self.elements:
                 
                indexes = charges.int() == e
                 
                batch_indexes = torch.where(indexes)[0].type(torch.int)
                 
                sub = torch_rep[indexes]
                
                if (sub.shape[0] == 0): continue
                
                sub_radial = sub[:,:self.nradial]
                sub_angular = sub[:, self.nradial:]
                
                sub_radial = project_representation(sub_radial, self.radial_reductors[e])
                sub_angular = project_representation(sub_angular, self.angular_reductors[e])
            
                coeffs_radial = SORFTransformCuda.apply(sub_radial, self.Dmat_radial[e], cn_radial, self.ntransforms)
                coeffs_radial = coeffs_radial.view(coeffs_radial.shape[0], coeffs_radial.shape[1] * coeffs_radial.shape[2])
                Ztest_radial += CosFeatures.apply(coeffs_radial, self.b_radial[e], coordinates.shape[0], batch_indexes)
                
                coeffs_angular = SORFTransformCuda.apply(sub_angular, self.Dmat_angular[e], cn_angular, self.ntransforms)
                coeffs_angular = coeffs_angular.view(coeffs_angular.shape[0], coeffs_angular.shape[1] * coeffs_angular.shape[2])
                Ztest_angular += CosFeatures.apply(coeffs_angular, self.b_angular[e], coordinates.shape[0], batch_indexes)
   
            Ztest = torch.cat((Ztest_radial, Ztest_angular), dim=1)
            
            total_energies = torch.matmul(Ztest, self.alpha.float())
            
            if (forces):
                forces_torch, = torch.autograd.grad(-total_energies.sum(), coordinates)
            
            end.record()
            torch.cuda.synchronize()
            
        # if (profiler):
            # <FunctionEventAvg key=cudaEventCreateWithFlags self_cpu_time=6.000us cpu_time=1.500us  self_cuda_time=0.000us cuda_time=0.000us input_shapes= cpu_memory_usage=0 cuda_memory_usage=0>
            # print(prof.key_averages()[0])
            # print(prof.key_averages(group_by_stack_n=8).table(sort_by='self_cuda_time_total', row_limit=15)['Name'])
            
        if (print_info):
            print("prediction for", coordinates.shape[0], "molecules time: ", start.elapsed_time(end), "ms")
        
        if (forces):
            result = total_energies, forces_torch

        else:
            result = total_energies
    
        if (profiler):
            result = (result,) + (prof,)
            
        return result

    def save_model(self, file_name="model"):
  
        data = {'elements': self.elements,
                'ntransforms': self.ntransforms,
                'nfeatures': self.nfeatures,
                'nbatch_train': self.nbatch_train,
                'nbatch_test': self.nbatch_test,
                'npcas': self.npcas,
                'nstacks': self.nstacks,
                'is_trained': self.is_trained,
                '_subtract_self_energies':self._subtract_self_energies,
                'sigma': self.sigma,
                'llambda': self.llambda,
                'alpha': self.alpha.cpu().numpy()
                }

        if (self._subtract_self_energies):
            data['self_energies'] = self.self_energy.cpu().numpy()
            
        for e in self.elements:
            data[f'dmat_{e}'] = self.Dmat[e].cpu().numpy()
            data[f'bk_{e}'] = self.bk[e].cpu().numpy()
            data[f'reductors_{e}'] = self.reductors[e].cpu().numpy()
            
        np.save(file_name, data)
    
    def load_model(self, file_name="model"):
        
        data = np.load(file_name  if ".npy" in file_name else file_name + ".npy", allow_pickle=True)[()]
        
        self.elements = data['elements']
        self.species = torch.from_numpy(self.elements).float().cuda()
        self.sigma = data['sigma']
        self.llambda = data['llambda']
        
        self.ntransforms = data['ntransforms']
        self.nfeatures = data['nfeatures']
        self.nbatch_train = data['nbatch_train']
        self.nbatch_test = data['nbatch_test']
        
        self.npcas = data['npcas']
        self.nstacks = data['nstacks']
        
        self.is_trained = data['is_trained']
        self.alpha = torch.from_numpy(data['alpha']).double().cuda()
        
        self._subtract_self_energies = data['_subtract_self_energies']

        if (self._subtract_self_energies):
            self.self_energy = torch.from_numpy(data['self_energies']).float()
            
        self.Dmat = {}
        self.bk = {}
        self.reductors = {}
        
        for e in self.elements:
            self.Dmat[e] = torch.from_numpy(data[f'dmat_{e}']).cuda()
            self.bk[e] = torch.from_numpy(data[f'bk_{e}']).cuda()
            self.reductors[e] = torch.from_numpy(data[f'reductors_{e}']).cuda()
        
