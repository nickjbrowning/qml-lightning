
import torch
import numpy as np
import gto_cuda
import hadamard_flavours

from qml_lightning.representations.EGTO import get_elemental_gto
from qml_lightning.features.SORF import *
from time import time


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


def get_element_types(coordinates: torch.Tensor, charges: torch.Tensor, species: torch.Tensor):
    return gto_cuda.get_element_types_gpu(coordinates, charges, species)


def compute_elemental_gto_shared(coordinates: torch.Tensor, charges: torch.Tensor, species: torch.Tensor, ngaussians: int, eta: float, lmax: int, rcut: float, gradients=False):
    
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
    
    print (element_types)
    
    if (gradients):
        start.record()
        output = gto_cuda.elemental_gto_gpu_shared(coordinates, charges, species, element_types, mbody_list,
                                        orbital_components, orbital_weights, orbital_indexes, offset, eta, lmax, rcut, gradients)
        rep, grad = output[0], output[1]
        end.record()
        torch.cuda.synchronize()
        return rep, grad
    else: 
        start.record()
        output = gto_cuda.elemental_gto_gpu_shared(coordinates, charges, species, element_types, mbody_list,
                                        orbital_components, orbital_weights, orbital_indexes, offset, eta, lmax, rcut, gradients)
        end.record()
        torch.cuda.synchronize()
    
        return output[0]


def sample_sorf_elemental_basis(nfeatures, repsize, Dmat, sigma=20.0, elements=[1.0, 6.0, 8.0], normalize=True):
    from scipy.linalg import hadamard

    M = nfeatures
    d = repsize
    
    W = dict()
    b = dict()
    
    H = hadamard(d)

    m = int(np.log2(d))
    
    if (normalize):
        H = H / (2 ** (m / 2))
    
    # Dmat: nelements : {ntransforms, nstacks, d}
    
    D = {}
    for e in elements:
        
        A = None
        V = None
        
        for i in range(np.int(np.float(M) / d)):
            
            D1 = np.zeros((d, d))
            # D2 = np.zeros((d, d))
            
#             dd1 = np.random.uniform(-1, 1, d)
#             dd2 = np.random.uniform(-1, 1, d)
#             
#             dd1[dd1 < 0] = -1
#             dd1[dd1 > 0] = 1
#             
#             dd2[dd2 < 0] = -1
#             dd2[dd2 > 0] = 1

            np.fill_diagonal(D1, Dmat[e][0, i])
            # np.fill_diagonal(D2, Dmat[e][1, i])

            # W_i = np.matmul(H, np.matmul(D2, np.matmul(H, D1)))
            W_i = np.matmul(H, D1)
            
            if A is None: 
                A = W_i
            else: A = np.concatenate((A, W_i), axis=1)
            
        W[e] = (np.sqrt(d) / sigma) * A

        b[e] = np.random.uniform(0.0, 1.0, [M]) * 2.0 * np.pi  
        
    return W, b


def get_reductors(X, charges, npcas, elements):
    
    reductors = {}
    
    for e in elements:
        
        indexes = charges == e
    
        batch_indexes = torch.where(indexes)[0].type(torch.int)
    
        sub = X[indexes]
        
        perm = torch.randperm(sub.size(0))
        idx = perm[:500]

        choice_input = sub[idx]

        eigvecs, eigvals, vh = torch.linalg.svd(choice_input.T, full_matrices=False, compute_uv=True)
    
        cev = 100 - (torch.sum(eigvals) - torch.sum(eigvals[:npcas])) / torch.sum(eigvals) * 100
    
        reductor = eigvecs[:,:npcas]
        size_from = reductor.shape[0]
        size_to = reductor.shape[1]
    
        print (f"{size_from} -> {size_to}  Cumulative Explained Feature Variance = {cev:6.2f} %%")
        
        reductors[e] = reductor
        
    return reductors


if __name__ == "__main__":
    from kitchen import get_elemental_kitchen_sinks
    from forcepack import EnergyTrainedModel
    from forcepack import calc_mae
    from forcepack import calc_rmse
    from forcepack import calc_pearsonr

    ntrain = 1000
    nbatch = 4
    
    ntest = 1000
    
    print ("NTRAIN: ", ntrain, "NTEST: ", ntest)
    
    ngaussians = 20
    eta = 2.3
    lmax = 3
    rcut = 6.0
    
    ntransforms = 1
    nfeatures = 8192
    
    nsamples_proj = 500
    npcas = 256
    
    sigma = 18.0
    llambda = 1e-10
    
    coeff_normalisation = np.sqrt(npcas) / sigma
    
    elements = [1.0, 6.0]
    
    from deepdistribution.deepdist.loader import DeepDistDataloader
    
    path = '/home/nick/git/EGTO/ElementalGTO/data/benzene_dft.npz'
    
    cuda = torch.cuda.is_available()
    n_gpus = 1 if cuda else None
    device = torch.device('cuda' if cuda else 'cpu')
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    data = np.load(path)
    
    coords = data['R']
    nuclear_charges = data['z']
    energies = data['E'].flatten()
    
    forces = data['F']
    uq_nc = np.unique(data['z'])
    nuclear_charges = np.repeat(nuclear_charges[np.newaxis,:], data['R'].shape[0], axis=0)
    
    dl = DeepDistDataloader(coords, nuclear_charges, energies, forces)
    
    all_indexes = np.arange(50000, coords.shape[0])
    
    np.random.shuffle(all_indexes)
    
    train_indexes = all_indexes[:ntrain]
    test_indexes = all_indexes[ntrain:ntrain + ntest]
    
    train_coordinates, train_charges, train_energies, train_forces = dl[train_indexes]
    
    print (train_coordinates)
    
    train_coordinates = train_coordinates.type(torch.float32)
    train_charges = train_charges.type(torch.float32)
    train_energies = train_energies.type(torch.float32)
    train_forces = train_forces.type(torch.float32)
    
    self_energy = torch.Tensor([0., -0.500273, 0., 0., 0., 0., -37.845355, -54.583861, -75.064579, -99.718730]) * 627.5095
    self_energy = self_energy.cuda()
    train_interaction = torch.sum(self_energy[train_charges.type(torch.long)], dim=1)
    
    train_energies = train_energies - train_interaction
    
    test_coordinates, test_charges, test_energies, test_forces = dl[test_indexes]
    
    test_coordinates = test_coordinates.type(torch.float32)
    test_charges = test_charges.type(torch.float32)
    test_energies = test_energies.type(torch.float32)
    test_interaction = torch.sum(self_energy[test_charges.type(torch.long)], dim=1)
    
    test_energies = test_energies - test_interaction
    
    test_forces = test_forces.type(torch.float32)
    
    species = torch.from_numpy(uq_nc).type(torch.float32).cuda()
    
    print (species)
    # element_types = get_element_types(coordinates, charges, species)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    gto = compute_elemental_gto_shared(train_coordinates, train_charges, species, ngaussians, eta, lmax, rcut, gradients=False)
    
    print (gto)
    
    reductors = get_reductors(gto, train_charges, npcas, elements)
    
    normalization = np.sqrt(2.0 / nfeatures)
    
    natoms = train_coordinates.shape[1]
    
    print (natoms)
    
    Dmat = {}
    
    for e  in elements:
        Dmat[e] = np.random.uniform(-1, 1, (ntransforms, np.int(np.float(nfeatures) / npcas), npcas))
        Dmat[e][Dmat[e] > 0.0] = 1.0
        Dmat[e][Dmat[e] < 0.0] = -1.0
    
    Wk, bk = sample_sorf_elemental_basis(nfeatures, npcas, Dmat, sigma=sigma)
    
    ZTZ = torch.zeros(nfeatures, nfeatures, device=device, dtype=torch.float64)
    
    ZtrainY = torch.zeros(nfeatures, device=device, dtype=torch.float64)
    GtrainY = torch.zeros(nfeatures, device=device, dtype=torch.float64)
    
    start.record()
    for i in range(nbatch):
        
        zbatch = np.int(np.ceil(ntrain / nbatch))
        
        print (zbatch)
        
        batch_train_charges = train_charges[i * zbatch:(i + 1) * zbatch]
        batch_train_coordinates = train_coordinates[i * zbatch:(i + 1) * zbatch]
        batch_energies = train_energies[i * zbatch:(i + 1) * zbatch]
        batch_forces = train_forces[i * zbatch:(i + 1) * zbatch]
        
        train_gto, train_gto_derivative = compute_elemental_gto_shared(batch_train_coordinates, batch_train_charges, species, ngaussians, eta, lmax, rcut, gradients=True)
        
        Ztrain = torch.zeros(zbatch, nfeatures, device=device, dtype=torch.float32)
        Gtrain_derivative = torch.zeros(zbatch, gto.shape[1], 3, nfeatures, device=device, dtype=torch.float32)
    
        for e in elements:
            
            indexes = batch_train_charges == e
    
            batch_indexes = torch.where(indexes)[0].type(torch.int)
            
            sub = train_gto[indexes]
            sub_grad = train_gto_derivative[indexes]
    
            sub = torch.matmul(sub, reductors[e])
            sub_grad = torch.einsum('jmnk, kl->jmnl', sub_grad, reductors[e])
            
            coeffs = coeff_normalisation * hadamard_flavours.sorf_matrix_gpu(sub, torch.from_numpy(Dmat[e]).float().cuda(), nfeatures)
            
            Ztrain += hadamard_flavours.molecular_featurisation_gpu(coeffs, torch.from_numpy(bk[e]).float().cuda(), batch_indexes, zbatch)
        
            Gtrain_derivative -= coeff_normalisation * hadamard_flavours.molecular_featurisation_derivative_gpu(coeffs,
                                                            torch.from_numpy(bk[e]).float().cuda(),
                                                            torch.from_numpy(Dmat[e]).float().cuda(), sub_grad, batch_indexes, zbatch)

        ZtrainY += torch.matmul(Ztrain.double().T, batch_energies.double())
        GtrainY += torch.matmul(Gtrain_derivative.reshape(zbatch * natoms * 3, nfeatures).double().T, batch_forces.double().flatten())
        
        ZTZ += torch.matmul(Ztrain.double().T, Ztrain.double()) + torch.matmul(Gtrain_derivative.reshape(zbatch * natoms * 3, nfeatures).double().T,
                                                         Gtrain_derivative.reshape(zbatch * natoms * 3, nfeatures).double()) 
    end.record()
    torch.cuda.synchronize()
    print (ZTZ)
    print("batched ZTZ time: ", start.elapsed_time(end), "ms")
    
    ZTZ[torch.eye(nfeatures).bool()] += llambda
    
    start.record()
    Y = ZtrainY + GtrainY
    # alpha = torch.solve(Y[:, None], ZTZ.double()).solution
    
    alpha = torch.solve(Y[:, None], ZTZ).solution
    
    end.record()
    torch.cuda.synchronize()

    print (alpha)

    print("coefficients time: ", start.elapsed_time(end), "ms")
    
    start.record()
    gto_test, gto_test_grad = compute_elemental_gto_shared(test_coordinates, test_charges, species, ngaussians, eta, lmax, rcut, gradients=True)
    
    E = torch.zeros(ntest, device=device, dtype=torch.float32)
    F = torch.zeros(ntest * natoms * 3, device=device, dtype=torch.float32)
    
    alpha = alpha[:, 0].float()
    for e in elements:
        
        indexes = test_charges == e

        batch_indexes = torch.where(indexes)[0].type(torch.int)
    
        sub = gto_test[indexes]
        sub_grad = gto_test_grad[indexes]
        
        sub = torch.matmul(sub, reductors[e])
        sub_grad = torch.einsum('jmnk, kl->jmnl', sub_grad, reductors[e])
        
        coeffs = coeff_normalisation * hadamard_flavours.sorf_matrix_gpu(sub, torch.from_numpy(Dmat[e]).float().cuda(), nfeatures)
    
        Ztest = hadamard_flavours.molecular_featurisation_gpu(coeffs, torch.from_numpy(bk[e]).float().cuda(), batch_indexes, ntest)
        Gtest = -coeff_normalisation * hadamard_flavours.molecular_featurisation_derivative_gpu(coeffs,
                                                        torch.from_numpy(bk[e]).float().cuda(),
                                                        torch.from_numpy(Dmat[e]).float().cuda(), sub_grad, batch_indexes, ntest)
        
        F += torch.matmul(Gtest.reshape(ntest * natoms * 3, nfeatures), alpha)
        E += torch.matmul(Ztest, alpha)

    end.record()
    torch.cuda.synchronize()
    
    print("prediction for", ntest, "molecules time: ", start.elapsed_time(end), "ms")
    
    print("Energy MAE:", calc_mae(E.cpu().numpy(), test_energies.detach().cpu().numpy()))
    print("Energy RMSE:", calc_rmse(E.cpu().numpy(), test_energies.detach().cpu().numpy()))
    print("pearsons R:", calc_pearsonr(E.cpu().numpy(), test_energies.detach().cpu().numpy()))
    
    print("Force MAE:", calc_mae(F.cpu().numpy(), test_forces.flatten().detach().cpu().numpy()))
    print("Force RMSE:", calc_rmse(F.cpu().numpy(), test_forces.flatten().detach().cpu().numpy()))
    print("pearsons R:", calc_pearsonr(F.cpu().numpy(), test_forces.flatten().detach().cpu().numpy()))
    
