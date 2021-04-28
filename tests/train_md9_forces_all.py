import torch
import numpy as np

from qml_lightning.representations.EGTO import get_elemental_gto
from qml_lightning.features.SORF import get_SORF_diagonals, get_bias, get_SORF_coefficients, get_features, get_feature_derivatives, SORFTransformCuda
from qml_lightning.representations.dimensionality_reduction import get_reductors, project_representation, project_derivative
from qml_lightning.representations.EGTO import ElementalGTO
import argparse

self_energy = torch.Tensor([0., -0.500273, 0., 0., 0., 0., -37.845355, -54.583861, -75.064579, -99.718730]).cuda()
hartree2kcalmol = 627.5095

ngaussians = 20
eta = 3.0
lmax = 2
rcut = 6.0

ntransforms = 1
nfeatures = 8192

npcas = 128

sigma = 30.0
llambda = 1e-13

species = np.array([1, 6, 7, 8])


def get_reductors(X, Z, npcas, species):
    
    reductors = {}
    
    for e in species:
        
        inputs = []
        
        for coordinates, charges in zip(X, Z):
            
            gto = get_elemental_gto(coordinates, charges, torch.from_numpy(species).cuda().float(), ngaussians, eta, lmax, rcut, gradients=False)
            
            indexes = charges == e
        
            batch_indexes = torch.where(indexes)[0].type(torch.int)
            
            sub = gto[indexes]
            
            if (sub.shape[0] == 0):
                continue
            
            perm = torch.randperm(sub.size(0))
            idx = perm[:512]
    
            choice_input = sub[idx]
            
            inputs.append(choice_input)
            
        mat = torch.cat(inputs)
        
        print (mat.shape)
    
        eigvecs, eigvals, vh = torch.linalg.svd(mat.T, full_matrices=False, compute_uv=True)
    
        cev = 100 - (torch.sum(eigvals) - torch.sum(eigvals[:npcas])) / torch.sum(eigvals) * 100
    
        reductor = eigvecs[:,:npcas]
        size_from = reductor.shape[0]
        size_to = reductor.shape[1]
    
        print (f"{size_from} -> {size_to}  Cumulative Explained Feature Variance = {cev:6.2f} %%")
        
        reductors[e] = reductor
    
    return reductors


def grab_data(npz_path, indexes):
    data = np.load(npz_path)

    coords = data['R']
    nuclear_charges = data['z']
    energies = data['E'].flatten()
    forces = data['F']
    
    nuclear_charges = np.repeat(nuclear_charges[np.newaxis,:], data['R'].shape[0], axis=0)
  
    coordinates = torch.from_numpy(coords[train_indexes]).float().cuda()
    coordinates.requires_grad = True
    charges = torch.from_numpy(nuclear_charges[train_indexes]).float().cuda()
    energies = torch.from_numpy(energies[train_indexes]).float().cuda()
    forces = torch.from_numpy(forces[train_indexes]).float().cuda()
    
    self_interaction = self_energy[charges.long()].sum(axis=1) * hartree2kcalmol
    energies = energies - self_interaction
    
    return coordinates, charges, energies, forces
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-ntrain", type=int, default=1000)
    parser.add_argument("-ntest", type=int, default=100)
    parser.add_argument("-nbatch", type=int, default=4)
    parser.add_argument("-datas", type=list, default=['data/benzene_dft.npz', 'data/ethanol_dft.npz', 'data/malonaldehyde_dft.npz',
                                                     'data/naphthalene_dft.npz', 'data/salicylic_dft.npz', 'data/toluene_dft.npz', 'data/uracil_dft.npz', 'data/aspirin_dft.npz' ])
    
    '''model parameters'''
    parser.add_argument("-sigma", type=float, default=20.0)
    parser.add_argument("-llambda", type=float, default=1e-11)
    parser.add_argument("-npcas", type=int, default=128)
    parser.add_argument("-ntransforms", type=int, default=1)
    parser.add_argument("-nfeatures", type=int, default=8192)
    
    '''representation parameters'''
    parser.add_argument("-eta", type=float, default=2.3)
    parser.add_argument("-rcut", type=float, default=6.0)
    parser.add_argument("-lmax", type=int, default=2)
    parser.add_argument("-ngaussians", type=int, default=20)
    
    args = parser.parse_args()
    
    print ("---Argument Summary---")
    print (args)
    
    ntrain = args.ntrain
    nbatch = args.nbatch
    ntest = args.ntest

    ngaussians = args.ngaussians
    eta = args.eta
    lmax = args.lmax
    rcut = args.rcut
    
    ntransforms = args.ntransforms
    nfeatures = args.nfeatures
    
    npcas = args.npcas
    
    sigma = args.sigma
    llambda = args.llambda
    
    datas = args.datas
    
    cuda = torch.cuda.is_available()
    n_gpus = 1 if cuda else None
    device = torch.device('cuda' if cuda else 'cpu')
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    train_coordinates = []
    train_charges = []
    train_forces = []
    train_energies = []
    
    test_coordinates = []
    test_charges = []
    test_forces = []
    test_energies = []

    print ("--- Loading Data ---")
    for npz_file in datas:
        
        ALL_IDX = np.arange(100000)
    
        np.random.shuffle(ALL_IDX)
        
        train_indexes = ALL_IDX[:ntrain]
        test_indexes = ALL_IDX[ntrain:ntrain + ntest]
    
        coordinates, charges, energies, forces = grab_data(npz_file, train_indexes)
        
        train_coordinates.append(coordinates)
        train_charges.append(charges)
        train_forces.append(forces)
        train_energies.append(energies)
        
        coordinates, charges, energies, forces = grab_data(npz_file, test_indexes)
        
        test_coordinates.append(coordinates)
        test_charges.append(charges)
        test_forces.append(forces)
        test_energies.append(energies)
        
    print ("--- Calculating Projection Matrices ---")
    
    reductors = get_reductors(train_coordinates, train_charges, npcas, species)
    
    coeff_normalisation = np.sqrt(npcas) / sigma
    feature_normalisation = np.sqrt(2.0 / float(nfeatures))
    
    Dmat = get_SORF_diagonals(species, ntransforms, nfeatures, npcas)
    bk = get_bias(species, nfeatures)
    
    ZTZ = torch.zeros(nfeatures, nfeatures, device=device, dtype=torch.float64)
    ZtrainY = torch.zeros(nfeatures, device=device, dtype=torch.float64)
    GtrainY = torch.zeros(nfeatures, device=device, dtype=torch.float64)
    
    print ("--- Computing Gramm Matrix ---")
    
    for i, (coordinates, charges, energies, forces) in enumerate(zip(train_coordinates, train_charges, train_energies, train_forces)):
        
        print ("Computing sub-Gramm Matrix for: ", datas[i])
        natoms = coordinates.shape[1]
        
        for i in range(nbatch):
        
            zbatch = np.int(np.ceil(ntrain / nbatch))
            
            batch_train_charges = charges[i * zbatch:(i + 1) * zbatch]
            batch_train_coordinates = coordinates[i * zbatch:(i + 1) * zbatch]
            batch_energies = energies[i * zbatch:(i + 1) * zbatch]
            batch_forces = forces[i * zbatch:(i + 1) * zbatch]
            
            train_gto, train_gto_derivative = get_elemental_gto(batch_train_coordinates, batch_train_charges, torch.from_numpy(species).cuda().float(), ngaussians, eta, lmax, rcut, gradients=True)
            
            Ztrain = torch.zeros(zbatch, nfeatures, device=device, dtype=torch.float32)
            Gtrain_derivative = torch.zeros(zbatch, natoms, 3, nfeatures, device=device, dtype=torch.float32)
            
            for e in species:
                
                indexes = batch_train_charges == e
        
                batch_indexes = torch.where(indexes)[0].type(torch.int)
                
                sub = train_gto[indexes]
                sub_grad = train_gto_derivative[indexes]
                
                if (sub.shape[0] == 0):
                    continue
                
                sub = project_representation(sub, reductors[e])
                sub_grad = project_derivative(sub_grad, reductors[e])
                
                coeffs = get_SORF_coefficients(sub, nfeatures, Dmat[e], coeff_normalisation)
            
                Ztrain += get_features(coeffs, bk[e], batch_indexes, zbatch)
                
                Gtrain_derivative -= get_feature_derivatives(coeffs, bk[e], Dmat[e], sub_grad, batch_indexes, zbatch, coeff_normalisation)
                
            ZtrainY += torch.matmul(Ztrain.double().T, batch_energies.double())
            GtrainY += torch.matmul(Gtrain_derivative.reshape(zbatch * natoms * 3, nfeatures).double().T, batch_forces.double().flatten())
            
            ZTZ += torch.matmul(Ztrain.double().T, Ztrain.double()) + torch.matmul(Gtrain_derivative.reshape(zbatch * natoms * 3, nfeatures).double().T,
                                                             Gtrain_derivative.reshape(zbatch * natoms * 3, nfeatures).double())
            
    ZTZ[torch.eye(nfeatures).bool()] += llambda
    
    Y = ZtrainY + GtrainY
    
    alpha = torch.solve(Y[:, None], ZTZ).solution
    alpha = alpha[:, 0]
    
    del train_charges, train_coordinates, train_energies, train_forces
    del ZTZ, ZtrainY, GtrainY, Ztrain, Gtrain_derivative
    
    torch.cuda.empty_cache()
    
    print ("--- Calculating Errors ---")
    
    nstacks = int(float(nfeatures) / npcas)
    
    for i, (coordinates, charges, energies, forces) in enumerate(zip(test_coordinates, test_charges, test_energies, test_forces)):
        
        natoms = coordinates.shape[1]
        ntest = coordinates.shape[0]
        
        fingerprint = ElementalGTO(species=species, low_cutoff=0.0, high_cutoff=rcut, n_gaussians=ngaussians, eta=eta, Lmax=lmax, device=device)
        
        rep = fingerprint.forward(coordinates, charges.int())

        Ztest = torch.zeros(ntest, nfeatures, device=device, dtype=torch.float32)
        
        for e in species:
  
            indexes = charges == e
            batch_indexes = torch.where(indexes)[0].type(torch.int)
            
            sub = rep[indexes]
            
            if (sub.shape[0] == 0):
                continue
            
            sub = project_representation(sub, reductors[e])
            
            sub = sub.repeat(1, nstacks).reshape(sub.shape[0], nstacks, npcas)
            
            coeffs = coeff_normalisation * SORFTransformCuda.apply(Dmat[e] * sub)
            coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1] * coeffs.shape[2])
            
            Ztest.index_add_(0, batch_indexes, feature_normalisation * torch.cos(coeffs + bk[e]))

        energies_predict = torch.matmul(Ztest.double(), alpha)
        
        forces_predict, = torch.autograd.grad(-energies_predict.sum(), coordinates, retain_graph=True)
        
        print ("---", datas[i], "---")
        print("Energy MAE:", torch.mean(torch.abs(energies_predict - energies)))
        print("Force MAE:", torch.mean(torch.abs(forces_predict.flatten() - forces.flatten())))
            
