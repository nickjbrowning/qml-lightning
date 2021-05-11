import torch
import numpy as np
from tqdm import tqdm

from qml_lightning.representations.EGTO import get_elemental_gto
from qml_lightning.features.SORF import get_SORF_diagonals, get_bias, get_SORF_coefficients, get_features, get_feature_derivatives, SORFTransformCuda
from qml_lightning.representations.dimensionality_reduction import project_representation, project_derivative
from qml_lightning.representations.EGTO import ElementalGTO
from qml_lightning.utils.ani_dataloader import iter_data_buckets
import argparse

path_to_h5file = 'data/ani-1x.h5'


def get_reductors(X, Z, npcas, species):
    
    reductors = {}
    
    for e in species:
        
        inputs = []
        
        for coordinates, charges in zip(X, Z):
            
            coordinates = coordinates.unsqueeze(0)
            charges = charges.unsqueeze(0)
            
            gto = get_elemental_gto(coordinates, charges, torch.from_numpy(species).cuda().float(), ngaussians, eta, lmax, rcut, gradients=False)
            
            indexes = charges == e
        
            batch_indexes = torch.where(indexes)[0].type(torch.int)
            
            sub = gto[indexes]
            
            if (sub.shape[0] == 0):
                continue
            
            inputs.append(sub)
        
        if (len(inputs) == 0):
            continue
        
        mat = torch.cat(inputs)
        
        perm = torch.randperm(mat.size(0))
         
        idx = perm[:512]

        choice_input = mat[idx]

        eigvecs, eigvals, vh = torch.linalg.svd(choice_input.T, full_matrices=False, compute_uv=True)
    
        cev = 100 - (torch.sum(eigvals) - torch.sum(eigvals[:npcas])) / torch.sum(eigvals) * 100
    
        reductor = eigvecs[:,:npcas]
        size_from = reductor.shape[0]
        size_to = reductor.shape[1]
    
        print (f"{size_from} -> {size_to}  Cumulative Explained Feature Variance = {cev:6.2f} %%")
        
        reductors[e] = reductor
    
    return reductors


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-ntrain", type=int, default=1000)
    parser.add_argument("-ntest", type=int, default=250)
    parser.add_argument("-num_batch", type=int, default=1000)
    
    '''model parameters'''
    parser.add_argument("-sigma", type=float, default=20.0)
    parser.add_argument("-llambda", type=float, default=1e-10)
    parser.add_argument("-npcas", type=int, default=256)
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
    num_batch = args.num_batch
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
    
    feature_normalisation = np.sqrt(2.0 / nfeatures)
    coeff_normalisation = np.sqrt(npcas) / sigma
    
    cuda = torch.cuda.is_available()
    n_gpus = 1 if cuda else None
    device = torch.device('cuda' if cuda else 'cpu')
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    elements = np.array([1, 6, 7, 8, 9])
    
    data_keys = ['wb97x_tz.energy', 'wb97x_tz.forces'] 
    
    Xs = []
    Zs = []
    Es = []
    Fs = []
    
    for data in iter_data_buckets(path_to_h5file, keys=data_keys):
        X = data['coordinates']
        Z = data['atomic_numbers']
        E = data['wb97x_tz.energy']
        F = data['wb97x_tz.forces']
        
        for i in range(X.shape[0]):
            Xs.append(X[i])
            Zs.append(Z)
            Es.append(E[i])
            Fs.append(F[i])
    
    ALL_IDX = np.arange(len(Xs))
    np.random.shuffle(ALL_IDX)

    # elements = np.array([1, 6, 7, 8])
    train_coordinates = []
    train_charges = []
    train_energies = torch.zeros(ntrain, dtype=torch.float64, device=device)
    
    test_coordinates = []
    test_charges = []
    test_energies = torch.zeros(ntest, dtype=torch.float64, device=device)
    
    train_indexes = ALL_IDX[:ntrain]
    test_indexes = ALL_IDX[ntrain:ntrain + ntest]
    
    self_energy = torch.Tensor([0., -0.500273, 0., 0., 0., 0., -37.845355, -54.583861, -75.064579, -99.718730]).double().cuda()
    hartree2kcalmol = 627.5095
    
    for i in range(ntrain):
        charges = torch.from_numpy(Zs[train_indexes[i]]).float().cuda()
        coordinates = torch.from_numpy(Xs[train_indexes[i]]).float().cuda()
        energy = Es[train_indexes[i]] * hartree2kcalmol
        
        self_interaction = self_energy[charges.long()].sum(axis=0) * hartree2kcalmol
        energy = energy - self_interaction
        
        train_coordinates.append(coordinates)
        train_charges.append(charges)
    
        train_energies[i] = energy
    
    for i in range(ntest):
        
        charges = torch.from_numpy(Zs[test_indexes[i]]).float().cuda()
        coordinates = torch.from_numpy(Xs[test_indexes[i]]).float().cuda()
        energy = Es[test_indexes[i]] * hartree2kcalmol
        
        self_interaction = self_energy[charges.long()].sum(axis=0) * hartree2kcalmol
        energy = energy - self_interaction
        
        test_coordinates.append(coordinates)
        test_charges.append(charges)
    
        test_energies[i] = energy
    
    reductors = get_reductors(train_coordinates[:10000], train_charges[:10000], npcas, elements)
  
    feature_normalization = np.sqrt(2.0 / nfeatures)
   
    Dmat = get_SORF_diagonals(elements, ntransforms, nfeatures, npcas)
    bk = get_bias(elements, nfeatures)

    ZTZ = torch.zeros(nfeatures, nfeatures, device=device, dtype=torch.float64)
    
    ZtrainY = torch.zeros(nfeatures, 1, device=device, dtype=torch.float64)
    
    fingerprint = ElementalGTO(species=elements, low_cutoff=0.0, high_cutoff=rcut, n_gaussians=ngaussians, eta=eta, Lmax=lmax, device=device)
    
    nstacks = int(float(nfeatures) / npcas)
    
    start.record()
    
    Ztrain = torch.zeros(num_batch, nfeatures, device=device, dtype=torch.float64)
    batch_energies = torch.zeros(num_batch, 1, device=device, dtype=torch.float64)
    
    idx = 0
    
    for i, (coordinates, charges) in tqdm(enumerate(zip(train_coordinates, train_charges)), ascii=True, desc="Building Feature Matrix"): 
        
        coordinates = coordinates.unsqueeze(0)
        charges = charges.unsqueeze(0)
        
        rep = fingerprint.forward(coordinates, charges.int())

        for e in elements:
            
            indexes = charges.int() == e
            
            batch_indexes = torch.where(indexes)[0].type(torch.int)
            
            sub = rep[indexes]
            
            if (sub.shape[0] == 0):
                continue
            
            sub = project_representation(sub, reductors[e])
            
            sub = sub.repeat(1, nstacks).reshape(sub.shape[0], nstacks, npcas)
            
            coeffs = coeff_normalisation * SORFTransformCuda.apply(Dmat[e] * sub)
            coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1] * coeffs.shape[2])
            
            Ztrain[idx,:] += feature_normalisation * torch.sum(torch.cos(coeffs + bk[e]).double(), dim=0)
            batch_energies[idx, 0] = train_energies.double()[i]
            
        idx = idx + 1
        
        if (idx % num_batch == 0):
            idx = 0
            
            ZTZ += torch.matmul(Ztrain.T, Ztrain)
            ZtrainY += torch.matmul(Ztrain.T, batch_energies)
            
            Ztrain = torch.zeros(num_batch, nfeatures, device=device, dtype=torch.float64)
            batch_energies = torch.zeros(num_batch, 1, device=device, dtype=torch.float64)
            
    end.record()
    torch.cuda.synchronize()

    print("batched ZTZ time: ", start.elapsed_time(end), "ms")
    
    ZTZ[torch.eye(nfeatures).bool()] += llambda
    
    print (ZTZ)
    
    start.record()
    
    Y = ZtrainY 
    
    alpha = torch.solve(Y, ZTZ).solution
    
    end.record()
    torch.cuda.synchronize()

    print("coefficients time: ", start.elapsed_time(end), "ms")
    
    start.record()

    alpha = alpha[:, 0]
    
    alpha.cpu().numpy().tofile("alpha.npy")
    
    for e in elements:
        if (e in reductors):
            Dmat[e].cpu().numpy().tofile("W_" + str(e) + ".npy")
            bk[e].cpu().numpy().tofile("b_" + str(e) + ".npy")
            reductors[e].cpu().numpy().tofile("reductor_" + str(e) + ".npy")
    
    start.record()
    
    predictions = torch.zeros(ntest, device=device, dtype=torch.float64)
    
    for i, (coordinates, charges) in tqdm(enumerate(zip(test_coordinates, test_charges)), ascii=True, desc="Computing Predictions"): 
        
        coordinates = coordinates.unsqueeze(0)
        charges = charges.unsqueeze(0)
        
        Ztest = torch.zeros(1, nfeatures, device=device, dtype=torch.float64)
        
        rep = fingerprint.forward(coordinates, charges.int())
        
        for e in elements:
            
            indexes = charges.int() == e
            
            batch_indexes = torch.where(indexes)[0].type(torch.int)
            
            sub = rep[indexes]
            
            if (sub.shape[0] == 0):
                continue
            
            sub = project_representation(sub, reductors[e])
            
            sub = sub.repeat(1, nstacks).reshape(sub.shape[0], nstacks, npcas)
            
            coeffs = coeff_normalisation * SORFTransformCuda.apply(Dmat[e] * sub)
            coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1] * coeffs.shape[2])
            
            Ztest[0,:] += feature_normalisation * torch.sum(torch.cos(coeffs + bk[e]).double(), dim=0)
        
        predictions[i] = torch.matmul(Ztest, alpha)
    
    end.record()
    torch.cuda.synchronize()
    
    print("prediction for", ntest, "molecules time: ", start.elapsed_time(end), "ms")
    
    print("Energy MAE TORCH:", torch.mean(torch.abs(predictions - test_energies)))
  
