import torch
import numpy as np

from qml_lightning.representations.EGTO import get_elemental_gto
from qml_lightning.features.SORF import get_SORF_diagonals, get_bias, get_SORF_coefficients, get_features, get_feature_derivatives, SORFTransformCuda
from qml_lightning.representations.dimensionality_reduction import project_representation, project_derivative
from qml_lightning.representations.EGTO import ElementalGTO
import argparse


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
    parser.add_argument("-nbatch", type=int, default=4)
    
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
    
    feature_normalisation = np.sqrt(2.0 / nfeatures)
    coeff_normalisation = np.sqrt(npcas) / sigma
    
    path = '/home/nick/git/EGTO/ElementalGTO/tools/qm7_data.npz'
    
    cuda = torch.cuda.is_available()
    n_gpus = 1 if cuda else None
    device = torch.device('cuda' if cuda else 'cpu')
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    data = np.load(path, allow_pickle=True)
    
    coords = data['R']
    nuclear_charges = data['Q']
    energies = data['hof'].flatten()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    elements = np.array([1, 6, 7, 8, 9])

    # elements = np.array([1, 6, 7, 8])
    train_coordinates = []
    train_charges = []
    train_energy = []
    
    test_coordinates = []
    test_charges = []
    test_energy = []
    
    all_indexes = np.arange(0, coords.shape[0])
    
    del_indexes = []
    
    for i in range(len(all_indexes)):
        if (16 in nuclear_charges[i]):
            del_indexes.append(i)
    
    all_indexes = np.delete(all_indexes, del_indexes)
    
    np.random.shuffle(all_indexes)
    
    train_indexes = all_indexes[:ntrain]
    test_indexes = all_indexes[ntrain:ntrain + ntest]
    
    for i in range(ntrain):
        train_coordinates.append(torch.from_numpy(coords[train_indexes[i]]).float().cuda())
        train_charges.append(torch.from_numpy(nuclear_charges[train_indexes[i]]).float().cuda())
        train_energy.append(energies[train_indexes[i]])
    
    train_energies = torch.Tensor(train_energy).float().cuda()
    
    for i in range(ntest):
        
        test_coordinates.append(torch.from_numpy(coords[test_indexes[i]]).float().cuda())
        test_charges.append(torch.from_numpy(nuclear_charges[test_indexes[i]]).float().cuda())
        test_energy.append(energies[test_indexes[i]])
    
    test_energies = torch.Tensor(test_energy).float().cuda()
    
    reductors = get_reductors(train_coordinates, train_charges, npcas, elements)
  
    feature_normalization = np.sqrt(2.0 / nfeatures)
   
    Dmat = get_SORF_diagonals(elements, ntransforms, nfeatures, npcas)
    bk = get_bias(elements, nfeatures)

    ZTZ = torch.zeros(nfeatures, nfeatures, device=device, dtype=torch.float64)
    
    ZtrainY = torch.zeros(nfeatures, 1, device=device, dtype=torch.float64)
    
    Ztrain = torch.zeros(ntrain, nfeatures, device=device, dtype=torch.float64)
    
    fingerprint = ElementalGTO(species=elements, low_cutoff=0.0, high_cutoff=rcut, n_gaussians=ngaussians, eta=eta, Lmax=lmax, device=device)
    
    nstacks = int(float(nfeatures) / npcas)
    
    start.record()
    for i, (coordinates, charges) in enumerate(zip(train_coordinates, train_charges)): 
        
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
            
            Ztrain[i,:] += feature_normalisation * torch.sum(torch.cos(coeffs + bk[e]).double(), dim=0)

    ZtrainY = torch.matmul(Ztrain.T, train_energies.double()[:, None])

    ZTZ = torch.matmul(Ztrain.T, Ztrain)
        
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
    
    Ztest = torch.zeros(ntest, nfeatures, device=device, dtype=torch.float64)
    
    start.record()
    
    for i, (coordinates, charges) in enumerate(zip(test_coordinates, test_charges)): 
        
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
            
            Ztest[i,:] += feature_normalisation * torch.sum(torch.cos(coeffs + bk[e]).double(), dim=0)
    
    energies_predict = torch.matmul(Ztest, alpha)
    
    end.record()
    torch.cuda.synchronize()
    
    print("prediction for", ntest, "molecules time: ", start.elapsed_time(end), "ms")
    
    print("Energy MAE TORCH:", torch.mean(torch.abs(energies_predict - test_energies)))
  
