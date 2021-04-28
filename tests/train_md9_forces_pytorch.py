import torch
import numpy as np

from qml_lightning.representations.EGTO import get_elemental_gto
from qml_lightning.features.SORF import get_SORF_diagonals, get_bias, get_SORF_coefficients, get_features, get_feature_derivatives, SORFTransformCuda
from qml_lightning.representations.dimensionality_reduction import get_reductors, project_representation, project_derivative
from qml_lightning.representations.EGTO import ElementalGTO
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-ntrain", type=int, default=1000)
    parser.add_argument("-ntest", type=int, default=250)
    parser.add_argument("-nbatch", type=int, default=4)
    parser.add_argument("-data", type=str, default='data/aspirin_dft.npz')
    
    '''model parameters'''
    parser.add_argument("-sigma", type=float, default=20.0)
    parser.add_argument("-llambda", type=float, default=1e-10)
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
    
    data_path = args.data

    ngaussians = args.ngaussians
    eta = args.eta
    lmax = args.lmax
    rcut = args.rcut
    
    ntransforms = args.ntransforms
    nfeatures = args.nfeatures
    
    npcas = args.npcas
    
    sigma = args.sigma
    llambda = args.llambda
    
    coeff_normalisation = np.sqrt(npcas) / sigma
    
    path = args.data
    
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
    elements = np.unique(data['z'])
    nuclear_charges = np.repeat(nuclear_charges[np.newaxis,:], data['R'].shape[0], axis=0)
    
    ALL_IDX = np.arange(coords.shape[0])
    
    np.random.shuffle(ALL_IDX)
    
    train_indexes = ALL_IDX[:ntrain]
    test_indexes = ALL_IDX[ntrain:ntrain + ntest]
    
    self_energy = torch.Tensor([0., -0.500273, 0., 0., 0., 0., -37.845355, -54.583861, -75.064579, -99.718730]).cuda()
    hartree2kcalmol = 627.5095
    
    train_coordinates = torch.from_numpy(coords[train_indexes]).float().cuda()
    train_charges = torch.from_numpy(nuclear_charges[train_indexes]).float().cuda()
    train_energies = torch.from_numpy(energies[train_indexes]).float().cuda()
    train_forces = torch.from_numpy(forces[train_indexes]).float().cuda()
    
    self_interaction = self_energy[train_charges.long()].sum(axis=1) * hartree2kcalmol
    train_energies = train_energies - self_interaction
    
    test_coordinates = torch.from_numpy(coords[test_indexes]).float().cuda()
    test_coordinates.requires_grad = True
    test_charges = torch.from_numpy(nuclear_charges[test_indexes]).float().cuda()
    test_energies = torch.from_numpy(energies[test_indexes]).cuda()
    test_forces = torch.from_numpy(forces[test_indexes]).cuda()
    
    self_interaction = self_energy[test_charges.long()].sum(axis=1) * hartree2kcalmol
    test_energies = test_energies - self_interaction
        
    species = torch.from_numpy(elements).type(torch.float32).cuda()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    gto = get_elemental_gto(train_coordinates, train_charges, species, ngaussians, eta, lmax, rcut, gradients=False)
    
    reductors = get_reductors(gto, train_charges, npcas, elements)
    
    feature_normalization = np.sqrt(2.0 / nfeatures)
    
    natoms = train_coordinates.shape[1]
    
    Dmat = get_SORF_diagonals(elements, ntransforms, nfeatures, npcas)
    bk = get_bias(elements, nfeatures)

    ZTZ = torch.zeros(nfeatures, nfeatures, device=device, dtype=torch.float64)
    
    ZtrainY = torch.zeros(nfeatures, 1, device=device, dtype=torch.float64)
    GtrainY = torch.zeros(nfeatures, 1, device=device, dtype=torch.float64)
    
    start.record()
    for i in range(nbatch):
        
        zbatch = np.int(np.ceil(ntrain / nbatch))
        
        batch_train_charges = train_charges[i * zbatch:(i + 1) * zbatch]
        batch_train_coordinates = train_coordinates[i * zbatch:(i + 1) * zbatch]
        batch_energies = train_energies[i * zbatch:(i + 1) * zbatch]
        batch_forces = train_forces[i * zbatch:(i + 1) * zbatch]
        
        train_gto, train_gto_derivative = get_elemental_gto(batch_train_coordinates, batch_train_charges, species, ngaussians, eta, lmax, rcut, gradients=True)
        
        Ztrain = torch.zeros(zbatch, nfeatures, device=device, dtype=torch.float64)
        Gtrain_derivative = torch.zeros(zbatch, train_gto.shape[1], 3, nfeatures, device=device, dtype=torch.float64)
    
        for e in elements:
            
            indexes = batch_train_charges == e
    
            batch_indexes = torch.where(indexes)[0].type(torch.int)
            
            sub = train_gto[indexes]
            sub_grad = train_gto_derivative[indexes]
    
            sub = project_representation(sub, reductors[e])
            sub_grad = project_derivative(sub_grad, reductors[e])
            
            coeffs = get_SORF_coefficients(sub, nfeatures, Dmat[e], coeff_normalisation)
            
            Ztrain += get_features(coeffs, bk[e], batch_indexes, zbatch).double()
            
            Gtrain_derivative -= get_feature_derivatives(coeffs, bk[e], Dmat[e], sub_grad, batch_indexes, zbatch, coeff_normalisation).double()
        
        Gtrain_derivative = Gtrain_derivative.reshape(zbatch * natoms * 3, nfeatures)
        
        # ZtrainY += torch.matmul(Ztrain.double().T, batch_energies.double())
        # GtrainY += torch.matmul(Gtrain_derivative.reshape(zbatch * natoms * 3, nfeatures).double().T, batch_forces.double().flatten())
        
        ZtrainY += torch.matmul(Ztrain.T, batch_energies.double()[:, None])
        GtrainY += torch.matmul(Gtrain_derivative.T, batch_forces.double().flatten()[:, None])

        ZTZ += torch.matmul(Ztrain.T, Ztrain)
        ZTZ += torch.matmul(Gtrain_derivative.T, Gtrain_derivative)
        # ZTZ += torch.matmul(Ztrain.double().T, Ztrain.double()) + torch.matmul(Gtrain_derivative.reshape(zbatch * natoms * 3, nfeatures).double().T,
        #                                                 Gtrain_derivative.reshape(zbatch * natoms * 3, nfeatures).double()) 
    end.record()
    torch.cuda.synchronize()
    print ("---Gramm Matrix---")

    print("batched ZTZ time: ", start.elapsed_time(end), "ms")
    
    ZTZ[torch.eye(nfeatures).bool()] += llambda
    
    start.record()
    Y = ZtrainY + GtrainY
    
    alpha = torch.solve(Y, ZTZ).solution
    
    del ZTZ, ZtrainY, GtrainY, Ztrain, Gtrain_derivative
    
    end.record()
    torch.cuda.synchronize()

    print (alpha)

    print("coefficients time: ", start.elapsed_time(end), "ms")
    
    start.record()

    alpha = alpha[:, 0]
    
    alpha.cpu().numpy().tofile("alpha.npy")
    
    for e in elements:
        Dmat[e].cpu().numpy().tofile("W_" + str(e) + ".npy")
        bk[e].cpu().numpy().tofile("b_" + str(e) + ".npy")
        reductors[e].cpu().numpy().tofile("reductor_" + str(e) + ".npy")
    
    startg = torch.cuda.Event(enable_timing=True)
    endg = torch.cuda.Event(enable_timing=True)
    
    fingerprint = ElementalGTO(species=elements, low_cutoff=0.0, high_cutoff=rcut, n_gaussians=ngaussians, eta=eta, Lmax=lmax, device=device)
        
    rep = fingerprint.forward(test_coordinates, test_charges.int())
    
    Ztest = torch.zeros(ntest, nfeatures, device=device, dtype=torch.float64)
    
    nstacks = int(float(nfeatures) / npcas)
    
    coeff_normalisation = np.sqrt(npcas) / sigma
    feature_normalisation = np.sqrt(2.0 / float(nfeatures))
    
    for e in elements:
  
        indexes = test_charges == e
        batch_indexes = torch.where(indexes)[0].type(torch.int)
        
        sub = rep[indexes]
        
        if (sub.shape[0] == 0):
            continue
        
        sub = project_representation(sub, reductors[e])
        
        sub = sub.repeat(1, nstacks).reshape(sub.shape[0], nstacks, npcas)
        
        coeffs = coeff_normalisation * SORFTransformCuda.apply(Dmat[e] * sub)
        coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1] * coeffs.shape[2])
        
        Ztest.index_add_(0, batch_indexes, feature_normalisation * torch.cos(coeffs + bk[e]).double())

    energies_predict = torch.matmul(Ztest, alpha)
    
    end.record()
    torch.cuda.synchronize()
    
    print("prediction for", ntest, "molecules time: ", start.elapsed_time(end), "ms")
    
    forces_predict, = torch.autograd.grad(-energies_predict.sum(), test_coordinates, retain_graph=True)
    
    print("Energy MAE TORCH:", torch.mean(torch.abs(energies_predict - test_energies)))
    print("Force MAE TORCH:", torch.mean(torch.abs(forces_predict.flatten() - test_forces.flatten())))
