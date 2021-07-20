import torch
import numpy as np

import argparse

from qml_lightning.representations.EGTO import EGTOCuda
from qml_lightning.models.random_features import RandomFourrierFeaturesModel


def concatenate_npzs(outfile, paths):
    import os
    
    if (os.path.exists(outfile)):
        return
    
    print ("Concatenating data into singular NPZ file... this will take some time.")
    print ("Output file will be: ", outfile)
    
    all_coords = []
    all_charges = []
    all_energies = []
    all_forces = []
    
    for i, v in enumerate(paths):
        print ("Parsing data file:", v)
        data = np.load(v)
        
        coords = data['R']
        nuclear_charges = data['z']
        energies = data['E'].flatten()
        forces = data['F']
        
        for j in range(coords.shape[0]):
            all_coords.append(coords[j])
            all_charges.append(nuclear_charges)
            all_energies.append(energies[j])
            all_forces.append(forces[j])
        
    np.savez(outfile, coords=all_coords, z=all_charges, E=all_energies, F=all_forces)
    print ("Finished data prep ...")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-ntrain", type=int, default=1000)
    parser.add_argument("-ntest", type=int, default=250)
    parser.add_argument("-nreductor_samples", type=int, default=2048)
    parser.add_argument("-nbatch", type=int, default=128)
    parser.add_argument("-datas", type=str, default=['../../../data/aspirin_dft.npz', '../../../data/benzene_dft.npz', '../../../data/ethanol_dft.npz', '../../../data/malonaldehyde_dft.npz',
                                                     '../../../data/naphthalene_dft.npz', '../../../data/salicylic_dft.npz', '../../../data/toluene_dft.npz', '../../../data/uracil_dft.npz'])
    
    '''model parameters'''
    parser.add_argument("-sigma", type=float, default=3.0)
    parser.add_argument("-llambda", type=float, default=1e-12)
    parser.add_argument("-npcas", type=int, default=128)
    parser.add_argument("-nfeatures", type=int, default=8192)
    
    '''representation parameters'''
    parser.add_argument("-eta", type=float, default=2.0)
    parser.add_argument("-rcut", type=float, default=6.0)
    parser.add_argument("-lmax", type=int, default=2)
    parser.add_argument("-ngaussians", type=int, default=20)
    parser.add_argument("-forces", type=int, default=1)
    
    args = parser.parse_args()
    
    print ("---Argument Summary---")
    print (args)
    
    nreductor_samples = args.nreductor_samples
    ntrain = args.ntrain
    nbatch = args.nbatch
    ntest = args.ntest

    ngaussians = args.ngaussians
    eta = args.eta
    lmax = args.lmax
    rcut = args.rcut

    nfeatures = args.nfeatures
    npcas = args.npcas
    sigma = args.sigma
    llambda = args.llambda
    
    use_forces = args.forces
    
    paths = args.datas
    
    cuda = torch.cuda.is_available()
    n_gpus = 1 if cuda else None
    device = torch.device('cuda' if cuda else 'cpu')
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    unique_z = np.array([1.0, 6.0, 7.0, 8.0])
    
    from pathlib import Path
    path = Path(paths[0])

    concatenate_npzs(str(path.parent.absolute()) + "/all_md9.npz", paths)
    
    print ("Loading data...")
    data = np.load(str(path.parent.absolute()) + "/all_md9.npz", allow_pickle=True)
    
    coords = data['coords']
    nuclear_charges = data['z']
    energies = data['E']
    forces = data['F']
    
    print ("Finished loading data...")
    
    ALL_IDX = np.arange(len(coords))
    
    np.random.shuffle(ALL_IDX)
    
    train_indexes = ALL_IDX[:ntrain]
    test_indexes = ALL_IDX[ntrain:ntrain + ntest]
    reductor_samples = ALL_IDX[ntrain + ntest: ntrain + ntest + nreductor_samples]
    
    train_coordinates = [coords[i] for i in train_indexes]
    train_charges = [nuclear_charges[i] for i in train_indexes]
    train_energies = [energies[i] for i in train_indexes]
    train_forces = [forces[i] for i in train_indexes]
    
    test_coordinates = [coords[i] for i in test_indexes]
    test_charges = [nuclear_charges[i] for i in test_indexes]
    test_energies = [energies[i] for i in test_indexes]
    test_forces = [forces[i] for i in test_indexes]
    
    rep = EGTOCuda(species=unique_z, high_cutoff=rcut, ngaussians=ngaussians, eta=eta, lmax=lmax)

    model = RandomFourrierFeaturesModel(rep, elements=unique_z, sigma=sigma, llambda=llambda,
                                nfeatures=nfeatures, npcas=npcas, nbatch=nbatch)
    
    print ("Calculating projection matrices...")
    model.get_reductors([coords[i] for i in reductor_samples], [nuclear_charges[i]for i in reductor_samples], npcas=npcas)
    
    print ("Removing linear atomic contributions to properties...")
    model.set_subtract_self_energies(True)
    model.self_energy = torch.Tensor([0., -0.500273, 0., 0., 0., 0., -37.845355, -54.583861, -75.064579, -99.718730]).double() * 627.5095
    
    print ("Training model...")
    model.train(train_coordinates, train_charges, train_energies, train_forces if use_forces else None)
    
    data = model.format_data(test_coordinates, test_charges, test_energies, test_forces)

    test_energies = data['energies']  # self_energies are removed here also
    test_forces = data['forces']
    
    max_natoms = data['natom_counts'].max().item()
    
    energy_predictions, force_predictions = model.predict_cuda(test_coordinates, test_charges, max_natoms, forces=True)

    print("Energy MAE CUDA:", torch.mean(torch.abs(energy_predictions - test_energies)))
    print("Force MAE CUDA:", torch.mean(torch.abs(force_predictions.flatten() - test_forces.flatten())))
