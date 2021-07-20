import torch
import numpy as np

import argparse

from qml_lightning.models.random_features import RandomFeaturesModel

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-ntrain", type=int, default=1000)
    parser.add_argument("-ntest", type=int, default=250)
    parser.add_argument("-nreductor_samples", type=int, default=1024)
    parser.add_argument("-nbatch", type=int, default=128)
    parser.add_argument("-data", type=str, default='../../data/aspirin_dft.npz')
    
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
    
    nreductor_samples = args.nreductor_samples
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
    # elements = np.array([1, 6, 7, 8])
    nuclear_charges = np.repeat(nuclear_charges[np.newaxis,:], data['R'].shape[0], axis=0)
    
    ALL_IDX = np.arange(coords.shape[0])
    
    np.random.shuffle(ALL_IDX)
    
    train_indexes = ALL_IDX[:ntrain]
    test_indexes = ALL_IDX[ntrain:ntrain + ntest]
    
    train_indexes = ALL_IDX[:ntrain]
    test_indexes = ALL_IDX[ntrain:ntrain + ntest]
    reductor_samples = ALL_IDX[ntrain + ntest: ntrain + ntest + nreductor_samples]
    
    test_coordinates = [coords[i] for i in test_indexes]
    test_charges = [nuclear_charges[i] for i in test_indexes]
    test_energies = [energies[i] for i in test_indexes]
    test_forces = [forces[i] for i in test_indexes]
    
    model = RandomFeaturesModel(elements=elements, ntransforms=ntransforms, sigma=sigma, llambda=llambda,
                                nfeatures=nfeatures, npcas=npcas, nbatch=nbatch, npca_choice=nreductor_samples,
                                ngaussians=ngaussians, eta=eta, lmax=lmax, rcut=rcut)
    
    model.convert_from_hartree_to_kcal = False
    
    data = model.format_data(test_coordinates, test_charges, test_energies, test_forces)

    test_energies = data['energies']
    test_forces = data['forces']
    
    max_natoms = data['natom_counts'].max().item()
    
    model.get_reductors([coords[i] for i in reductor_samples], [nuclear_charges[i]for i in reductor_samples])

    model.train([coords[i] for i in train_indexes], [nuclear_charges[i] for i in train_indexes],
                [energies[i] for i in train_indexes], [forces[i] for i in train_indexes])
    
    energy_predictions, force_predictions = model.predict_torch(test_coordinates, test_charges, forces=True)

    energy_predictions = torch.cat(energy_predictions)
    force_predictions = torch.cat(force_predictions)
    
    print("Energy MAE Torch:", torch.mean(torch.abs(energy_predictions - test_energies)))
    print("Force MAE Torch:", torch.mean(torch.abs(force_predictions.flatten() - test_forces.flatten())))
    
    energy_predictions, force_predictions = model.predict_cuda(test_coordinates, test_charges, max_natoms, forces=True)
    
    print("Energy MAE CUDA:", torch.mean(torch.abs(energy_predictions - test_energies)))
    print("Force MAE CUDA:", torch.mean(torch.abs(force_predictions.flatten() - test_forces.flatten())))
