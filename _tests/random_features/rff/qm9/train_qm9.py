import torch
import numpy as np

import argparse

from qml_lightning.representations.EGTO import EGTOCuda
from qml_lightning.models.random_features import RandomFourrierFeaturesModel

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-ntrain", type=int, default=1000)
    parser.add_argument("-ntest", type=int, default=1000)
    parser.add_argument("-nreductor_samples", type=int, default=1000)
    parser.add_argument("-nbatch", type=int, default=64)
    
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
    
    path = '../../../data/qm9_data.npz'
    
    cuda = torch.cuda.is_available()
    n_gpus = 1 if cuda else None
    device = torch.device('cuda' if cuda else 'cpu')
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    data = np.load(path, allow_pickle=True)
    
    coords = data['coordinates']
    nuclear_charges = data['charges']
    energies = np.array(data['H_atomization']) * 627.503
    
    new_coords = []
    new_charges = []
    new_energies = []
    
    for i in range(len(coords)):
        
        if (9.0 in nuclear_charges[i]):
            continue
        
        new_coords.append(coords[i])
        new_charges.append(nuclear_charges[i])
        new_energies.append(energies[i])
        
    unique_z = np.unique(np.concatenate(new_charges))

    ALL_IDX = np.arange(len(new_coords))
    
    np.random.shuffle(ALL_IDX)
    
    train_indexes = ALL_IDX[:ntrain]
    test_indexes = ALL_IDX[ntrain:ntrain + ntest]
    
    train_indexes = ALL_IDX[:ntrain]
    test_indexes = ALL_IDX[ntrain:ntrain + ntest]
    reductor_samples = ALL_IDX[ntrain + ntest: ntrain + ntest + nreductor_samples]
    
    train_coordinates = [new_coords[i] for i in train_indexes]
    train_charges = [new_charges[i] for i in train_indexes]
    train_energies = [new_energies[i] for i in train_indexes]
    
    test_coordinates = [new_coords[i] for i in test_indexes]
    test_charges = [new_charges[i] for i in test_indexes]
    test_energies = [new_energies[i] for i in test_indexes]

    rep = EGTOCuda(species=unique_z, high_cutoff=rcut, ngaussians=ngaussians, eta=eta, lmax=lmax)

    model = RandomFourrierFeaturesModel(rep, elements=unique_z, sigma=sigma, llambda=llambda,
                                nfeatures=nfeatures, npcas=npcas, nbatch=nbatch)
    
    print ("Calculating projection matrices...")
    model.get_reductors([new_coords[i] for i in reductor_samples], [new_charges[i]for i in reductor_samples], npcas=npcas)
    
    # print ("Subtracting linear atomic property contributions ...")
    # model.set_subtract_self_energies(True)
    # model.calculate_self_energy(train_charges, train_energies)
    
    print ("Training model...")

    model.train(train_coordinates, train_charges, train_energies)
    
    data = model.format_data(test_coordinates, test_charges, test_energies)

    test_energies = data['energies']

    max_natoms = data['natom_counts'].max().item()
    
    energy_predictions = model.predict_cuda(test_coordinates, test_charges, max_natoms, forces=False)
    
    print("Energy MAE CUDA:", torch.mean(torch.abs(energy_predictions - test_energies)))
    
    energy_predictions = model.predict_torch(test_coordinates, test_charges, max_natoms, forces=False)
    
    print("Energy MAE torch:", torch.mean(torch.abs(energy_predictions - test_energies)))
