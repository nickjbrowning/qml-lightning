import torch
import numpy as np

import argparse

from qml_lightning.representations.FCHL import FCHLCuda
from qml_lightning.models.random_features import RandomFourrierFeaturesModel

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-ntrain", type=int, default=1000)
    parser.add_argument("-nbatch_train", type=int, default=64)
    parser.add_argument("-nbatch_test", type=int, default=256)
    
    '''model parameters'''
    parser.add_argument("-sigma", type=float, default=2.0)
    parser.add_argument("-llambda", type=float, default=1e-5)
    parser.add_argument("-npcas", type=int, default=128)
    parser.add_argument("-nstacks", type=int, default=128)
    
    parser.add_argument('-rcut', type=float, default=6.0)
    parser.add_argument("-nRs2", type=int, default=24)
    parser.add_argument("-nRs3", type=int, default=20)
    parser.add_argument("-eta2", type=float, default=0.32)
    parser.add_argument("-eta3", type=float, default=2.7)
    parser.add_argument("-two_body_decay", type=float, default=1.8)
    parser.add_argument("-three_body_decay", type=float, default=0.57)
    parser.add_argument("-three_body_weight", type=float, default=13.4)
    
    parser.add_argument("-hyperparam_opt", type=int, choices=[0, 1], default=0)
    parser.add_argument("-forces", type=int, choices=[0, 1], default=1)
    
    parser.add_argument("-train_ids", type=str, default="splits/aspirin_train_ids.npy")
    parser.add_argument("-test_ids", type=str, default="splits/aspirin_test_ids.npy")
    parser.add_argument("-path", type=str, default="../data/aspirin_dft.npz")
    
    args = parser.parse_args()
    
    print ("---Argument Summary---")
    print (args)

    ntrain = args.ntrain
    nbatch_train = args.nbatch_train
    nbatch_test = args.nbatch_test
    
    nstacks = args.nstacks
    npcas = args.npcas
    
    rcut = args.rcut
    nRs2 = args.nRs2
    nRs3 = args.nRs3
    eta2 = args.eta2
    eta3 = args.eta3
    two_body_decay = args.two_body_decay
    three_body_decay = args.three_body_decay
    three_body_weight = args.three_body_weight
    
    sigma = args.sigma
    llambda = args.llambda
    
    cuda = torch.cuda.is_available()
    n_gpus = 1 if cuda else None
    device = torch.device('cuda' if cuda else 'cpu')
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    data = np.load(args.path)
    
    coords = data['R']
    nuclear_charges = data['z']
    energies = data['E'].flatten()
    forces = data['F']

    nuclear_charges = np.repeat(nuclear_charges[np.newaxis,:], coords.shape[0], axis=0)
    
    train_IDs = np.fromfile(args.train_ids, dtype=int)
    test_indexes = np.fromfile(args.test_ids, dtype=int)
    
    unique_z = np.unique(np.concatenate(nuclear_charges)).astype(int)

    ALL_IDX = np.arange(len(coords))
    
    train_indexes = train_IDs[:ntrain]
    
    train_coordinates = [coords[i] for i in train_indexes]
    train_charges = [nuclear_charges[i] for i in train_indexes]
    train_energies = [energies[i] for i in train_indexes]
    train_forces = [forces[i] for i in train_indexes]
    
    test_coordinates = [coords[i] for i in test_indexes]
    test_charges = [nuclear_charges[i] for i in test_indexes]
    test_energies = [energies[i] for i in test_indexes]
    test_forces = [forces[i] for i in test_indexes]
    
    rep = FCHLCuda(species=unique_z, rcut=rcut, nRs2=nRs2, nRs3=nRs3, eta2=eta2, eta3=eta3,
                   two_body_decay=two_body_decay, three_body_decay=three_body_decay, three_body_weight=three_body_weight)
    
    model = RandomFourrierFeaturesModel(rep, elements=unique_z, sigma=sigma, llambda=llambda,
                                nstacks=nstacks, npcas=npcas,
                                nbatch_train=nbatch_train, nbatch_test=nbatch_test)
    
    print ("Calculating projection matrices...")
    model.get_reductors(coords, nuclear_charges, npcas=npcas)
    
    print ("Subtracting linear atomic property contributions ...")
    model.set_subtract_self_energies(True)
    model.calculate_self_energy(train_charges, train_energies)
    
    if (args.hyperparam_opt):
        model.hyperparam_opt_nested_cv(train_coordinates, train_charges, train_energies, F=train_forces if args.forces else None)
    
    model.train(train_coordinates, train_charges, train_energies, train_forces if args.forces else None)
    
    data = model.format_data(test_coordinates, test_charges, test_energies, test_forces if args.forces else None)

    test_energies = data['energies']
    max_natoms = data['natom_counts'].max().item()

    if (args.forces):
        test_forces = data['forces']
        energy_predictions, force_predictions = model.predict(test_coordinates, test_charges, max_natoms, forces=True)
    else:
        energy_predictions = model.predict(test_coordinates, test_charges, max_natoms, forces=False)

    print("Energy MAE /w backwards:", torch.mean(torch.abs(energy_predictions - test_energies)))
    
    if (args.forces):
        print("Force MAE /w backwards:", torch.mean(torch.abs(force_predictions - test_forces)))
    
    # model.save_jit_model()
