import torch
import numpy as np

import argparse

from qml_lightning.representations.FCHL import FCHLCuda
from qml_lightning.models.hadamard_features import HadamardFeaturesModel

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-ntrain", type=int, default=10000)
    parser.add_argument("-nbatch_train", type=int, default=256)
    parser.add_argument("-nbatch_test", type=int, default=256)
    
    '''model parameters'''
    parser.add_argument("-sigma", type=float, default=6.0)
    parser.add_argument("-llambda", type=float, default=1e-8)
    parser.add_argument("-npcas", type=int, default=128)
    parser.add_argument("-ntransforms", type=int, default=2)
    parser.add_argument("-nstacks", type=int, default=128)
  
    parser.add_argument("-rcut", type=float, default=6.0)
    parser.add_argument("-nRs2", type=int, default=23)
    parser.add_argument("-nRs3", type=int, default=22)
    parser.add_argument("-eta2", type=float, default=0.27)
    parser.add_argument("-eta3", type=float, default=5.6)
    parser.add_argument("-two_body_decay", type=float, default=2.78)
    parser.add_argument("-three_body_decay", type=float, default=2.1)
    parser.add_argument("-three_body_weight", type=float, default=60.1)

    parser.add_argument("-train_ids", type=str, default="./train_ids.npy")
    parser.add_argument("-test_ids", type=str, default="./test_ids.npy")
    args = parser.parse_args()
    
    print ("---Argument Summary---")
    print (args)

    ntrain = args.ntrain
    nbatch_train = args.nbatch_train
    nbatch_test = args.nbatch_test

    nstacks = args.nstacks
    ntransforms = args.ntransforms
    npcas = args.npcas
    
    sigma = args.sigma
    llambda = args.llambda
    
    path = '../data/qm9_data.npz'
    
    cuda = torch.cuda.is_available()
    n_gpus = 1 if cuda else None
    device = torch.device('cuda' if cuda else 'cpu')
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    data = np.load(path, allow_pickle=True)
    
    coords = data['coordinates']
    nuclear_charges = data['charges']
    energies = np.array(data['H_atomization']) * 627.5095
    
    train_IDs = np.fromfile(args.train_ids, dtype=int)
    test_indexes = np.fromfile(args.test_ids, dtype=int)
    
    unique_z = np.unique(np.concatenate(nuclear_charges)).astype(int)

    ALL_IDX = np.arange(len(coords))
    
    np.random.shuffle(train_IDs)
    
    train_indexes = train_IDs[:ntrain]
    
    train_coordinates = [coords[i] for i in train_indexes]
    train_charges = [nuclear_charges[i] for i in train_indexes]
    train_energies = [energies[i] for i in train_indexes]
  
    test_coordinates = [coords[i] for i in test_indexes]
    test_charges = [nuclear_charges[i] for i in test_indexes]
    test_energies = [energies[i] for i in test_indexes]

    rep = FCHLCuda(species=unique_z, high_cutoff=args.rcut, nRs2=args.nRs2, nRs3=args.nRs3,
                 eta2=args.eta2, eta3=args.eta3, two_body_decay=args.two_body_decay,
                 three_body_weight=args.three_body_weight, three_body_decay=args.three_body_decay)
    
    model = HadamardFeaturesModel(rep, elements=unique_z, sigma=sigma, llambda=llambda,
                                nstacks=nstacks, npcas=npcas,
                                nbatch_train=nbatch_train, nbatch_test=nbatch_test)
    
    print ("Calculating projection matrices...")
    model.get_reductors(train_coordinates, train_charges, npcas=npcas)
    
    ''' 
    For models of ntrain < 10k, switch the above to:
    
    model.get_reductors(coords, nuclear_charges , npcas=npcas)
    
    not enough sulphur environments in the training set in low data regime.
    '''
    
    model.set_subtract_self_energies(True)
    model.calculate_self_energy(train_charges, train_energies)
    
    model.train(train_coordinates, train_charges, train_energies)
    
    data = model.format_data(test_coordinates, test_charges, test_energies)

    test_energies = data['energies']

    max_natoms = data['natom_counts'].max().item()
    
    energy_predictions = model.predict(test_coordinates, test_charges, max_natoms, forces=False, use_backward=True)

    print("Energy MAE /w backwards:", torch.mean(torch.abs(energy_predictions - test_energies)))

