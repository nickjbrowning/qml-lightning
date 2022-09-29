import torch
import numpy as np

import argparse

from qml_lightning.representations.FCHL import FCHLCuda
from qml_lightning.utils import util_ops

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-rcut', type=float, default=6.0)
    parser.add_argument("-nRs2", type=int, default=100)
    parser.add_argument("-nRs3", type=int, default=50)
    parser.add_argument("-eta2", type=float, default=0.32)
    parser.add_argument("-eta3", type=float, default=2.7)
    parser.add_argument("-two_body_decay", type=float, default=1.8)
    parser.add_argument("-three_body_decay", type=float, default=0.57)
    parser.add_argument("-three_body_weight", type=float, default=13.4)

    parser.add_argument("-path", type=str, default="../../examples/data/aspirin_dft.npz")
    
    args = parser.parse_args()
    
    print ("---Argument Summary---")
    print (args)

    rcut = args.rcut
    nRs2 = args.nRs2
    nRs3 = args.nRs3
    eta2 = args.eta2
    eta3 = args.eta3
    two_body_decay = args.two_body_decay
    three_body_decay = args.three_body_decay
    three_body_weight = args.three_body_weight
    
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
    
    unique_z = np.unique(np.concatenate(nuclear_charges)).astype(int)

    ALL_IDX = np.arange(len(coords))
    
    coordinates = [coords[i] for i in range (1)]
    charges = [nuclear_charges[i] for i in range (1)]
    energies = [energies[i] for i in range (1)]
    forces = [forces[i] for i in range (1)]
    
    rep = FCHLCuda(species=unique_z, rcut=rcut, nRs2=nRs2, nRs3=nRs3, eta2=eta2, eta3=eta3,
                   two_body_decay=two_body_decay, three_body_decay=three_body_decay, three_body_weight=three_body_weight)
    
    data = util_ops.format_data(coordinates, charges)
    
    X = data['coordinates']
    Z = data['charges']
    atomIDs = data['atomIDs']
    molIDs = data['molIDs']
    natom_counts = data['natom_counts']

    rep = rep.get_representation(X, Z, atomIDs, molIDs, natom_counts)

    rep = rep.cpu().numpy()
    
    print (rep.shape)
    
    import matplotlib.pyplot as plt
    
    for i in range(rep.shape[1]):
        plt.plot(rep[0, i])
        
    plt.show()
