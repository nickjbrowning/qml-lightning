import torch
import numpy as np

import argparse

from qml_lightning.representations.FCHL import FCHLCuda
from qml_lightning.models.hadamard_features import HadamardFeaturesModel

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-ntrain", type=int, default=400)
    parser.add_argument("-ntest", type=int, default=1000)
    parser.add_argument("-nbatch_train", type=int, default=64)
    parser.add_argument("-nbatch_test", type=int, default=256)
    
    '''model parameters'''
    parser.add_argument("-sigma", type=float, default=2.0)
    parser.add_argument("-llambda", type=float, default=1e-5)
    parser.add_argument("-npcas", type=int, default=128)
    parser.add_argument("-ntransforms", type=int, default=2)
    parser.add_argument("-nstacks", type=int, default=128)
    
    parser.add_argument('-rcut', type=float, default=6.0)
    parser.add_argument("-nRs2", type=int, default=24)
    parser.add_argument("-nRs3", type=int, default=20)
    parser.add_argument("-eta2", type=float, default=0.32)
    parser.add_argument("-eta3", type=float, default=2.7)
    parser.add_argument("-two_body_decay", type=float, default=1.8)
    parser.add_argument("-three_body_decay", type=float, default=0.57)
    parser.add_argument("-three_body_weight", type=float, default=13.4)

    mols = ['aspirin', 'salicylic', 'benzene', 'ethanol', 'malonaldehyde', 'naphthalene', 'toluene', 'uracil']
    
    args = parser.parse_args()
    
    print ("---Argument Summary---")
    print (args)

    ntrain = args.ntrain
    ntest = args.ntest
    
    nbatch_train = args.nbatch_train
    nbatch_test = args.nbatch_test
    
    nstacks = args.nstacks
    ntransforms = args.ntransforms
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
    
    train_coordinates = []
    train_charges = []
    train_energies = []
    train_forces = []
    
    test_coordinates = []
    test_charges = []
    test_energies = []
    test_forces = []
    
    for v in mols:
        
        print ('Loading:', v)
        data = np.load('../data/' + v + '_dft.npz')
        
        coords = data['R']
        nuclear_charges = data['z']
        E = data['E'].flatten()
        F = data['F']
    
        nuclear_charges = np.repeat(nuclear_charges[np.newaxis,:], coords.shape[0], axis=0)
        
        sub_train_ids = np.fromfile('splits/' + v + '_train_ids.npy', dtype=int)
        sub_test_ids = np.fromfile('splits/' + v + '_test_ids.npy', dtype=int)
        
        np.random.shuffle(sub_train_ids)
        np.random.shuffle(sub_test_ids)
        
        for j in sub_train_ids[:ntrain]:

            train_coordinates.append(coords[j])
            train_charges.append(nuclear_charges[j])
            train_energies.append(E[j])
            train_forces.append(F[j])
            
        for j in sub_test_ids[:ntest]:

            test_coordinates.append(coords[j])
            test_charges.append(nuclear_charges[j])
            test_energies.append(E[j])
            test_forces.append(F[j])
       
    unique_z = np.unique(np.concatenate(train_charges)).astype(int)
    
    rep = FCHLCuda(species=unique_z, rcut=rcut, nRs2=nRs2, nRs3=nRs3, eta2=eta2, eta3=eta3,
                   two_body_decay=two_body_decay, three_body_decay=three_body_decay, three_body_weight=three_body_weight)
    
    model = HadamardFeaturesModel(rep, elements=unique_z, sigma=sigma, llambda=llambda,
                                nstacks=nstacks, ntransforms=ntransforms, npcas=npcas,
                                nbatch_train=nbatch_train, nbatch_test=nbatch_test)
    
    print ("Calculating projection matrices...")
    model.get_reductors(train_coordinates, train_charges, npcas=npcas)
    
    print ("Subtracting linear atomic property contributions ...")
    model.set_subtract_self_energies(True)
    model.calculate_self_energy(train_charges, train_energies)
   
    model.train(train_coordinates, train_charges, train_energies, train_forces)
    
    model.save_jit_model(file_name='all_md17.pt')
    
    data = model.format_data(test_coordinates, test_charges, test_energies, test_forces)

    test_energies = data['energies']
    test_forces = data['forces']
    max_natoms = data['natom_counts'].max().item()

    energy_predictions, force_predictions = model.predict(test_coordinates, test_charges, max_natoms, forces=True)

    print("Energy MAE /w backwards:", torch.mean(torch.abs(energy_predictions - test_energies)))
    print("Force MAE /w backwards:", torch.mean(torch.abs(force_predictions - test_forces)))
    
