import torch
import numpy as np

import argparse
import json
from qml_lightning.representations.FCHL import FCHLCuda
from qml_lightning.models.hadamard_features import HadamardFeaturesModel


def read_datafile(fname):
    f = open(fname, 'r')
    
    lines = f.readlines()
    
    curr = 0
    element2charge = {'H': 1.0, 'C': 6.0, 'N': 7.0, 'O': 8.0}
    
    all_charges = []
    all_coordinates = []
    all_energies = []
    all_forces = []
    
    while (curr < len(lines)):
        natoms = int(lines[curr])
        
        energy = float(lines[curr + 1].split("=")[3].split(" ")[0])
        
        elements = []
        coordinates = np.zeros((natoms, 3))
        forces = np.zeros((natoms, 3))
        
        for i in range(curr + 2, curr + 2 + natoms):
      
            data = lines[i].split()
            
            element = data[0]
            c = np.array([float(v) for v in data[1:4]])
            f = np.array([float(v) for v in data[4:]])
            
            elements.append(element)
            coordinates[i - (curr + 2)] = c
            forces[i - (curr + 2)] = f
        
        charges = np.array([element2charge[v] for v in elements])
       
        all_charges.append(charges)
        all_coordinates.append(coordinates)
        all_energies.append(energy)
        all_forces.append(forces)
        
        curr += 2 + natoms
    return np.array(all_charges), np.array(all_coordinates), np.array(all_energies), np.array(all_forces)

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-nbatch_train", type=int, default=64)
    parser.add_argument("-nbatch_test", type=int, default=256)
    
    '''model parameters'''
    parser.add_argument("-sigma", type=float, default=2.0)
    parser.add_argument("-llambda", type=float, default=1e-5)
    parser.add_argument("-npcas", type=int, default=256)
    parser.add_argument("-ntransforms", type=int, default=1)
    parser.add_argument("-nstacks", type=int, default=64)
    
    parser.add_argument('-rcut', type=float, default=6.0)
    parser.add_argument("-nRs2", type=int, default=24)
    parser.add_argument("-nRs3", type=int, default=20)
    parser.add_argument("-eta2", type=float, default=0.32)
    parser.add_argument("-eta3", type=float, default=2.7)
    parser.add_argument("-two_body_decay", type=float, default=1.8)
    parser.add_argument("-three_body_decay", type=float, default=0.57)
    parser.add_argument("-three_body_weight", type=float, default=13.4)
    
    parser.add_argument("-train_data", type=str, default="../data/dataset_3BPA/train_300K.xyz")
    parser.add_argument("-test_data", type=str, default="../data/dataset_3BPA/test_300K.xyz")
    
    args = parser.parse_args()
    
    print ("---Argument Summary---")
    print (args)

    nbatch_train = args.nbatch_train
    nbatch_test = args.nbatch_test
    
    train_data = args.train_data
    test_data = args.test_data
    
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
   
    train_charges, train_coordinates, train_energies, train_forces = read_datafile(args.train_data)
    test_charges, test_coordinates, test_energies, test_forces = read_datafile(args.test_data)
    
    unique_z = np.unique(np.concatenate(train_charges)).astype(int)
        
    rep = FCHLCuda(species=unique_z, rcut=rcut, nRs2=nRs2, nRs3=nRs3, eta2=eta2, eta3=eta3,
                   two_body_decay=two_body_decay, three_body_decay=three_body_decay, three_body_weight=three_body_weight)
    
    model = HadamardFeaturesModel(rep, elements=unique_z, sigma=sigma, llambda=llambda,
                                nstacks=nstacks, ntransforms=ntransforms, npcas=npcas,
                                nbatch_train=nbatch_train, nbatch_test=nbatch_test)
    
    print ("Note: results are in eV")
    
    print ("Calculating projection matrices...")
    model.get_reductors(train_coordinates, train_charges, npcas=npcas)
    
    print ("Subtracting linear atomic property contributions ...")
    model.set_subtract_self_energies(True)
    model.self_energy = torch.Tensor([0., -13.587222780835477, 0., 0., 0., 0., -1029.4889999855063, -1484.9814568572233, -2041.9816003861047]).double()
  
    model.train(train_coordinates, train_charges, train_energies, train_forces)
    
    data = model.format_data(test_coordinates, test_charges, test_energies, test_forces)

    test_energies = data['energies']
    test_forces = data['forces']
    max_natoms = data['natom_counts'].max().item()

    energy_predictions, force_predictions = model.predict(test_coordinates, test_charges, max_natoms, forces=True)

    print("Energy MAE:", torch.mean(torch.abs(energy_predictions - test_energies)))
    print("Force MAE:", torch.mean(torch.abs(force_predictions - test_forces)))
    
    print("Energy RMSE:", torch.sqrt(torch.mean(torch.pow(energy_predictions - test_energies, 2))))
    print("Force RMSE:", torch.sqrt(torch.mean(torch.pow(force_predictions - test_forces, 2))))
