import torch
import numpy as np

import argparse

import qml_lightning # load shared libs for torchscript

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-ntrain", type=int, default=1000)
    parser.add_argument("-ntest", type=int, default=1000)
    
    parser.add_argument("-train_ids", type=str, default="splits/aspirin_train_ids.npy")
    parser.add_argument("-test_ids", type=str, default="splits/aspirin_test_ids.npy")
    parser.add_argument("-path", type=str, default="../data/aspirin_dft.npz")
    
    args = parser.parse_args()
    
    print ("---Argument Summary---")
    print (args)

    ntrain = args.ntrain
    ntest = args.ntest
    
    cuda = torch.cuda.is_available()
    n_gpus = 1 if cuda else None
    device = torch.device('cuda' if cuda else 'cpu')
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    data = np.load(args.path, allow_pickle=True)
    
    if ('R' in data.keys()):
        coords = data['R']
        nuclear_charges = data['z']
        energies = data['E'].flatten()
        forces = data['F']
    else:
        coords = data['coords']
        nuclear_charges = data['nuclear_charges']
        energies = data['energies'].flatten()
        forces = data['forces']

    nuclear_charges = np.repeat(nuclear_charges[np.newaxis,:], coords.shape[0], axis=0)
    
    train_IDs = np.fromfile(args.train_ids, dtype=int)
    test_indexes = np.fromfile(args.test_ids, dtype=int)[:args.ntest]
    
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
    
    # load model
    model = torch.load("model_sorf.pt")

    #setup torch tensors from numpy objects
    xyz = torch.tensor(test_coordinates[0], device='cuda', dtype=torch.float32, requires_grad=True)
    charges = torch.tensor(test_charges[0], device='cuda', dtype=torch.float32)
    atomIDs =torch.arange(xyz.shape[0],device='cuda', dtype=torch.int32)
    molIDs = torch.zeros(xyz.shape[0], device='cuda',dtype=torch.int32)
    atom_counts= torch.tensor([xyz.shape[0]], device='cuda', dtype=torch.int32)

    #forward expects e.g [nbatch, natoms, 3], (where nbatch >= 1)
    energy = model.forward(xyz[None], charges[None], atomIDs, molIDs, atom_counts)

    #get forces from autograd
    forces_torch, = torch.autograd.grad(-energy.sum(), xyz)

    #compare
    print (forces_torch.cpu().detach().numpy())
    print (test_forces[0])
