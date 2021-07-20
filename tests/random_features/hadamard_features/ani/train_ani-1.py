import torch
import numpy as np
from tqdm import tqdm

from qml_lightning.utils.ani1_dataloader import iter_data_buckets
import argparse

from qml_lightning.models.hadamard_features import HadamardFeaturesModel
from qml_lightning.representations.EGTO import EGTOCuda

import h5py

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-ntrain", type=int, default=1000)
    parser.add_argument("-ntest", type=int, default=10000)
    parser.add_argument("-nreductor_samples", type=int, default=1000)
    
    parser.add_argument("-nbatch", type=int, default=64)
    
    '''model parameters'''
    parser.add_argument("-sigma", type=float, default=2.0)
    parser.add_argument("-llambda", type=float, default=1e-12)
    parser.add_argument("-npcas", type=int, default=128)
    parser.add_argument("-nfeatures", type=int, default=8192)
    parser.add_argument("-ntransforms", type=int, default=1)
    
    '''representation parameters'''
    parser.add_argument("-eta", type=float, default=2.0)
    parser.add_argument("-rcut", type=float, default=6.0)
    parser.add_argument("-lmax", type=int, default=2)
    parser.add_argument("-ngaussians", type=int, default=20)
    
    args = parser.parse_args()
    
    print ("---Argument Summary---")
    print (args)
    
    ntrain = args.ntrain
    ntest = args.ntest
    nbatch = args.nbatch
    nreductor_samples = args.nreductor_samples
    
    ngaussians = args.ngaussians
    eta = args.eta
    lmax = args.lmax
    rcut = args.rcut
   
    nfeatures = args.nfeatures
    ntransforms = args.ntransforms
    npcas = args.npcas
    
    sigma = args.sigma
    llambda = args.llambda
    
    cuda = torch.cuda.is_available()
    n_gpus = 1 if cuda else None
    device = torch.device('cuda' if cuda else 'cpu')
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    elements = np.array([1, 6, 7, 8])
   
    Xs = []
    Zs = []
    Es = []
    Fs = []
    
    element2charge = {'H': 1.0, 'C': 6.0, 'N': 7.0, 'O':8.0}
    
    path_to_h5file = '../../../data/ani-1/'
    h5files = ['ani_gdb_s01.h5', 'ani_gdb_s03.h5', 'ani_gdb_s04.h5', 'ani_gdb_s06.h5', 'ani_gdb_s07.h5', 'ani_gdb_s08.h5']
    
    print ("Loading data...")
    for file in h5files:
        with h5py.File(path_to_h5file + file, 'r') as f:
            
            for grp in f.values():
                for key_group in grp.keys():
                    
                    sub_grp = grp[key_group]
                    
                    keys = sub_grp.keys()
                    
                    data = dict((k, sub_grp[k][()]) for k in keys)
    
                    coordinates = data['coordinates']
                    energies = data['energies']
                    charges = np.array([element2charge[v.decode('UTF-8')] for v in  data['species']])
                    
                    for i in range(coordinates.shape[0]):
                        Xs.append(coordinates[i])
                        Zs.append(charges)
                        Es.append(energies[i])
    
    print ("Finished loading data...")
    
    ALL_IDX = np.arange(len(Xs))
    np.random.shuffle(ALL_IDX)

    train_indexes = ALL_IDX[:ntrain]
    test_indexes = ALL_IDX[ntrain:ntrain + ntest]
    reductor_samples = ALL_IDX[ntrain + ntest: ntrain + ntest + nreductor_samples]

    train_coordinates = [Xs[i]  for i in train_indexes]
    train_charges = [Zs[i] for i in train_indexes]
    train_energies = [Es[i] for i in train_indexes] 

    test_coordinates = [Xs[i] for i in test_indexes]
    test_charges = [Zs[i] for i in test_indexes]
    test_energies = [Es[i] for i in test_indexes]

    representation = EGTOCuda(species=elements, high_cutoff=rcut, ngaussians=ngaussians, eta=eta, lmax=lmax)
    
    model = HadamardFeaturesModel(representation, elements=elements, ntransforms=ntransforms, sigma=sigma, llambda=llambda,
                                nfeatures=nfeatures, npcas=npcas, nbatch=nbatch)
    
    print ("Calculating projection matrices...")
    model.get_reductors([Xs[i] for i in reductor_samples], [Zs[i]for i in reductor_samples], npcas=npcas)
    
    print ("Subtracting linear atomic contributions...")
    model.set_convert_hartree2kcal(True)
    model.set_subtract_self_energies(True)
    model.calculate_self_energy(train_charges, train_energies)
    # model.self_energy = torch.Tensor([0., -0.600952980000, 0., 0., 0., 0., -38.08316124000, -54.70775770000, -75.19446356000, 0]).double() * 627.503
    
    print ("Training model...")
    model.train(train_coordinates, train_charges, train_energies)
    
    data = model.format_data(test_coordinates, test_charges, test_energies)
    
    test_energies = data['energies']
    test_forces = data['forces']
    
    max_natoms = data['natom_counts'].max().item()

    energy_predictions = model.predict_cuda(test_coordinates, test_charges, max_natoms)
    
    print("Energy MAE CUDA:", torch.mean(torch.abs(energy_predictions - test_energies)))
    print("Energy RMSE CUDA:", torch.sqrt(torch.mean(torch.pow(energy_predictions - test_energies, 2.0))))
