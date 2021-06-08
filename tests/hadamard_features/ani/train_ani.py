import torch
import numpy as np
from tqdm import tqdm

from qml_lightning.representations.EGTO import get_elemental_gto
from qml_lightning.features.SORF import get_SORF_diagonals, get_bias, get_SORF_coefficients, get_features, get_feature_derivatives, SORFTransformCuda
from qml_lightning.representations.dimensionality_reduction import project_representation, project_derivative
from qml_lightning.representations.EGTO import ElementalGTO
from qml_lightning.utils.ani_dataloader import iter_data_buckets
import argparse

from qml_lightning.cuda import pairlist_gpu2
from qml_lightning.cuda import egto_gpu2
from qml_lightning.representations.EGTO import get_egto

from train_ani_energies import format_data
from train_ani_energies import get_reductors

from qml_lightning.models.random_features import RandomFeaturesModel

path_to_h5file = '../../data/ani-1x.h5'

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-ntrain", type=int, default=1000)
    parser.add_argument("-ntest", type=int, default=1000)
    parser.add_argument("-nreductor_samples", type=int, default=1000)
    
    parser.add_argument("-nbatch", type=int, default=64)
    
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
    
    ntrain = args.ntrain
    ntest = args.ntest
    nbatch = args.nbatch
    nreductor_samples = args.nreductor_samples
    
    ngaussians = args.ngaussians
    eta = args.eta
    lmax = args.lmax
    rcut = args.rcut
    
    ntransforms = args.ntransforms
    nfeatures = args.nfeatures
    
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
    
    data_keys = ['wb97x_tz.energy', 'wb97x_tz.forces'] 
    
    Xs = []
    Zs = []
    Es = []
    Fs = []
    
    for data in iter_data_buckets(path_to_h5file, keys=data_keys):
        Xt = data['coordinates']
        Zt = data['atomic_numbers']
        Et = data['wb97x_tz.energy']
        Ft = data['wb97x_tz.forces']
        
        for i in range(Xt.shape[0]):
            Xs.append(Xt[i])
            Zs.append(Zt)
            Es.append(Et[i])
            Fs.append(Ft[i])
    
    print ("Num Data Total:", len(Xs))
    
    ALL_IDX = np.arange(len(Xs))
    np.random.shuffle(ALL_IDX)

    train_indexes = ALL_IDX[:ntrain]
    test_indexes = ALL_IDX[ntrain:ntrain + ntest]
    reductor_samples = ALL_IDX[ntrain + ntest: ntrain + ntest + nreductor_samples]
    
    train_coordinates = [Xs[i] for i in train_indexes]
    train_charges = [Zs[i] for i in train_indexes]
    train_energies = [Es[i] for i in train_indexes]
    train_forces = [Fs[i] for i in train_indexes]
    
    test_coordinates = [Xs[i] for i in test_indexes]
    test_charges = [Zs[i] for i in test_indexes]
    test_energies = [Es[i] for i in test_indexes]
    test_forces = [Fs[i] for i in test_indexes]
    
    model = RandomFeaturesModel(elements=elements, ntransforms=ntransforms, sigma=sigma, llambda=llambda,
                                nfeatures=nfeatures, npcas=npcas, nbatch=nbatch, npca_choice=nreductor_samples,
                                ngaussians=ngaussians, eta=eta, lmax=lmax, rcut=rcut)
    
    model.calculate_self_energy(train_charges, train_energies)
    
    model.get_reductors([Xs[i] for i in reductor_samples], [Zs[i]for i in reductor_samples])
    
    model.train(train_coordinates, train_charges, train_energies, train_forces)
    
    data = model.format_data(test_coordinates, test_charges, test_energies, test_forces)
    
    test_energies = data['energies']
    test_forces = data['forces']
    
    max_natoms = data['natom_counts'].max().item()

    energy_predictions, force_predictions = model.predict_cuda(test_coordinates, test_charges, max_natoms, forces=True)
    
    print("Energy MAE CUDA:", torch.mean(torch.abs(energy_predictions - test_energies)))
    print("Force MAE CUDA:", torch.mean(torch.abs(force_predictions.flatten() - test_forces.flatten())))
    
