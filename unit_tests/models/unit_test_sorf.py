import torch
import numpy as np

import argparse
from qml_lightning.torchscript import setup # called to load torchscript libs from QML_LIGHTNING_TORCHSCRIPT_LIB

if __name__ == "__main__":
    
    energy = -406746.26498811337
    
    coordinates = np.array([[ 2.25571364, -0.38562082,  0.25521779],
                         [ 0.83507879,  1.9715759 , -0.41464916],
                         [ 2.88876512,  0.84852166,  0.26907964],
                         [ 2.14876936,  2.01964345, -0.04776138],
                         [-3.48410053,  0.49229042, -0.10826158],
                         [ 0.92033708, -0.50883859, -0.10168934],
                         [ 0.21159328,  0.68818845, -0.42443103],
                         [-0.96446081, -1.83211358, -0.39899052],
                         [-1.65259035,  0.8639225 ,  1.35082265],
                         [ 0.88341369, -2.88346755,  0.19632983],
                         [ 0.22071292, -1.79338057, -0.09240105],
                         [-2.05297017,  0.69330798,  0.24948384],
                         [-1.12429142,  0.6204384 , -0.81430244],
                         [ 0.29509341, -3.64658463,  0.43020677],
                         [ 2.66286461, -1.38969562,  0.4938589 ],
                         [ 0.26339145,  2.78767702, -0.91261049],
                         [ 3.94534538,  0.95482407,  0.58319346],
                         [ 2.74120081,  2.93792725, -0.25633347],
                         [-3.86742258, -0.54453643, -0.04496527],
                         [-4.02107013,  1.29272799,  0.46787286],
                         [-3.64746502,  0.93253809, -1.11074989]])
    
    actual_forces = np.array(
                        [[  7.89330996, -23.78912768,   4.75720496],
                         [-38.27593486, -31.65179859, -23.22267552],
                         [  5.21347896,  10.2603322 ,  13.09139135],
                         [ 50.5363484 ,   6.61211691, -15.55595476],
                         [-12.72959065,  46.13122006, -25.23051101],
                         [ -9.58919845,  34.00624946,  -2.14151264],
                         [ -4.02681556,   3.9549125 ,   2.12373134],
                         [-12.01427352, -16.60541945,   6.77979629],
                         [ 11.33458458,  11.20022527,  55.77121139],
                         [ 47.83550316, -43.44880385,  27.73867976],
                         [-59.10512603,  24.5090502 , -24.75256802],
                         [ 12.76864844, -23.085553  , -30.7630836 ],
                         [-12.56774628,  14.78405338,  -7.55859095],
                         [  0.42237233,  18.79314136, -12.5848658 ],
                         [ 15.32161419,  20.22681044,   1.20607638],
                         [ 10.33235702,  -6.90958355,  17.34088206],
                         [-11.3081238 ,  -3.24890291,  -6.62375556],
                         [-10.26088399,  -7.22121239,  13.23349569],
                         [  5.65715804,  12.01787416,   4.4747182 ],
                         [  0.27872805, -28.06318756,   8.05143185],
                         [  2.28359002, -18.47239697,  -6.13510139],])

    
    charges = np.array([6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1])
    
    ref_energy = -406746.2188
    ref_forces = np.array([[ 7.1108e+00, -2.3761e+01,  4.6022e+00],
                                 [-3.9776e+01, -3.1590e+01, -2.3346e+01],
                                 [ 5.2500e+00,  1.0356e+01,  1.2197e+01],
                                 [ 5.0961e+01,  5.8797e+00, -1.4790e+01],
                                 [-1.3412e+01,  4.5461e+01, -2.4878e+01],
                                 [-8.1629e+00,  3.4137e+01, -1.9793e+00],
                                 [-4.0663e+00,  2.8512e+00,  1.7828e+00],
                                 [-1.2617e+01, -1.6348e+01,  6.7924e+00],
                                 [ 1.1322e+01,  1.2017e+01,  5.5691e+01],
                                 [ 4.8178e+01, -4.3340e+01,  2.7648e+01],
                                 [-5.8656e+01,  2.4451e+01, -2.4338e+01],
                                 [ 1.2962e+01, -2.4175e+01, -3.0994e+01],
                                 [-1.1406e+01,  1.6746e+01, -7.4394e+00],
                                 [ 2.1097e-01,  1.8720e+01, -1.2756e+01],
                                 [ 1.5738e+01,  1.9531e+01,  1.7070e+00],
                                 [ 1.0520e+01, -6.9458e+00,  1.7073e+01],
                                 [-1.1871e+01, -3.0293e+00, -6.6352e+00],
                                 [-1.0281e+01, -6.9400e+00,  1.3242e+01],
                                 [ 5.4357e+00,  1.2351e+01,  4.4594e+00],
                                 [ 2.7494e-02, -2.8147e+01,  8.2276e+00],
                                 [ 2.5313e+00, -1.8224e+01, -6.2653e+00]])
    
    X = torch.from_numpy(coordinates).float().cuda()
    Z = torch.from_numpy(charges).float().cuda()
    
    X.requires_grad = True
    X = X.unsqueeze(0)
    Z = Z.unsqueeze(0)
    
    molIDs = torch.zeros(coordinates.shape[0], dtype=torch.int).cuda()
    atomIDs = torch.arange(coordinates.shape[0], dtype=torch.int).cuda()
    atom_counts = torch.zeros(1).fill_(coordinates.shape[0]).int().cuda()
    
    loaded = torch.jit.load('saved_models/model_sorf.pt')
    
    energy_prediction = loaded.forward(X, Z, atomIDs, molIDs, atom_counts)
    
    forces_prediction, = torch.autograd.grad(-energy_prediction.sum(), X)

    
    assert np.isclose(ref_energy,  loaded.self_energy[Z.long()].sum() + energy_prediction.cpu().item()), "predicted energy is not close to ref_energy"
    assert np.isclose(ref_forces,  forces_prediction.cpu().squeeze(0).numpy(), rtol=1e-4, atol=1e-4).all(), "predicted forces is not close to ref_forces"
    
    print ("All is well!")