import torch
from torch import nn
import numpy as np


class ElementNet(nn.Module):

    def __init__(self, input_size, output_size, n_layers, activation, net_type, layersizes=None):
        super(ElementNet, self).__init__()
        
        if net_type == 'embedding':
            n_neurons = [input_size * i for i in range(2, 2 * (n_layers + 1), 2)]
            n_neurons.insert(0, input_size)
        elif net_type == 'fitting' or net_type == 'resnet':
            
            if layersizes is not None:
                if net_type == 'resnet':
                    assert all(x == layersizes[0] for x in layersizes), \
                        'For resnet style networks, all layers need the same size'
                n_neurons = [input_size, 1024, *layersizes]
            else:
                n_neurons = [input_size] * n_layers
        else:
            raise ValueError('Wrong Net Type. Try embedding or fitting.')

        if net_type == 'fitting':
            layers = [RandomFeaturesLayer(input_size, 1024)]

            # for i in range(1, len(n_neurons) - 1):
            #    layers.append(DenseLayer(n_neurons[i], n_neurons[i + 1], activation=activation, net_type=net_type))
                
            for i in range(1, len(n_neurons) - 1):
                layers.append(DenseLayer(n_neurons[i], n_neurons[i + 1], activation=activation, net_type=net_type))
                # layers.append(Rffmodule(n_neurons[i], 512, n_neurons[i + 1]))
                
            print ("NUMLAYERS:", len(layers))
        elif net_type == 'resnet':
            layers = [DenseLayer(n_neurons[i], n_neurons[i + 1], activation=activation, net_type=net_type)
                      for i in range(1, len(n_neurons) - 1)]
            layers.insert(0, DenseLayer(n_neurons[0], n_neurons[1], activation=activation, net_type='fitting'))
            # layers.insert(0, RandomFeaturesLayer(input_size, 128))
        
        layers.append(DenseLayer(n_neurons[-1], output_size, activation=None, net_type=None))

        self.element_net = nn.Sequential(*layers)

    def forward(self, features):
        return self.element_net(features)


class RandomFeaturesLayer(nn.Module):

    def __init__(self, n_input_features, n_output_features):
        super(RandomFeaturesLayer, self).__init__()
        
        self.W = torch.randn(n_input_features, n_output_features, requires_grad=True).cuda()
        self.b = torch.rand(n_output_features, requires_grad=True).cuda()

    def forward(self, input_features):

        # return torch.cos(torch.matmul(input_features, self.W) + (self.b * 2.0 * np.pi))
        
        return torch.cos(torch.matmul(input_features, self.W) + (self.b * 2.0 * np.pi))


from qml_lightning.features.SORF import SORFTransformCuda


class HadamardModule(nn.Module):

    def __init__(self, input_dim, npcas, out_dim):
        
        super(HadamardModule, self).__init__()
        
        D = np.random.uniform(-1, 1, (2, int(out_dim / npcas), npcas))
        D[D > 0.0] = 1.0
        D[D < 0.0] = -1.0
        
        self.out_dim = out_dim
        self.d = torch.from_numpy(D).float().cuda()
        
        self.coeff_normalisation = np.sqrt(npcas) / 3.0
        
        self.projector = torch.rand(input_dim, npcas, requires_grad=True, device=torch.device('cuda'))
        
        self.b = torch.rand(out_dim).cuda()

    def forward(self, x):
        
        x_ = torch.matmul(x, self.projector)

        test = SORFTransformCuda.apply(x_.flatten(start_dim=0, end_dim=1), self.d, self.coeff_normalisation, 2)
        
        return torch.cos(test.flatten(start_dim=1, end_dim=2).reshape(x_.shape[0], x.shape[1], self.out_dim) + (self.b * 2.0 * np.pi))


class Rffmodule(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim):
        super(Rffmodule, self).__init__()

        self.lin_in_rff = torch.nn.Linear(input_dim, hidden_dim)
        nn.init.uniform_(self.lin_in_rff.bias, 0, 2 * torch.pi) 
        nn.init.normal_(self.lin_in_rff.weight, 0, 1) 

        self.lin_out_rff = torch.nn.Linear(hidden_dim, out_dim)

        self.lin_in_sig = torch.nn.Linear(input_dim, hidden_dim)
        self.lin_out_sig = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):

        x_rff = self.lin_in_rff(x)
        x_rff = torch.cos(x_rff)
        x_rff = self.lin_out_rff(x_rff)

        x_sig = self.lin_in_sig(x)
        x_sig = torch.tanh(x_sig)
        x_sig = self.lin_out_sig(x_sig)

        return x_rff * x_sig

    
class DenseLayer(nn.Linear):

    def __init__(self, n_input_features, n_output_features, activation, net_type, bias=True):
        super(DenseLayer, self).__init__(n_input_features, n_output_features, bias)
        self.activation = activation
        self.net_type = net_type

    def forward(self, input_features):
        
        y = super(DenseLayer, self).forward(input_features)
        
        if self.activation:
            y = self.activation(y)
            
        if self.net_type == 'embedding':
            y += torch.cat(input_features, input_features)
        elif self.net_type == 'resnet':
            y = y + input_features
        return y


class ElementMask(nn.Module):

    def __init__(self, nuclear_charges, device=torch.device('cpu')):
        super(ElementMask, self).__init__()
        self.n_elements = len(nuclear_charges)
        max_elements = int(max(nuclear_charges) + 1)

        self.gate = nn.Embedding(max_elements, self.n_elements)
        
        weights = torch.zeros(max_elements, self.n_elements, device=device, dtype=torch.float)
        
        for idx, nc in enumerate(nuclear_charges):
            weights[nc, idx] = 1.0
            
        self.gate.weight.data = weights
        self.gate.weight.requires_grad = False

    def forward(self, atomic_numbers):
        return self.gate(atomic_numbers)
