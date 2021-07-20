import torch
from torch import nn


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
                n_neurons = [input_size, *layersizes]
            else:
                n_neurons = [input_size] * n_layers
        else:
            raise ValueError('Wrong Net Type. Try embedding or fitting.')

        if net_type == 'fitting':
            layers = [DenseLayer(n_neurons[i], n_neurons[i + 1], activation=activation, net_type=net_type)
                      for i in range(len(n_neurons) - 1)]
        elif net_type == 'resnet':
            layers = [DenseLayer(n_neurons[i], n_neurons[i + 1], activation=activation, net_type=net_type)
                      for i in range(1, len(n_neurons) - 1)]
            layers.insert(0, DenseLayer(n_neurons[0], n_neurons[1], activation=activation, net_type='fitting'))

        layers.append(DenseLayer(n_neurons[-1], output_size, activation=None, net_type=None))

        self.element_net = nn.Sequential(*layers)

    def forward(self, features):
        return self.element_net(features)


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
