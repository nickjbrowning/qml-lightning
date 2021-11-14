import itertools

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from qml_lightning.models.neural_net.net_utils import ElementNet, ElementMask


class SimpleFeedForwardNetwork(pl.LightningModule):

    def __init__(self, species, fingerprint, layersizes, activation, learning_rate, loss_function,
                 force_training=False, n_layers=2, output_size=1, net_type='fitting'):
        
        super(SimpleFeedForwardNetwork, self).__init__()

        n_nets = len(species)
      
        self.element_mask = ElementMask(species, device=fingerprint.device)
        self.fingerprint = fingerprint
        self.element_to_id = self.fingerprint.element_to_id
        
        self.networks = nn.ModuleList(
            [
                ElementNet(
                    input_size=self.fingerprint.fp_size,
                    output_size=output_size,
                    layersizes=layersizes,
                    n_layers=n_layers,
                    activation=activation,
                    net_type=net_type
                )
                for _ in range(n_nets)
            ]
        )
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.force_training = force_training
    
    def forward(self, coordinates, nuclear_charges, natom_counts):

        fingerprint = self.fingerprint(coordinates, nuclear_charges, natom_counts)

        return torch.sum(self.element_mask(nuclear_charges) * 
                         torch.cat([net(fingerprint) for net in self.networks], dim=2), dim=2)

    def get_forces(self, coordinates, energy):
        derivative = torch.autograd.grad(energy.sum(), coordinates,
                                         create_graph=True,
                                         retain_graph=True)[0]
        return -derivative

    def training_step(self, batch, batch_idx):
        coordinates, nuclear_charges, ref_energies, ref_forces, natom_counts = batch
        
        energies = self.forward(coordinates, nuclear_charges, natom_counts)
        
        forces = None

        if self.force_training:
            forces = self.get_forces(coordinates, energies)

        loss = self.loss_function(torch.sum(energies, dim=1), ref_energies, forces, ref_forces)
 
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        coordinates, nuclear_charges, ref_energies, ref_forces, natom_counts = batch
        coordinates.requires_grad = True
        energies = self.forward(coordinates, nuclear_charges, natom_counts)
        forces = None

        if self.force_training:
            forces = self.get_forces(coordinates, energies)
            
        torch.set_grad_enabled(False)
        
        loss = self.loss_function(torch.sum(energies, dim=1), ref_energies, forces, ref_forces)
        
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)

        energy_mae = torch.mean(torch.abs(torch.sum(energies, dim=1) - ref_energies))
        energy_rmse = torch.sqrt(torch.mean(torch.pow(torch.sum(energies, dim=1) - ref_energies, 2)))
        
        self.log('val_energy_mae', energy_mae)
        
        self.log('val_energy_rmse', energy_rmse)
        
        if self.force_training:
            force_mae = torch.mean(torch.abs(forces - ref_forces))
            force_rmse = torch.sqrt(torch.mean(torch.pow(forces - ref_forces, 2)))
            self.log('val_force_mae', force_mae)
            self.log('val_force_rmse', force_rmse)
            
        return loss

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        coordinates, nuclear_charges, ref_energies, ref_forces, natom_counts = batch
        coordinates.requires_grad = True
        energies = self.forward(coordinates, nuclear_charges, natom_counts)
        forces = None

        if self.force_training:
            forces = self.get_forces(coordinates, energies)
            
        torch.set_grad_enabled(False)
        
        loss = self.loss_function(torch.sum(energies, dim=1), ref_energies, forces, ref_forces)
        
        energy_mae = torch.mean(torch.abs(torch.sum(energies, dim=1) - ref_energies))

        energy_rmse = torch.sqrt(torch.mean(torch.pow(torch.sum(energies, dim=1) - ref_energies, 2)))

        self.log('test_energy_mae', energy_mae)
        self.log('test_energy_rmse', energy_rmse)
        
        if self.force_training:
            force_mae = torch.mean(torch.abs(forces - ref_forces))
            force_rmse = torch.sqrt(torch.mean(torch.pow(forces - ref_forces, 2)))
            self.log('test_force_mae', force_mae)
            self.log('test_force_rmse', force_rmse)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # return {
       # 'optimizer': optimizer,
       # 'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, threshold=1e-4),
       # 'monitor': 'val_loss'
        # }
        
        return optimizer
