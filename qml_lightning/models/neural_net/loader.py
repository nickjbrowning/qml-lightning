import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


class SimpleNetDataModule(pl.LightningDataModule):

    def __init__(self, coords, nuclear_charges, energies, forces, ntrain=1000, nvalidation=1000, ntest=5000, batch_size=32, num_workers=1):
        super().__init__()
        self.coords = coords
        self.nuclear_charges = nuclear_charges
        self.energies = energies
        self.forces = forces
        self.batch_size = batch_size
        
        self.ntrain = ntrain
        self.nvalidation = nvalidation
        self.ntest = ntest
        self.num_workers = num_workers

    def __len__(self):
        return len(self.energies)

    def split_data(self, train_split=0.7):
        self.train_size = self.ntrain
        self.val_size = self.nvalidation
        self.test_size = self.coords.shape[0] - self.train_size - self.val_size

    def setup(self, train_split=0.7):
        data_full = SimpleNetDataloader(self.coords, self.nuclear_charges, self.energies, self.forces)
        self.data_train, self.data_val, self.data_test = random_split(data_full, [self.train_size, self.val_size, self.test_size])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)


class SimpleNetDataloader(Dataset):

    def __init__(self, coords, nuclear_charges, energies, forces):
        self.coords = coords
        self.nuclear_charges = nuclear_charges
        self.energies = energies
        self.forces = forces
        
        self.hartree2kcalmol = 627.5095
        self.ev2kcalmol = 23.06
        
        self.self_energy = np.array([0., -0.500273, 0., 0., 0., 0., -37.845355, -54.583861, -75.064579, -99.718730])
        
        self.convert_energies()

    def __len__(self):
        return len(self.energies)

    def convert_energies(self):
        self_interaction = self.self_energy[self.nuclear_charges].sum(axis=1) 
        self.energies = (self.energies) - self_interaction
        self.forces = self.forces 

    def __getitem__(self, sample):
        return torch.tensor(self.coords[sample], requires_grad=True).float().cuda(), \
               torch.tensor(self.nuclear_charges[sample]).long().cuda(), \
               torch.tensor(self.energies[sample]).float().cuda(), \
               torch.tensor(self.forces[sample]).float().cuda()
