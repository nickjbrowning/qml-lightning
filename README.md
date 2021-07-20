# QML-Lightning - Do Not Share Publically!

<img src="./images/qml_lightning.png" alt="drawing" width="400"/>

GPU-Accelerated Kernel Methods and Representations for Quantum Machine Learning. 

This package includes both the Random Fourrier Feature method (cite: TODO), as well as Structured Orthogonal Random Features (cite: TODO). it additionally contains a highly efficient 3-body representation thats designed specifically to work well on GPUs.

# Requirements

```
pytorch
numpy
scipy
tqdm
h5py
cuda C compilers, e.g https://developer.nvidia.com/hpc-sdk
```

# Installation

cd to the directory containing setup.py and type

```bash
python3 setup.py build
```

if you've got multiple versions of CUDA, you may need to prepend this command to let nvcc know where to find the correct CUDA libraries, eg:

```bash
LDFLAGS=-L/opt/nvidia/hpc_sdk/Linux_x86_64/20.11/cuda python3 setup.py build

```

Once built, set `PYTHONPATH` to the following build directory, e.g in your `.bashrc` file:

```bash
export PYTHONPATH=/home/nick/git/QMLightning/build/lib.linux-x86_64-3.8:$PYTHONPATH

```

Now you should be good to go!

# Use

Many examples of performing property learning for both random and Hadamard features can be found in the qml_lightning/tests folder. In this example we'll use the MD9 dataset (TODO: cite) with Hadamard features located here: `tests/hadamard_features/md9/train_single.py`

First, download the MD9 dataset by typing the following command in the `tests/` folder

```
make get-data
```

This will download the MD9 datasets, as well as the QM9 dataset.

now run the following command to do some ML, learning only energies:

```python
python3 train_single.py -ntrain 1000 -ntest 500 -nfeatures 8192 -npcas 128 -sigma 3.0 -llambda 1e-12 -forces 0
```

and learning energies + forces:

```python
python3 train_single.py -ntrain 1000 -ntest 500 -nfeatures 8192 -npcas 128 -sigma 3.0 -llambda 1e-12 -forces 1
```

Additionally, in this folder you'll find `train_multiple.py`, which trains on all the MD9 data simultaneously.

The modifiable parameters are as follows:

```
Data Parameters

data: path to npz data file, default = data/aspirin_dft.npz
ntrain: number of training configurations to use, default = 1000.
ntest: number of test samples to measure errors against, default = 250.
nbatch: number of molecules to batch over, necessary to set this when doing force learning (e.g 64, 128) due to memory constraints, default = 128.

Model + Training Parameters

nfeatures: number of features for the kernel approxmation. Must be a power of 2 in the case of Hadamard features, default = 8192.
npcas: project the representation using dimenstionality reduction onto a new basis of this size. Must be a power of 2 in the case of Hadamard features, default = 128.
ntransforms: number of times to apply the [HDx]_j hadamard transform to build the features, default = 1. 
sigma: kernel width parameter of the kernel we're approximating, default= 3.0.
llambda: regularization parameter for modifying the diagonals of the ZTZ Gramm Matrix, default = 1e-12.
forces: 0/1, 0 - train only on energies, 1 - train on energies + forces, default = 1.

Representation Parameters

eta: gaussian smearing width, default = 2.0.
ngaussians: number of radial gridpoints in which to expand the atomic densities, default = 20.
lmax: maximum angular momentum number to consider (0: S-like oribtals, 1: S + P-like orbitals, 2: S + P + D-like orbitals...), default = 2.
rcut: radial cutoff value in Angstrom, default = 6.0.
```
# Development

The code is structured such that all of the CUDA C implementational details are hidden away in the two BaseKernel subclasses: RandomFourrierFeaturesModel and HadamardFeaturesModel. The CUDA C implementation of the EGTO representation is wrapped by the EGTOCuda class. Consequently, training models with this code is incredibly straightforward and is performed in a few lines:

```python
rep = EGTOCuda(species=unique_z, high_cutoff=rcut, ngaussians=ngaussians, eta=eta, lmax=lmax)

model = RandomFourrierFeaturesModel(rep, elements=unique_z, sigma=sigma, llambda=llambda, nfeatures=nfeatures, npcas=npcas, nbatch=nbatch)
    
model.get_reductors([coords[i] for i in reductor_samples], [nuclear_charges[i]for i in reductor_samples], npcas=npcas)
    
model.train(train_coordinates, train_charges, train_energies, train_forces if use_forces else None)
```

The BaseKernel subclasses expect python lists of numpy ndarrays containing the relevant input data. These are then converted to torch CUDA tensors internally. In the above example, the train_coordinates, train_energies and train_forces python lists might have the following structure:

```
train_coordinates: [ndarray(5, 3), ndarray(11,3), ndarray(21,3)...]
train_charges: [ndarray(5), ndarray(11), ndarray(21)]
train_energies: [-1843, -1024, -765]
train_forces: [ndarray(5, 3), ndarray(11,3), ndarray(21,3)...]
```
The EGTOCuda is capable of building the atomic densities for multiple different types of molecules simultaneously, and this again is all hidden away!

# Caveats

Since Hadamard transforms have dimension 2^{m}, the representation also needs to be this length. This is achieved using an SVD on a subsample of the atomic representations. Examples of this are provided in the test codes code.

Periodic boundary conditions are currently not supported, but I'll modify the GPU pairlist code when I get some time.

For learning size-extensive properties, I recommend subtracting the atomic contributions (e.g single-atom energies) from the total energy. The code can automatically do this for you with the following call:

```python
model.set_subtract_self_energies(True)
model.self_energy = torch.Tensor([0., -0.500273, 0., 0., 0., 0., -37.845355, -54.583861, -75.064579, -99.718730]).double() * 627.5095
```

where self_energy is a torch Tensor that has shape (max(elements) + 1). If these single-atom properties are unavailable, you can tell the code to linearly fit them for you:

```python
model.set_subtract_self_energies(True)
model.calculate_self_energy(train_charges, train_energies)
````
