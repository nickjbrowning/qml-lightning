<div style="text-align:center"><img src="./images/qml_lightning.png" alt="drawing" width="400"/></div>

# QML-Lightning!

GPU-Accelerated Approximate Kernel Methods and Representations for Quantum Machine Learning. 


# Requirements

```
pytorch
numpy
scipy
tqdm
h5py
cuda C compilers and runtime libraries, e.g https://developer.nvidia.com/cuda-downloads
```

# Installation

cd to the directory containing setup.py and type

```bash
python3 setup.py build
```

if you've got multiple versions of CUDA, you may need to prepend the LDFLAGS variable to let nvcc know where to find the correct CUDA libraries, eg:

```bash
LDFLAGS=-L/usr/local/cuda-11.4 python3 setup.py build
```

The above may also apply if you're using an environment manager, e.g conda/miniconda. Alternatively, you can make sure your LD_LIBRARY_PATH is set correctly. Once built, set `PYTHONPATH` to the following build directory, e.g in your `.bashrc` file:

```bash
export PYTHONPATH=/path/to/qml_lightning/build/lib.XXX:$PYTHONPATH
```

Source your .bashrc and you should be good to go!

# Use

First, download the datasets by typing the following command in the `tests/` folder

```
bash get_data.sh
```

This will download the MD9 datasets, as well as the QM9 dataset and provide you with the link to the 3BPA dataset. Examples for various datasets are provided in the qml_lightning/tests folder. In this example we'll use the MD17 dataset with Hadamard features located here: `tests/MD17/train_md17.py`


now run the following command to do some ML, learning only energies + forces:

```python
python3 train_md17.py -ntrain 1000 -nstacks 128 -npcas 128 -sigma 2.0 -llambda 1e-5
```

# Development

The code is structured such that all of the CUDA C implementational details are hidden away in the two BaseKernel subclasses: RandomFourrierFeaturesModel and HadamardFeaturesModel. The CUDA C implementation of the FCHL19 representation is wrapped by the FCHLCuda class. Consequently, training models with this code is incredibly straightforward and is performed in a few lines:

```python
rep = FCHLCuda(species=unique_z, high_cutoff=rcut, nRs2=nRs2, nRs3=nRs3, eta2=eta2, eta3=eta3,
                   two_body_decay=two_body_decay, three_body_decay=three_body_decay, three_body_weight=three_body_weight)
    
model = HadamardFeaturesModel(rep, elements=unique_z, sigma=sigma, llambda=llambda,
                                nstacks=nstacks, ntransforms=ntransforms, npcas=npcas,
                                nbatch_train=nbatch_train, nbatch_test=nbatch_test)
    
model.get_reductors(train_coordinates, train_charges, npcas=npcas)
    
model.set_subtract_self_energies(True)
model.self_energy = torch.Tensor([0., -13.587222780835477, 0., 0., 0., 0., -1029.4889999855063, -1484.9814568572233, -2041.9816003861047]).double()
  
model.train(train_coordinates, train_charges, train_energies, train_forces)
    
data = model.format_data(test_coordinates, test_charges, test_energies, test_forces)

test_energies = data['energies']
test_forces = data['forces']
max_natoms = data['natom_counts'].max().item()

energy_predictions, force_predictions = model.predict(test_coordinates, test_charges, max_natoms, forces=True, use_backward=True)
```

The BaseKernel subclasses expect python lists of numpy ndarrays containing the relevant input data. These are then converted to torch CUDA tensors internally by the format_data method. Note that this method uses zero padding when relevant for datasets containing molecules with different sizes. In the above example, the train_coordinates, train_energies and train_forces python lists might have the following structure:

```
train_coordinates: [ndarray(5, 3), ndarray(11,3), ndarray(21,3)...]
train_charges: [ndarray(5), ndarray(11), ndarray(21)]
train_energies: [-1843, -1024, -765]
train_forces: [ndarray(5, 3), ndarray(11,3), ndarray(21,3)...]
```

# Caveats

Since Hadamard transforms have dimension 2^{m}, where m is a positive integer, the representation also needs to be this length. This is achieved using an SVD on a subsample of the atomic representations. Examples of this are provided in the tests folder.

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
