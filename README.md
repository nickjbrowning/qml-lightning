# QML-Lightning

GPU-Accelerated Kernel Methods and Representations for Quantum Machine Learning.

<img src="./images/pogchamp.jpg" alt="drawing" width="200"/>

### Kernel Methods
Structured Orthogonal Random Features via Fast Hadamard Transforms

### Representations
Elemental Gaussian-Type Orbital Atomic Density

# Requirements

```
pytorch
numpy
scipy
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

An example of performing energy + force learning on an Aspirin trajectory can be found in `tests/`

First, download the MD9 dataset by typing the following command in the `tests/` folder

```
make get-data
```

now run the following command to do some ML,

```
python3 aspirin.py -ntrain 1000 -ntest 500 -nbatch 4 -sigma 12.0 -lambda 1e-10
```

nbatch specifies how many times to split the training and test sets. The code then loops over these spits to construct the Z^T Z Gramm matrix iteratively.

sigma corresponds to the "width" of the kernel the SORF method is approximating. Values are problem-specific but somewhere between 8.0 and 32.0 should be ok.

lambda is the level of regularization. Recommend values between 1e-8 and 1e-11 should be ok.

# Caveats

The GPU code assumes that the coordinate + charge info you're passing in are the same size across the batch. Therefore, if you want to train on multiple *different* types of molecules (e.g a set of molecules, each with multiple conformations /w energies + forces), you'll need to batch over these molecules individually when constructing the Gramm matrix.

Energy learning seems relatively poor at the moment, but forces look fine. This is most probably due to the representation being underdeveloped. I'm working on this at the moment so hopefully in a few iterations it should have comparable performance to FCHL19 on scalar-valued properties. 
