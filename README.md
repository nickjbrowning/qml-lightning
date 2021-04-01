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

Once built, set `PYTHONPATH` to build directory, e.g in your `.bashrc` file:

```bash
export PYTHONPATH=/home/nick/git/QMLightning/build:$PYHTONPATH

```

Now you should be good to go!

# Use

