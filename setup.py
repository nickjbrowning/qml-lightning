import torch.cuda
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME

ext_modules = []

__author__ = "Nicholas J. Browning"
__credits__ = "Nicholas J. Browning (2021), https:://TODO"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Nicholas J. Browning"
__email__ = "nickjbrowning@gmail.com"
__status__ = "Alpha"
__description__ = "GPU-Accelerated Kernel Methods for Quantum Machine Learning"
__url__ = "TODO"

# optimisation_level_host = '-g'
# optimisation_level_device = '-G'

optimisation_level_host = '-O2'
optimisation_level_device = '-O2'


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        return [line.rstrip() for line in f]


if torch.cuda.is_available() and CUDA_HOME is not None:
    
    gto_extension = CUDAExtension(
        '.cuda.egto_gpu', [
            'qml_lightning/cuda/gto_cuda.cpp',
            'qml_lightning/cuda/gto_cuda_kernel.cu'
        ],
         extra_compile_args={'cxx': [optimisation_level_host],
                            'nvcc': [optimisation_level_device, '-ftz=true']})
    
    fchl_extension = CUDAExtension(
        '.cuda.fchl_gpu', [
            'qml_lightning/cuda/fchl_cuda.cpp',
            'qml_lightning/cuda/fchl_cuda_kernel.cu'
        ],
         extra_compile_args={'cxx': [optimisation_level_host],
                            'nvcc': [optimisation_level_device, '-ftz=true']})
    
    hd_extension = CUDAExtension(
        '.cuda.sorf_gpu', [
            'qml_lightning/cuda/hadamard_cuda.cpp',
            'qml_lightning/cuda/hadamard_kernel.cu'
        ],
         extra_compile_args={'cxx': [optimisation_level_host],
                            'nvcc': [optimisation_level_device, '-ftz=true']})
    
    rff_extension = CUDAExtension(
        '.cuda.rff_gpu', [
            'qml_lightning/cuda/random_features.cpp',
            'qml_lightning/cuda/random_features_kernel.cu'
        ],
        extra_compile_args={'cxx': [optimisation_level_host],
                            'nvcc': [optimisation_level_device, '-ftz=true']})
    
    pairlist_extension = CUDAExtension(
        '.cuda.pairlist_gpu', [
            'qml_lightning/cuda/pairlist_cuda.cpp',
            'qml_lightning/cuda/pairlist_kernel.cu'
        ],
        extra_compile_args={'cxx': [optimisation_level_host],
                            'nvcc': [optimisation_level_device, '-ftz=true']})
    
    operator_extension = CUDAExtension(
        '.cuda.operator_gpu', [
            'qml_lightning/cuda/operator_cuda.cpp',
            'qml_lightning/cuda/operator_kernel.cu'
        ],
         extra_compile_args={'cxx': [optimisation_level_host],
                            'nvcc': [optimisation_level_device, '-ftz=true']})
    
    utils_extension = CUDAExtension(
        '.cuda.utils_gpu', [
            'qml_lightning/cuda/utils_cuda.cpp',
            'qml_lightning/cuda/utils_kernel.cu'
        ],
         extra_compile_args={'cxx': [optimisation_level_host],
                            'nvcc': [optimisation_level_device, '-ftz=true']})
    
    ext_modules.append(gto_extension)
    ext_modules.append(hd_extension)
    ext_modules.append(pairlist_extension)
    ext_modules.append(fchl_extension)
    ext_modules.append(rff_extension)
    ext_modules.append(operator_extension)
    ext_modules.append(utils_extension)
    
else:
    print("ERROR: cuda not available, or CUDA_HOME not set.")
    exit()
    
setup(
    name='qmlightning',
    packages=['qml_lightning',
              'qml_lightning.features',
              'qml_lightning.representations',
              'qml_lightning.models',
              'qml_lightning.models.neural_net',
              'qml_lightning.utils'],
    version=__version__,
    author=__author__,
    author_email=__email__,
    platforms='Any',
    description=__description__,
    long_description=readme(),
    keywords=['Machine Learning', 'Quantum Chemistry'],
    classifiers=[],
    url=__url__,
    install_requires=requirements(),
    
    ext_package='qml_lightning',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
