'''
Created on 7 Sep 2022

@author: Nicholas J. Browning
'''

import os
import sys
import torch

if os.getenv("QML_LIGHTNING_TORCHSCRIPT_LIB") is not None:
    
    path = os.getenv("QML_LIGHTNING_TORCHSCRIPT_LIB")
    
    print ("Attempting to load torchscript libs...")
    if (path[-1] != '/'):
        path += '/'
    
    if (not os.path.exists(path + 'torchscript_sorf.so') or not os.path.exists(path + 'torchscript_fchl19.so')):
        print ("FATAL: Could not find either torchscript_fchl19.so or torchscript_sorf.so in path QML_LIGHTNING_TORCHSCRIPT_LIB=", os.getenv("QML_LIGHTNING_TORCHSCRIPT_LIB"))
        sys.exit()
    else:
        torch.ops.load_library(path + 'torchscript_fchl19.so')
        torch.ops.load_library(path + 'torchscript_sorf.so')
        
        print ("Loading complete.")
    
else:
    print("FATAL: Must define QML_LIGHTNING_TORCHSCRIPT_LIB path. Example: export QML_LIGHTNING_TORCHSCRIPT_LIB=/path/to/build/lib.linux-x86_64-3.9/qml_lightning")
    sys.exit()
    
