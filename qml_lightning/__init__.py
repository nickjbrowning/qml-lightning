import os
import torch
import sysconfig
import sys

_HERE = os.path.realpath(os.path.dirname(__file__))
EXT_SUFFIX = sysconfig.get_config_var('EXT_SUFFIX')

if (not os.path.exists(_HERE + '/torchscript_fchl19.so') or not os.path.exists(_HERE + '/torchscript_sorf.so')):
    print (f"FATAL: Could not find either torchscript_fchl19.so or torchscript_sorf.so in path: {_HERE}, was 'pip install . --no-build-isolation' successful?")
    sys.exit()
else:
    torch.ops.load_library(_HERE + '/torchscript_fchl19.so')
    torch.ops.load_library(_HERE + '/torchscript_sorf.so')
    

