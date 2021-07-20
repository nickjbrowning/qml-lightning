import h5py
import numpy as np


def iter_data_buckets(h5filename, keys=['coordinates', 'energies', 'species']):
    
    '''reference: https://www.nature.com/articles/s41467-019-10827-4
       github: https://github.com/aiqm/ANI1x_datasets '''
    
    """ Iterate over buckets of data in ANI HDF5 file. 
    Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
    and other available properties specified by `keys` list, w/o NaN values.
    """
    keys = set(keys)
    keys.discard('atomic_numbers')
    keys.discard('coordinates')
    with h5py.File(h5filename, 'r') as f:
        
        for grp in f.values():
            for key_group in grp.keys():
                
                sub_grp = grp[key_group]
                
                data = dict((k, sub_grp[k][()]) for k in keys)
                
                print (sub_grp.keys())
                print (data)
