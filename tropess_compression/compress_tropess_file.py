import h5py
import numpy as np
import argparse

import sys
from tropess_compression.akc_compression import Multiple_Sounding_Compression 

def compress_file(args):
    
    filename = args.filename
    max_error = float(args.max_error) 
    
    data_file_orig = h5py.File(filename, 'r')
    data_file_new = h5py.File(filename[:-4] + '_compressed_v2.hdf', 'w')

    def replicate_groups(name, object):
        if isinstance(object, h5py.Group):
            data_file_new.create_group(name)

    def replicate_datasets(name, object):
        if isinstance(object, h5py.Dataset):
            print(name) 
            data_set = object
            if len(data_set.shape) == 3 and data_set.shape[1] == data_set.shape[2]: 
                data_nparray = np.array(data_set)
                data_nparray_MSC = Multiple_Sounding_Compression(data_nparray) 
                data_nparray_compbytes = data_nparray_MSC.compress_3D(abs_error=2*max_error)
                data_file_new.create_dataset(name + "_compressed_v2", data=data_nparray_compbytes)
            else:
                data_file_new.create_dataset(name, data=data_set)

    data_file_orig.visititems(replicate_groups)
    data_file_orig.visititems(replicate_datasets)

    data_file_new.close()
    data_file_orig.close() 

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--filename', type=str, default=None,
                        help='name of single file to compress')
    parser.add_argument('--max_error', type=str, default='0.00005',
                        help='Maximum tolerated error for entries of compressed objects. ')
    
    args = parser.parse_args()
    compress_file(args)

if __name__ == '__main__':
    main()