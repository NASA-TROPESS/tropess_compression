import h5py
import numpy as np
import argparse

import sys
from tropess_compression.akc_compression import Multiple_Sounding_Decompression 

def decompress_file(args):
    
    filename = args.filename 
    
    data_file_orig = h5py.File(filename, 'r')
    data_file_new = h5py.File(filename[:-18] + '_decompressed_v2.hdf', 'w')

    def replicate_groups(name, object):
        if isinstance(object, h5py.Group):
            data_file_new.create_group(name)

    def replicate_datasets(name, object):
        if isinstance(object, h5py.Dataset):
            print(name) 
            if name[-14:] == '_compressed_v2':
                data_bytes = object
                data_MSD = Multiple_Sounding_Decompression(data_bytes)
                data_decompressed = data_MSD.decompress_3D()
                data_file_new.create_dataset(name[:-11], data=data_decompressed, dtype='f4')
            else:
                data_file_new.create_dataset(name, data=object)

    data_file_orig.visititems(replicate_groups)
    data_file_orig.visititems(replicate_datasets)

    data_file_new.close()
    data_file_orig.close() 

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--filename', type=str, default=None,
                        help='name of single file to decompress')
    
    args = parser.parse_args()
    decompress_file(args)

if __name__ == '__main__':
    main()