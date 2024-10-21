import os
import re
import shutil
import tempfile
import argparse
import logging

import netCDF4
import numpy as np
from nco import Nco

from tropess_compression.akc_compression import Multiple_Sounding_Compression

DEFAULT_COMPRESSION_VAR_RE = r'^(.+_averaging_kernel)|(.+_covariance)$'

# Do not copy these variables from the original source variable
IGNORE_ATTRS = r'^(_FillValue)$'

logger = logging.getLogger()

def compress_variable(data_file_input, data_file_output, var_name, max_error, progress_bar=False):

    # Read input data
    fill_value = data_file_input[var_name]._FillValue
    data_input = data_file_input[var_name][...].filled(fill_value)

    # Perform compression
    compressor = Multiple_Sounding_Compression(data_input, fill_value=fill_value, progress_bar=progress_bar)
    compressed_data = compressor.compress_3D(max_error=max_error)

    # Create new variable with same name as original, requires this variable
    # to have been removed from the output data file
    dim_name = data_file_input[var_name].name + "_compressed_bytes"
    out_dim = data_file_output.createDimension(dim_name, len(compressed_data))
    out_var = data_file_output.createVariable(var_name, np.byte, (dim_name,), fill_value=fill_value)
    out_var[...] = compressed_data

    # Copy attributes from source variable, except for certain ignored ones
    for attr_name in data_file_input[var_name].ncattrs():
        if re.search(IGNORE_ATTRS, attr_name):
            continue

        setattr(out_var, attr_name, getattr(data_file_input[var_name], attr_name))

    # Annotate as a compressed variable
    out_var.compression_help = 'This value can not be read directly. Use decompress_tropess_file: https://github.com/NASA-TROPESS/tropess_compression'

def compression_variable_list(data_file_input):
    "Find variable names that match a certain regular expression"

    def find_compression_vars(root):
        for var_obj in root.variables.values():
            if re.match(DEFAULT_COMPRESSION_VAR_RE, var_obj.name):
                v_name = var_obj.group().path + "/" + var_obj.name
                yield v_name.lstrip('/')
        for grp_obj in root.groups.values():
            for v_name in find_compression_vars(grp_obj):
                yield v_name

    return list(find_compression_vars(data_file_input))

def remove_netcdf_variables(input_filename, var_removal_list, output_filename=None):

    nco = Nco()

    var_list_str = ",".join(var_removal_list)
    
    temp_filename = tempfile.mkstemp()[1]

    nco.ncks(input=input_filename, output=temp_filename, options=["-x", f"-v {var_list_str}"])

    if output_filename is not None:
        shutil.copyfile(temp_filename, output_filename)
        os.remove(temp_filename)

    return temp_filename

def compress_file(input_filename, output_filename, max_error, progress_bar=False):

    # Open input file, find which variables will be compressed    
    data_file_input = netCDF4.Dataset(input_filename, 'r')
    vars_to_compress = compression_variable_list(data_file_input)

    # Start with a copy of the original since some contents will not be compressed
    logger.debug(f"Creating modified destination file: {output_filename} from {input_filename}")
    
    # Remove the compression variable from the destination file
    remove_netcdf_variables(input_filename, vars_to_compress, output_filename)

    # Open output for copying compression output
    data_file_output = netCDF4.Dataset(output_filename, 'a')

    # Compress variables
    for var_name in vars_to_compress:
        logger.debug(f"Compressing: {var_name}")
        compress_variable(data_file_input, data_file_output, var_name, max_error, progress_bar=progress_bar)

    data_file_input.close()
    data_file_output.close()


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_filename', type=str,
                        help='File name of input TROPESS product to compress')

    parser.add_argument('output_filename', type=str,
                        help='File name for the compressed output TROPESS product file')

    parser.add_argument('--max_error', type=float, default=0.00005,
                        help='Maximum tolerated error for entries of compressed objects.')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable additional debug logging to screen')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)

    compress_file(args.input_filename, args.output_filename, args.max_error, progress_bar=args.verbose)

if __name__ == '__main__':
    main()