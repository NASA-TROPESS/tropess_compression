import re
import argparse
import logging

import netCDF4
import numpy as np

from tropess_compression.akc_compression import Multiple_Sounding_Compression
from tropess_compression.netcdf_util import remove_netcdf_variables, remove_unlimited_dims, copy_var_attributes

DEFAULT_COMPRESSION_VAR_RE = r'^(.*averaging_kernel)|(.+_covariance)$'

DEFAULT_MAX_ERROR = 0.00005 

# Support pre and post netCDF v1.6.0 compression kwarg formatting
compression_kwarg = {'compression': 'zlib'}
ver_parts = list(map(int, netCDF4.__version__.split('.')))
if ver_parts[0] == 1 and ver_parts[1] < 6:
    compression_kwarg = {'zlib': True}

logger = logging.getLogger()

def compress_variable(data_file_input, data_file_output, var_name, max_error=DEFAULT_MAX_ERROR, progress_bar=False):

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
    out_var = data_file_output.createVariable(var_name, np.byte, (dim_name,), fill_value=fill_value, **compression_kwarg)
    out_var[...] = compressed_data

    # Copy attributes from source variable, except for certain ignored ones
    copy_var_attributes(data_file_input[var_name], out_var)

    # Annotate as a compressed variable
    out_var.compression_comment = 'This value cannot be read directly. Use decompress_tropess_file: https://github.com/NASA-TROPESS/tropess_compression'
    out_var.compression_max_error = max_error
    out_var.uncompressed_dimensions = [ dim_name for dim_name in data_file_input[var_name].dimensions ]
    out_var.uncompressed_data_type = str(data_file_input[var_name].dtype)
    out_var.uncompressed_fill_value = fill_value

def compression_variable_list(data_file_input):
    "Find variable names that match a certain regular expression"

    def find_compression_vars(root):
        for var_obj in root.variables.values():
            if re.match(DEFAULT_COMPRESSION_VAR_RE, var_obj.name) and len(var_obj.shape) == 3:
                v_name = var_obj.group().path + "/" + var_obj.name
                yield v_name.lstrip('/')
        for grp_obj in root.groups.values():
            for v_name in find_compression_vars(grp_obj):
                yield v_name

    return list(find_compression_vars(data_file_input))

def compress_file(input_filename, output_filename, max_error=DEFAULT_MAX_ERROR, progress_bar=False):

    # Open input file, find which variables will be compressed    
    data_file_input = netCDF4.Dataset(input_filename, 'r')
    vars_to_compress = compression_variable_list(data_file_input)

    # Start with a copy of the original since some contents will not be compressed
    logger.debug(f"Creating modified destination file: {output_filename} from {input_filename}")
    
    # Remove the compression variable from the destination file
    remove_netcdf_variables(input_filename, output_filename, vars_to_compress)

    # Remove unlimited dimensions to improve traditional compression
    remove_unlimited_dims(output_filename, output_filename, overwrite=True)

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

    parser.add_argument('--max_error', type=float, default=DEFAULT_MAX_ERROR,
                        help=f'Maximum tolerated error for entries of compressed objects. Default value: {DEFAULT_MAX_ERROR}')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable additional debug logging to screen')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)

    compress_file(args.input_filename, args.output_filename, max_error=args.max_error, progress_bar=args.verbose)

if __name__ == '__main__':
    main()
