import re
import argparse
import logging

import netCDF4

from tropess_compression.akc_compression import Multiple_Sounding_Decompression 
from tropess_compression.netcdf_util import remove_netcdf_variables, copy_var_attributes

COMPRESS_DIMENSIONS_RE = r'.*_compressed_bytes'

logger = logging.getLogger()

# Support pre and post netCDF v1.6.0 compression kwarg formatting
compression_kwarg = {'compression': 'zlib'}
ver_parts = list(map(int, netCDF4.__version__.split('.')))
if ver_parts[0] == 1 and ver_parts[1] < 6:
    compression_kwarg = {'zlib': True}

def decompress_variable(data_file_input, data_file_output, var_name, progress_bar=False):

    # Read input data
    # Gracefully (maybe?) handle variables that are missing _FillValue. We should
    # fully shift away from using missing_value in the future. We should also use
    # an up-to-date version of netCDF4 so we could use variable.get_fill_value()
    # instead of this ... thing.
    fill_value = data_file_input[var_name]._FillValue = data_file_input[var_name]._FillValue if hasattr(data_file_input[var_name], '_FillValue') else data_file_input[var_name].missing_value
    compressed_input = data_file_input[var_name][...].filled(fill_value)

    # Perform decompression
    decompressor = Multiple_Sounding_Decompression(compressed_input, progress_bar=progress_bar)
    decompressed_data = decompressor.decompress_3D()

    # Create new variable with same name as original, requires this variable
    # to have been removed from the output data file
    decompress_dims = data_file_input[var_name].uncompressed_dimensions
    decompress_dtype = data_file_input[var_name].uncompressed_data_type
    decompress_fill_value = data_file_input[var_name].uncompressed_fill_value

    out_var = data_file_output.createVariable(var_name, decompress_dtype, decompress_dims, fill_value=decompress_fill_value, **compression_kwarg)
    out_var[...] = decompressed_data

    # Copy attributes from source variable, except for certain ignored ones
    copy_var_attributes(data_file_input[var_name], out_var)

def decompression_variable_list(data_file_input):
    "Find variable names that have a compressed_bytes dimension"

    def find_decompression_vars(root):
        for var_obj in root.variables.values():
            if re.match(COMPRESS_DIMENSIONS_RE, var_obj.dimensions[0]):
                v_name = var_obj.group().path + "/" + var_obj.name
                yield v_name.lstrip('/')
        for grp_obj in root.groups.values():
            for v_name in find_decompression_vars(grp_obj):
                yield v_name

    return list(find_decompression_vars(data_file_input))

def decompress_file(input_filename, output_filename, progress_bar=False):

    # Open input file, find which variables will be decompressed    
    data_file_input = netCDF4.Dataset(input_filename, 'r')
    vars_to_decompress = decompression_variable_list(data_file_input)

    # Start with a copy of the original since some contents will not be compressed
    logger.debug(f"Creating modified destination file: {output_filename} from {input_filename}")
    
    # Remove the compressed variables from the destination file to overwrite with decompressed variables
    remove_netcdf_variables(input_filename, output_filename, vars_to_decompress)

    # Open output for copying compression output
    data_file_output = netCDF4.Dataset(output_filename, 'a')

    # Compress variables
    for var_name in vars_to_decompress:
        logger.debug(f"Decompressing: {var_name}")
        decompress_variable(data_file_input, data_file_output, var_name, progress_bar=progress_bar)

    data_file_input.close()
    data_file_output.close()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_filename', type=str,
                        help='File name of input TROPESS product to decompress')

    parser.add_argument('output_filename', type=str,
                        help='File name for the decompressed output TROPESS product file')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable additional debug logging to screen')
    
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)

    decompress_file(args.input_filename, args.output_filename, progress_bar=args.verbose)

if __name__ == '__main__':
    main()
