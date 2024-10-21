
import os
import re
import shutil
import tempfile

from nco import Nco

# Do not copy these variables from the original source variable
IGNORE_ATTRS_PATTERN = r'^(_FillValue|compression_.*|uncompressed_.*)$'

def remove_netcdf_variables(input_filename, var_removal_list, output_filename=None):

    nco = Nco()

    var_list_str = ",".join(var_removal_list)
    
    temp_filename = tempfile.mkstemp()[1]

    nco.ncks(input=input_filename, output=temp_filename, options=["-x", f"-v {var_list_str}"])

    if output_filename is not None:
        shutil.copyfile(temp_filename, output_filename)
        os.remove(temp_filename)

    return temp_filename

def copy_var_attributes(source_var, dest_var, ignore_pattern=IGNORE_ATTRS_PATTERN):

    # Copy attributes from source variable, except for certain ignored ones
    for attr_name in source_var.ncattrs():
        if re.search(ignore_pattern, attr_name):
            continue

        setattr(dest_var, attr_name, getattr(source_var, attr_name))