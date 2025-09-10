
import os
import re
import shutil
import tempfile

from nco import Nco

# Do not copy these variables from the original source variable
IGNORE_ATTRS_PATTERN = r'^(_FillValue|compression_.*|uncompressed_.*)$'

def call_ncks(input_filename, output_filename, options, overwrite=False):

    nco = Nco()

    use_temp = False
    if os.path.realpath(input_filename) == os.path.realpath(output_filename):
        if not overwrite:
            raise IOError(f"Will not overwrite original file {input_filename}")
        
        temp_fd, dest_filename = tempfile.mkstemp()
        use_temp = True
    else:
        dest_filename = output_filename

    nco.ncks(input=input_filename, output=dest_filename, options=options)

    if not os.path.exists(dest_filename):
        raise IOError(f"ncks failed to create {dest_filename} from {input_filename}")
    
    if use_temp:
        shutil.copyfile(dest_filename, output_filename)
        os.remove(dest_filename)
        os.close(temp_fd)

def remove_netcdf_variables(input_filename, output_filename, var_removal_list, **kwargs):

    var_list_str = ",".join(var_removal_list)
    
    ncks_options=["-x", f"-v {var_list_str}"]

    return call_ncks(input_filename, output_filename, options=ncks_options, **kwargs)

def remove_unlimited_dims(input_filename, output_filename, **kwargs):
    
    ncks_options=["--fix_rec_dmn all"]

    return call_ncks(input_filename, output_filename, options=ncks_options, **kwargs)

def copy_var_attributes(source_var, dest_var, ignore_pattern=IGNORE_ATTRS_PATTERN):

    # Copy attributes from source variable, except for certain ignored ones
    for src_attr_name in source_var.ncattrs():
        if re.search(ignore_pattern, src_attr_name):
            continue

        # Since missing_value has a special, warnings occur when trying to cast from
        # the source uncompressed array into the destination compressed array.
        # Example warning:
        # UserWarning: WARNING: missing_value cannot be safely cast to variable dtype
        #
        # So instead rename attribute name back and forth when compressing and uncompressing
        # to reflect that the missing value is from the source variable
        if src_attr_name == "missing_value":
            dst_attr_name = "source_missing_value"
        elif src_attr_name == "source_missing_value":
            dst_attr_name = "missing_value"
        else:
            dst_attr_name = src_attr_name

        setattr(dest_var, dst_attr_name, getattr(source_var, src_attr_name))