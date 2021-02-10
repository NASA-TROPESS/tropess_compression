# tropess_compression

The primary MUSES compression algorithms are contained in code/MUSES_compression_v2.py. These currently are designed to compress all large (3D) data structures in L2 data files.

The algorithm and code usage is documented in documentation/MUSES_compression.pdf. The code requires the Python modules listed in requirements.txt. Instructions for setting up a Python virtual environment and installing these requirements are contained in documentation/MUSES_compression_instructions.txt.

Furthermore, sample programs code/MUSES_compress_to_file_v2.py and code/MUSES_decompress_from_file_v2.py have been written produced a compressed version of a typical L2 product data file, and subsequently decompress it. Usage of these programs is also documented in documentation/MUSES_compression_instructions.txt. These sample programs may need to be updated for new MUSES data file formats.

