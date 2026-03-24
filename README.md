# tropess_compression

TROPESS Averaging Kernel and Covariance Compression and Decompression Tool.

This tool can be used to decompress large (3D) data structures in TROPESS L2 full product files.

## Environment Setup

### Using Pixi (Recommended)

[Pixi](https://pixi.sh) provides a fast, cross-platform package manager built on conda-forge.

1. Install pixi by following the instructions at https://pixi.sh
2. Set up the environment:
   ```bash
   cd tropess-compression
   pixi install
   ```
3. Run commands using pixi:
   ```bash
   pixi run compress_tropess_file --help
   pixi run decompress_tropess_file --help
   ```

### Using Conda

Alternatively, you can use conda/mamba with the provided environment.yml file:

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```
   Or with mamba for faster installation:
   ```bash
   mamba env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate tropess-compression
   ```

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Command Line Tools

After environment setup, two command-line tools are available for compressing and decompressing TROPESS data files.

### compress_tropess_file

Compress a TROPESS product file by encoding large 3D data structures.

**Usage:**
```bash
compress_tropess_file [--max_error MAX_ERROR] [--verbose] input_filename output_filename
```

**Positional Arguments:**
- `input_filename` - File name of input TROPESS product to compress
- `output_filename` - File name for the compressed output TROPESS product file

**Options:**
- `--max_error MAX_ERROR` - Maximum tolerated error for entries of compressed objects (default: 5e-05)
- `--verbose`, `-v` - Enable additional debug logging to screen
- `-h`, `--help` - Show help message and exit

**Example:**
```bash
pixi run compress_tropess_file input.nc compressed_output.nc --max_error 1e-04 --verbose
```

Or with conda:
```bash
compress_tropess_file input.nc compressed_output.nc --max_error 1e-04 --verbose
```

### decompress_tropess_file

Decompress a previously compressed TROPESS product file.

**Usage:**
```bash
decompress_tropess_file [--verbose] input_filename output_filename
```

**Positional Arguments:**
- `input_filename` - File name of input TROPESS product to decompress
- `output_filename` - File name for the decompressed output TROPESS product file

**Options:**
- `--verbose`, `-v` - Enable additional debug logging to screen
- `-h`, `--help` - Show help message and exit

**Example:**
```bash
pixi run decompress_tropess_file compressed_input.nc decompressed_output.nc --verbose
```

Or with conda:
```bash
decompress_tropess_file compressed_input.nc decompressed_output.nc --verbose
```

## Additional Documentation

The algorithm is documented in the [Compression of Averaging Kernels and Covariance Matrices in TROPESS L2 Ozone Archival Data Products](documentation/MUSES_compression.pdf) writeup.

For an example of using the software package directly to decompress individual netCDF variables, see the [example decompression Jupyter notebook](documentation/example_decompression.ipynb). 

## Copyright and Licensing Info
Copyright (c) 2023-24 California Institute of Technology (“Caltech”). U.S. Government sponsorship acknowledged. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:

• Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.

• Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.

• Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the
names of its contributors may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Open Source License Approved by Caltech/JPL
APACHE LICENSE, VERSION 2.0
• Text version: https://www.apache.org/licenses/LICENSE-2.0.txt
• SPDX short identifier: Apache-2.0
• OSI Approved License: https://opensource.org/licenses/Apache-2.0
