import numpy as np
import netCDF4

from tropess_compression.compress_tropess_file import compress_file
from tropess_compression.decompress_tropess_file import decompress_file

N_SOUNDING = 4
N_LEVEL = 67
FILL = -999.0


def make_matrices():
    rng = np.random.default_rng(0)
    sym = np.empty((N_SOUNDING, N_LEVEL, N_LEVEL), dtype=np.float32)
    asym = np.empty_like(sym)
    for i in range(N_SOUNDING):
        a = rng.normal(size=(N_LEVEL, N_LEVEL)) * 0.01
        sym[i] = a @ a.T
        b = np.diag(rng.uniform(0.5, 1.0, N_LEVEL)) + rng.normal(size=(N_LEVEL, N_LEVEL)) * 0.05
        asym[i] = b @ b.T * 0.1 + np.triu(rng.normal(size=(N_LEVEL, N_LEVEL)) * 0.05, 1)
    return sym, asym


def write_input(path, sym, asym):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("sounding", N_SOUNDING)
        ds.createDimension("level_fm", N_LEVEL)
        # Real products have other variables on these dimensions, which keeps
        # them in the compressed file for decompression to use.
        ds.createVariable("latitude", "f4", ("sounding",))[...] = np.arange(N_SOUNDING)
        ds.createVariable("pressure_fm", "f4", ("sounding", "level_fm"))[...] = np.ones((N_SOUNDING, N_LEVEL))
        for name, data in (("observation_error_covariance", sym),
                           ("h2o_hdo_observation_error_covariance", asym)):
            var = ds.createVariable(name, "f4", ("sounding", "level_fm", "level_fm"), fill_value=FILL,
                                      chunksizes=(1, N_LEVEL, N_LEVEL))
            var[...] = data


def test_asymmetric_covariance_round_trip(tmp_path):
    sym, asym = make_matrices()
    assert np.abs(asym - asym.transpose(0, 2, 1)).max() > 0.01

    in_file = str(tmp_path / "in.nc")
    comp_file = str(tmp_path / "comp.nc")
    dec_file = str(tmp_path / "dec.nc")
    write_input(in_file, sym, asym)

    compress_file(in_file, comp_file)
    decompress_file(comp_file, dec_file)

    with netCDF4.Dataset(dec_file) as ds:
        for name, expected in (("observation_error_covariance", sym),
                               ("h2o_hdo_observation_error_covariance", asym)):
            assert np.abs(ds[name][...] - expected).max() < 5e-4, name
