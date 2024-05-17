import xarray as xr
from argparse import ArgumentParser

def calc_stat(path, filename):
    ds = xr.open_dataset(f"{path}/{filename}")
    ds_min = ds.min(dim=["time", "longitude", "latitude"])
    ds_max = ds.max(dim=["time", "longitude", "latitude"])
    ds_min.to_netcdf(f"{path}/{filename.split('.')[0]}_min.nc")
    ds_max.to_netcdf(f"{path}/{filename.split('.')[0]}_max.nc")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--filename")
    args = parser.parse_args()
    calc_stat(args.path, args.filename)
