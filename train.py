import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import get_model_size_mb
import math
import numpy as np
import xarray as xr
from argparse import ArgumentParser
from types import SimpleNamespace
from datetime import datetime
import matplotlib.pyplot as plt
import pyinterp
import pyinterp.backends.xarray
from scipy.interpolate import RegularGridInterpolator
from tqdm import trange, tqdm
import sys, os
from pathlib import Path

YEAR = 2016


class WeatherBenchDataset_sampling(Dataset):
    def __init__(self, file_name, data_path, nbatch, nsample, variable="z"):
        file_path = f"{data_path}/{file_name}"
        self.ds = xr.open_mfdataset(file_path).load()
        self.ds = self.ds.assign_coords(time=np.arange(len(self.ds.time)))
        #self.grid = pyinterp.Grid3D(pyinterp.Axis(self.ds.time), pyinterp.Axis(self.ds.lat), pyinterp.Axis(self.ds.lon, is_circle=True), self.ds[variable].data)
        self.interpolator = RegularGridInterpolator((self.ds.time, self.ds.lat, self.ds.lon), self.ds[variable].data, bounds_error=False, fill_value=None)
        self.variable = variable
        self.ntime = len(self.ds.time)
        self.nbatch = nbatch
        self.nsample = nsample
        self.rndeng = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
        self.mean = self.ds[variable].mean(dim=["time"]).to_numpy()
        self.std = (self.ds[variable].max(dim=["time"]) - self.ds[variable].min(dim=["time"])).to_numpy()
        self.interp_mean = RegularGridInterpolator((self.ds.lat, self.ds.lon), self.mean, bounds_error=False, fill_value=None)
        self.interp_std = RegularGridInterpolator((self.ds.lat, self.ds.lon), self.std, bounds_error=False, fill_value=None)
    
    def __len__(self):
        return self.nbatch

    def __getitem__(self, idx):
        if isinstance(idx, int):
            rnds = self.rndeng.draw(self.nsample)
            time = rnds[:, 0] * (self.ntime - 1)
            pind = torch.zeros_like(time) + float(self.ds.level.mean())
            latind = 90 - 180/math.pi*torch.acos(1 - 2 * rnds[:, 1])
            lonind = (rnds[:, 2] * 360)
            coord = torch.stack((time, pind, latind, lonind), dim=-1).to(torch.float32)
            #var_sampled = pyinterp.trivariate(self.grid, time.ravel(), latind.ravel(), lonind.ravel()).reshape(latind.shape)
            coord_in = torch.stack((time, latind, lonind), dim=-1)
            var_sampled = self.interpolator(coord_in).reshape(latind.shape)
            var_sampled = torch.as_tensor(var_sampled).unsqueeze(-1).to(torch.float32)
            mean = torch.as_tensor(self.interp_mean(coord_in[..., 1:]).reshape(var_sampled.shape)).to(torch.float32)
            std = torch.as_tensor(self.interp_std(coord_in[..., 1:]).reshape(var_sampled.shape)).to(torch.float32)
            return coord, var_sampled, mean, std

    def getslice(self, tind, pind):
        lat_v = torch.as_tensor(self.ds.lat.to_numpy())
        lon_v = torch.as_tensor(self.ds.lon.to_numpy())
        lat, lon = torch.meshgrid((lat_v, lon_v), indexing="ij")
        p = torch.zeros_like(lat) + float(self.ds.level.mean())
        t = torch.zeros_like(lat) + float(tind)
        coord = torch.stack((t, p, lat, lon), dim=-1).unsqueeze(0).to(torch.float32)
        var = torch.as_tensor(self.ds[self.variable].isel(time=tind).to_numpy()).unsqueeze(-1).unsqueeze(0).to(torch.float32)
        mean = torch.as_tensor(self.mean).reshape(var.shape).to(torch.float32)
        std = torch.as_tensor(self.std).reshape(var.shape).to(torch.float32)
        return coord, var, mean, std
        
class ERA5Dataset_sampling(Dataset):
    def __init__(self, file_name, data_path, nbatch, nsample, variable="z", use_stat=False):
        file_path = f"{data_path}/{file_name}"
        self.ds = xr.open_dataset(file_path)[variable].load()#{"time": 20}
        self.ds = self.ds.assign_coords(time=self.ds.time.dt.dayofyear-1)
        self.interpolator = pyinterp.backends.xarray.Grid4D(self.ds)
        self.variable = variable
        self.ntime = len(self.ds.time)
        self.nbatch = nbatch
        self.nsample = nsample
        self.rndeng = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
        self.use_stat = use_stat
        if use_stat:
            path = Path(file_path)
            base_path = f"{path.parent}/{path.stem}"
            self.ds_min = xr.load_dataset(f"{base_path}_min.nc")[variable]
            self.ds_max = xr.load_dataset(f"{base_path}_max.nc")[variable]
        #assert len(sample_block_size) == 3 # np, nlat, nlon

    def __len__(self):
        return self.nbatch

    def __getitem__(self, idx):
        if isinstance(idx, int):
            rnds = self.rndeng.draw(self.nsample)
            time = rnds[:, 0] * (self.ntime - 1)
            #pind = (torch.rand((self.nsample,)) * (1000-10) + 10)
            pind = torch.as_tensor(self.ds.level.to_numpy(), dtype=torch.float32)[torch.randperm(self.nsample) % len(self.ds.level)]
            #latind = (torch.rand((self.nsample,)) * 180 - 90)
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            latind = 90 - 180/math.pi*torch.acos(1 - 2 * rnds[:, 1])
            lonind = (rnds[:, 2] * 360)
            coord = torch.stack((time, pind, latind, lonind), dim=-1)
            var_sampled = self.interpolator.quadrivariate(
                dict(longitude=lonind.ravel(),
                     latitude=latind.ravel(),
                     time=time.ravel(),
                     level=pind.ravel())).reshape(latind.shape)
            var_sampled = torch.as_tensor(var_sampled).unsqueeze(-1)
            if not self.use_stat:
                return coord, var_sampled
            else:
                minv = torch.from_numpy(self.ds_min.sel(level=pind).to_numpy()).unsqueeze(-1)
                maxv = torch.from_numpy(self.ds_max.sel(level=pind).to_numpy()).unsqueeze(-1)
                mid = 0.5 * (minv + maxv) + torch.zeros_like(var_sampled)
                range_ = maxv - minv + torch.zeros_like(var_sampled)
                return coord, var_sampled, mid, range_

    def getslice(self, tind, pind):
        lat_v = torch.as_tensor(self.ds.latitude.to_numpy())
        lon_v = torch.as_tensor(self.ds.longitude.to_numpy())
        lat, lon = torch.meshgrid((lat_v, lon_v), indexing="ij")
        p = torch.zeros_like(lat) + self.ds.level.to_numpy()[pind]
        t = torch.zeros_like(lat) + float(tind)
        coord = torch.stack((t, p, lat, lon), dim=-1).to(torch.float32)
        var = torch.as_tensor(self.ds.isel(time=tind, level=pind).to_numpy()).to(torch.float32).unsqueeze(-1)
        coord, var = coord.unsqueeze(0), var.unsqueeze(0)
        if not self.use_stat:
            return coord, var
        else:
            minv = torch.from_numpy(self.ds_min.isel(level=pind).to_numpy())
            maxv = torch.from_numpy(self.ds_max.isel(level=pind).to_numpy())
            mid = 0.5 * (minv + maxv) + torch.zeros_like(var)
            range_ = maxv - minv + torch.zeros_like(var)
            return coord, var, mid, range_


class FourierFeature(nn.Module):
    def __init__(self, sigma, infeature, outfeature):
        super(FourierFeature, self).__init__()
        self.feature_map = nn.Parameter(torch.normal(0., sigma, (outfeature, infeature)) ,requires_grad=False)
    def forward(self, x, cos_only: bool = False):
        # x shape: (..., infeature)
        x = 2*math.pi*F.linear(x, self.feature_map)
        if cos_only:
            return torch.cos(x)
        else:
            return torch.cat((torch.sin(x), torch.cos(x)), dim=-1)
    
class LonLat2XYZ(nn.Module):
    def forward(self, x):
        # x shape: (..., (time, pressure, lat, lon))
        time = x[..., 0]
        p = x[..., 1]
        lat = x[..., 2]
        lon = x[..., 3]
        sinlat = torch.sin(lat)
        coslat = torch.cos(lat)
        sinlon = torch.sin(lon)
        coslon = torch.cos(lon)
        return torch.stack((time, p, sinlat, coslat*sinlon, coslat*coslon), dim=-1)
    
class NormalizeInput(nn.Module):
    def __init__(self, tscale, zscale):
        super(NormalizeInput, self).__init__()
        self.scale = nn.Parameter(torch.tensor([1.0/tscale, 1.0/zscale, math.pi/180., math.pi/180.]), requires_grad=False)
    def forward(self, x):
        return x*self.scale
    
class InvScale(nn.Module):
    def forward(self, coord, z_normalized):
        factor = 0.9
        p = coord[..., 1:2]
        std = 0.385e5-0.35e4*torch.log(p)
        mean = 4.315e5-6.15e4*torch.log(p)
        return (z_normalized / factor)*std + mean

class ResBlock(nn.Module):
    def __init__(self, width, use_batchnorm=True, use_skipconnect=True):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(width, width, bias=False)
        self.fc2 = nn.Linear(width, width, bias=True)
        self.use_batchnorm = use_batchnorm
        self.use_skipconnect = use_skipconnect
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(width)
            self.bn2 = nn.BatchNorm1d(width)

    def forward(self, x_original):
        # x shape: (batch_size, width)
        x = x_original
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.gelu(x)
        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.gelu(x)
        x = self.fc2(x)
        if self.use_skipconnect:
            return x + x_original
        else:
            return x

class MultiResolutionEmbedding(nn.Module):
    def __init__(self, feature_size, nfeature, tresolution, tscale):
        # tresolution: timestep size in hours
        super().__init__()
        self.tscale = tscale
        self.tresolution = tresolution
        self.embed1 = nn.Embedding(366, nfeature, max_norm=1.0)
        self.embed2 = nn.Embedding(24, nfeature, max_norm=1.0)
        self.embed3 = nn.Embedding(int(feature_size/tscale), nfeature, max_norm=1.0)

    def forward(self, idx):
        idx = idx.squeeze(-1)
        idx1 = torch.floor(idx * self.tresolution).long()
        idx2 = torch.floor(idx / self.tscale).long()
        embed1 = self.embed1((idx1 // 24) % 366)
        embed2 = self.embed2(idx1 % 24)
        embed3 = self.embed3(idx2)
        embed = torch.cat((embed1, embed2, embed3), dim=-1)
        return embed
    
class FitNet(nn.Module):
    __constants__ = ['use_xyztransform','use_fourierfeature','use_tembedding','use_invscale', 'depth']

    def __init__(self, args):
        super(FitNet, self).__init__()
        self.args = args
        if args.use_invscale:
            self.invscale = InvScale()
        if args.use_xyztransform:
            self.lonlat2xyz = LonLat2XYZ()
            ns = 3
        else:
            ns = 2
        if args.use_tembedding:
            ne = args.ntfeature * 3
            self.embed_t = MultiResolutionEmbedding(args.tembed_size, args.ntfeature, args.tresolution, args.tscale)
        else:
            ne = 0
        if args.use_fourierfeature:
            self.fourierfeature_t = FourierFeature(args.sigma, 1, args.ntfeature)
            self.fourierfeature_p = FourierFeature(args.sigma, 1, args.nfeature)
            self.fourierfeature_s = FourierFeature(args.sigma, ns, args.nfeature)
            nf = 2*(2*args.nfeature + args.ntfeature)
        else:
            nf = 2 + ns
        self.normalize = NormalizeInput(args.tscale, args.zscale)     
        self.depth = args.depth
        self.fci = nn.Linear(nf + ne, args.width)
        self.fcs = nn.ModuleList([ResBlock(args.width, args.use_batchnorm, args.use_skipconnect) for i in range(args.depth)])
        self.fco = nn.Linear(args.width, 1)

        self.use_xyztransform = self.args.use_xyztransform
        self.use_fourierfeature = self.args.use_fourierfeature
        self.use_tembedding = self.args.use_tembedding
        self.use_invscale = self.args.use_invscale

    def forward(self, coord):
        batch_size = coord.shape[:-1]
        x = self.normalize(coord)
        if self.use_xyztransform:
            x = self.lonlat2xyz(x)
        if self.use_fourierfeature:
            t = x[..., 0:1]
            p = x[..., 1:2]
            s = x[..., 2:]
            x = torch.cat((self.fourierfeature_t(t), self.fourierfeature_p(p), self.fourierfeature_s(s)), dim=-1)
        if self.use_tembedding:
            x = torch.cat((self.embed_t(coord[..., 0:1]), x), dim=-1)
        x = F.gelu(self.fci(x))
        x = x.flatten(end_dim=-2) # batchnorm 1d only accepts (N, C) shape
        for fc in self.fcs:        
            x = fc(x)
        x = F.gelu(x)
        x = self.fco(x)
        x = x.view(batch_size).unsqueeze(-1)
        if self.use_invscale or self.args.use_stat:
            x = torch.tanh(x)
        if self.use_invscale:
            x = self.invscale(coord, x)
        return x
    
class FitNetModule(pl.LightningModule):
    # sigma=1.5, omega=30., nfeature=256, width=512, depth=4, tscale=60.0, zscale=100., learning_rate=1e-3, batch_size=3
    def __init__(self, args):
        super(FitNetModule, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = FitNet(args)#torch.jit.script(FitNet(args))
        self.input_type = torch.float32
        
    def train_dataloader(self):
        if self.args.dataloader_mode == "sampling_nc":
            dataset = ERA5Dataset_sampling(self.args.file_name, self.args.data_path, 2677*9, 361*120, variable=self.args.variable, use_stat=self.args.use_stat)
        elif self.args.dataloader_mode == "weatherbench":
            dataset = WeatherBenchDataset_sampling(self.args.file_name, self.args.data_path, 2677*9, 361*120, variable=self.args.variable)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, prefetch_factor=8)
        return dataloader
    
    def val_dataloader(self):
        # TODO: use xarray unstack?
        it = 42
        ip = 4
        if self.args.dataloader_mode == "sampling_nc":
            data = ERA5Dataset_sampling(self.args.file_name, self.args.data_path, 2677*9, 361*120, variable=self.args.variable, use_stat=self.args.use_stat).getslice(it, ip)
        elif self.args.dataloader_mode == "weatherbench":
            data = WeatherBenchDataset_sampling(self.args.file_name, self.args.data_path, 2677*9, 361*120, variable=self.args.variable).getslice(it, ip)
        dataset = TensorDataset(*data)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        return dataloader
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25, patience=10000)
        sched = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'train_loss'
        }
        return [optimizer], [sched]
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        if self.args.use_stat:
            coord, var, mean, std = batch
            #print(coord.mean(), var.mean(), mean.mean(), std.mean())
            var_pred = self(coord) * 0.5 * 1.4 * std + mean
        else:
            coord, var = batch
            var_pred = self(coord)
        lat = coord[..., 2:3] / 180. * math.pi
        p = coord[..., 1:2]
        assert var.shape == var_pred.shape
        assert var.shape == lat.shape
        delta = var_pred - var
        delta_abs = torch.abs(delta)
        loss_linf = delta_abs.max()
        loss_l1 = delta_abs.mean()
        loss_l2 = delta.pow(2).mean()
        if self.args.loss_type == "scaled_mse":
            loss = (delta/(11 - torch.log(p))).pow(2).mean()
        elif self.args.loss_type == "mse":
            loss = loss_l2
        elif self.args.loss_type == "logsumexp":
            loss = torch.logsumexp(torch.abs(delta))

        self.log("train_loss", loss)
        #self.log("train_loss"+self.args.loss_type, loss)
        self.log("train_loss_l2", loss_l2)
        self.log("train_loss_l1", loss_l1)
        self.log("train_loss_linf", loss_linf)
        return loss

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.args.use_stat:
                coord, var, mean, std = batch
                var_pred = self(coord) * 0.5 * 1.4 * std + mean
            else:
                coord, var = batch
                var_pred = self(coord)
            lat = coord[..., 2:3] / 180. * math.pi
            assert var.shape == var_pred.shape
            assert var.shape == lat.shape
            delta_origin = var_pred - var
            delta = delta_origin
            val_loss = delta.abs().max()
            val_loss_l2 = delta.pow(2).mean()
            plt.figure(figsize=(10,8))
            X, Y = coord[..., 3].squeeze().detach().cpu(), coord[..., 2].squeeze().detach().cpu()
            plt.contour(X, Y, var.squeeze().detach().cpu(), colors="green")
            plt.contour(X, Y, var_pred.squeeze().detach().cpu(), colors="red")#cmap="Reds"
            plt.pcolormesh(X, Y, delta_origin.squeeze().detach().cpu(), cmap="coolwarm", shading='nearest')
            plt.axis('scaled')
            plt.title(f'p={torch.mean(coord[..., 1]).item()}')
            plt.colorbar(fraction=0.02, pad=0.04)
            if self.trainer.is_global_zero:
                self.log("val_loss", val_loss, rank_zero_only=True)
                self.log("val_loss_l2", val_loss_l2, rank_zero_only=True)

def test_on_wholedataset(file_name, data_path, output_path, output_file, model, device="cuda", variable="z"):
    ds = xr.open_dataset(f"{data_path}/{file_name}")
    ds_pred = xr.zeros_like(ds[variable]) - 9999
    ds = ds.assign_coords(time=ds.time.dt.dayofyear-1)
    dtype = model.input_type
    lat = torch.tensor(ds.latitude.to_numpy(), dtype=dtype, device=device)
    lon = torch.tensor(ds.longitude.to_numpy(), dtype=dtype, device=device)
    ps = ds.level.to_numpy().astype(float)
    ts = ds.time.to_numpy().astype(float)
    model = model.to(device)
    max_error = np.zeros(ps.shape[0])
    for i in trange(ts.shape[0]):
        for j in range(ps.shape[0]):
            ti = float(ts[i])
            pj = float(ps[j])
            t = torch.tensor([ti], dtype=dtype, device=device)
            p = torch.tensor([pj], dtype=dtype, device=device)
            coord = torch.stack(torch.meshgrid(t, p, lat, lon, indexing="ij"), dim=-1).squeeze(0).squeeze(0)
            with torch.no_grad():
                var_pred = model(coord)
                ds_pred.data[i, j, :, :] = var_pred.cpu().numpy().squeeze(-1)
                max_error[j] = max(max_error[j], np.abs(ds_pred.data[i, j, :, :] - ds[variable][i, j, :, :]).max())
    print(np.array_repr(max_error))
    ds_pred.to_netcdf(f"{output_path}/{output_file}")

def generate_outputs(model, output_path, output_file, device="cuda"):
    file_name = model.args.file_name
    data_path = model.args.data_path
    variable = model.args.variable #"z"
    ds = xr.open_mfdataset(f"{data_path}/{file_name}").load()
    out_ds = xr.zeros_like(ds)
    #mean = float(ds[variable].mean())
    #std = float(ds[variable].max() - ds[variable].min())
    mean = ds[variable].mean(dim=["time"]).to_numpy()
    std = (ds[variable].max(dim=["time"]) - ds[variable].min(dim=["time"])).to_numpy()
    assert len(ds[variable].shape) == 3
    lon_v = torch.as_tensor(ds.lon.to_numpy(), device=device, dtype=torch.float32)
    lat_v = torch.as_tensor(ds.lat.to_numpy(), device=device, dtype=torch.float32)
    lat, lon = torch.meshgrid((lat_v, lon_v), indexing="ij")
    p = torch.zeros_like(lat, device=device) + float(ds.level.mean())
    t = torch.zeros_like(lat, device=device)
    model = model.to(device)
    errors = np.zeros(len(ds.time))
    for it in tqdm(range(len(ds.time))):
        coord = torch.stack((t + it, p, lat, lon), dim=-1)
        with torch.no_grad():
            var_pred = model(coord).squeeze(-1).cpu().numpy() * 0.5 * 1.4 * std + mean
            out_ds[variable].data[it, :, :] = var_pred[:, :]
            var = ds[variable].isel(time=it).to_numpy()
            errors[it] = np.abs(var_pred - var).max()
    file_name = f"{output_path}/{output_file}"
    print(f"Saving to {file_name}")
    out_ds.to_netcdf(file_name)
    print(errors.max())

def main(args):
    model = FitNetModule(args)
    if args.ckpt_path != "":
        model_loaded = FitNetModule.load_from_checkpoint(args.ckpt_path)
        model.model.load_state_dict(model_loaded.model.state_dict())

    trainer = None
    if not args.notraining:
        strategy = pl.strategies.DDPStrategy(process_group_backend="nccl", find_unused_parameters=False)
        trainer = pl.Trainer(accumulate_grad_batches=args.accumulate_grad_batches, check_val_every_n_epoch=10, accelerator="gpu", auto_select_gpus=True, devices=args.num_gpu, strategy=strategy, min_epochs=10, max_epochs=args.nepoches, gradient_clip_val=0.5, sync_batchnorm=True)
        trainer.fit(model)

    model.eval()
    if (not trainer) or trainer.is_global_zero:
            print("Model size (MB):", get_model_size_mb(model))

    if args.quantizing:
        model.model.fcs = model.model.fcs.half()
        quantized_size = get_model_size_mb(model)
        model.model.fcs = model.model.fcs.float()
        print(f"Quantized (FP16) size (MB): {quantized_size}")

    if args.testing and ((not trainer) or trainer.is_global_zero):
        test_on_wholedataset(model.args.file_name, model.args.data_path, model.args.output_path, model.args.output_file, model, variable=model.args.variable)

    if args.generate_full_outputs and ((not trainer) or trainer.is_global_zero):
        generate_outputs(model, args.output_path, args.output_file)

    return model

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_gpu", default=-1, type=int)
    parser.add_argument("--nepoches", default=20, type=int)
    parser.add_argument("--batch_size", default=3, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--sigma", default=1.6, type=float)
    parser.add_argument("--nfeature", default=128, type=int)
    parser.add_argument("--ntfeature", default=16, type=int)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--tscale", default=60., type=float)
    parser.add_argument("--zscale", default=100., type=float)
    parser.add_argument("--variable", default="z", type=str)
    parser.add_argument("--dataloader_mode", default="sampling_nc", type=str)
    parser.add_argument("--data_path", default=".", type=str)
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--ckpt_path", default="", type=str)
    parser.add_argument('--use_batchnorm', action='store_true')
    parser.add_argument('--use_skipconnect', action='store_true')
    parser.add_argument('--use_invscale', action='store_true')
    parser.add_argument('--use_fourierfeature', action='store_true')
    parser.add_argument('--use_tembedding', action='store_true')
    parser.add_argument("--tembed_size", default=400, type=int) # number of time steps
    parser.add_argument("--tresolution", default=24, type=float)
    parser.add_argument('--use_xyztransform', action='store_true')
    parser.add_argument('--use_stat', action='store_true')
    parser.add_argument('--loss_type', default="scaled_mse", type=str)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--generate_full_outputs', action='store_true')
    parser.add_argument("--output_path", default=".", type=str)
    parser.add_argument("--output_file", default="output.nc", type=str)
    parser.add_argument('--notraining', action='store_true')
    parser.add_argument('--quantizing', action='store_true')
    args = parser.parse_args()
    if args.all:
        args.use_batchnorm = True
        args.use_invscale = not args.use_stat
        args.use_skipconnect = True
        args.use_xyztransform = True
        args.use_fourierfeature = True
    main(args)
