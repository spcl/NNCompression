# Download data
Execute `WeatherBench/download.py` with following arguments:
```bash
python download.py --variable=geopotential --mode=single --level_type=pressure --years=2016 --resolution=0.5 --time=00:00 --pressure_level 10 50 100 200 300 400 500 700 850 925 1000 --custom_fn=dataset1.nc
python download.py --variable=geopotential --mode=single --level_type=pressure --years=2016 --resolution=0.25 --time=00:00 --pressure_level 10 50 100 200 300 400 500 700 850 925 1000 --custom_fn=dataset2.nc
python download.py --variable=geopotential --mode=separate --level_type=pressure --resolution=5.625 --pressure_level=500 --custom_fn=dataset3_z.nc
python download.py --variable=temperature --mode=separate --level_type=pressure --resolution=5.625 --pressure_level=850 --custom_fn=dataset3_t.nc
python download.py --variable=geopotential --mode=separate --level_type=pressure --resolution=2.8125 --pressure_level=500 --custom_fn=dataset4_z.nc
python download.py --variable=temperature --mode=separate --level_type=pressure --resolution=2.8125 --pressure_level=850 --custom_fn=dataset4_t.nc
```
# Run experiments in Section 3.1
```bash
for W in 32 64 128 256 512
do
    python train.py --nepoches=20 --all  --quantizing --testing --variable=z  --dataloader_mode=sampling_nc --file_name=dataset1.nc --width=$W --output_file=dataset1_w${W}.nc
    python train.py --nepoches=20 --all  --quantizing --testing --variable=z  --dataloader_mode=sampling_nc --file_name=dataset2.nc --width=$W --output_file=dataset2_w${W}.nc
    python train.py --nepoches=20 --all  --quantizing --generate_full_outputs --variable=z --use_stat --tscale=360 --dataloader_mode=weatherbench  --file_name=dataset3_z_*.nc --width=$W --output_file=dataset3_z_w${W}.nc
    python train.py --nepoches=20 --all  --quantizing --generate_full_outputs --variable=z --use_stat --tscale=360 --dataloader_mode=weatherbench  --file_name=dataset4_z_*.nc --width=$W --output_file=dataset4_z_w${W}.nc
done
```

# Run experiments in Section 3.2
```bash
python train.py --nepoches=20 --all  --quantizing --generate_full_outputs --variable=z --use_stat --tscale=360 --dataloader_mode=weatherbench  --file_name=dataset4_z_*.nc --width=512 --output_file=dataset4_z_w512.nc
python train.py --nepoches=20 --all  --quantizing --generate_full_outputs --variable=t --use_stat --tscale=360 --dataloader_mode=weatherbench  --file_name=dataset4_t_*.nc --width=512 --output_file=dataset4_t_w512.nc
cdo selyear,1979/2015 dataset4_z_w512.nc train_data_path/geopotential_500/geopotential_500_1979_2015.nc
cdo selyear,2016/2018 dataset4_z.nc train_data_path/geopotential_500/geopotential_500_2016_2018.nc
cdo selyear,1979/2015 dataset4_t_w512.nc train_data_path/temperature_850/temperature_850_1979_2015.nc
cdo selyear,2016/2018 dataset4_t.nc train_data_path/temperature_850/temperature_850_2016_2018.nc
cd WeatherBench
python -m src.train_nn -c config.yml --datadir=train_data_path
```