import cdsapi

c = cdsapi.Client()

resolution = [0.5, 0.25]

for i, res in enumerate(resolution):
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['geopotential', 'temperature'], #'divergence', 'potential_vorticity', 'relative_humidity',
            'year': '2016',
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': '00:00',
            'pressure_level': [
            '10', '50', '100',
            '200', '300', '400',
            '500', '700', '850',
            '925', '1000',
        ],
        'grid': [res, res],  
        },
        f'dataset_{i+1}.nc')

# to get dataset 3 4, please retrieve from the WeatherBench tools: https://github.com/pangeo-data/WeatherBench/blob/master/src/download.py 