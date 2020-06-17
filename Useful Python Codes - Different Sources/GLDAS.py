import urllib.request
response = urllib.request.urlopen('http://www.example.com/')
html = response.read()
urllib.request.urlretrieve('http://www.example.com/songs/mp3.mp3', 'mp3.mp3')

dirr = "subset_GLDAS_CLSM025_D_2.0_20190929_224249.txt"
text_file = open(str(dirr), "r")
lines = text_file.readlines()

urllib.request.urlretrieve(lines[0])

user, password = 'miladpanahi@email.arizona.edu', 'Mm91104748Mm'
import requests
resp = requests.get(lines[1], auth=(user, password))

s = requests.Session()
s.get(lines[1])
s.post(lines[1], data={'_username': user, '_password': password})
s.get(lines[1])

# Set the URL string to point to a specific data URL. Some generic examples are:
#   https://servername/data/path/file
#   https://servername/opendap/path/file[.format[?subset]]
#   https://servername/daac-bin/OTF/HTTP_services.cgi?KEYWORD=value[&KEYWORD=value]
import requests
URL = lines[1].rstrip()

# Set the FILENAME string to the data file name, the LABEL keyword value, or any customized name.
FILENAME = 'gls'

import requests

result = requests.get(URL)
try:
    result.raise_for_status()
    f = open(FILENAME, 'wb')
    f.write(result.content)
    f.close()
    print('contents of URL written to ' + FILENAME)
except:
    print('requests.get() returned an error code ' + str(result.status_code))






import requests  # get the requsts library from https://github.com/requests/requests


# overriding requests.Session.rebuild_auth to mantain headers when redirected

class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'

    def __init__(self, username, password):

        super().__init__()

        self.auth = (username, password)

    # Overrides from the library to keep headers when redirected to or from

    # the NASA auth host.

    def rebuild_auth(self, prepared_request, response):

        headers = prepared_request.headers

        url = prepared_request.url

        if 'Authorization' in headers:

            original_parsed = requests.utils.urlparse(response.request.url)

            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and \
 \
                    redirect_parsed.hostname != self.AUTH_HOST and \
 \
                    original_parsed.hostname != self.AUTH_HOST:

                del headers['Authorization']

        return


# create session with the user credentials that will be used to authenticate access to the data

username = "miladpanahi@email.arizona.edu"

password = 

session = SessionWithHeaderRedirection(username, password)

# the url of the file we wish to retrieve
for i in range(24471):
    url = lines[i+1].rstrip()

    # extract the filename from the url to be used when saving the file

    filename = url[229:237]

    try:

        # submit the request using the session

        response = session.get(url, stream=True)

        print(response.status_code)

        # raise an exception in case of http errors

        response.raise_for_status()

        # save the file

        with open(filename, 'wb') as fd:

            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)



    except requests.exceptions.HTTPError as e:

        # handle any errors here

        print(e)




import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type':'monthly_averaged_reanalysis',
        'variable':[
            '100m_u_component_of_wind','100m_v_component_of_wind','10m_u_component_of_neutral_wind',
            '10m_u_component_of_wind','10m_v_component_of_neutral_wind','10m_v_component_of_wind',
            '10m_wind_speed','2m_dewpoint_temperature','2m_temperature',
            'evaporation','instantaneous_10m_wind_gust','instantaneous_large_scale_surface_precipitation_fraction',
            'land_sea_mask','large_scale_precipitation','large_scale_precipitation_fraction',
            'large_scale_rain_rate','large_scale_snowfall','large_scale_snowfall_rate_water_equivalent',
            'mean_evaporation_rate','mean_large_scale_precipitation_fraction','mean_large_scale_precipitation_rate',
            'mean_large_scale_snowfall_rate','mean_potential_evaporation_rate','mean_sea_level_pressure',
            'mean_snow_evaporation_rate','mean_snowfall_rate','mean_snowmelt_rate',
            'mean_total_precipitation_rate','orography','potential_evaporation',
            'precipitation_type','skin_temperature','slope_of_sub_gridscale_orography',
            'snow_albedo','snow_density','snow_depth',
            'snow_evaporation','snowfall','snowmelt',
            'standard_deviation_of_filtered_subgrid_orography','standard_deviation_of_orography','temperature_of_snow_layer',
            'total_column_snow_water','total_precipitation'
        ],
        'year':[
            '1979','1980','1981',
            '1982','1983','1984',
            '1985','1986','1987',
            '1988','1989','1990',
            '1991','1992','1993',
            '1994','1995','1996',
            '1997','1998','1999',
            '2000','2001','2002',
            '2003','2004','2005',
            '2006','2007','2008',
            '2009','2010','2011',
            '2012','2013','2014',
            '2015','2016','2017',
            '2018','2019'
        ],
        'month':[
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12'
        ],
        'time':'00:00',
        'format':'netcdf'
    },
    'era5.nc')