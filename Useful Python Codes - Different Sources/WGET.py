import wget

print('Beginning file download with wget module')
a = [i[:-1] if i.endswith('\n') else i for i in open('C:/Users/miladpanahi/PycharmProjects/Era5/subset_GLDAS_CLSM025_D_V2.0_20190820_025247.txt')]

a = open('C:/Users/miladpanahi/PycharmProjects/Era5/subset_GLDAS_CLSM025_D_V2.0_20190820_025247.txt').readlines()
aa = [x.strip() for x in open('C:/Users/miladpanahi/PycharmProjects/Era5/subset_GLDAS_CLSM025_D_V2.0_20190820_025247.txt').readlines()]

wget.download("https://username:password@example.com/")

url = aa[1]
wget.download('https://miladpanahi:*Mm91104748Mm*@hydro1.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FGLDAS%2FGLDAS_CLSM025_D.2.0%2F2014%2F09%2FGLDAS_CLSM025_D.A20140902.020.nc4&FORMAT=bmM0Lw&BBOX=-60%2C-180%2C90%2C180&LABEL=GLDAS_CLSM025_D.A20140902.020.nc4.SUB.nc4&SHORTNAME=GLDAS_CLSM025_D&SERVICE=L34RS_LDAS&VERSION=1.02&DATASET_VERSION=2.0&VARIABLES=AvgSurfT_tavg%2CRainf_f_tavg%2CRainf_tavg%2CSnowf_tavg%2CSnowT_tavg%2CTair_f_tavg%2CWind_f_tavg')



import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-land-monthly-means',
    {
        'format':'netcdf',
        'product_type':'monthly_averaged_reanalysis',
        'variable':[
            '10m_u_component_of_wind','10m_v_component_of_wind','2m_dewpoint_temperature',
            '2m_temperature','skin_temperature','snowfall',
            'total_precipitation'
        ],
        'year':[
            '2001','2002','2003',
            '2004','2005','2006',
            '2007','2008','2009',
            '2010','2011','2012',
            '2013','2014','2015',
            '2016','2017','2018',
            '2019'
        ],
        'month':[
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12'
        ],
        'time':'00:00'
    },
    'download.nc')