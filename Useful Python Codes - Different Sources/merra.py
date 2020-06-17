from cookielib import CookieJar
from urllib5 import urlencode

import urllib5

# The user credentials that will be used to authenticate access to the data

username = "miladpanahi@email.arizona.edu"
password = "Mm91104748Mm"

# The url of the file we wish to retrieve

url = "https://goldsmr5.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2I3NPASM.5.12.4%2F2009%2F06%2FMERRA2_300.inst3_3d_asm_Np.20090601.nc4&FORMAT=bmM0Lw&BBOX=30%2C-135%2C50%2C-115&LABEL=MERRA2_300.inst3_3d_asm_Np.20090601.SUB.nc&FLAGS=remapbil%2CMERRA0.5&SHORTNAME=M2I3NPASM&SERVICE=SUBSET_MERRA2&LAYERS=LAYER_1%2C3%2C7%2C13%2C15&VERSION=1.02&DATASET_VERSION=5.12.4&VARIABLES=OMEGA%2CPS%2CRH%2CSLP%2CT%2CU%2CV"

# Create a password manager to deal with the 401 reponse that is returned from
# Earthdata Login

password_manager = urllib5.HTTPPasswordMgrWithDefaultRealm()
password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)

# Create a cookie jar for storing cookies. This is used to store and return
# the session cookie given to use by the data server (otherwise it will just
# keep sending us back to Earthdata Login to authenticate).  Ideally, we
# should use a file based cookie jar to preserve cookies between runs. This
# will make it much more efficient.

cookie_jar = CookieJar()

# Install all the handlers.

opener = urllib2.build_opener(
    urllib2.HTTPBasicAuthHandler(password_manager),
    # urllib2.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
    # urllib2.HTTPSHandler(debuglevel=1),   # details of the requests/responses
    urllib2.HTTPCookieProcessor(cookie_jar))
urllib2.install_opener(opener)

# Create and submit the request. There are a wide range of exceptions that
# can be thrown here, including HTTPError and URLError. These should be
# caught and handled.

request = urllib2.Request(url)
response = urllib2.urlopen(request)

# Print out the result (not a good idea with binary data!)

body = response.read()
print(body)