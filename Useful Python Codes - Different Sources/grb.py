import gdal
import numpy as np
import statistics
import osr
import math

# Open file
dataset = gdal.Open('C:/Users/miladpanahi/Desktop/Master/Paper/NCEP-EMC-Stage IV/2002-03/ST4.2002103112.24h.grb', gdal.GA_ReadOnly)
message_count = dataset.RasterCount
x_size = dataset.RasterXSize
y_size = dataset.RasterYSize

src_srs = osr.SpatialReference()
src_srs.ImportFromWkt(dataset.GetProjection())
tgt_srs = osr.SpatialReference()
tgt_srs.ImportFromEPSG(4326)
transform = osr.CoordinateTransformation(src_srs, tgt_srs)

# Parsing for valid data points
message = dataset.GetRasterBand(1)
data_array = message.ReadAsArray()
data_points = []
for row in range(y_size):
    for col in range(x_size):
        temperature = data_array[row][col]
        if temperature != message.GetNoDataValue():
            lat_long_point = transform.TransformPoint(row, col)
            lat = lat_long_point[1]
            long = lat_long_point[0]
            data_points.append([lat, long, temperature])

# Display statistics for temperature
temperatures = [data_point[2] for data_point in data_points]
print("Count: " + str(len(temperatures)))
print("Max: " + str(np.max(temperatures)))
print("Min: " + str(np.min(temperatures)))
print("Mean: " + str(statistics.mean(temperatures)))
print("Standard Deviation: " + str(statistics.stdev(temperatures)))

# Show 1/20000 of the data points. Each data point holds a temperature and its corresponding lat/long
print("\nData Points:")
for i in range(math.floor(len(data_points) / 20000)):
    print(data_points[i * 20000])