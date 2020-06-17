import pupygrib
with open('ST4.2002103112.24h', 'rb') as stream:
    for i, msg in enumerate(pupygrib.read(stream), 1):
        lons, lats = msg.get_coordinates()
        values = msg.get_values()
        print("Message {}: {:.3f} {}".format(i, values.mean(), lons.shape))