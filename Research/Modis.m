hdfinfo('MOD08_D3.A2008153.061.2017291133014.hdf');
data1 = double(hdfread('MOD08_D3.A2008153.061.2017291133014.hdf','Cloud_Fraction_Mean'))*1.0000e-04;
lat1 = hdfread('MOD08_D3.A2008153.061.2017291133014.hdf','YDim');
lon1 = hdfread('MOD08_D3.A2008153.061.2017291133014.hdf','XDim');
pcolor(lon1,lat1,data1)
hold on
data2 = double(hdfread('C:\Users\miladpanahi\Downloads\MOD06_L2.A2019079.1925.061.2019080073439.hdf','Cloud_Fraction'));
lat2 = hdfread('C:\Users\miladpanahi\Downloads\MOD06_L2.A2019079.1925.061.2019080073439.hdf','Latitude');
lon2 = hdfread('C:\Users\miladpanahi\Downloads\MOD06_L2.A2019079.1925.061.2019080073439.hdf','Longitude');
pcolor(lon2,lat2,data2)
data3 = double(hdfread('C:\Users\miladpanahi\Downloads\MOD06_L2.A2019079.1930.061.2019080073114.hdf','Cloud_Fraction'));
lat3 = hdfread('C:\Users\miladpanahi\Downloads\MOD06_L2.A2019079.1930.061.2019080073114.hdf','Latitude');
lon3 = hdfread('C:\Users\miladpanahi\Downloads\MOD06_L2.A2019079.1930.061.2019080073114.hdf','Longitude');
states = shaperead('usastatehi', 'UseGeoCoords', true);
pcolor(lon3,lat3,data3)
ylim([0 90])
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[1 1 1;cmap];
colormap(cmap);
colorbar;