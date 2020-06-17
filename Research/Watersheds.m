bilFiles = dir('*.dat'); 
numfiles = length(bilFiles);
DF = cell(1, numfiles);
for k = 1:numfiles 
   temp = restore_idl(bilFiles(k).name);
   DF{k} = temp.DATA.PRECIP;
end
St4Pr = cat(3,DF{1,:});
LonSt4 = double(temp.DATA.LON);
LatSt4 = double(temp.DATA.LAT);

tmp = sum(St4Pr,3);
int = griddata(LatSt4(:),LonSt4(:),tmp(:),latM ,lonM);

figure
h = pcolor(LonSt4,LatSt4,tmp);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
cmap=jet;
cmap=[1 1 1;cmap];
colormap(cmap);
colorbar;
set(h, 'EdgeColor', 'none');
caxis([0 300])


X = cell(671,1);
Y = cell(671,1);

Watersheds = shaperead('HCDN_nhru_final_671.shp','UseGeoCoords',true);
for i = 1:671
    Y{i} = (Watersheds(i).Lat);
    X{i} = (Watersheds(i).Lon);
end
WSLat = cat(2,Y{:,1});
WSLon = cat(2,X{:,1});

figure;

plot(WSLon,WSLat)
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');

hold on

figure
imagesc(lonM(1,:),latM(:,1),int);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[1 1 1;cmap];
colormap(cmap);
colorbar;
set(gca,'Ydir','Normal');
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
caxis([0 300])


latlim = [latM(49,1)-(latM(1,1)-latM(2,1))/2 latM(1,1)+(latM(1,1)-latM(2,1))/2];
lonlim = [lonM(1,1)-(latM(1,1)-latM(2,1))/2 lonM(1,116)+(latM(1,1)-latM(2,1))/2];
rasterSize = [588 1392];
R = georefcells(latlim,lonlim,rasterSize,'ColumnsStartFrom','north');
geotiffwrite('2015-16-R6.tif',mask6MM,R,'CoordRefSysCode',4326)


gamma1 = nanmean(SWEDF(23:610,2:1393,107:109),3).*mask3;
gamma1(gamma1==0) = NaN;
gamma2 = nanmean(SWEDF(23:610,2:1393,110:112),3).*mask4;
gamma2(gamma2==0) = NaN;

latlim = [v2(588) v2(1)];
lonlim = [v1(1) v1(1392)];
rasterSize = [588 1392];
R = georefcells(latlim,lonlim,rasterSize,'ColumnsStartFrom','north');
geotiffwrite('g7216R1.tif',gamma1,R,'CoordRefSysCode',4326)
geotiffwrite('g7220R4.tif',gamma2,R,'CoordRefSysCode',4326)




BW = imbinarize(mask1MM);
[B,L] = bwboundaries(BW,'noholes');
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on
for k = 1:length(B)
   boundary = B{k};
   plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
end
