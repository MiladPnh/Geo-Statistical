figure
% latlim = [40 90];
% lonlim = [0 150];
h1 = subplot(4,3,1);
imagesc(lonGPCC1,latGPCC1,Fdjf);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(h1,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(h1, cmap);
%colorbar;
caxis([1 2.5])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Wind Speed - DJF')
%xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
xticks(-180:30:180) 
yticks(-90:45:90)
xtickangle(90) 
set(gca,'Xticklabel',[])
hold on; plot(long,lat,'k-')
h2 = subplot(4,3,2);
imagesc(lonGPCC1,latGPCC1,DJFLeg);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(h2,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(h2, cmap);
%colorbar;
caxis([1 2.5])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Air Temperature - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
xticks(-180:30:180) 
yticks(-90:45:90)
xtickangle(90)
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[])
hold on; plot(long,lat,'k-')
h3 = subplot(4,3,3);
imagesc(lonGPCC1,latGPCC1,RelDiffDJF);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(h3,'Ydir','Normal');
cmap=jet;
cmap(33,:)=[1 1 1];
colormap(h3, cmap);
%colorbar;
caxis([-50 50])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Dewpoint Temperature - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
xticks(-180:30:180) 
yticks(-90:45:90)
xtickangle(90)
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[])
hold on; plot(long,lat,'k-')
h4 = subplot(4,3,4);
imagesc(lonGPCC1,latGPCC1,Fmam);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(h4,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(h4, cmap);
%colorbar;
caxis([1 2.5])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Wind Speed - DJF')
%xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
xticks(-180:30:180) 
yticks(-90:45:90)
xtickangle(90) 
set(gca,'Xticklabel',[])
hold on; plot(long,lat,'k-')
h5 = subplot(4,3,5);
imagesc(lonGPCC1,latGPCC1,MAMLeg);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(h5,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(h5, cmap);
%colorbar;
caxis([1 2.5])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Air Temperature - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
xticks(-180:30:180) 
yticks(-90:45:90)
xtickangle(90) 
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[])
hold on; plot(long,lat,'k-')
h6 = subplot(4,3,6);
imagesc(lonGPCC1,latGPCC1,RelDiffMAM);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(h6,'Ydir','Normal');
cmap=jet;
cmap(33,:)=[1 1 1];
colormap(h6, cmap);
%colorbar;
caxis([-50 50])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Dewpoint Temperature - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
xticks(-180:30:180) 
yticks(-90:45:90)
xtickangle(90)
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[])
hold on; plot(long,lat,'k-')
h7 = subplot(4,3,7);
imagesc(lonGPCC1,latGPCC1,Fjja);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(h7,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(h7, cmap);
%colorbar;
caxis([1 2.5])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Wind Speed - DJF')
%xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
xticks(-180:30:180) 
yticks(-90:45:90)
xtickangle(90) 
set(gca,'Xticklabel',[])
hold on; plot(long,lat,'k-')
h8 = subplot(4,3,8);
imagesc(lonGPCC1,latGPCC1,JJALeg);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(h8,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(h8, cmap);
%colorbar;
caxis([1 2.5])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Air Temperature - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
xticks(-180:30:180) 
yticks(-90:45:90)
xtickangle(90)
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[])
hold on; plot(long,lat,'k-')
h9 = subplot(4,3,9);
imagesc(lonGPCC1,latGPCC1,RelDiffJJA);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(h9,'Ydir','Normal');
cmap=jet;
cmap(33,:)=[1 1 1];
colormap(h9, cmap);
%colorbar;
caxis([-50 50])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Dewpoint Temperature - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
xticks(-180:30:180) 
yticks(-90:45:90)
xtickangle(90) 
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[])
hold on; plot(long,lat,'k-')
h10 = subplot(4,3,10);
imagesc(lonGPCC1,latGPCC1,Fson);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(h10,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(h10, cmap);
colorbar;
caxis([1 2.5])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Wind Speed - DJF')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
xticks(-180:30:180) 
yticks(-90:45:90)
xtickangle(90) 
hold on; plot(long,lat,'k-')
h11 = subplot(4,3,11);
imagesc(lonGPCC1,latGPCC1,SONLeg);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(h11,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(h11, cmap);
colorbar;
caxis([1 2.5])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Air Temperature - DJF')
xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
xticks(-180:30:180) 
yticks(-90:45:90)
xtickangle(90) 
set(gca,'Yticklabel',[]) 
hold on; plot(long,lat,'k-')
h12 = subplot(4,3,12);
imagesc(lonGPCC1,latGPCC1,RelDiffSON);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(h12,'Ydir','Normal');
cmap=jet;
cmap(33,:)=[1 1 1];
colormap(h12, cmap);
colorbar;
caxis([-50 50])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Dewpoint Temperature - DJF')
xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
set(gca,'Yticklabel',[]) 
hold on; plot(long,lat,'k-')
xticks(-180:30:180) 
yticks(-90:45:90)
xtickangle(90) 
set(gca,'Yticklabel',[]) 


print(gcf,'cnt3d','-dpng','-r1600');
