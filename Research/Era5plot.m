figure
% latlim = [40 90];
% lonlim = [0 150];
subplot(4,3,1)
imagesc(lonGPCC1,latGPCC1,WindEra5InttDJFF);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
%colorbar;
caxis([0 3.5])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Wind Speed - DJF')
%xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
subplot(4,3,2)
imagesc(lonGPCC1,latGPCC1,TEra5InttDJFF);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
%colorbar;
caxis([-45 45])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Air Temperature - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
subplot(4,3,3)
imagesc(lonGPCC1,latGPCC1,RhDJF);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
%cmap(33,:)=[1 1 1];
colormap(cmap);
%colorbar;
caxis([0 100])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Dewpoint Temperature - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
subplot(4,3,4)
imagesc(lonGPCC1,latGPCC1,WindEra5InttMAMM);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
%colorbar;
caxis([0 3.5])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Wind Speed - DJF')
%xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
subplot(4,3,5)
imagesc(lonGPCC1,latGPCC1,TEra5InttMAMM);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
%colorbar;
caxis([-45 45])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Air Temperature - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
subplot(4,3,6)
imagesc(lonGPCC1,latGPCC1,RhMAM);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
%cmap(33,:)=[1 1 1];
colormap(cmap);
%colorbar;
caxis([0 100])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Dewpoint Temperature - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
subplot(4,3,7)
imagesc(lonGPCC1,latGPCC1,WindEra5InttJJAA);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
%colorbar;
caxis([0 3.5])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Wind Speed - DJF')
%xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
subplot(4,3,8)
imagesc(lonGPCC1,latGPCC1,TEra5InttJJAA);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
%colorbar;
caxis([-45 45])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Air Temperature - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
subplot(4,3,9)
imagesc(lonGPCC1,latGPCC1,RhJJA);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
%cmap(33,:)=[1 1 1];
colormap(cmap);
%colorbar;
caxis([0 100])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Dewpoint Temperature - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
subplot(4,3,10)
imagesc(lonGPCC1,latGPCC1,WindEra5InttSONN);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
colorbar;
caxis([0 3.5])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Wind Speed - DJF')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
subplot(4,3,11)
imagesc(lonGPCC1,latGPCC1,TEra5InttSONN);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
colorbar;
caxis([-45 45])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Air Temperature - DJF')
xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
subplot(4,3,12)
imagesc(lonGPCC1,latGPCC1,RhSON);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
%cmap(33,:)=[1 1 1];
colormap(cmap);
colorbar;
caxis([0 100])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Dewpoint Temperature - DJF')
xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
%%
%1-D
scatterhist(RhSON(:),RelDiffSON(:),'Kernel','on','Location','SouthEast',...
    'Direction','out','Color','kbr','LineStyle',{'-','-.',':'},...
    'LineWidth',[2,2,2],'Marker','+od','MarkerSize',[4,5,6]);


h = histogram2(RhSON(:),RelDiffSON(:),'DisplayStyle','tile','ShowEmptyBins','on');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
colorbar;

%2-D
tmp = nanmean(RelDiffSONss,3);


[h,ch]=plot3c(erawSONint(:),eratSONint(:),tmp(:),[-100, -75, -50, -25, 0, 25, 50, 75, 100]);
[h,ch]=plot3c(erawSONint(:),RhSONint(:),tmp(:),[-100, -75, -50, -25, 0, 25, 50, 75, 100]);
[h,ch]=plot3c(RhSONint(:),eratSONint(:),tmp(:),[-100, -75, -50, -25, 0, 25, 50, 75, 100]);


RhDJFint
RhMAMint
RhJJAint
RhSONint

erawDJFint
erawMAMint
erawJJAint
erawSONint

eratDJFint
eratMAMint
eratJJAint
eratSONint


ymsPrDJF = cell(size(RelDiffSONss,3),1);
for i = 1:size(RelDiffSONss,3)
    tmp = RelDiffSONss(:,:,i);
    h = binscatter(RhSONint(:),tmp(:),[249 249]);
    %colormap(gca,'parula')
    %h.ShowEmptyBins = 'on';
    counts = flipud(transpose(h.Values));
    x = h.XBinEdges;
    y = h.YBinEdges;
    xm = movmean(x,2);xm = xm(2:end);
    ym = movmean(y,2);ym = flip(ym(2:end));
    ym1 = (ym*counts)./sum(counts,1);
    ymsPrDJF{i} = ym1;
    %colormap(gca,'jet')
end
ymsPrDJFs = cat(1,ymsPrDJF{:,1});

% for i = 1:37
%     plot(xm,ymsWDJFs(i,:))
%     hold on
% end


ymsPrDJFsavg = nanmean(ymsPrDJFs,1);
maxymsPrDJFsavg = zeros(37,1);
minymsPrDJFsavg = zeros(37,1);

for i = 1:size(ymsPrDJFs,2)
    maxymsPrDJFsavg(i) = nanmax(ymsPrDJFs(:,i));
    minymsPrDJFsavg(i) = nanmin(ymsPrDJFs(:,i));
end


tmp = nanmean(RelDiffSONss,3);
figure
subplot(2,1,1)
%yyaxis left
h = binscatter(RhSONint(:),tmp(:),[249 249]);
%hold on
%yyaxis right
x = h.XBinEdges;
xm = movmean(x,2);xm = xm(2:end);
counts = flipud(transpose(h.Values));
%bar(xm, sum(counts,1),'FaceColor',[1 1 1],'EdgeColor',[0 0 0],'LineWidth',.1)
%axis('ij');
subplot(2,1,2)
shade_plot(xm',minymsPrDJFsavg,maxymsPrDJFsavg,ymsPrDJFsavg')
xlabel('RH')
ylabel('RD')












%%
figure
plot(nanmean(RelDiffDJF,2),latGPCC1)
xlabel('RD')
ylabel('Lat [\circ]')
hold on
plot(nanmean(RelDiffMAM,2),latGPCC1)
xlabel('RD')
ylabel('Lat [\circ]')
hold on
plot(nanmean(RelDiffJJA,2),latGPCC1)
xlabel('RD')
ylabel('Lat [\circ]')
hold on
plot(nanmean(RelDiffSON,2),latGPCC1)
xlabel('RD')
ylabel('Lat [\circ]')
legend('DJF', 'MAM', 'JJA', 'SON')

%%