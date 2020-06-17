a = figure;
% latlim = [40 50];
% lonlim = [-105 -80];
subplot(4,1,1);
imagesc(lonM(1,:),latM(:,1),RD1);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
set(gca,'Ydir','Normal');
%hcb1=colorbar;
caxis([-50 50])
% set(gca, 'XLim', lonlim, 'YLim',latlim);
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
title('c)')
subplot(4,1,2);
imagesc(lonM(1,:),latM(:,1),RD2);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
set(gca,'Ydir','Normal');
%hcb1=colorbar;
caxis([-50 50])
% set(gca, 'XLim', lonlim, 'YLim',latlim);
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
title('f)')
subplot(4,1,3);
imagesc(lonM(1,:),latM(:,1),RD3);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
set(gca,'Ydir','Normal');
%hcb1=colorbar;
caxis([-50 50])
% set(gca, 'XLim', lonlim, 'YLim',latlim);
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
title('j)')
subplot(4,1,4);
imagesc(lonM(1,:),latM(:,1),RD4);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
set(gca,'Ydir','Normal');
%cmap1=jet;
%cmap1(33,:)=[1 1 1];
colormap(cmap1);
%cmap1(33,:)=[1 1 1];
caxis([-50 50])
% set(gca, 'XLim', lonlim, 'YLim',latlim);
xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
title('l)')
colorbar('southoutside')

% hp3 = get(subplot(4,1,4),'Position');

print(gcf,'FL3.png','-dpng','-r1000');


FuchsCorrFacsDJF = FuchsCorrFacsDJF(40:67,55:114,:);
FuchsCorrFacsJJA = FuchsCorrFacsJJA(40:67,55:114,:);
FuchsCorrFacsMAM = FuchsCorrFacsMAM(40:67,55:114,:);
FuchsCorrFacsSON = FuchsCorrFacsSON(40:67,55:114,:);

[llat, llon] = meshgrid(latGPCC,lonGPCC);llat = llat';llon = llon';
tmp = nanmean(FuchsCorrFacsSON,3);
SON=griddata(llat(:),llon(:),tmp(:),latM ,lonM);

mask = DJF;
mask(mask>0) = 1;


X = [-178.75 : 2.5 : 178.75];X = X(23:46);
Y = [-88.75 : 2.5 : 88.75];Y = Y(47:56);
[llat, llon] = meshgrid(Y,X);llat = llat';llon = llon';
tmp1 = (Leg(:,:,9) + Leg(:,:,10) + Leg(:,:,11))./3;
tmpp4=griddata(llat(:),llon(:),tmp1(:),latM ,lonM);
tmpp4 = tmpp4.*mask;


RD1 = 200*(DJF - tmpp1)./(DJF+tmpp1);
RD1(isnan(RD1))=0;

RD2 = 200*(MAM - tmpp2)./(MAM+tmpp2);
RD2(isnan(RD2))=0;

RD3 = 200*(JJA - tmpp3)./(JJA+tmpp3);
RD3(isnan(RD3))=0;

RD4 = 200*(SON - tmpp4)./(DJF+tmpp4);
RD4(isnan(RD4))=0;


hFig=gcf;
subplot(2,1,1)
image(X);
colormap(gca,'gray')
hcb1=colorbar;
subplot(2,1,2)
image(P);
colormap(gca,'jet')
hcb2=colorbar;


M = csvread('FBRatio.csv');
L = {'PRISM','GPCC','GPCC-F','GPCC-LW','GPCP','IMERG-F-Cal','IMERG-F-Uncal','IMERG-IR','IMERG-HQ','ERA-Interim', 'ERA5', 'MERRA2','UA-SWE','UA-SWE-NA'};

figure
imagesc(M); % plot the matrix
% set(gca, 'XTick', 1:12); % center x-axis ticks on bins
% set(gca, 'YTick', 1:12); % center y-axis ticks on bins
%set(gca, 'XTickLabel', L); % set x-axis labels
set(gca,'XTickLabel',[]);
xticks(0:1:14)
yticks(0:1:14)
set(gca,'YTickLabel',[]);
%set(gca, 'YTickLabel', L); % set y-axis labels
%xtickangle(90)
% set title
cmap=jet;
cmap=[.8 .8 .8;cmap];
colormap(cmap);
colorbar;
caxis([.25 2.5])


print(gcf,'BRaa.png','-dpng','-r1200');
