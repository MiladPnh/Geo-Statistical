ncvars =  {'longitude','latitude','v10','u10','si10',...
          'z','d2m','t2m','e','mer','lsrr','mlspr','msr',...
          'lssfr','mlssr','mtpr','rsn','tp','es','mser'};
info = ncinfo('era5.nc'); 
      
%%Main Params
eralat = double(ncread("era5.nc", 'latitude'));
eralon = double(ncread("era5.nc", 'longitude'))-179.875;
[latera, lonera] = meshgrid(eralat,eralon);latera = latera';lonera = lonera';

% erav = double(ncread("era5.nc", 'v10'));erav = permute([erav(721:1440,:,:);erav(1:720,:,:)],[2 1 3]);
% erau = double(ncread("era5.nc", 'u10'));erau = permute([erau(721:1440,:,:);erau(1:720,:,:)],[2 1 3]);
%eraww = sqrt(erau.^2+erav.^2);

eraw = double(ncread("era5.nc", 'si10'));eraw = permute([eraw(721:1440,:,:);eraw(1:720,:,:)],[2 1 3]);
eraw1 = cell(12,1);
for i = 1:12
    eraw1{i} = nanmean(eraw(:,:,i:12:end),3);
end
eraw11 = cat(3,eraw1{:,1});
clear eraw eraw1

erawDJF = (eraw11(:,:,12)+eraw11(:,:,1)+eraw11(:,:,2))./3;
erawMAM = (eraw11(:,:,3)+eraw11(:,:,4)+eraw11(:,:,5))./3;
erawJJA = (eraw11(:,:,6)+eraw11(:,:,7)+eraw11(:,:,8))./3;
erawSON = (eraw11(:,:,9)+eraw11(:,:,10)+eraw11(:,:,11))./3;

erawDJFint = griddata(latera(:),lonera(:),erawDJF(:),llattt ,llonnn);
erawMAMint = griddata(latera(:),lonera(:),erawMAM(:),llattt ,llonnn);
erawJJAint = griddata(latera(:),lonera(:),erawJJA(:),llattt ,llonnn);
erawSONint = griddata(latera(:),lonera(:),erawSON(:),llattt ,llonnn);

load('UndercatchDB.mat', 'llattt','llonnn', 'RelDiffDJFss', 'RelDiffMAMss','RelDiffJJAss','RelDiffSONss');


eradew = double(ncread("era5.nc", 'd2m'));eradew = permute([eradew(721:1440,:,:);eradew(1:720,:,:)],[2 1 3]);
eradew1 = cell(12,1);
for i = 1:12
    eradew1{i} = nanmean(eradew(:,:,i:12:end),3);
end
eradew11 = cat(3,eradew1{:,1});
clear eradew eradew1

eradewDJF = (eradew11(:,:,12)+eradew11(:,:,1)+eradew11(:,:,2))./3;eradewDJF = eradewDJF-273.15;
eradewMAM = (eradew11(:,:,3)+eradew11(:,:,4)+eradew11(:,:,5))./3;eradewMAM = eradewMAM-273.15;
eradewJJA = (eradew11(:,:,6)+eradew11(:,:,7)+eradew11(:,:,8))./3;eradewJJA = eradewJJA-273.15;
eradewSON = (eradew11(:,:,9)+eradew11(:,:,10)+eradew11(:,:,11))./3;eradewSON = eradewSON-273.15;


erat = double(ncread("era5.nc", 't2m'));erat = permute([erat(721:1440,:,:);erat(1:720,:,:)],[2 1 3]);
erat1 = cell(12,1);
for i = 1:12
    erat1{i} = nanmean(erat(:,:,i:12:end),3);
end
erat11 = cat(3,erat1{:,1});
clear erat erat1

eratDJF = (erat11(:,:,12)+erat11(:,:,1)+erat11(:,:,2))./3;eratDJF = eratDJF-273.15;
eratMAM = (erat11(:,:,3)+erat11(:,:,4)+erat11(:,:,5))./3;eratMAM = eratMAM-273.15;
eratJJA = (erat11(:,:,6)+erat11(:,:,7)+erat11(:,:,8))./3;eratJJA = eratJJA-273.15;
eratSON = (erat11(:,:,9)+erat11(:,:,10)+erat11(:,:,11))./3;eratSON = eratSON-273.15;

eratDJFint = griddata(latera(:),lonera(:),eratDJF(:),llattt ,llonnn);
eratMAMint = griddata(latera(:),lonera(:),eratMAM(:),llattt ,llonnn);
eratJJAint = griddata(latera(:),lonera(:),eratJJA(:),llattt ,llonnn);
eratSONint = griddata(latera(:),lonera(:),eratSON(:),llattt ,llonnn);

RhDJF =100*(exp((17.625.*eradewDJF)./(243.04+eradewDJF))./exp((17.625.*eratDJF)./(243.04+eratDJF)));
RhMAM =100*(exp((17.625.*eradewMAM)./(243.04+eradewMAM))./exp((17.625.*eratMAM)./(243.04+eratMAM)));
RhJJA =100*(exp((17.625.*eradewJJA)./(243.04+eradewJJA))./exp((17.625.*eratJJA)./(243.04+eratJJA)));
RhSON =100*(exp((17.625.*eradewSON)./(243.04+eradewSON))./exp((17.625.*eratSON)./(243.04+eratSON)));

RhDJFint = griddata(latera(:),lonera(:),RhDJF(:),llattt ,llonnn);
RhMAMint = griddata(latera(:),lonera(:),RhMAM(:),llattt ,llonnn);
RhJJAint = griddata(latera(:),lonera(:),RhJJA(:),llattt ,llonnn);
RhSONint = griddata(latera(:),lonera(:),RhSON(:),llattt ,llonnn);

%%Altitude Params
eraZ = double(ncread("era5.nc", 'z'));eraZ = permute([eraZ(721:1440,:,:);eraZ(1:720,:,:)],[2 1 3]);
eraGeomH = ((eraZ/9.81)*6356000)./(eraZ/9.81-6356000);
eraZ1 = cell(12,1);
for i = 1:12
    eraZ1{i} = nanmean(eraGeomH(:,:,i:12:end),3);
end
eraZ11 = cat(3,eraZ1{:,1});
clear eraZ eraZ1

eraz = nanmean(eraZ11,3);

eraZint = griddata(latera(:),lonera(:),eraz(:),llattt ,llonnn);

eraZintt = -eraZint.*maskdjf;

%%Evaporation Params
erae = double(ncread("era5.nc", 'e'));erae = permute([erae(721:1440,:,:);erae(1:720,:,:)],[2 1 3]);
erae1 = cell(12,1);
for i = 1:12
    erae1{i} = nanmean(erae(:,:,i:12:end),3);
end
erae11 = cat(3,erae1{:,1});
clear erae erae1

eraeDJF = (erae11(:,:,12)+erae11(:,:,1)+erae11(:,:,2))./3;
eraeMAM = (erae11(:,:,3)+erae11(:,:,4)+erae11(:,:,5))./3;
eraeJJA = (erae11(:,:,6)+erae11(:,:,7)+erae11(:,:,8))./3;
eraeSON = (erae11(:,:,9)+erae11(:,:,10)+erae11(:,:,11))./3;

eraeDJFint = griddata(latera(:),lonera(:),eraeDJF(:),llattt ,llonnn);
eraeMAMint = griddata(latera(:),lonera(:),eraeMAM(:),llattt ,llonnn);
eraeJJAint = griddata(latera(:),lonera(:),eraeJJA(:),llattt ,llonnn);
eraeSONint = griddata(latera(:),lonera(:),eraeSON(:),llattt ,llonnn);


eramer = double(ncread("era5.nc", 'mer'));eramer = permute([eramer(721:1440,:,:);eramer(1:720,:,:)],[2 1 3]);
eramer1 = cell(12,1);
for i = 1:12
    eramer1{i} = nanmean(eramer(:,:,i:12:end),3);
end
eramer11 = cat(3,eramer1{:,1});
clear eramer eramer1

eramerDJF = (eramer11(:,:,12)+eramer11(:,:,1)+eramer11(:,:,2))./3;
eramerMAM = (eramer11(:,:,3)+eramer11(:,:,4)+eramer11(:,:,5))./3;
eramerJJA = (eramer11(:,:,6)+eramer11(:,:,7)+eramer11(:,:,8))./3;
eramerSON = (eramer11(:,:,9)+eramer11(:,:,10)+eramer11(:,:,11))./3;

eramerDJFint = griddata(latera(:),lonera(:),eramerDJF(:),llattt ,llonnn);
eramerDJFint = eramerDJFint*3600; %mm/hour
eramerMAMint = griddata(latera(:),lonera(:),eramerMAM(:),llattt ,llonnn);
eramerMAMint = eramerMAMint*3600; %mm/hour
eramerJJAint = griddata(latera(:),lonera(:),eramerJJA(:),llattt ,llonnn);
eramerJJAint = eramerJJAint*3600; %mm/hour
eramerSONint = griddata(latera(:),lonera(:),eramerSON(:),llattt ,llonnn);
eramerSONint = eramerSONint*3600; %mm/hour

eramerSONint(eramerSONint<0)=0;



%%Rain Rate Params
eramtpr = double(ncread("era5.nc", 'mtpr'));eramtpr = permute([eramtpr(721:1440,:,:);eramtpr(1:720,:,:)],[2 1 3]);
eramtpr1 = cell(12,1);
for i = 1:12
    eramtpr1{i} = nanmean(eramtpr(:,:,i:12:end),3);
end
eramtpr11 = cat(3,eramtpr1{:,1});
clear eramtpr eramtpr1

eramtprDJF = (eramtpr11(:,:,12)+eramtpr11(:,:,1)+eramtpr11(:,:,2))./3;
eramtprMAM = (eramtpr11(:,:,3)+eramtpr11(:,:,4)+eramtpr11(:,:,5))./3;
eramtprJJA = (eramtpr11(:,:,6)+eramtpr11(:,:,7)+eramtpr11(:,:,8))./3;
eramtprSON = (eramtpr11(:,:,9)+eramtpr11(:,:,10)+eramtpr11(:,:,11))./3;

eramtprDJFint = griddata(latera(:),lonera(:),eramtprDJF(:),llattt ,llonnn);
eramtprDJFint = eramtprDJFint*3600; %mm/hour
eramtprMAMint = griddata(latera(:),lonera(:),eramtprMAM(:),llattt ,llonnn);
eramtprMAMint = eramtprMAMint*3600; %mm/hour
eramtprJJAint = griddata(latera(:),lonera(:),eramtprJJA(:),llattt ,llonnn);
eramtprJJAint = eramtprJJAint*3600; %mm/hour
eramtprSONint = griddata(latera(:),lonera(:),eramtprSON(:),llattt ,llonnn);
eramtprSONint = eramtprSONint*3600; %mm/hour


erarr = double(ncread("era5.nc", 'lsrr'));erarr = permute([erarr(721:1440,:,:);erarr(1:720,:,:)],[2 1 3]);
erarr1 = cell(12,1);
for i = 1:12
    erarr1{i} = nanmean(erarr(:,:,i:12:end),3);
end
erarr11 = cat(3,erarr1{:,1});
clear erarr erarr1

eramlspr = double(ncread("era5.nc", 'mlspr'));eramlspr = permute([eramlspr(721:1440,:,:);eramlspr(1:720,:,:)],[2 1 3]);
eramlspr1 = cell(12,1);
for i = 1:12
    eramlspr1{i} = nanmean(eramlspr(:,:,i:12:end),3);
end
eramlspr11 = cat(3,eramlspr1{:,1});
clear eramlspr eramlspr1

eratp = double(ncread("era5.nc", 'tp'));eratp = permute([eratp(721:1440,:,:);eratp(1:720,:,:)],[2 1 3]);
eratp1 = cell(12,1);
for i = 1:12
    eratp1{i} = nanmean(eratp(:,:,i:12:end),3);
end
eratp11 = cat(3,eratp1{:,1});
clear eratp eratp1




%%Snow Rate Params
erasdens = double(ncread("era5.nc", 'rsn'));erasdens = permute([erasdens(721:1440,:,:);erasdens(1:720,:,:)],[2 1 3]);
erasdens1 = cell(12,1);
for i = 1:12
    erasdens1{i} = nanmean(erasdens(:,:,i:12:end),3);
end
erasdens11 = cat(3,erasdens1{:,1});
clear erasdens erasdens1

erasdensDJF = (erasdens11(:,:,12)+erasdens11(:,:,1)+erasdens11(:,:,2))./3;
erasdensMAM = (erasdens11(:,:,3)+erasdens11(:,:,4)+erasdens11(:,:,5))./3;
erasdensJJA = (erasdens11(:,:,6)+erasdens11(:,:,7)+erasdens11(:,:,8))./3;
erasdensSON = (erasdens11(:,:,9)+erasdens11(:,:,10)+erasdens11(:,:,11))./3;

erasdensDJFint = griddata(latera(:),lonera(:),erasdensDJF(:),llattt ,llonnn);
erasdensMAMint = griddata(latera(:),lonera(:),erasdensMAM(:),llattt ,llonnn);
erasdensJJAint = griddata(latera(:),lonera(:),erasdensJJA(:),llattt ,llonnn);
erasdensSONint = griddata(latera(:),lonera(:),erasdensSON(:),llattt ,llonnn);

eramsr = double(ncread("era5.nc", 'msr'));eramsr = permute([eramsr(721:1440,:,:);eramsr(1:720,:,:)],[2 1 3]);
eramsr1 = cell(12,1);
for i = 1:12
    eramsr1{i} = nanmean(eramsr(:,:,i:12:end),3);
end
eramsr11 = cat(3,eramsr1{:,1});
clear eramsr eramsr1


eramsrDJF = (eramsr11(:,:,12)+eramsr11(:,:,1)+eramsr11(:,:,2))./3;
eramsrMAM = (eramsr11(:,:,3)+eramsr11(:,:,4)+eramsr11(:,:,5))./3;
eramsrJJA = (eramsr11(:,:,6)+eramsr11(:,:,7)+eramsr11(:,:,8))./3;
eramsrSON = (eramsr11(:,:,9)+eramsr11(:,:,10)+eramsr11(:,:,11))./3;

eramsrDJFint = griddata(latera(:),lonera(:),eramsrDJF(:),llattt ,llonnn);
eramsrDJFint = eramsrDJFint*3600; %mm/hour
eramsrMAMint = griddata(latera(:),lonera(:),eramsrMAM(:),llattt ,llonnn);
eramsrMAMint = eramsrMAMint*3600; %mm/hour
eramsrJJAint = griddata(latera(:),lonera(:),eramsrJJA(:),llattt ,llonnn);
eramsrJJAint = eramsrJJAint*3600; %mm/hour
eramsrSONint = griddata(latera(:),lonera(:),eramsrSON(:),llattt ,llonnn);
eramsrSONint = eramsrSONint*3600; %mm/hour


eralssfr = double(ncread("era5.nc", 'lssfr'));eralssfr = permute([eralssfr(721:1440,:,:);eralssfr(1:720,:,:)],[2 1 3]);
eralssfr1 = cell(12,1);
for i = 1:12
    eralssfr1{i} = nanmean(eralssfr(:,:,i:12:end),3);
end
eralssfr11 = cat(3,eralssfr1{:,1});
clear eralssfr eralssfr1


eramlssr = double(ncread("era5.nc", 'mlssr'));eramlssr = permute([eramlssr(721:1440,:,:);eramlssr(1:720,:,:)],[2 1 3]);
eramlssr1 = cell(12,1);
for i = 1:12
    eramlssr1{i} = nanmean(eramlssr(:,:,i:12:end),3);
end
eramlssr11 = cat(3,eramlssr1{:,1});
clear eramlssr eramlssr1


eraes = double(ncread("era5.nc", 'es'));eraes = permute([eraes(721:1440,:,:);eraes(1:720,:,:)],[2 1 3]);
eraes1 = cell(12,1);
for i = 1:12
    eraes1{i} = nanmean(eraes(:,:,i:12:end),3);
end
eraes11 = cat(3,eraes1{:,1});
clear eraes eraes1

eraesDJF = (eraes11(:,:,12)+eraes11(:,:,1)+eraes11(:,:,2))./3;
eraesMAM = (eraes11(:,:,3)+eraes11(:,:,4)+eraes11(:,:,5))./3;
eraesJJA = (eraes11(:,:,6)+eraes11(:,:,7)+eraes11(:,:,8))./3;
eraesSON = (eraes11(:,:,9)+eraes11(:,:,10)+eraes11(:,:,11))./3;

eraesDJFint = griddata(latera(:),lonera(:),eraesDJF(:),llattt ,llonnn);eraesDJFint=eraesDJFint/4.2468e-08;
eraesMAMint = griddata(latera(:),lonera(:),eraesMAM(:),llattt ,llonnn);
eraesJJAint = griddata(latera(:),lonera(:),eraesJJA(:),llattt ,llonnn);
eraesSONint = griddata(latera(:),lonera(:),eraesSON(:),llattt ,llonnn);


eramser = double(ncread("era5.nc", 'mser'));eramser = permute([eramser(721:1440,:,:);eramser(1:720,:,:)],[2 1 3]);
eramser1 = cell(12,1);
for i = 1:12
    eramser1{i} = nanmean(eramser(:,:,i:12:end),3);
end
eramser11 = cat(3,eramser1{:,1});
clear eramser eramser1


eramserDJF = (eramser11(:,:,12)+eramser11(:,:,1)+eramser11(:,:,2))./3;
eramserMAM = (eramser11(:,:,3)+eramser11(:,:,4)+eramser11(:,:,5))./3;
eramserJJA = (eramser11(:,:,6)+eramser11(:,:,7)+eramser11(:,:,8))./3;
eramserSON = (eramser11(:,:,9)+eramser11(:,:,10)+eramser11(:,:,11))./3;

eramserDJFint = griddata(latera(:),lonera(:),eramserDJF(:),llattt ,llonnn);
eramserDJFint = eramserDJFint*3600; %mm/hour
eramserMAMint = griddata(latera(:),lonera(:),eramserMAM(:),llattt ,llonnn);
eramserMAMint = eramserMAMint*3600; %mm/hour
eramserJJAint = griddata(latera(:),lonera(:),eramserJJA(:),llattt ,llonnn);
eramserJJAint = eramserJJAint*3600; %mm/hour
eramserSONint = griddata(latera(:),lonera(:),eramserSON(:),llattt ,llonnn);
eramserSONint = eramserSONint*3600; %mm/hour

eramserDJFint(eramserDJFint<0)=0;


eralandmask = double(ncread("era5.nc", 'lsm'));eralandmask = permute([eralandmask(721:1440,:,:);eralandmask(1:720,:,:)],[2 1 3]);
eralandmask1 = cell(12,1);
for i = 1:12
    eralandmask1{i} = nanmean(eralandmask(:,:,i:12:end),3);
end
eralandmask11 = cat(3,eralandmask1{:,1});
clear eralandmask eralandmask1
eramask = nanmean(eralandmask11,3);eramask(eramask<.5)=NaN;eramask(eramask>.5)=1;

%% plots

load coast
figure
% latlim = [40 90];
% lonlim = [0 150];
subplot(4,1,1)
imagesc(eralon,eralat,RhDJF.*eramask);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
%colorbar;
caxis([0 100])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Wind Speed - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')

subplot(4,1,2)
imagesc(eralon,eralat,RhMAM.*eramask);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
%colorbar;
caxis([0 100])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Air Temperature - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')


subplot(4,1,3)
imagesc(eralon,eralat,RhJJA.*eramask);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
%colorbar;
caxis([0 100])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Dewpoint Temperature - DJF')
%xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')

subplot(4,1,4)
imagesc(eralon,eralat,RhSON.*eramask);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
%colorbar;
caxis([0 100])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Dewpoint Temperature - DJF')
xlabel('Lon [\circ]')
%ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')

%% Machine Learning

load('UndercatchDB.mat','FuchsCorrFacsDJFss','FuchsCorrFacsMAMss','FuchsCorrFacsJJAss','FuchsCorrFacsSONss')
load('UndercatchDB.mat','DJFLeg','MAMLeg','JJALeg','SONLeg')

Fdjf = nanmean(FuchsCorrFacsDJFss,3);
Fmam = nanmean(FuchsCorrFacsMAMss,3);
Fjja = nanmean(FuchsCorrFacsJJAss,3);
Fson = nanmean(FuchsCorrFacsSONss,3);

tmp = Fdjf(1:180,:);
DBdjf(:,1) = tmp(:);
tmp = DJFLeg(1:180,:);
DBdjf(:,2) = tmp(:);
tmp = erawDJFint(1:180,:);
DBdjf(:,3) = tmp(:);
tmp = eratDJFint(1:180,:);
DBdjf(:,4) = tmp(:);
tmp = RhDJFint(1:180,:);
DBdjf(:,5) = tmp(:);
tmp = eramtprDJFint(1:180,:);
DBdjf(:,6) = tmp(:);
tmp = erasdensDJFint(1:180,:);
DBdjf(:,7) = tmp(:);
tmp = eramsrDJFint(1:180,:);
DBdjf(:,8) = tmp(:);


tmp = Fjja(1:180,:);
DBjja(:,1) = tmp(:);
tmp = JJALeg(1:180,:);
DBjja(:,2) = tmp(:);
tmp = erawJJAint(1:180,:);
DBjja(:,3) = tmp(:);
tmp = eratJJAint(1:180,:);
DBjja(:,4) = tmp(:);
tmp = RhJJAint(1:180,:);
DBjja(:,5) = tmp(:);
tmp = eramtprJJAint(1:180,:);
DBjja(:,6) = tmp(:);
tmp = erasdensJJAint(1:180,:);
DBjja(:,7) = tmp(:);
tmp = eramsrJJAint(1:180,:);
DBjja(:,8) = tmp(:);

%% Zonal Plots
maskland = Fson>0;
tmp = SONLeg.*maskland;tmp(tmp==0)=NaN;
allData = cat(3,Fson,tmp);
CrcSON = nanmean(allData,3);
CrcZonalSON = nanmean(CrcSON,2);
%CrcZonalDJF = (CrcZonalDJF-1).*100;
LegZonalson = nanmean(tmp,2);
FZonalson = nanmean(Fson,2);
%LegZonal = (LegZonal-1).*100;
%FZonal = (FZonal-1).*100;
%RngCrcZonalDJF = FZonal - LegZonal;
%plot(89.5:-.5:-90,(CrcZonalDJF-1).*100);hold on;plot(89.5:-.5:-90,(RngCrcZonalDJF-1).*100);
%x = 89.5:-.5:-90;y = (CrcZonalDJF-1).*100;
%err = RngCrcZonalDJF.*100;
%errp = max(FZonal-1,LegZonal-1).*100;errn = min(FZonal-1,LegZonal-1).*100;
% errp = err;errp(errp<0)=NaN;
% errn = err;errn(errn>0)=NaN;

% errorbar(x,y,errn,errp,[],[],'--h','MarkerSize',6,...
%     'MarkerEdgeColor','k','MarkerFaceColor','k')

% errp(isnan(errp))=0;errp = errp./2+y;
% errn(isnan(errn))=0;errn = y - errn./2;
x = 89.5:-.5:-60;

figure
subplot(2,2,1)
plot(x,LegZonal(1:300))
hold on
plot(x,FZonal(1:300))
hold on
plot(x,CrcZonalDJF(1:300))
legend('Legates', 'Fuchs', 'Zonal Average Correction')
grid on
subplot(2,2,2)
plot(x,LegZonalmam(1:300))
hold on
plot(x,FZonalmam(1:300))
hold on
plot(x,CrcZonalMAM(1:300))
grid on
subplot(2,2,3)
plot(x,LegZonaljja(1:300))
hold on
plot(x,FZonaljja(1:300))
hold on
plot(x,CrcZonalJJA(1:300))
grid on
subplot(2,2,4)
plot(x,LegZonalson(1:300))
hold on
plot(x,FZonalson(1:300))
hold on
plot(x,CrcZonalSON(1:300))
grid on
ylabel('Lat [\circ]')

print(gcf,'PercentRD.png','-dpng','-r1200');


figure
plot(200*(FZonal(1:300)-LegZonal(1:300))./((FZonal(1:300)+LegZonal(1:300))),x)
hold on
plot(200*(FZonalmam(1:300)-LegZonalmam(1:300))./((FZonalmam(1:300)+LegZonalmam(1:300))),x)
hold on
plot(200*(FZonaljja(1:300)-LegZonaljja(1:300))./((FZonaljja(1:300)+LegZonaljja(1:300))),x)
hold on
plot(200*(FZonalson(1:300)-LegZonalson(1:300))./((FZonalson(1:300)+LegZonalson(1:300))),x)
legend('DJF', 'MAM', 'JJA', 'SON')
grid on


figure
x = -90:.5:89.5;
plot(landcnts', flipud(x'));
% hold on
% plot(flipud(x'), GPCCAnnualL);
% hold on
% plot(flipud(x'), GPCCAnnualF);
ylim([-65 90])
ylabel('Lat [\circ]')
%ylabel('Daily Precipitation [mm/day]')
xlabel('Zonal Land Pixel Counts')
%legend('GPCC', 'GPCC-L', 'GPCC-F')
grid on


figure
shade_plot(x',errn,errp,y)
legend('Uncertainty (Fuchs - Legates)','Zonal Average Correction')

