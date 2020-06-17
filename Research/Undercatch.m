%%
lat=flipud(linspace(-90,90,3600)');lon=linspace(-180,180,7200);
figure
imagesc(lon,lat,RD1);
colormap(jet);
cmap=jet;
cmap(33,:)=[1 1 1];
colormap(cmap);
colorbar;
caxis([-50 50])
colorbar;
title('Relative Diff CHELSA Vs CHPclim')

figure
imagesc(lon,lat,RD2);
colormap(jet);
cmap=jet;
cmap(33,:)=[1 1 1];
colormap(cmap);
colorbar;
caxis([-50 50])
colorbar;
title('Relative Diff CHELSA Vs WorldClim')

figure
imagesc(lon,lat,RD3);
colormap(jet);
cmap=jet;
cmap(33,:)=[1 1 1];
colormap(cmap);
colorbar;
caxis([-50 50])
colorbar;
title('Relative Diff CHPclim Vs WorldClim')


RD1 = 200*(CHELSA_V12_corr_fac - CHPclim_V1_corr_fac)./(CHELSA_V12_corr_fac + CHPclim_V1_corr_fac);
RD1(isnan(RD1))=0;

RD2 = 200*(CHELSA_V12_corr_fac - WorldClim_V2_corr_fac)./(CHELSA_V12_corr_fac + WorldClim_V2_corr_fac);
RD2(isnan(RD2))=0;

RD3 = 200*(CHPclim_V1_corr_fac - WorldClim_V2_corr_fac)./(CHPclim_V1_corr_fac + WorldClim_V2_corr_fac);
RD3(isnan(RD3))=0;



zonal1 = nanmean(CHELSA_V12_corr_fac,2);
zonal2 = nanmean(CHPclim_V1_corr_fac,2);
zonal3 = nanmean(WorldClim_V2_corr_fac,2);
zonmean = (zonal1+zonal2+zonal3)./3;

figure
plot(zonmean,lat)
hold on
plot(zonal2,lat)
hold on
plot(zonal3,lat)
legend('CHELSA','CHPclim','WorldClim')

%%
projectdir = 'C:\Users\miladpanahi\Desktop\Master\Paper\GPCC\Monitoring Full dataset\SON';
dinfo = dir( fullfile(projectdir, '*.nc') );
num_files = length(dinfo);
filenames = fullfile( projectdir, {dinfo.name} );
precipsInt = cell(num_files, 1);
for K = 1 : num_files
  this_file = filenames{K};
  precipsInt{K} = transpose(ncread(this_file, 'corr_fac'));
  %precipsInt{1,K}(isnan(mydata{1,K}))=0;
end
FuchsCorrFacsSON = cat(3,precipsInt{:,1});



FuchsCorrFacsSONs = cell(size(FuchsCorrFacsSON,3)/3,1);
j = 0;
for i = 1:3:size(FuchsCorrFacsSON,3)
    tmp = nanmean(FuchsCorrFacsSON(:,:,i:i+2),3);
    j = j+1;
    FuchsCorrFacsSONs{j} = griddata(llatt(:),llonn(:),tmp(:),llattt ,llonnn);
end    
FuchsCorrFacsSONss = cat(3,FuchsCorrFacsSONs{:,1});



avgFuchsDJF = nanmean(FuchsCorrFacsDJFss,3);
avgFuchsMAM = nanmean(FuchsCorrFacsMAMss,3);
avgFuchsJJA = nanmean(FuchsCorrFacsJJAss,3);
avgFuchsSON = nanmean(FuchsCorrFacsSONss,3);




avgFuchsJJA = (nanmean(FuchsCorrFacs(:,:,246:248),3)+nanmean(FuchsCorrFacs(:,:,258:260),3)+nanmean(FuchsCorrFacs(:,:,270:272),3)+nanmean(FuchsCorrFacs(:,:,282:284),3)+nanmean(FuchsCorrFacs(:,:,294:296),3)+nanmean(FuchsCorrFacs(:,:,306:308),3)+nanmean(FuchsCorrFacs(:,:,318:320),3)+nanmean(FuchsCorrFacs(:,:,330:332),3)+nanmean(FuchsCorrFacs(:,:,342:344),3))./9;
[llattt, llonnn] = meshgrid(latGPCC1,lonGPCC1);llattt = llattt';llonnn = llonnn';
avgFuchsDJF1 = griddata(llatt(:),llonn(:),avgFuchsDJF(:),llattt ,llonnn);


figure
imagesc(lonGPCC,latGPCC,avgFuchsSON);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[1 1 1;cmap];
colormap(cmap);
colorbar;
caxis([1 2])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
title('CF-F , SON, 2010-11')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
load coast
hold on; plot(long,lat,'w-')

avgFuchs = nanmean(FuchsCorrFacs,3);

latGPCC = double(ncread("monitoring_v6_10_2002_11.nc", 'lat'));
lonGPCC = double(ncread("monitoring_v6_10_2002_11.nc", 'lon'));
[llatt, llonn] = meshgrid(latGPCC,lonGPCC);llatt = llatt';llonn = llonn';


figure
imagesc(lonGPCC,latGPCC,avgFuchs);
colormap(jet);
cmap=jet;
cmap(1,:)=[1 1 1];
colormap(cmap);
colorbar;
caxis([0 2])
colorbar;
set(gca,'Ydir','Normal');
title('Fuchs')


X = [-178.75 : 2.5 : 178.75];
Y = [-88.75 : 2.5 : 88.75];
[llat, llon] = meshgrid(Y,X);llat = llat';llon = llon';
Leg = flipud(mcof);
AvgLeg = nanmean(Leg(:,:,6:8),3);
AvgLegInt = griddata(llat(:),llon(:),AvgLeg(:),llattt ,llonnn);
mask = avgFuchs;
mask(mask>0) = 1;
AvgLegInt = AvgLegInt.*mask;


figure
imagesc(lonGPCC,latGPCC,AvgLegInt);
colormap(jet);
cmap=jet;
cmap(1,:)=[1 1 1];
colormap(cmap);
colorbar;
caxis([0 2])
colorbar;
set(gca,'Ydir','Normal');
title('Leg')


RdFL = 200*(AvgLegInt - avgFuchs)./(AvgLegInt+avgFuchs);
figure
imagesc(lonGPCC,latGPCC,RdFL);
colormap(jet);
cmap=jet;
cmap(33,:)=[1 1 1];
colormap(cmap);
colorbar;
caxis([-50 50])
colorbar;
set(gca,'Ydir','Normal');
title('Relative Difference F & LW')

dinfo = dir( fullfile('C:\Users\miladpanahi\Desktop\Master\Paper\GPCP', '*.nc') );


%%

projectdir = 'C:\Users\miladpanahi\Desktop\Master\Paper\GPCC\GPCC Full full dataset';
dinfo = dir( fullfile(projectdir, '*.nc') );
num_files = length(dinfo);
filenames = fullfile( projectdir, {dinfo.name} );
precipsInt = cell(num_files, 1);
for K = 1 : num_files
  this_file = filenames{K};
  precipsInt{K} = flipud(permute(ncread(this_file, 'precip'),[2 1 3]));
  %precipsInt{1,K}(isnan(mydata{1,K}))=0;
end

GPCCrawDJF = cell(num_files, 1);
GPCCrawMAM = cell(num_files, 1);
GPCCrawJJA = cell(num_files, 1);
GPCCrawSON = cell(num_files, 1);

for i = 2:35
    temp = precipsInt{i-1};
    tempo = precipsInt{i};    
    GPCCrawDJF{i} = nansum(temp(:,:,end-31:end),3) + nansum(tempo(:,:,1:59),3);
    GPCCrawMAM{i} = nansum(tempo(:,:,60:152),3);
    GPCCrawJJA{i} = nansum(tempo(:,:,153:245),3);
    GPCCrawSON{i} = nansum(tempo(:,:,246:end-32),3);
end

GPCCrawDJF1 = cat(3,GPCCrawDJF{:,1});
GPCCrawMAM1 = cat(3,GPCCrawMAM{:,1});
GPCCrawJJA1 = cat(3,GPCCrawJJA{:,1});
GPCCrawSON1 = cat(3,GPCCrawSON{:,1});

GPCCrawDJF1(GPCCrawDJF1==0) = NaN;
GPCCrawMAM1(GPCCrawMAM1==0) = NaN;
GPCCrawJJA1(GPCCrawJJA1==0) = NaN;
GPCCrawSON1(GPCCrawSON1==0) = NaN;

GPCCdjf = nanmean(GPCCrawDJF1,3)/91;
GPCCmam = nanmean(GPCCrawMAM1,3)/91;
GPCCjja = nanmean(GPCCrawJJA1,3)/91;
GPCCson = nanmean(GPCCrawSON1,3)/91;


GPCCrawDJF_L = nanmean(GPCCrawDJF1,3).*DJFLeg;
GPCCrawMAM_L = nanmean(GPCCrawMAM1,3).*MAMLeg;
GPCCrawJJA_L = nanmean(GPCCrawJJA1,3).*JJALeg;
GPCCrawSON_L = nanmean(GPCCrawSON1,3).*SONLeg;

GPCCrawDJF_F = nanmean(GPCCrawDJF1,3).*Fdjf;
GPCCrawMAM_F = nanmean(GPCCrawMAM1,3).*Fmam;
GPCCrawJJA_F = nanmean(GPCCrawJJA1,3).*Fjja;
GPCCrawSON_F = nanmean(GPCCrawSON1,3).*Fson;


GPCCdjfF = GPCCrawDJF_F./91;
GPCCmamF = GPCCrawMAM_F./91;
GPCCjjaF = GPCCrawJJA_F./91;
GPCCsonF = GPCCrawSON_F./91;

GPCCdjfL = GPCCrawDJF_L./91;
GPCCmamL = GPCCrawMAM_L./91;
GPCCjjaL = GPCCrawJJA_L./91;
GPCCsonL = GPCCrawSON_L./91;

maskdjf = Fdjf;maskdjf(maskdjf>0)=1;
maskmam = Fmam;maskmam(maskmam>0)=1;
maskjja = Fjja;maskjja(maskjja>0)=1;
maskson = Fson;maskson(maskson>0)=1;

GPCCdjf=GPCCdjf.*maskdjf;
GPCCmam=GPCCmam.*maskmam;
GPCCjja=GPCCjja.*maskjja;
GPCCson=GPCCson.*maskson;

GPCCdjfL=GPCCdjfL.*maskdjf;
GPCCmamL=GPCCmamL.*maskmam;
GPCCjjaL=GPCCjjaL.*maskjja;
GPCCsonL=GPCCsonL.*maskson;

GPCCdjfZ = nanmean(GPCCdjf,2);
GPCCmamZ = nanmean(GPCCmam,2);
GPCCjjaZ = nanmean(GPCCjja,2);
GPCCsonZ = nanmean(GPCCson,2);


GPCCdjfZL = nanmean(GPCCdjfL,2);
GPCCmamZL = nanmean(GPCCmamL,2);
GPCCjjaZL = nanmean(GPCCjjaL,2);
GPCCsonZL = nanmean(GPCCsonL,2);


GPCCdjfZF = nanmean(GPCCdjfF,2);
GPCCmamZF = nanmean(GPCCmamF,2);
GPCCjjaZF = nanmean(GPCCjjaF,2);
GPCCsonZF = nanmean(GPCCsonF,2);


GPCCAnnual = (GPCCdjfZ+GPCCmamZ+GPCCjjaZ+GPCCsonZ)./4;
GPCCAnnualL = (GPCCdjfZL+GPCCmamZL+GPCCjjaZL+GPCCsonZL)./4;
GPCCAnnualF = (GPCCdjfZF+GPCCmamZF+GPCCjjaZF+GPCCsonZF)./4;



figure
x = -90:.5:89.5;
plot(flipud(x'), landcnts');
% hold on
% plot(flipud(x'), GPCCAnnualL);
% hold on
% plot(flipud(x'), GPCCAnnualF);
xlim([-65 90])
xlabel('Lat [\circ]')
%ylabel('Daily Precipitation [mm/day]')
ylabel('Zonal Land Pixel Counts')
%legend('GPCC', 'GPCC-L', 'GPCC-F')
grid on

save('DrBehrangiCheck', 'GPCCdjf', 'GPCCmam', 'GPCCjja', 'GPCCson', 'GPCCrawDJF_L', 'GPCCrawMAM_L',...
'GPCCrawJJA_L', 'GPCCrawSON_L', 'GPCCrawDJF_F', 'GPCCrawMAM_F', 'GPCCrawJJA_F', 'GPCCrawSON_F', 'maskdjf',...
'maskmam', 'maskjja', 'maskson', 'DJFLeg', 'MAMLeg', 'JJALeg', 'SONLeg', 'Fdjf', 'Fmam', 'Fjja', 'Fson')

nansum(nansum(WorldClim_V2_Corr_SON_intt.*wn))


a = 2.1618;
b = 2.3337;
(b-a)/a*100


% var=GPCCdjfF;
% maskdjff = maskdjf;
% var(maskdjff<.5)=NaN;
v1=linspace(89.75,-89.75,360);
mlat=v1'*ones(1,720);
mlat(isnan(maskdjff)==1)=NaN;
figure;imagesc(mlat);
w=cos(mlat*pi/180);wn=w./nansum(w(:));

nansum(nansum(GPCCdjf.*wn))

GPCCdjf
GPCCmam
GPCCjja
GPCCson
GPCCdjfF
GPCCmamF
GPCCjjaF
GPCCsonF
GPCCdjfL
GPCCmamL
GPCCjjaL
GPCCsonL




figure(1)
h = binscatter(eraZintt(:),GPCCmam(:),[249 249]);
figure(2)
hf = binscatter(eraZintt(:),GPCCmamF(:),[249 249]);
figure(3)
hl = binscatter(eraZintt(:),GPCCmamL(:),[249 249]);
%colormap(gca,'parula')
%h.ShowEmptyBins = 'on';
counts = flipud(transpose(h.Values));countsf = flipud(transpose(hf.Values));countsl = flipud(transpose(hl.Values));
x = h.XBinEdges;xf = hf.XBinEdges;xl = hl.XBinEdges;
y = h.YBinEdges;yf = hf.YBinEdges;yl = hl.YBinEdges;
xm = movmean(x,2);xm = xm(2:end);xmf = movmean(xf,2);xmf = xmf(2:end);xml = movmean(xl,2);xml = xml(2:end);
ym = movmean(y,2);ym = flip(ym(2:end));ymf = movmean(yf,2);ymf = flip(ymf(2:end));yml = movmean(yl,2);yml = flip(yml(2:end));
ym1 = (ym*counts)./sum(counts,1);ym1f = (ymf*countsf)./sum(countsf,1);ym1l = (yml*countsl)./sum(countsl,1);

figure
plot(xm,ym1)
hold on
plot(xmf,ym1f)
hold on
plot(xml,ym1l)
legend('GPCC', 'GPCC-L', 'GPCC-F')
grid on


RelDdjf = 200*(GPCCdjfF-GPCCdjfL)./(GPCCdjfF+GPCCdjfL);
RelDmam = 200*(GPCCmamF-GPCCmamL)./(GPCCmamF+GPCCmamL);
RelDjja = 200*(GPCCjjaF-GPCCjjaL)./(GPCCjjaF+GPCCjjaL);
RelDson = 200*(GPCCsonF-GPCCsonL)./(GPCCsonF+GPCCsonL);

figure(1)
h = binscatter(eraZintt(:),RelDdjf(:),[249 249]);
counts = flipud(transpose(h.Values));
x = h.XBinEdges;
y = h.YBinEdges;
xm = movmean(x,2);xm = xm(2:end);
ym = movmean(y,2);ym = flip(ym(2:end));
ymm = (ym*counts)./sum(counts,1);
figure(2)
h1 = binscatter(eraZintt(:),RelDmam(:),[249 249]);
counts1 = flipud(transpose(h1.Values));
x1 = h1.XBinEdges;
y1 = h1.YBinEdges;
xm1 = movmean(x1,2);xm1 = xm1(2:end);
ym1 = movmean(y1,2);ym1 = flip(ym1(2:end));
ym11 = (ym1*counts1)./sum(counts1,1);
figure(3)
h2 = binscatter(eraZintt(:),RelDjja(:),[249 249]);
counts2 = flipud(transpose(h2.Values));
x2 = h2.XBinEdges;
y2 = h2.YBinEdges;
xm2 = movmean(x2,2);xm2 = xm2(2:end);
ym2 = movmean(y2,2);ym2 = flip(ym2(2:end));
ym12 = (ym2*counts2)./sum(counts2,1);
figure(4)
h3 = binscatter(eraZintt(:),RelDson(:),[249 249]);
counts3 = flipud(transpose(h3.Values));
x3 = h3.XBinEdges;
y3 = h3.YBinEdges;
xm3 = movmean(x3,2);xm3 = xm3(2:end);
ym3 = movmean(y3,2);ym3 = flip(ym3(2:end));
ymm3 = (ym3*counts3)./sum(counts3,1);

figure
plot(xm,ymm)
hold on
plot(xm1,ym11)
hold on
plot(xm2,ym12)
hold on
plot(xm3,ymm3)
grid on
legend('DJF', 'MAM', 'JJA', 'SON')


phasepercentageDJFint
phasepercentageMAMint
phasepercentageJJAint
phasepercentageSONint

% mesh(phasepercentageDJFint,eraZintt,RelDdjf)
% 
% [OX, OY] = ndgrid(phasepercentageDJFint, eraZintt);
% [OXYv] = [OX(:), OY(:)];
% Rx = reshape(Rxv, size(OX));
% Ry = reshape(Ryv, size(OX));
% pcolor(Rx, Ry, RelDdjf)

RelDann(:,:,1) = RelDdjf;RelDann(:,:,2) = RelDmam;RelDann(:,:,3) = RelDjja;RelDann(:,:,4) = RelDson;
RelDann1 = nanmean(RelDann,3);

PhaseAnn(:,:,1) = phasepercentageDJFint;PhaseAnn(:,:,2) = phasepercentageMAMint;PhaseAnn(:,:,3) = phasepercentageJJAint;PhaseAnn(:,:,4) = phasepercentageSONint;
PhaseAnn1 = nanmean(PhaseAnn,3);


maskk = RelDann1;maskk(maskk>0)=1;maskk(maskk<=0)=1;
PhaseAnn11 = PhaseAnn1.*maskk;
eraZintt1 = eraZintt.*maskk;

[Aphase,Iphase] = sort(PhaseAnn11(:));
[Aalt,Ialt] = sort(eraZintt1(:));
[Ard,Ird] = sort(RelDann1(:));

sampleNum = 3200;
size = 259200/3200;

AphaseMapped = sepblockfun(Aphase(:), [sampleNum,1], @nanmean);
AaltMapped = sepblockfun(Aalt(:), [sampleNum,1], @nanmean); 
ArdMapped = sepblockfun(Ard(:), [sampleNum,1], @nanmean); 

realSize = length(ArdMapped(~isnan(ArdMapped)));


whos ArdMapped

latcell = cell(realSize,1);
loncell = cell(realSize,1);

for i = 1:realSize
    loncell{i} = Ialt(sampleNum*(i-1)+1:sampleNum*i);
end
loncellc = cat(2,loncell{:,1});

for i = 1:realSize
    latcell{i} = Iphase(sampleNum*(i-1)+1:sampleNum*i);
end
latcellc = cat(2,latcell{:,1});

G = zeros(sampleNum,realSize);


for i = 1:sampleNum
    for j = 1:realSize
        [row, col] = find(latcellc==loncellc(i,j));
        G(i,j) = col;
    end
end

for i = 1:realSize
    for j = 1:realSize
        GG(i,j).data = [];
    end
end

tmp = RelDdjf(:);
for i = 1:sampleNum
    for j = 1:realSize
        GG(G(i,j),j).data = [GG(G(i,j),j).data tmp(loncellc(i,j))];
    end
end

figure
[h,ch]=plot3c(eraZintt1(:),phasepercentageDJFint1(:),RelDdjf(:),[-100, -75, -50, -25, 0, 25, 50, 75, 100]);

GGG = zeros(realSize,realSize);
for i = 1:realSize
    for j = 1:realSize
        GGG(i,j) = nanmean(GG(i,j).data);
    end
end

GGGcnt = zeros(realSize,realSize);
for i = 1:realSize
    for j = 1:realSize
        GGGcnt(i,j) = length(GG(i,j).data);
    end
end

xlon = AaltMapped(1:realSize);
ylat = AphaseMapped(1:realSize);

figure
imagesc(xlon, ylat, GGG);
colormap jet
colorbar
%set(gca,'Ydir','Normal');

figure
imagesc(xlon, ylat, GGGcnt);
colormap jet
colorbar
%set(gca,'Ydir','Normal');


GPCCdjf
GPCCmam
GPCCjja
GPCCson
GPCCdjfF
GPCCmamF
GPCCjjaF
GPCCsonF
GPCCdjfL
GPCCmamL
GPCCjjaL
GPCCsonL


varl=GPCCsonL;
varf=GPCCsonF;
var=GPCCson;
a=maskdjf; sa=nansum(a');fs=find(sa<2);
f=find(isnan(a)==1);
var(f)=NaN;
varl(f)=NaN;
varf(f)=NaN;
v1=linspace(89.75,-89.75,360);
v2=nanmean(var');
v2f=nanmean(varf');
v2l=nanmean(varl');
b=cos(v1*pi/180);
b(fs)=NaN;
b2=b./nansum(b(:));
gl=nansum(v2l.*b2);
gf=nansum(v2f.*b2);
g=nansum(v2.*b2);
[g gf gl]
[(gf-g)/g*100 (gl-g)/g*100 ]

%% GPCP Global Numbers

projectdir = 'C:\Users\miladpanahi\Desktop\Master\Paper\GPCP';
dinfo = dir( fullfile(projectdir, '*.nc') );
precipsInt = permute(ncread('precip.mon.mean.nc', 'precip'),[2 1 3]);
latGPCP = flipud(ncread('precip.mon.mean.nc', 'lat'));
lonGPCP = ncread('precip.mon.mean.nc', 'lon')-180;

precipsIntt = [precipsInt(:,73:144,:),precipsInt(:,1:72,:)];
precipsInttt = flipud(precipsIntt);
% imagesc(lonGPCP,latGPCP,precipsInttt(:,:,1))
% set(gca,'Ydir','Normal');
% colormap jet
% colorbar

mask2_5deg1 = [mask2_5deg(:,73:144),mask2_5deg(:,1:72)];
mask2_5deg2 = flipud(mask2_5deg1);
mask = mask2_5deg2;
mask(mask2_5deg2>95) = NaN;
mask(mask<=95)=1;
% figure
% imagesc(tmp)
% set(gca,'Ydir','Normal');
GPCP = precipsInttt.*mask;


GPCPson = cell(size(GPCP,3)/12,1);
j = 0;
for i = 9:12:size(GPCP,3)
    j = j+1;
    GPCPson{j} = nanmean(GPCP(:,:,i:i+2),3);
end    
GPCPdjf1 = cat(3,GPCPdjf{:,1}); GPCPdjf2 = nanmean(GPCPdjf1,3);
GPCPmam1 = cat(3,GPCPmam{:,1}); GPCPmam2 = nanmean(GPCPmam1,3);
GPCPjja1 = cat(3,GPCPjja{:,1}); GPCPjja2 = nanmean(GPCPjja1,3);
GPCPson1 = cat(3,GPCPson{:,1}); GPCPson2 = nanmean(GPCPson1,3);
GPCPdjf{41} = nanmean(GPCP(:,:,1:2),3);

%Pixel Method
v1=linspace(88.75,-88.75,72);
mlat=v1'*ones(1,144);
mlat(isnan(mask)==1)=NaN;
figure;imagesc(mlat);
w=cos(mlat*pi/180);wn=w./nansum(w(:));

nansum(nansum(GPCPson2.*wn))

%Zonal Method

var=GPCPson2;
a=mask; sa=nansum(a');fs=find(sa<2);
f=find(isnan(a)==1);
var(f)=NaN;
v1=linspace(88.75,-88.75,72);
v2=nanmean(var');
b=cos(v1*pi/180);
b(fs)=NaN;
b2=b./nansum(b(:));
g=nansum(v2.*b2)




%%
WorldClim_V2_Corr_DJF_int = WorldClim_V2_Corr_DJF_int./30;
WorldClim_V2_Corr_MAM_int = WorldClim_V2_Corr_MAM_int./30;
WorldClim_V2_Corr_JJA_int = WorldClim_V2_Corr_JJA_int./30;
WorldClim_V2_Corr_SON_int = WorldClim_V2_Corr_SON_int./30;

WorldClim_V2_Corr_DJF_intt = WorldClim_V2_Corr_DJF_int.*maskdjf;
WorldClim_V2_Corr_MAM_intt = WorldClim_V2_Corr_MAM_int.*maskmam;
WorldClim_V2_Corr_JJA_intt = WorldClim_V2_Corr_JJA_int.*maskjja;
WorldClim_V2_Corr_SON_intt = WorldClim_V2_Corr_SON_int.*maskson;


WorldClim_V2_Corr_DJF_inttZ = nansum(WorldClim_V2_Corr_DJF_intt,2);
WorldClim_V2_Corr_MAM_inttZ = nansum(WorldClim_V2_Corr_MAM_intt,2);
WorldClim_V2_Corr_JJA_inttZ = nansum(WorldClim_V2_Corr_JJA_intt,2);
WorldClim_V2_Corr_SON_inttZ = nansum(WorldClim_V2_Corr_SON_intt,2);

maskdjf(isnan(maskdjf))=0;
landcnts = zeros(1,360);
for i = 1:360
    landcnts(i) = nnz(maskdjf(i,:));
end    
landcnts(landcnts==0)=NaN;
landcntss = landcnts./nanmean(landcnts);
landcntss = landcntss';

GPCCdjfZ = nansum(GPCCdjf,2);GPCCdjfZ=GPCCdjfZ.*landcntss;
GPCCmamZ = nansum(GPCCmam,2);GPCCmamZ=GPCCmamZ.*landcntss;
GPCCjjaZ = nansum(GPCCjja,2);GPCCjjaZ=GPCCjjaZ.*landcntss;
GPCCsonZ = nansum(GPCCson,2);GPCCsonZ=GPCCsonZ.*landcntss;





e = cos(latGPCC1./180*pi);
ee = e(1:180)/nansum(e(1:180));eee = [ee;flipud(ee)];

GPCCdjfZZ = WorldClim_V2_Corr_DJF_inttZ.*eee;
GPCCmamZZ = WorldClim_V2_Corr_MAM_inttZ.*eee;
GPCCjjaZZ = WorldClim_V2_Corr_JJA_inttZ.*eee;
GPCCsonZZ = WorldClim_V2_Corr_SON_inttZ.*eee;

GPCCdjfZZ = GPCCdjfZ.*eee;
GPCCmamZZ = GPCCmamZ.*eee;
GPCCjjaZZ = GPCCjjaZ.*eee;
GPCCsonZZ = GPCCsonZ.*eee;


GPCCdjfZZZ = nanmean(GPCCdjfZZ);
GPCCmamZZZ = nanmean(GPCCmamZZ);
GPCCjjaZZZ = nanmean(GPCCjjaZZ);
GPCCsonZZZ = nanmean(GPCCsonZZ);

GlobalGPCCraw = (GPCCdjfZZZ+GPCCmamZZZ+GPCCjjaZZZ+GPCCsonZZZ)./4;





CHELSA_V12_Corr = ncread('CHELSA_V12.nc','corr_P_monthly');
CHELSA_V12_Org = ncread('CHELSA_V12.nc','orig_P_monthly');
ChelsaCF = CHELSA_V12_Corr./CHELSA_V12_Org;
CHPclim_V1_Corr = ncread('CHPclim_V1.nc','corr_P_monthly');
CHPclim_V1_Org = ncread('CHPclim_V1.nc','orig_P_monthly');ChpCF = CHPclim_V1_Corr./CHPclim_V1_Org;
WorldClim_V2_Corr = ncread('WorldClim_V2.nc','corr_P_monthly');
WorldClim_V2_Org = ncread('WorldClim_V2.nc','orig_P_monthly');WcCF = WorldClim_V2_Corr./WorldClim_V2_Org;

DJFLeg = (legmonth(:,:,12)+legmonth(:,:,1)+legmonth(:,:,2))./3;
MAMLeg = (legmonth(:,:,3)+legmonth(:,:,4)+legmonth(:,:,5))./3;
JJALeg = (legmonth(:,:,6)+legmonth(:,:,7)+legmonth(:,:,8))./3;
SONLeg = (legmonth(:,:,9)+legmonth(:,:,10)+legmonth(:,:,11))./3;




%%
CHELSA_V12_Corr = permute(CHELSA_V12_Corr,[2 1 3]);
ChelsaCF = permute(ChelsaCF,[2 1 3]);
CHPclim_V1_Corr = permute(CHPclim_V1_Corr,[2 1 3]);
ChpCF = permute(ChpCF,[2 1 3]);
WorldClim_V2_Corr = permute(WorldClim_V2_Corr,[2 1 3]);
WcCF = permute(WcCF,[2 1 3]);
%%
lonBeck = ncread('WorldClim_V2.nc','lon');
latBeck = ncread('WorldClim_V2.nc','lat');
[latBeckk, lonBeckk] = meshgrid(latBeck,lonBeck);lonBeckk = lonBeckk';latBeckk = latBeckk';


CHELSA_V12_Corr_DJF = (CHELSA_V12_Corr(:,:,12)+CHELSA_V12_Corr(:,:,1)+CHELSA_V12_Corr(:,:,2))./3;
CHELSA_V12_Corr_MAM = (CHELSA_V12_Corr(:,:,3)+CHELSA_V12_Corr(:,:,4)+CHELSA_V12_Corr(:,:,5))./3;
CHELSA_V12_Corr_JJA = (CHELSA_V12_Corr(:,:,6)+CHELSA_V12_Corr(:,:,7)+CHELSA_V12_Corr(:,:,8))./3;
CHELSA_V12_Corr_SON = (CHELSA_V12_Corr(:,:,9)+CHELSA_V12_Corr(:,:,10)+CHELSA_V12_Corr(:,:,11))./3;

CHPclim_V1_Corr_DJF = (CHPclim_V1_Corr(:,:,12)+CHPclim_V1_Corr(:,:,1)+CHPclim_V1_Corr(:,:,2))./3;
CHPclim_V1_Corr_MAM = (CHPclim_V1_Corr(:,:,3)+CHPclim_V1_Corr(:,:,4)+CHPclim_V1_Corr(:,:,5))./3;
CHPclim_V1_Corr_JJA = (CHPclim_V1_Corr(:,:,6)+CHPclim_V1_Corr(:,:,7)+CHPclim_V1_Corr(:,:,8))./3;
CHPclim_V1_Corr_SON = (CHPclim_V1_Corr(:,:,9)+CHPclim_V1_Corr(:,:,10)+CHPclim_V1_Corr(:,:,11))./3;

WorldClim_V2_Corr_DJF = (WorldClim_V2_Corr(:,:,12)+WorldClim_V2_Corr(:,:,1)+WorldClim_V2_Corr(:,:,2))./3;
WorldClim_V2_Corr_MAM = (WorldClim_V2_Corr(:,:,3)+WorldClim_V2_Corr(:,:,4)+WorldClim_V2_Corr(:,:,5))./3;
WorldClim_V2_Corr_JJA = (WorldClim_V2_Corr(:,:,6)+WorldClim_V2_Corr(:,:,7)+WorldClim_V2_Corr(:,:,8))./3;
WorldClim_V2_Corr_SON = (WorldClim_V2_Corr(:,:,9)+WorldClim_V2_Corr(:,:,10)+WorldClim_V2_Corr(:,:,11))./3;

ChelsaCF_DJF = (ChelsaCF(:,:,12)+ChelsaCF(:,:,1)+ChelsaCF(:,:,2))./3;
ChelsaCF_MAM = (ChelsaCF(:,:,3)+ChelsaCF(:,:,4)+ChelsaCF(:,:,5))./3;
ChelsaCF_JJA = (ChelsaCF(:,:,6)+ChelsaCF(:,:,7)+ChelsaCF(:,:,8))./3;
ChelsaCF_SON = (ChelsaCF(:,:,9)+ChelsaCF(:,:,10)+ChelsaCF(:,:,11))./3;

ChpCF_DJF = (ChpCF(:,:,12)+ChpCF(:,:,1)+ChpCF(:,:,2))./3;
ChpCF_MAM = (ChpCF(:,:,3)+ChpCF(:,:,4)+ChpCF(:,:,5))./3;
ChpCF_JJA = (ChpCF(:,:,6)+ChpCF(:,:,7)+ChpCF(:,:,8))./3;
ChpCF_SON = (ChpCF(:,:,9)+ChpCF(:,:,10)+ChpCF(:,:,11))./3;

WcCF_DJF = (WcCF(:,:,12)+WcCF(:,:,1)+WcCF(:,:,2))./3;
WcCF_MAM = (WcCF(:,:,3)+WcCF(:,:,4)+WcCF(:,:,5))./3;
WcCF_JJA = (WcCF(:,:,6)+WcCF(:,:,7)+WcCF(:,:,8))./3;
WcCF_SON = (WcCF(:,:,9)+WcCF(:,:,10)+WcCF(:,:,11))./3;

CHELSA_V12_Corr_DJF_int = griddata(latBeckk(:),lonBeckk(:),double(CHELSA_V12_Corr_DJF(:)),llattt ,llonnn);
CHELSA_V12_Corr_MAM_int = griddata(latBeckk(:),lonBeckk(:),double(CHELSA_V12_Corr_MAM(:)),llattt ,llonnn);
CHELSA_V12_Corr_JJA_int = griddata(latBeckk(:),lonBeckk(:),double(CHELSA_V12_Corr_JJA(:)),llattt ,llonnn);
CHELSA_V12_Corr_SON_int = griddata(latBeckk(:),lonBeckk(:),double(CHELSA_V12_Corr_SON(:)),llattt ,llonnn);

CHPclim_V1_Corr_DJF_int = griddata(latBeckk(:),lonBeckk(:),double(CHPclim_V1_Corr_DJF(:)),llattt ,llonnn);
CHPclim_V1_Corr_MAM_int = griddata(latBeckk(:),lonBeckk(:),double(CHPclim_V1_Corr_MAM(:)),llattt ,llonnn);
CHPclim_V1_Corr_JJA_int = griddata(latBeckk(:),lonBeckk(:),double(CHPclim_V1_Corr_JJA(:)),llattt ,llonnn);
CHPclim_V1_Corr_SON_int = griddata(latBeckk(:),lonBeckk(:),double(CHPclim_V1_Corr_SON(:)),llattt ,llonnn);

WorldClim_V2_Corr_DJF_int = griddata(latBeckk(:),lonBeckk(:),double(WorldClim_V2_Corr_DJF(:)),llattt ,llonnn);
WorldClim_V2_Corr_MAM_int = griddata(latBeckk(:),lonBeckk(:),double(WorldClim_V2_Corr_MAM(:)),llattt ,llonnn);
WorldClim_V2_Corr_JJA_int = griddata(latBeckk(:),lonBeckk(:),double(WorldClim_V2_Corr_JJA(:)),llattt ,llonnn);
WorldClim_V2_Corr_SON_int = griddata(latBeckk(:),lonBeckk(:),double(WorldClim_V2_Corr_SON(:)),llattt ,llonnn);

ChelsaCF_DJF_int = griddata(latBeckk(:),lonBeckk(:),double(ChelsaCF_DJF(:)),llattt ,llonnn);
ChelsaCF_MAM_int = griddata(latBeckk(:),lonBeckk(:),double(ChelsaCF_MAM(:)),llattt ,llonnn);
ChelsaCF_JJA_int = griddata(latBeckk(:),lonBeckk(:),double(ChelsaCF_JJA(:)),llattt ,llonnn);
ChelsaCF_SON_int = griddata(latBeckk(:),lonBeckk(:),double(ChelsaCF_SON(:)),llattt ,llonnn);

ChpCF_DJF_int = griddata(latBeckk(:),lonBeckk(:),double(ChpCF_DJF(:)),llattt ,llonnn);
ChpCF_MAM_int = griddata(latBeckk(:),lonBeckk(:),double(ChpCF_MAM(:)),llattt ,llonnn);
ChpCF_JJA_int = griddata(latBeckk(:),lonBeckk(:),double(ChpCF_JJA(:)),llattt ,llonnn);
ChpCF_SON_int = griddata(latBeckk(:),lonBeckk(:),double(ChpCF_SON(:)),llattt ,llonnn);

WcCF_DJF_int = griddata(latBeckk(:),lonBeckk(:),double(WcCF_DJF(:)),llattt ,llonnn);
WcCF_MAM_int = griddata(latBeckk(:),lonBeckk(:),double(WcCF_MAM(:)),llattt ,llonnn);
WcCF_JJA_int = griddata(latBeckk(:),lonBeckk(:),double(WcCF_JJA(:)),llattt ,llonnn);
WcCF_SON_int = griddata(latBeckk(:),lonBeckk(:),double(WcCF_SON(:)),llattt ,llonnn);





EnsembleBeck_DJF = nanmean(cat(3,CHELSA_V12_Corr_DJF_int,CHPclim_V1_Corr_DJF_int,WorldClim_V2_Corr_DJF_int),3);
EnsembleBeck_MAM = nanmean(cat(3,CHELSA_V12_Corr_MAM_int,CHPclim_V1_Corr_MAM_int,WorldClim_V2_Corr_MAM_int),3);
EnsembleBeck_JJA = nanmean(cat(3,CHELSA_V12_Corr_JJA_int,CHPclim_V1_Corr_JJA_int,WorldClim_V2_Corr_JJA_int),3);
EnsembleBeck_SON = nanmean(cat(3,CHELSA_V12_Corr_SON_int,CHPclim_V1_Corr_SON_int,WorldClim_V2_Corr_SON_int),3);

EnsembleBeck_DJF(isnan(EnsembleBeck_DJF)) = 1;
EnsembleBeck_MAM(isnan(EnsembleBeck_MAM)) = 1;
EnsembleBeck_JJA(isnan(EnsembleBeck_JJA)) = 1;
EnsembleBeck_SON(isnan(EnsembleBeck_SON)) = 1;


GPCC_CF_EnsBeck_DJF = EnsembleBeck_DJF./nanmean(GPCCrawDJF1,3);
GPCC_CF_EnsBeck_MAM = EnsembleBeck_MAM./nanmean(GPCCrawMAM1,3);
GPCC_CF_EnsBeck_JJA = EnsembleBeck_JJA./nanmean(GPCCrawJJA1,3);
GPCC_CF_EnsBeck_SON = EnsembleBeck_SON./nanmean(GPCCrawSON1,3);

GPCC_CF_EnsBeck_DJF(GPCC_CF_EnsBeck_DJF<1)=1;
GPCC_CF_EnsBeck_MAM(GPCC_CF_EnsBeck_MAM<1)=1;
GPCC_CF_EnsBeck_JJA(GPCC_CF_EnsBeck_JJA<1)=1;
GPCC_CF_EnsBeck_SON(GPCC_CF_EnsBeck_SON<1)=1;


x = 1:12;
yleg = nanmean(nanmean(legmonth));
ychelsa = nanmean(nanmean(ChelsaCF));
ychp = nanmean(nanmean(ChpCF));
ywc = nanmean(nanmean(WcCF));
ybeck = (ychelsa+ychp+ywc)/3;
%%
GPCCraw = cat(3,precipsInt{:,1});
GPCCraw = nanmean(FuchsCorrFacsSON,3);
latGPCC1 = flipud(double(ncread("full_data_daily_v2018_05_1982.nc", 'lat')));
lonGPCC1 = double(ncread("full_data_daily_v2018_05_1982.nc", 'lon'));
[lat1, lon1] = meshgrid(latGPCC1,lonGPCC1);lat1 = lat1';lon1 = lon1';
AvgGPCCInt = griddata(lat1(:),lon1(:),GPCCraw(:),llatt ,llonn);


RdLwF = 200*(AvgGPCCInt.*AvgLegInt - AvgGPCCInt.*avgFuchs)./(AvgGPCCInt.*AvgLegInt+AvgGPCCInt.*avgFuchs);
figure
imagesc(lonGPCC,latGPCC,AvgGPCCInt);
colormap(jet);
cmap=jet;
cmap(1,:)=[1 1 1];
colormap(cmap);
colorbar;
caxis([0 1500])
colorbar;
set(gca,'Ydir','Normal');
title('Relative Diff LW & F')

%%
[lat2, lon2] = meshgrid(lat,lon);lon2 = lon2';lat2 = lat2';

CHELSA_V12_corr_P_Int = griddata(lat2(:),lon2(:),double(CHELSA_V12_corr_P(:)),llatt ,llonn);
CHPclim_V1_corr_P_Int = griddata(lat2(:),lon2(:),double(CHPclim_V1_corr_P(:)),llatt ,llonn);
WorldClim_V2_corr_P_Int = griddata(lat2(:),lon2(:),double(WorldClim_V2_corr_P(:)),llatt ,llonn);


fcos = linspace(-90,90,180); fcos = flipud(fcos'); f = cosd(fcos); 
ff = repmat(f,1,360);

gpccf = AvgGPCCInt.*avgFuchs;
gpccl = AvgGPCCInt.*AvgLegInt;
gpccfavg = nanmean(nanmean(ff.*gpccf));
gpcclavg = nanmean(nanmean(ff.*gpccl));
gpccrawavg = nanmean(nanmean(ff.*AvgGPCCInt));
CHELSA_V12_corr_P_Int(150:180,:) = NaN;CHPclim_V1_corr_P_Int(150:180,:) = NaN;WorldClim_V2_corr_P_Int(150:180,:) = NaN;
Chavg = nanmean(nanmean(ff.*CHELSA_V12_corr_P_Int));
CHPavg = nanmean(nanmean(ff.*CHPclim_V1_corr_P_Int));
WCavg = nanmean(nanmean(ff.*WorldClim_V2_corr_P_Int));

%%



lat2=lat;lon2=lon; fm=find(mask>0);wt=NaN*ones(size(mask));mcos=cos(lat2*pi/180);wt(fm)=mcos(fm)./sum(mcos(fm));wto=wt;
fob=find(mask>0);

lat2=lat;lon2=lon; fm=find(mask>0);wt=NaN*ones(size(mask));mcos=cos(lat2*pi/180);wt(fm)=mcos(fm)./sum(mcos(fm));wty=wt;
fy=find(mask>0);

lat2=lat;lon2=lon; fm=find(mask>0);wt=NaN*ones(size(mask));mcos=cos(lat2*pi/180);wt(fm)=mcos(fm)./sum(mcos(fm));wtl=wt;
fl=find(mask>0);

lat2=lat;lon2=lon; fm=find(mask>0);wt=NaN*ones(size(mask));mcos=cos(lat2*pi/180);wt(fm)=mcos(fm)./sum(mcos(fm));wtdo=wt;
fdo=find(mask>0);

lat2=lat;lon2=lon; fm=find(mask>0);wt=NaN*ones(size(mask));mcos=cos(lat2*pi/180);wt(fm)=mcos(fm)./sum(mcos(fm));wtdv=wt;
fdv=find(mask>0);

lat2=lat;lon2=lon; fm=find(mask>0);wt=NaN*ones(size(mask));mcos=cos(lat2*pi/180);wt(fm)=mcos(fm)./sum(mcos(fm));wtmz=wt;
fmz=find(mask>0);

lat2=lat;lon2=lon; fm=find(mask>0);wt=NaN*ones(size(mask));mcos=cos(lat2*pi/180);wt(fm)=mcos(fm)./sum(mcos(fm));wtpe=wt;
fpe=find(mask>0);

lat2=lat;lon2=lon; fm=find(mask>0);wt=NaN*ones(size(mask));mcos=cos(lat2*pi/180);wt(fm)=mcos(fm)./sum(mcos(fm));wtvo=wt;
fvo=find(mask>0);
m=mask;m(fl)=1;m(fy)=2;m(fob)=3;m(fmz)=6;m(fpe)=7;m(fvo)=8;

pech = mmm;
pech(pech~=7) = NaN;


figure
imagesc(lon(1,:),lat(:,1),m);
%states = shaperead('usastatehi', 'UseGeoCoords', true);
%geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
colorbar;
set(gca, 'XLim', [0 180], 'YLim',[20 90]);
%title('Lena')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
load coast
figure 1;hold on; plot(long,lat,'k-')

mask(mask==0)=NaN;

%%
A = importdata('legates_correction_factors_05_jan-dec');
B = importdata('legates_correction_factors_10_jan-dec');
C = importdata('legates_correction_factors_25_jan-dec');
legates = cell(12, 1);
for i = 1:12
    legates{i} = reshape(A(:,i),[720,360])';
end

legmonth = cat(3,legates{:,1});
avglegg = nanmean(legmonth,3);

avglegDJF = (legmonth(:,:,12)+legmonth(:,:,1)+legmonth(:,:,2))./3;
avglegMAM = (legmonth(:,:,3)+legmonth(:,:,4)+legmonth(:,:,5))./3;
avglegJJA = (legmonth(:,:,6)+legmonth(:,:,7)+legmonth(:,:,8))./3;
avglegSON = (legmonth(:,:,9)+legmonth(:,:,10)+legmonth(:,:,11))./3;

figure
imagesc(lonGPCC1,latGPCC1,200*(avglegg-AvgLegInt)./(avglegg+AvgLegInt));
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap(33,:)=[1 1 1];
colormap(cmap);
colorbar;
caxis([-50 50])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
title('CF-L , DJF')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
figure
imagesc(lonGPCC1,latGPCC1,avglegMAM);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[1 1 1;cmap];
colormap(cmap);
colorbar;
caxis([1 2])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
title('CF-L , MAM')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
figure
imagesc(lonGPCC1,latGPCC1,avglegJJA);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[1 1 1;cmap];
colormap(cmap);
colorbar;
caxis([1 2])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
title('CF-L , JJA')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
figure
imagesc(lonGPCC1,latGPCC1,avglegSON);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[1 1 1;cmap];
colormap(cmap);
colorbar;
caxis([1 2])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
title('CF-L , SON')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')

load coast
figure(1); hold on; plot(long,lat,'w-')

%% Era5-Land

windu = ncread('download.nc', 'u10');
windv = ncread('download.nc', 'v10');

%%

wcDJF = (CFwc(:,:,12)+CFwc(:,:,1)+CFwc(:,:,2))./3;wcDJF = wcDJF';
wcMAM = (CFwc(:,:,3)+CFwc(:,:,4)+CFwc(:,:,5))./3;wcMAM = wcMAM';
wcJJA = (CFwc(:,:,6)+CFwc(:,:,7)+CFwc(:,:,8))./3;wcJJA = wcJJA';
wcSON = (CFwc(:,:,9)+CFwc(:,:,10)+CFwc(:,:,11))./3;wcSON = wcSON';

load coast

latitude=flipud(linspace(-90,90,3600)');lon=linspace(-180,180,7200);

figure
imagesc(lon,latitude,wcDJF);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[1 1 1;cmap];
colormap(cmap);
colorbar;
caxis([1 2])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
title('CFwc , DJF')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')

hold on; plot(long,lat,'w-')

%%
avglegDJFM = (legmonth(:,:,6)+legmonth(:,:,7)+legmonth(:,:,8)+legmonth(:,:,9))./4; avglegDJFM = avglegDJFM.*lena;


avgFuchsDJFM = nanmean(FuchsCorrFacs(:,:,348:351),3);
[llattt, llonnn] = meshgrid(latGPCC1,lonGPCC1);llattt = llattt';llonnn = llonnn';
avgFuchsDJFMint = griddata(llatt(:),llonn(:),avgFuchsDJFM(:),llattt ,llonnn);avgFuchsDJFMint = avgFuchsDJFMint.*lena;



latlim = [40 90];
lonlim = [0 150];
figure
imagesc(lonGPCC1,latGPCC1,200*(avgFuchsDJFMint-avglegDJFM)./(avgFuchsDJFMint+avglegDJFM));
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap(33,:)=[1 1 1];
colormap(cmap);
colorbar;
caxis([-50 50])
set(gca, 'XLim', lonlim, 'YLim',latlim);
title('Relative Difference , DJFM, 2011')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')



[latera, lonera] = meshgrid(latEra5,lonEra5);latera = latera';lonera = lonera';

winduJJA = (windu(:,:,18)+windu(:,:,19)+windu(:,:,20))./3;
windvJJA = (windv(:,:,18)+windv(:,:,19)+windv(:,:,20))./3;
wind = sqrt(winduJJA.^2+windvJJA.^2);
TJJA = (T(:,:,18)+T(:,:,19)+T(:,:,20))./3;
dewTJJA = (dewT(:,:,18)+dewT(:,:,19)+dewT(:,:,20))./3;

windint = griddata(latera(:),lonera(:),wind(:),llattt ,llonnn);windintpech = windint.*pech;
Tint = griddata(latera(:),lonera(:),TJJA(:),llattt ,llonnn);Tintpech = Tint.*pech;
dewTint = griddata(latera(:),lonera(:),dewTJJA(:),llattt ,llonnn);dewTintpech = dewTint.*pech;



%%

WinduEra5Int = cell(220, 1);
WindvEra5Int = cell(220, 1);
TEra5Int = cell(220, 1);
dewTindEra5Int = cell(220, 1);
for i = 1:220
    tmp = windu(:,:,i);
    tmpp = griddata(latera(:),lonera(:),tmp(:),llattt ,llonnn);
    WinduEra5Int{i} = tmpp;
    tmp1 = windv(:,:,i);
    tmpp1 = griddata(latera(:),lonera(:),tmp1(:),llattt ,llonnn);
    WindvEra5Int{i} = tmpp1;
    tmp2 = T(:,:,i);
    tmpp2 = griddata(latera(:),lonera(:),tmp2(:),llattt ,llonnn);
    TEra5Int{i} = tmpp2;
    tmp3 = dewT(:,:,i);
    tmpp3 = griddata(latera(:),lonera(:),tmp3(:),llattt ,llonnn);
    dewTindEra5Int{i} = tmpp3;
end

WinduEra5Intt = double(cat(3,WinduEra5Int{:,1}));
WindvEra5Intt = double(cat(3,WindvEra5Int{:,1}));
TEra5Intt = double(cat(3,TEra5Int{:,1}));
dewTindEra5Intt = double(cat(3,dewTindEra5Int{:,1}));

wind = sqrt(winduJJA.^2+windvJJA.^2);



a1=12;a2=14;a3=24;a4=26;a5=36;a6=38;a7=48;a8=50;a9=60;a10=62;a11=72;a12=74;a13=84;a14=86;a15=96;a16=98;a17=108;a18=110;a19=120;a20=122;a21=132;a22=134;a23=144;a24=146;a25=156;a26=158;a27=168;a28=170;a29=180;a30=182;a31=192;a32=194;a33=204;a34=206;a35=216;a36=218;
a1=15;a2=17;a3=27;a4=29;a5=39;a6=41;a7=51;a8=53;a9=63;a10=65;a11=75;a12=77;a13=87;a14=89;a15=99;a16=101;a17=111;a18=113;a19=123;a20=125;a21=135;a22=137;a23=147;a24=149;a25=159;a26=161;a27=171;a28=173;a29=183;a30=185;a31=195;a32=197;a33=207;a34=209;a35=219;a36=221;
a1=18;a2=20;a3=30;a4=32;a5=42;a6=44;a7=54;a8=56;a9=66;a10=68;a11=78;a12=80;a13=90;a14=92;a15=102;a16=104;a17=114;a18=116;a19=126;a20=128;a21=138;a22=140;a23=150;a24=152;a25=162;a26=164;a27=174;a28=176;a29=186;a30=188;a31=198;a32=200;a33=210;a34=212;a35=222;a36=224;
a1=21;a2=23;a3=33;a4=35;a5=45;a6=47;a7=57;a8=59;a9=69;a10=71;a11=81;a12=83;a13=93;a14=95;a15=105;a16=107;a17=117;a18=119;a19=129;a20=131;a21=141;a22=143;a23=153;a24=155;a25=165;a26=167;a27=177;a28=179;a29=189;a30=191;a31=201;a32=203;a33=213;a34=215;a35=225;a36=227;

a1=12;a2=14;a3=24;a4=26;a5=36;a6=38;a7=48;a8=50;a9=60;a10=62;a11=72;a12=74;a13=84;a14=86;a15=96;a16=98;a17=108;a18=110;a19=120;a20=122;a21=132;a22=134;a23=144;a24=146;a25=156;a26=158;a27=168;a28=170;a29=180;a30=182;a31=192;a32=194;a33=204;a34=206;a35=216;a36=218;
WindEra5InttDJF = (nanmean(WindEra5Intt(:,:,a1:a2),3)+nanmean(WindEra5Intt(:,:,a3:a4),3)+nanmean(WindEra5Intt(:,:,a5:a6),3)+nanmean(WindEra5Intt(:,:,a7:a8),3)+nanmean(WindEra5Intt(:,:,a9:a10),3)+nanmean(WindEra5Intt(:,:,a11:a12),3)+nanmean(WindEra5Intt(:,:,a13:a14),3)+nanmean(WindEra5Intt(:,:,a15:a16),3)+nanmean(WindEra5Intt(:,:,a17:a18),3)+nanmean(WindEra5Intt(:,:,a19:a20),3)+nanmean(WindEra5Intt(:,:,a21:a22),3)+nanmean(WindEra5Intt(:,:,a23:a24),3)+nanmean(WindEra5Intt(:,:,a25:a26),3)+nanmean(WindEra5Intt(:,:,a27:a28),3)+nanmean(WindEra5Intt(:,:,a29:a30),3)+nanmean(WindEra5Intt(:,:,a31:a32),3)+nanmean(WindEra5Intt(:,:,a33:a34),3)+nanmean(WindEra5Intt(:,:,a35:a36),3))./18;
TEra5InttDJF = (nanmean(TEra5Intt(:,:,a1:a2),3)+nanmean(TEra5Intt(:,:,a3:a4),3)+nanmean(TEra5Intt(:,:,a5:a6),3)+nanmean(TEra5Intt(:,:,a7:a8),3)+nanmean(TEra5Intt(:,:,a9:a10),3)+nanmean(TEra5Intt(:,:,a11:a12),3)+nanmean(TEra5Intt(:,:,a13:a14),3)+nanmean(TEra5Intt(:,:,a15:a16),3)+nanmean(TEra5Intt(:,:,a17:a18),3)+nanmean(TEra5Intt(:,:,a19:a20),3)+nanmean(TEra5Intt(:,:,a21:a22),3)+nanmean(TEra5Intt(:,:,a23:a24),3)+nanmean(TEra5Intt(:,:,a25:a26),3)+nanmean(TEra5Intt(:,:,a27:a28),3)+nanmean(TEra5Intt(:,:,a29:a30),3)+nanmean(TEra5Intt(:,:,a31:a32),3)+nanmean(TEra5Intt(:,:,a33:a34),3)+nanmean(TEra5Intt(:,:,a35:a36),3))./18;
dewTindEra5InttDJF = (nanmean(dewTindEra5Intt(:,:,a1:a2),3)+nanmean(dewTindEra5Intt(:,:,a3:a4),3)+nanmean(dewTindEra5Intt(:,:,a5:a6),3)+nanmean(dewTindEra5Intt(:,:,a7:a8),3)+nanmean(dewTindEra5Intt(:,:,a9:a10),3)+nanmean(dewTindEra5Intt(:,:,a11:a12),3)+nanmean(dewTindEra5Intt(:,:,a13:a14),3)+nanmean(dewTindEra5Intt(:,:,a15:a16),3)+nanmean(dewTindEra5Intt(:,:,a17:a18),3)+nanmean(dewTindEra5Intt(:,:,a19:a20),3)+nanmean(dewTindEra5Intt(:,:,a21:a22),3)+nanmean(dewTindEra5Intt(:,:,a23:a24),3)+nanmean(dewTindEra5Intt(:,:,a25:a26),3)+nanmean(dewTindEra5Intt(:,:,a27:a28),3)+nanmean(dewTindEra5Intt(:,:,a29:a30),3)+nanmean(dewTindEra5Intt(:,:,a31:a32),3)+nanmean(dewTindEra5Intt(:,:,a33:a34),3)+nanmean(dewTindEra5Intt(:,:,a35:a36),3))./18;


a1=15;a2=17;a3=27;a4=29;a5=39;a6=41;a7=51;a8=53;a9=63;a10=65;a11=75;a12=77;a13=87;a14=89;a15=99;a16=101;a17=111;a18=113;a19=123;a20=125;a21=135;a22=137;a23=147;a24=149;a25=159;a26=161;a27=171;a28=173;a29=183;a30=185;a31=195;a32=197;a33=207;a34=209;a35=219;a36=221;
WindEra5InttMAM = (nanmean(WindEra5Intt(:,:,a1:a2),3)+nanmean(WindEra5Intt(:,:,a3:a4),3)+nanmean(WindEra5Intt(:,:,a5:a6),3)+nanmean(WindEra5Intt(:,:,a7:a8),3)+nanmean(WindEra5Intt(:,:,a9:a10),3)+nanmean(WindEra5Intt(:,:,a11:a12),3)+nanmean(WindEra5Intt(:,:,a13:a14),3)+nanmean(WindEra5Intt(:,:,a15:a16),3)+nanmean(WindEra5Intt(:,:,a17:a18),3)+nanmean(WindEra5Intt(:,:,a19:a20),3)+nanmean(WindEra5Intt(:,:,a21:a22),3)+nanmean(WindEra5Intt(:,:,a23:a24),3)+nanmean(WindEra5Intt(:,:,a25:a26),3)+nanmean(WindEra5Intt(:,:,a27:a28),3)+nanmean(WindEra5Intt(:,:,a29:a30),3)+nanmean(WindEra5Intt(:,:,a31:a32),3)+nanmean(WindEra5Intt(:,:,a33:a34),3)+nanmean(WindEra5Intt(:,:,a35:a36),3))./18;
TEra5InttMAM = (nanmean(TEra5Intt(:,:,a1:a2),3)+nanmean(TEra5Intt(:,:,a3:a4),3)+nanmean(TEra5Intt(:,:,a5:a6),3)+nanmean(TEra5Intt(:,:,a7:a8),3)+nanmean(TEra5Intt(:,:,a9:a10),3)+nanmean(TEra5Intt(:,:,a11:a12),3)+nanmean(TEra5Intt(:,:,a13:a14),3)+nanmean(TEra5Intt(:,:,a15:a16),3)+nanmean(TEra5Intt(:,:,a17:a18),3)+nanmean(TEra5Intt(:,:,a19:a20),3)+nanmean(TEra5Intt(:,:,a21:a22),3)+nanmean(TEra5Intt(:,:,a23:a24),3)+nanmean(TEra5Intt(:,:,a25:a26),3)+nanmean(TEra5Intt(:,:,a27:a28),3)+nanmean(TEra5Intt(:,:,a29:a30),3)+nanmean(TEra5Intt(:,:,a31:a32),3)+nanmean(TEra5Intt(:,:,a33:a34),3)+nanmean(TEra5Intt(:,:,a35:a36),3))./18;
dewTindEra5InttMAM = (nanmean(dewTindEra5Intt(:,:,a1:a2),3)+nanmean(dewTindEra5Intt(:,:,a3:a4),3)+nanmean(dewTindEra5Intt(:,:,a5:a6),3)+nanmean(dewTindEra5Intt(:,:,a7:a8),3)+nanmean(dewTindEra5Intt(:,:,a9:a10),3)+nanmean(dewTindEra5Intt(:,:,a11:a12),3)+nanmean(dewTindEra5Intt(:,:,a13:a14),3)+nanmean(dewTindEra5Intt(:,:,a15:a16),3)+nanmean(dewTindEra5Intt(:,:,a17:a18),3)+nanmean(dewTindEra5Intt(:,:,a19:a20),3)+nanmean(dewTindEra5Intt(:,:,a21:a22),3)+nanmean(dewTindEra5Intt(:,:,a23:a24),3)+nanmean(dewTindEra5Intt(:,:,a25:a26),3)+nanmean(dewTindEra5Intt(:,:,a27:a28),3)+nanmean(dewTindEra5Intt(:,:,a29:a30),3)+nanmean(dewTindEra5Intt(:,:,a31:a32),3)+nanmean(dewTindEra5Intt(:,:,a33:a34),3)+nanmean(dewTindEra5Intt(:,:,a35:a36),3))./18;


a1=18;a2=20;a3=30;a4=32;a5=42;a6=44;a7=54;a8=56;a9=66;a10=68;a11=78;a12=80;a13=90;a14=92;a15=102;a16=104;a17=114;a18=116;a19=126;a20=128;a21=138;a22=140;a23=150;a24=152;a25=162;a26=164;a27=174;a28=176;a29=186;a30=188;a31=198;a32=200;a33=210;a34=212;a35=222;a36=224;
WindEra5InttJJA = (nanmean(WindEra5Intt(:,:,a1:a2),3)+nanmean(WindEra5Intt(:,:,a3:a4),3)+nanmean(WindEra5Intt(:,:,a5:a6),3)+nanmean(WindEra5Intt(:,:,a7:a8),3)+nanmean(WindEra5Intt(:,:,a9:a10),3)+nanmean(WindEra5Intt(:,:,a11:a12),3)+nanmean(WindEra5Intt(:,:,a13:a14),3)+nanmean(WindEra5Intt(:,:,a15:a16),3)+nanmean(WindEra5Intt(:,:,a17:a18),3)+nanmean(WindEra5Intt(:,:,a19:a20),3)+nanmean(WindEra5Intt(:,:,a21:a22),3)+nanmean(WindEra5Intt(:,:,a23:a24),3)+nanmean(WindEra5Intt(:,:,a25:a26),3)+nanmean(WindEra5Intt(:,:,a27:a28),3)+nanmean(WindEra5Intt(:,:,a29:a30),3)+nanmean(WindEra5Intt(:,:,a31:a32),3)+nanmean(WindEra5Intt(:,:,a33:a34),3)+nanmean(WindEra5Intt(:,:,a35:a36),3))./18;
TEra5InttJJA = (nanmean(TEra5Intt(:,:,a1:a2),3)+nanmean(TEra5Intt(:,:,a3:a4),3)+nanmean(TEra5Intt(:,:,a5:a6),3)+nanmean(TEra5Intt(:,:,a7:a8),3)+nanmean(TEra5Intt(:,:,a9:a10),3)+nanmean(TEra5Intt(:,:,a11:a12),3)+nanmean(TEra5Intt(:,:,a13:a14),3)+nanmean(TEra5Intt(:,:,a15:a16),3)+nanmean(TEra5Intt(:,:,a17:a18),3)+nanmean(TEra5Intt(:,:,a19:a20),3)+nanmean(TEra5Intt(:,:,a21:a22),3)+nanmean(TEra5Intt(:,:,a23:a24),3)+nanmean(TEra5Intt(:,:,a25:a26),3)+nanmean(TEra5Intt(:,:,a27:a28),3)+nanmean(TEra5Intt(:,:,a29:a30),3)+nanmean(TEra5Intt(:,:,a31:a32),3)+nanmean(TEra5Intt(:,:,a33:a34),3)+nanmean(TEra5Intt(:,:,a35:a36),3))./18;
dewTindEra5InttJJA = (nanmean(dewTindEra5Intt(:,:,a1:a2),3)+nanmean(dewTindEra5Intt(:,:,a3:a4),3)+nanmean(dewTindEra5Intt(:,:,a5:a6),3)+nanmean(dewTindEra5Intt(:,:,a7:a8),3)+nanmean(dewTindEra5Intt(:,:,a9:a10),3)+nanmean(dewTindEra5Intt(:,:,a11:a12),3)+nanmean(dewTindEra5Intt(:,:,a13:a14),3)+nanmean(dewTindEra5Intt(:,:,a15:a16),3)+nanmean(dewTindEra5Intt(:,:,a17:a18),3)+nanmean(dewTindEra5Intt(:,:,a19:a20),3)+nanmean(dewTindEra5Intt(:,:,a21:a22),3)+nanmean(dewTindEra5Intt(:,:,a23:a24),3)+nanmean(dewTindEra5Intt(:,:,a25:a26),3)+nanmean(dewTindEra5Intt(:,:,a27:a28),3)+nanmean(dewTindEra5Intt(:,:,a29:a30),3)+nanmean(dewTindEra5Intt(:,:,a31:a32),3)+nanmean(dewTindEra5Intt(:,:,a33:a34),3)+nanmean(dewTindEra5Intt(:,:,a35:a36),3))./18;

a1=21;a2=23;a3=33;a4=35;a5=45;a6=47;a7=57;a8=59;a9=69;a10=71;a11=81;a12=83;a13=93;a14=95;a15=105;a16=107;a17=117;a18=119;a19=129;a20=131;a21=141;a22=143;a23=153;a24=155;a25=165;a26=167;a27=177;a28=179;a29=189;a30=191;a31=201;a32=203;a33=213;a34=215;a35=225;a36=227;
WindEra5InttSON = (nanmean(WindEra5Intt(:,:,a1:a2),3)+nanmean(WindEra5Intt(:,:,a3:a4),3)+nanmean(WindEra5Intt(:,:,a5:a6),3)+nanmean(WindEra5Intt(:,:,a7:a8),3)+nanmean(WindEra5Intt(:,:,a9:a10),3)+nanmean(WindEra5Intt(:,:,a11:a12),3)+nanmean(WindEra5Intt(:,:,a13:a14),3)+nanmean(WindEra5Intt(:,:,a15:a16),3)+nanmean(WindEra5Intt(:,:,a17:a18),3)+nanmean(WindEra5Intt(:,:,a19:a20),3)+nanmean(WindEra5Intt(:,:,a21:a22),3)+nanmean(WindEra5Intt(:,:,a23:a24),3)+nanmean(WindEra5Intt(:,:,a25:a26),3)+nanmean(WindEra5Intt(:,:,a27:a28),3)+nanmean(WindEra5Intt(:,:,a29:a30),3)+nanmean(WindEra5Intt(:,:,a31:a32),3)+nanmean(WindEra5Intt(:,:,a33:a34),3)+nanmean(WindEra5Intt(:,:,a35:a36),3))./18;
TEra5InttSON = (nanmean(TEra5Intt(:,:,a1:a2),3)+nanmean(TEra5Intt(:,:,a3:a4),3)+nanmean(TEra5Intt(:,:,a5:a6),3)+nanmean(TEra5Intt(:,:,a7:a8),3)+nanmean(TEra5Intt(:,:,a9:a10),3)+nanmean(TEra5Intt(:,:,a11:a12),3)+nanmean(TEra5Intt(:,:,a13:a14),3)+nanmean(TEra5Intt(:,:,a15:a16),3)+nanmean(TEra5Intt(:,:,a17:a18),3)+nanmean(TEra5Intt(:,:,a19:a20),3)+nanmean(TEra5Intt(:,:,a21:a22),3)+nanmean(TEra5Intt(:,:,a23:a24),3)+nanmean(TEra5Intt(:,:,a25:a26),3)+nanmean(TEra5Intt(:,:,a27:a28),3)+nanmean(TEra5Intt(:,:,a29:a30),3)+nanmean(TEra5Intt(:,:,a31:a32),3)+nanmean(TEra5Intt(:,:,a33:a34),3)+nanmean(TEra5Intt(:,:,a35:a36),3))./18;
dewTindEra5InttSON = (nanmean(dewTindEra5Intt(:,:,a1:a2),3)+nanmean(dewTindEra5Intt(:,:,a3:a4),3)+nanmean(dewTindEra5Intt(:,:,a5:a6),3)+nanmean(dewTindEra5Intt(:,:,a7:a8),3)+nanmean(dewTindEra5Intt(:,:,a9:a10),3)+nanmean(dewTindEra5Intt(:,:,a11:a12),3)+nanmean(dewTindEra5Intt(:,:,a13:a14),3)+nanmean(dewTindEra5Intt(:,:,a15:a16),3)+nanmean(dewTindEra5Intt(:,:,a17:a18),3)+nanmean(dewTindEra5Intt(:,:,a19:a20),3)+nanmean(dewTindEra5Intt(:,:,a21:a22),3)+nanmean(dewTindEra5Intt(:,:,a23:a24),3)+nanmean(dewTindEra5Intt(:,:,a25:a26),3)+nanmean(dewTindEra5Intt(:,:,a27:a28),3)+nanmean(dewTindEra5Intt(:,:,a29:a30),3)+nanmean(dewTindEra5Intt(:,:,a31:a32),3)+nanmean(dewTindEra5Intt(:,:,a33:a34),3)+nanmean(dewTindEra5Intt(:,:,a35:a36),3))./18;

WindEra5InttDJFF = [WindEra5InttDJF(:,361:720),WindEra5InttDJF(:,1:360)];
%WindEra5InttDJFF = WindEra5InttDJFF.*mmm;
TEra5InttDJFF = [TEra5InttDJF(:,361:720),TEra5InttDJF(:,1:360)];TEra5InttDJFF = TEra5InttDJFF-273.15;
%TEra5InttDJFF = TEra5InttDJFF.*mmm;
dewTindEra5InttDJFF = [dewTindEra5InttDJF(:,361:720),dewTindEra5InttDJF(:,1:360)];dewTindEra5InttDJFF = dewTindEra5InttDJFF-273.15;
%dewTindEra5InttDJFF = dewTindEra5InttDJFF.*mmm;
RhDJF =100*(exp((17.625.*dewTindEra5InttDJFF)./(243.04+dewTindEra5InttDJFF))./exp((17.625.*TEra5InttDJFF)./(243.04+TEra5InttDJFF)));

WindEra5InttMAMM = [WindEra5InttMAM(:,361:720),WindEra5InttMAM(:,1:360)];
%WindEra5InttMAMM = WindEra5InttMAMM.*mmm;
TEra5InttMAMM = [TEra5InttMAM(:,361:720),TEra5InttMAM(:,1:360)];TEra5InttMAMM = TEra5InttMAMM-273.15;
%TEra5InttMAMM = TEra5InttMAMM.*mmm;
dewTindEra5InttMAMM = [dewTindEra5InttMAM(:,361:720),dewTindEra5InttMAM(:,1:360)];dewTindEra5InttMAMM = dewTindEra5InttMAMM-273.15;
%dewTindEra5InttMAMM = dewTindEra5InttMAMM.*mmm;
RhMAM =100*(exp((17.625.*dewTindEra5InttMAMM)./(243.04+dewTindEra5InttMAMM))./exp((17.625.*TEra5InttMAMM)./(243.04+TEra5InttMAMM)));


WindEra5InttJJAA = [WindEra5InttJJA(:,361:720),WindEra5InttJJA(:,1:360)];
%WindEra5InttJJAA = WindEra5InttJJAA.*mmm;
TEra5InttJJAA = [TEra5InttJJA(:,361:720),TEra5InttJJA(:,1:360)];TEra5InttJJAA = TEra5InttJJAA-273.15;
%TEra5InttJJAA = TEra5InttJJAA.*mmm;
dewTindEra5InttJJAA = [dewTindEra5InttJJA(:,361:720),dewTindEra5InttJJA(:,1:360)];dewTindEra5InttJJAA = dewTindEra5InttJJAA-273.15;
%dewTindEra5InttJJAA = dewTindEra5InttJJAA.*mmm;
RhJJA =100*(exp((17.625.*dewTindEra5InttJJAA)./(243.04+dewTindEra5InttJJAA))./exp((17.625.*TEra5InttJJAA)./(243.04+TEra5InttJJAA)));


WindEra5InttSONN = [WindEra5InttSON(:,361:720),WindEra5InttSON(:,1:360)];
%WindEra5InttSONN = WindEra5InttSONN.*mmm;
TEra5InttSONN = [TEra5InttSON(:,361:720),TEra5InttSON(:,1:360)];TEra5InttSONN = TEra5InttSONN-273.15;
%TEra5InttSONN = TEra5InttSONN.*mmm;
dewTindEra5InttSONN = [dewTindEra5InttSON(:,361:720),dewTindEra5InttSON(:,1:360)];dewTindEra5InttSONN = dewTindEra5InttSONN-273.15;
%dewTindEra5InttSONN = dewTindEra5InttSONN.*mmm;
RhSON =100*(exp((17.625.*dewTindEra5InttSONN)./(243.04+dewTindEra5InttSONN))./exp((17.625.*TEra5InttSONN)./(243.04+TEra5InttSONN)));




pech=m;
pech(pech<7) = NaN;pech(pech>7) = NaN;pech(pech==7) = 1;

legleg = 200*(AvgLegInt-avglegJJA)./(AvgLegInt+avglegJJA); leglegg = legleg.*pech;



figure
latlim = [40 90];
lonlim = [0 150];
imagesc(lonGPCC1,latGPCC1,leglegg);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap(33,:)=[1 1 1];
colormap(cmap);
colorbar;
caxis([-15 15])
set(gca, 'XLim', lonlim, 'YLim',latlim);
title('CF-L , DJF')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')









%%
latlim = [40 90];
lonlim = [0 150];
figure
imagesc(lonGPCC1,latGPCC1,WindEra5InttSON);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
colorbar;
caxis([0 2])
set(gca, 'XLim', lonlim, 'YLim',latlim);
title('wind , DJFM, 2011')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')

latlim = [40 90];
lonlim = [0 150];
figure
imagesc(lonGPCC1,latGPCC1,Tintpech);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
colorbar;
caxis([230 275])
set(gca, 'XLim', lonlim, 'YLim',latlim);
title('T , DJFM, 2011')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')

latlim = [40 90];
lonlim = [0 150];
figure
imagesc(lonGPCC1,latGPCC1,dewTintpech);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
colorbar;
caxis([230 275])
set(gca, 'XLim', lonlim, 'YLim',latlim);
title('dewT , DJFM, 2011')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')
%%

projectdir = 'C:\Users\miladpanahi\Desktop\Master\Paper\GPCC\Monitoring Full dataset\SON';
dinfo = dir( fullfile(projectdir, '*.nc') );
num_files = length(dinfo);
filenames = fullfile( projectdir, {dinfo.name} );
Spr = cell(num_files, 1);
Lpr = cell(num_files, 1);

for K = 1 : num_files
  this_file = filenames{K};
  Spr{K} = transpose(ncread(this_file, 'solid_p'));
  Lpr{K} = transpose(ncread(this_file, 'liquid_p'));
  %precipsInt{1,K}(isnan(mydata{1,K}))=0;
end
SolidPr = cat(3,Spr{:,1});
LiquidPr = cat(3,Lpr{:,1});

% avgSolidPrDJF = (nanmean(SolidPr(:,:,240:242),3)+nanmean(SolidPr(:,:,252:254),3)+nanmean(SolidPr(:,:,264:266),3)+nanmean(SolidPr(:,:,276:278),3)+nanmean(SolidPr(:,:,288:290),3)+nanmean(SolidPr(:,:,300:302),3)+nanmean(SolidPr(:,:,312:314),3)+nanmean(SolidPr(:,:,324:326),3)+nanmean(SolidPr(:,:,336:338),3))./9;
% avgLiquidPrDJF = (nanmean(LiquidPr(:,:,240:242),3)+nanmean(LiquidPr(:,:,252:254),3)+nanmean(LiquidPr(:,:,264:266),3)+nanmean(LiquidPr(:,:,276:278),3)+nanmean(LiquidPr(:,:,288:290),3)+nanmean(LiquidPr(:,:,300:302),3)+nanmean(LiquidPr(:,:,312:314),3)+nanmean(LiquidPr(:,:,324:326),3)+nanmean(LiquidPr(:,:,336:338),3))./9;
% 
% phasepercentageDJF = avgSolidPrDJF./(avgSolidPrDJF+avgLiquidPrDJF);

phasepercentageDJF = nanmean(SolidPr,3)./(nanmean(SolidPr,3)+nanmean(LiquidPr,3));
phasepercentageDJFint = 100*griddata(llatt(:),llonn(:),phasepercentageDJF(:),llattt ,llonnn);

phasepercentageMAM = nanmean(SolidPr,3)./(nanmean(SolidPr,3)+nanmean(LiquidPr,3));
phasepercentageMAMint = 100*griddata(llatt(:),llonn(:),phasepercentageMAM(:),llattt ,llonnn);

phasepercentageJJA = nanmean(SolidPr,3)./(nanmean(SolidPr,3)+nanmean(LiquidPr,3));
phasepercentageJJAint = 100*griddata(llatt(:),llonn(:),phasepercentageJJA(:),llattt ,llonnn);

phasepercentageSON = nanmean(SolidPr,3)./(nanmean(SolidPr,3)+nanmean(LiquidPr,3));
phasepercentageSONint = 100*griddata(llatt(:),llonn(:),phasepercentageSON(:),llattt ,llonnn);

% avgSolidPrMMA = (nanmean(SolidPr(:,:,243:245),3)+nanmean(SolidPr(:,:,255:257),3)+nanmean(SolidPr(:,:,267:269),3)+nanmean(SolidPr(:,:,279:281),3)+nanmean(SolidPr(:,:,291:293),3)+nanmean(SolidPr(:,:,303:305),3)+nanmean(SolidPr(:,:,315:317),3)+nanmean(SolidPr(:,:,327:329),3)+nanmean(SolidPr(:,:,339:341),3))./9;
% avgLiquidPrMMA = (nanmean(LiquidPr(:,:,243:245),3)+nanmean(LiquidPr(:,:,255:257),3)+nanmean(LiquidPr(:,:,267:269),3)+nanmean(LiquidPr(:,:,279:281),3)+nanmean(LiquidPr(:,:,291:293),3)+nanmean(LiquidPr(:,:,303:305),3)+nanmean(LiquidPr(:,:,315:317),3)+nanmean(LiquidPr(:,:,327:329),3)+nanmean(LiquidPr(:,:,339:341),3))./9;
% 
% phasepercentageMMA = avgSolidPrMMA./(avgSolidPrMMA+avgLiquidPrMMA);
% 
% avgSolidPrJJA = (nanmean(SolidPr(:,:,246:248),3)+nanmean(SolidPr(:,:,258:260),3)+nanmean(SolidPr(:,:,270:272),3)+nanmean(SolidPr(:,:,282:284),3)+nanmean(SolidPr(:,:,294:296),3)+nanmean(SolidPr(:,:,306:308),3)+nanmean(SolidPr(:,:,318:320),3)+nanmean(SolidPr(:,:,330:332),3)+nanmean(SolidPr(:,:,342:344),3))./9;
% avgLiquidPrJJA = (nanmean(LiquidPr(:,:,246:248),3)+nanmean(LiquidPr(:,:,258:260),3)+nanmean(LiquidPr(:,:,270:272),3)+nanmean(LiquidPr(:,:,282:284),3)+nanmean(LiquidPr(:,:,294:296),3)+nanmean(LiquidPr(:,:,306:308),3)+nanmean(LiquidPr(:,:,318:320),3)+nanmean(LiquidPr(:,:,330:332),3)+nanmean(LiquidPr(:,:,342:344),3))./9;
% 
% phasepercentageJJA = avgSolidPrJJA./(avgSolidPrJJA+avgLiquidPrJJA);

image(eraZintt(:),phasepercentageDJFint(:),RelDdjf(:))
dotsize=25
scatter3(eraZintt(:), phasepercentageDJFint(:), RelDdjf(:), dotsize, RelDdjf(:), 'filled')
[h,ch]=plot3c(eraZintt(:),phasepercentageDJFint(:),RelDdjf(:),[-100, -75, -50, -25, 0, 25, 50, 75, 100]);


figure
latlim = [40 90];
lonlim = [0 150];
imagesc(lonGPCC1,latGPCC1,phasepercentageJJAint.*100);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[.65 .65 .65;cmap];
colormap(cmap);
colorbar;
caxis([0 10])
set(gca, 'XLim', lonlim, 'YLim',latlim);
%title('Average Wind Speed - DJF')
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
hold on; plot(long,lat,'w-')

phasepercentageJJAintpech = phasepercentageJJAint.*pech;

%%

x=rand(100,1);
y=sin(x);
z=cos(x);
[h,ch]=plot3c(x,y,z);