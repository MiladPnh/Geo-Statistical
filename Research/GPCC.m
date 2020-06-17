ncvars =  {'precip'};
projectdir = 'C:\Users\miladpanahi\Desktop\Master\Paper\GPCC\monitoring\SON';
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
latGPCC = double(ncread("monitoring_v6_10_2002_11.nc", 'lat'));
lonGPCC = double(ncread("monitoring_v6_10_2002_11.nc", 'lon'));

imagesc(X,latGPCC,nanmean(Leg,3));
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[1 1 1;cmap];
colormap(cmap);
colorbar;
caxis([0 4])
%set(gca, 'XLim', lonlim, 'YLim',latlim);
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')

FuchsCorrFacs = FuchsCorrFacs(41:67,55:114,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ncdisp('full_data_daily_v2018_2016.nc');
% ncvars =  {'latitude', 'longitude', 'precip'};
% projectdir = 'C:\Users\miladpanahi\Desktop\Master\Paper\GPCP\2015-16';
% dinfo = dir( fullfile(projectdir, '*.nc') );
% num_files = length(dinfo);
% filenames = fullfile( projectdir, {dinfo.name} );
% precipsInt = cell(num_files, 1);
%if 151 ==> 305:365 & 1:90
%if 152 ==> 306:366 & 1:91
% for K = 1 : num_files
%   tmp = ncread(filenames{K}, ncvars{3});
%   tmp1 = tmp(:,:,305:365);
%   tmp2 = tmp(:,:,1:90);
%   
%   %precipsInt{1,K}(isnan(mydata{1,K}))=0;
% end
latGPCC = flipud(double(ncread("full_data_daily_v2018_2010.nc", 'lat')));latGPCC = latGPCC(40:67);
lonGPCC = double(ncread("full_data_daily_v2018_2010.nc", 'lon'));lonGPCC = lonGPCC(55:114);

tmp = flipud(permute(ncread('full_data_daily_v2018_2008.nc', 'precip'),[2 1 3]));tmp = tmp(41:67,55:114,305:365);
tmpp = flipud(permute(ncread('full_data_daily_v2018_2009.nc', 'precip'),[2 1 3]));tmpp = tmpp(41:67,55:114,1:91);
GPCCraw = cat(3,tmp,tmpp);
FuchsCorrFacs(isnan(FuchsCorrFacs))=1;
for i = 1:30
    tmp(:,:,i) = tmp(:,:,i).*FuchsCorrFacs(:,:,31);
end    
for i = 31:61
    tmp(:,:,i) = tmp(:,:,i).*FuchsCorrFacs(:,:,32);
end 
for i = 1:31
    tmpp(:,:,i) = tmpp(:,:,i).*FuchsCorrFacs(:,:,33);
end
for i = 32:59
    tmpp(:,:,i) = tmpp(:,:,i).*FuchsCorrFacs(:,:,34);
end
for i = 60:91
    tmpp(:,:,i) = tmpp(:,:,i).*FuchsCorrFacs(:,:,35);
end
    
GPCCPrFuschCrc = cat(3,tmp,tmpp);

[llat, llon] = meshgrid(latGPCC,lonGPCC);llat = llat';llon = llon';
PrInt1 = cell(152, 1);
PrInt2 = cell(152, 1);
PrInt3 = cell(152, 1);
PrInt4 = cell(152, 1);
PrInt5 = cell(152, 1);
PrInt6 = cell(152, 1);

for i = 1:151
    tmp = GPCCPrFuschCrc(:,:,i);
    %[X, Y]=meshgrid(v2,v1);
    %X=X';Y=Y';
    tmpp=griddata(llat(:),llon(:),tmp(:),latM ,lonM);
    tmpp1 = tmpp.*PrCrctF1;
    tmpp1 = tmpp1.*mask1MM;
    PrInt1{i} = tmpp1;
    tmpp2 = tmpp.*PrCrctF2;
    tmpp2 = tmpp2.*mask2MM;
    PrInt2{i} = tmpp2;
      tmpp3 = tmpp.*PrCrctF3;
      tmpp3 = tmpp3.*mask3MM;
      PrInt3{i} = tmpp3;
%     tmpp4 = tmpp.*PrCrctF4;
%     tmpp4 = tmpp4.*mask4MM;
%     PrInt4{i} = tmpp4;
%     tmpp5 = tmpp.*PrCrctF5;
%     tmpp5 = tmpp5.*mask5MM;
%     PrInt5{i} = tmpp5;
%     tmpp6 = tmpp.*PrCrctF6;
%     tmpp6 = tmpp6.*mask6MM;
%     PrInt6{i} = tmpp6;
end    
GPCCPrMasked1F = cat(3,PrInt1{:,1});
GPCCPrMasked2F = cat(3,PrInt2{:,1});
GPCCPrMasked3F = cat(3,PrInt3{:,1});
GPCCPrMasked4F = cat(3,PrInt4{:,1});
GPCCPrMasked5F = cat(3,PrInt5{:,1});
GPCCPrMasked6F = cat(3,PrInt6{:,1});

GPCC1Ts = nanmean(nanmean(GPCCPrMasked1F,1),2);
GPCC1AccTsFus = cumsum(GPCC1Ts);
GPCC1AccTsFus = GPCC1AccTsFus - GPCC1AccTsFus(1);
GPCC2Ts = nanmean(nanmean(GPCCPrMasked2F,1),2);
GPCC2AccTsFus = cumsum(GPCC2Ts);
GPCC2AccTsFus = GPCC2AccTsFus - GPCC2AccTsFus(1);
GPCC3Ts = nanmean(nanmean(GPCCPrMasked3F,1),2);
GPCC3AccTsFus = cumsum(GPCC3Ts);
GPCC3AccTsFus = GPCC3AccTsFus - GPCC3AccTsFus(1);
GPCC4Ts = nanmean(nanmean(GPCCPrMasked4F,1),2);
GPCC4AccTsFus = cumsum(GPCC4Ts);
GPCC4AccTsFus = GPCC4AccTsFus - GPCC4AccTsFus(1);
GPCC5Ts = nanmean(nanmean(GPCCPrMasked5,1),2);
GPCC5AccTsFus = cumsum(GPCC5Ts);
GPCC5AccTsFus = GPCC5AccTsFus - GPCC5AccTsFus(1);
GPCC6Ts = nanmean(nanmean(GPCCPrMasked6,1),2);
GPCC6AccTsFus = cumsum(GPCC6Ts);
GPCC6AccTsFus = GPCC6AccTsFus - GPCC6AccTsFus(1);





%%%%%%%%%%%%%%%%%%Legates%%%%%%%%%%%%%%%%%

X = [-178.75 : 2.5 : 178.75];X = X(23:46);
Y = [-88.75 : 2.5 : 88.75];Y = Y(47:56);
[llat, llon] = meshgrid(Y,X);llat = llat';llon = llon';
Leg = flipud(mcof);Leg = Leg(47:56,23:46,:);

fac = cell(12, 1);
for i=1:12
    tmp = Leg(:,:,i);
    fac{i} = griddata(llat(:),llon(:),tmp(:),latM ,lonM);
end
LegCorF = cat(3,fac{:,1});
LegCorF(isnan(LegCorF))=1;
LegCorF(LegCorF==0)=1;


latGPCC = flipud(double(ncread("full_data_daily_v2018_2003.nc", 'lat')));latGPCC = latGPCC(41:67);
lonGPCC = double(ncread("full_data_daily_v2018_2002.nc", 'lon'));lonGPCC = lonGPCC(55:114);

tmp = flipud(permute(ncread('full_data_daily_v2018_2010.nc', 'precip'),[2 1 3]));tmp = tmp(41:67,55:114,305:365);
tmpp = flipud(permute(ncread('full_data_daily_v2018_2011.nc', 'precip'),[2 1 3]));tmpp = tmpp(41:67,55:114,1:91);
GPCCtmp = cat(3,tmp,tmpp);
% for i = 1:30
%     tmp(:,:,i) = tmp(:,:,i).*LegCorF(:,:,11);
% end    
% for i = 31:61
%     tmp(:,:,i) = tmp(:,:,i).*LegCorF(:,:,12);
% end 
% for i = 1:31
%     tmpp(:,:,i) = tmpp(:,:,i).*LegCorF(:,:,1);
% end
% for i = 32:60
%     tmpp(:,:,i) = tmpp(:,:,i).*LegCorF(:,:,2);
% end
% for i = 61:91
%     tmpp(:,:,i) = tmpp(:,:,i).*LegCorF(:,:,3);
% end
% GPCCPrLegCrc = cat(3,tmp,tmpp);

[llat, llon] = meshgrid(latGPCC,lonGPCC);llat = llat';llon = llon';
PrInt1 = cell(152, 1);
PrInt2 = cell(152, 1);
PrInt3 = cell(152, 1);
PrInt4 = cell(152, 1);
PrInt5 = cell(152, 1);
PrInt6 = cell(152, 1);

for i = 1:30
    tmp = GPCCtmp(:,:,i);
    %[X, Y]=meshgrid(v2,v1);
    %X=X';Y=Y';
    tmpp=griddata(llat(:),llon(:),tmp(:),latM ,lonM);
    tmpp = tmpp.*LegCorF(:,:,11);
    tmpp1 = tmpp.*PrCrctF1;
    tmpp1 = tmpp1.*mask1MM;
    PrInt1{i} = tmpp1;
    tmpp2 = tmpp.*PrCrctF2;
    tmpp2 = tmpp2.*mask2MM;
    PrInt2{i} = tmpp2;
     tmpp3 = tmpp.*PrCrctF3;
     tmpp3 = tmpp3.*mask3MM;
     PrInt3{i} = tmpp3;
    tmpp4 = tmpp.*PrCrctF4;
    tmpp4 = tmpp4.*mask4MM;
    PrInt4{i} = tmpp4;
    tmpp5 = tmpp.*PrCrctF5;
    tmpp5 = tmpp5.*mask5MM;
    PrInt5{i} = tmpp5;
    tmpp6 = tmpp.*PrCrctF6;
    tmpp6 = tmpp6.*mask6MM;
    PrInt6{i} = tmpp6;
end
for i = 31:61
    tmp = GPCCtmp(:,:,i);
    %[X, Y]=meshgrid(v2,v1);
    %X=X';Y=Y';
    tmpp=griddata(llat(:),llon(:),tmp(:),latM ,lonM);
    tmpp = tmpp.*LegCorF(:,:,12);
    tmpp1 = tmpp.*PrCrctF1;
    tmpp1 = tmpp1.*mask1MM;
    PrInt1{i} = tmpp1;
    tmpp2 = tmpp.*PrCrctF2;
    tmpp2 = tmpp2.*mask2MM;
    PrInt2{i} = tmpp2;
    tmpp3 = tmpp.*PrCrctF3;
     tmpp3 = tmpp3.*mask3MM;
     PrInt3{i} = tmpp3;
    tmpp4 = tmpp.*PrCrctF4;
    tmpp4 = tmpp4.*mask4MM;
    PrInt4{i} = tmpp4;
    tmpp5 = tmpp.*PrCrctF5;
    tmpp5 = tmpp5.*mask5MM;
    PrInt5{i} = tmpp5;
    tmpp6 = tmpp.*PrCrctF6;
    tmpp6 = tmpp6.*mask6MM;
    PrInt6{i} = tmpp6;
end  
for i = 62:92
    tmp = GPCCtmp(:,:,i);
    %[X, Y]=meshgrid(v2,v1);
    %X=X';Y=Y';
    tmpp=griddata(llat(:),llon(:),tmp(:),latM ,lonM);
    tmpp = tmpp.*LegCorF(:,:,1);
    tmpp1 = tmpp.*PrCrctF1;
    tmpp1 = tmpp1.*mask1MM;
    PrInt1{i} = tmpp1;
    tmpp2 = tmpp.*PrCrctF2;
    tmpp2 = tmpp2.*mask2MM;
    PrInt2{i} = tmpp2;
     tmpp3 = tmpp.*PrCrctF3;
     tmpp3 = tmpp3.*mask3MM;
     PrInt3{i} = tmpp3;
    tmpp4 = tmpp.*PrCrctF4;
    tmpp4 = tmpp4.*mask4MM;
    PrInt4{i} = tmpp4;
    tmpp5 = tmpp.*PrCrctF5;
    tmpp5 = tmpp5.*mask5MM;
    PrInt5{i} = tmpp5;
    tmpp6 = tmpp.*PrCrctF6;
    tmpp6 = tmpp6.*mask6MM;
    PrInt6{i} = tmpp6;
end  
for i = 93:120
    tmp = GPCCtmp(:,:,i);
    %[X, Y]=meshgrid(v2,v1);
    %X=X';Y=Y';
    tmpp=griddata(llat(:),llon(:),tmp(:),latM ,lonM);
    tmpp = tmpp.*LegCorF(:,:,2);
    tmpp1 = tmpp.*PrCrctF1;
    tmpp1 = tmpp1.*mask1MM;
    PrInt1{i} = tmpp1;
    tmpp2 = tmpp.*PrCrctF2;
    tmpp2 = tmpp2.*mask2MM;
    PrInt2{i} = tmpp2;
     tmpp3 = tmpp.*PrCrctF3;
     tmpp3 = tmpp3.*mask3MM;
     PrInt3{i} = tmpp3;
    tmpp4 = tmpp.*PrCrctF4;
    tmpp4 = tmpp4.*mask4MM;
    PrInt4{i} = tmpp4;
    tmpp5 = tmpp.*PrCrctF5;
    tmpp5 = tmpp5.*mask5MM;
    PrInt5{i} = tmpp5;
    tmpp6 = tmpp.*PrCrctF6;
    tmpp6 = tmpp6.*mask6MM;
    PrInt6{i} = tmpp6;
end  
for i = 121:152
    tmp = GPCCtmp(:,:,i);
    %[X, Y]=meshgrid(v2,v1);
    %X=X';Y=Y';
    tmpp=griddata(llat(:),llon(:),tmp(:),latM ,lonM);
    tmpp = tmpp.*LegCorF(:,:,3);
    tmpp1 = tmpp.*PrCrctF1;
    tmpp1 = tmpp1.*mask1MM;
    PrInt1{i} = tmpp1;
    tmpp2 = tmpp.*PrCrctF2;
    tmpp2 = tmpp2.*mask2MM;
    PrInt2{i} = tmpp2;
     tmpp3 = tmpp.*PrCrctF3;
     tmpp3 = tmpp3.*mask3MM;
     PrInt3{i} = tmpp3;
    tmpp4 = tmpp.*PrCrctF4;
    tmpp4 = tmpp4.*mask4MM;
    PrInt4{i} = tmpp4;
    tmpp5 = tmpp.*PrCrctF5;
    tmpp5 = tmpp5.*mask5MM;
    PrInt5{i} = tmpp5;
    tmpp6 = tmpp.*PrCrctF6;
    tmpp6 = tmpp6.*mask6MM;
    PrInt6{i} = tmpp6;
end  
GPCCPrMasked1L = cat(3,PrInt1{:,1});
GPCCPrMasked2L = cat(3,PrInt2{:,1});
GPCCPrMasked3L = cat(3,PrInt3{:,1});
GPCCPrMasked4L = cat(3,PrInt4{:,1});
GPCCPrMasked5L = cat(3,PrInt5{:,1});
GPCCPrMasked6L = cat(3,PrInt6{:,1});

GPCC1Ts = nanmean(nanmean(GPCCPrMasked1L,1),2);
GPCC1AccTsLeg = cumsum(GPCC1Ts);
GPCC1AccTsLeg = GPCC1AccTsLeg - GPCC1AccTsLeg(1);
GPCC2Ts = nanmean(nanmean(GPCCPrMasked2L,1),2);
GPCC2AccTsLeg = cumsum(GPCC2Ts);
GPCC2AccTsLeg = GPCC2AccTsLeg - GPCC2AccTsLeg(1);
GPCC3Ts = nanmean(nanmean(GPCCPrMasked3L,1),2);
GPCC3AccTsLeg = cumsum(GPCC3Ts);
GPCC3AccTsLeg = GPCC3AccTsLeg - GPCC3AccTsLeg(1);
GPCC4Ts = nanmean(nanmean(GPCCPrMasked4L,1),2);
GPCC4AccTsLeg = cumsum(GPCC4Ts);
GPCC4AccTsLeg = GPCC4AccTsLeg - GPCC4AccTsLeg(1);
GPCC5Ts = nanmean(nanmean(GPCCPrMasked5,1),2);
GPCC5AccTsLeg = cumsum(GPCC5Ts);
GPCC5AccTsLeg = GPCC5AccTsLeg - GPCC5AccTsLeg(1);
GPCC6Ts = nanmean(nanmean(GPCCPrMasked6,1),2);
GPCC6AccTsLeg = cumsum(GPCC6Ts);
GPCC6AccTsLeg = GPCC6AccTsLeg - GPCC6AccTsLeg(1);



%%%%%%%%%%%%%%%%%%%%%GPCC Raw%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[llat, llon] = meshgrid(latGPCC,lonGPCC);llat = llat';llon = llon';
PrInt1 = cell(152, 1);
PrInt2 = cell(152, 1);
PrInt3 = cell(152, 1);
PrInt4 = cell(152, 1);
PrInt5 = cell(152, 1);
PrInt6 = cell(152, 1);

for i = 1:151
    tmp = GPCCraw(:,:,i);
    %[X, Y]=meshgrid(v2,v1);
    %X=X';Y=Y';
    tmpp=griddata(llat(:),llon(:),tmp(:),latM ,lonM);
    tmpp1 = tmpp.*PrCrctF1;
    tmpp1 = tmpp1.*mask1MM;
    PrInt1{i} = tmpp1;
    tmpp2 = tmpp.*PrCrctF2;
    tmpp2 = tmpp2.*mask2MM;
    PrInt2{i} = tmpp2;
      tmpp3 = tmpp.*PrCrctF3;
      tmpp3 = tmpp3.*mask3MM;
      PrInt3{i} = tmpp3;
    tmpp4 = tmpp.*PrCrctF4;
    tmpp4 = tmpp4.*mask4MM;
    PrInt4{i} = tmpp4;
    tmpp5 = tmpp.*PrCrctF5;
    tmpp5 = tmpp5.*mask5MM;
    PrInt5{i} = tmpp5;
    tmpp6 = tmpp.*PrCrctF6;
    tmpp6 = tmpp6.*mask6MM;
    PrInt6{i} = tmpp6;
end    
GPCCPrMasked1R = cat(3,PrInt1{:,1});
GPCCPrMasked2R = cat(3,PrInt2{:,1});
GPCCPrMasked3R = cat(3,PrInt3{:,1});
GPCCPrMasked4R = cat(3,PrInt4{:,1});
GPCCPrMasked5R = cat(3,PrInt5{:,1});
GPCCPrMasked6R = cat(3,PrInt6{:,1});

GPCC1Ts = nanmean(nanmean(GPCCPrMasked1R,1),2);
GPCC1AccTs = cumsum(GPCC1Ts);
GPCC1AccTs = GPCC1AccTs - GPCC1AccTs(1);
GPCC2Ts = nanmean(nanmean(GPCCPrMasked2R,1),2);
GPCC2AccTs = cumsum(GPCC2Ts);
GPCC2AccTs = GPCC2AccTs - GPCC2AccTs(1);
GPCC3Ts = nanmean(nanmean(GPCCPrMasked3R,1),2);
GPCC3AccTs = cumsum(GPCC3Ts);
GPCC3AccTs = GPCC3AccTs - GPCC3AccTs(1);
GPCC4Ts = nanmean(nanmean(GPCCPrMasked4R,1),2);
GPCC4AccTs = cumsum(GPCC4Ts);
GPCC4AccTs = GPCC4AccTs - GPCC4AccTs(1);
GPCC5Ts = nanmean(nanmean(GPCCPrMasked5,1),2);
GPCC5AccTs = cumsum(GPCC5Ts);
GPCC5AccTs = GPCC5AccTs - GPCC5AccTs(1);
GPCC6Ts = nanmean(nanmean(GPCCPrMasked6,1),2);
GPCC6AccTs = cumsum(GPCC6Ts);
GPCC6AccTs = GPCC6AccTs - GPCC6AccTs(1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imagesc(X,Y,Leg(:,:,1));
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
set(gca, {'YDir'}, {'reverse'}); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

latGPCC = flipud(double(ncread("full_data_daily_v2018_2002.nc", 'lat')));latGPCC = latGPCC(41:67);
lonGPCC = double(ncread("full_data_daily_v2018_2002.nc", 'lon'));lonGPCC = lonGPCC(55:114);

[llat, llon] = meshgrid(latGPCC,lonGPCC);llat = llat';llon = llon';

facc = cell(80, 1);
for i=1:80
    tmp = FuchsCorrFacs(:,:,i);
    facc{i} = griddata(llat(:),llon(:),tmp(:),latM ,lonM);
end
FuchsCorF = cat(3,facc{:,1});
FuchsCorF(isnan(FuchsCorF))=1;
FuchsCorF(FuchsCorF==0)=1;

LFcorDiff = cell(80, 1);
for i = 1:80
    if rem(i,5)==1
        LFcorDiff{i} = FuchsCorF(:,:,i)-LegCorF(:,:,11);
    elseif rem(i,5)==2
        LFcorDiff{i} = FuchsCorF(:,:,i)-LegCorF(:,:,12);    
    elseif rem(i,5)==3
        LFcorDiff{i} = FuchsCorF(:,:,i)-LegCorF(:,:,1);
    elseif rem(i,5)==4
        LFcorDiff{i} = FuchsCorF(:,:,i)-LegCorF(:,:,2);
    elseif rem(i,5)==0
       LFcorDiff{i} = FuchsCorF(:,:,i)-LegCorF(:,:,3);
    end
end
LFcorDiffs = cat(3,LFcorDiff{:,1});
LFcorDiffs = LFcorDiffs(1:19,1:30,:);
mmm = mean(mean(LFcorDiffs));

Nov02 = nanmean(nanmean(nanmean(mydataa(1:239,1:361))));
Dec02 = nanmean(nanmean(nanmean(mydataa(1:239,1:361))));
Jan03 = nanmean(nanmean(nanmean(mydataa(1:239,1:361))));
Feb03 = nanmean(nanmean(nanmean(mydataa(1:239,1:361))));
March03 = nanmean(nanmean(nanmean(mydataa(1:239,1:361))));
