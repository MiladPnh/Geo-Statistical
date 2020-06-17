% ncdisp('Noah.dailymean.20151110.nc');
ncvars =  {'QSUBCAN', 'QSUBGRD'};
projectdir = 'C:\Users\miladpanahi\Desktop\Master\Paper\Sublimation Noah\2014-15';
dinfo = dir( fullfile(projectdir, '*.nc') );
num_files = length(dinfo);
filenames = fullfile( projectdir, {dinfo.name} );
subCAN = cell(num_files, 1);
subGRD = cell(num_files, 1);
for K = 1 : num_files
  this_file = filenames{K};
  subCAN{K}=ncread(this_file,ncvars{1}); subGRD{K}=ncread(this_file,ncvars{2});
  subCAN{K}(subCAN{K}<-100)=0; subGRD{K}(subGRD{K}<-100)=0;
  subCAN{K} = double(flipud(permute(subCAN{K},[2 1 3]))); subGRD{K} = double(flipud(permute(subGRD{K},[2 1 3])));
  subCAN{K} = subCAN{K}*24*60*60; subGRD{K} = subGRD{K}*24*60*60;
  subCAN{K} = subCAN{K}(33:224,:);subGRD{K} = subGRD{K}(33:224,:);
end
NoahSublimationCAN = double(cat(3,subCAN{:,1}));
NoahSublimationGRD = double(cat(3,subGRD{:,1}));
NoahSublimationTot = NoahSublimationGRD+NoahSublimationCAN;

lon = double(ncread('Noah.dailymean.20141101.nc','lon'));
lat = double(flipud(ncread('Noah.dailymean.20141101.nc','lat')));lat=lat(33:224);

[llat, llon] = meshgrid(lat,lon);llat = llat';llon = llon';

PrInt1 = cell(num_files, 1);
PrInt2 = cell(num_files, 1);
PrInt3 = cell(num_files, 1);
PrInt4 = cell(num_files, 1);
PrInt5 = cell(num_files, 1);
PrInt6 = cell(num_files, 1);

for i = 1:num_files
    tmp = NoahSublimationTot(:,:,i);
    %[X, Y]=meshgrid(v2,v1);
    %X=X';Y=Y';
    tmpp=griddata(llat(:),llon(:),tmp(:),latM ,lonM);
    tmpp1 = tmpp.*mask1MM;
    PrInt1{i} = tmpp1;
    tmpp2 = tmpp.*mask2MM;
    PrInt2{i} = tmpp2;
      tmpp3 = tmpp.*mask3MM;
      PrInt3{i} = tmpp3;
    tmpp4 = tmpp.*mask4MM;
    PrInt4{i} = tmpp4;
    tmpp5 = tmpp.*mask5MM;
    PrInt5{i} = tmpp5;
%     tmpp6 = tmpp.*mask6MM;
%     PrInt6{i} = tmpp6;
end

NoahSub1 = cat(3,PrInt1{:,1});
NoahSub2 = cat(3,PrInt2{:,1});
NoahSub3 = cat(3,PrInt3{:,1});
NoahSub4 = cat(3,PrInt4{:,1});
NoahSub5 = cat(3,PrInt5{:,1});
NoahSub6 = cat(3,PrInt6{:,1});

E1 = nanmean(nanmean(NoahSub1));sub1a = sum(E1(83:108));sub1b = sum(E1(108:138));
E2 = nanmean(nanmean(NoahSub2));sub2a = sum(E2(83:108));sub2b = sum(E2(108:138));
E3 = nanmean(nanmean(NoahSub3));sub3a = sum(E3(83:108));sub3b = sum(E3(108:138));
E4 = nanmean(nanmean(NoahSub4));sub4a = sum(E4(83:108));sub4b = sum(E4(108:138));
E5 = nanmean(nanmean(NoahSub5));sub5a = sum(E5(83:108));sub5b = sum(E5(108:138));
E6 = nanmean(nanmean(NoahSub6));sub6a = sum(E6(46:77));sub6b = sum(E6(78:138));sub6c = sum(E6(16:46));

figure; imagesc(lonEra,latEra,MeanEraPr); caxis([0 10]);colorbar;colormap(jet);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
set(gca,'YDIR','Normal');
