info = ncinfo('ST4.2003103112.nc');
info.Variables.Name(:)
pr = ncread('ST4.2003103112.nc','A_PCP_240_SFC_acc24h');
lon = ncread('ST4.2003103112.nc','gridlon_240');
lat = ncread('ST4.2003103112.nc','gridlat_240');

%% IDL

projectdir = 'C:\Users\miladpanahi\Desktop\Master\Paper\undercatch\Canada SPICE';
dinfo = dir( fullfile(projectdir, '*.dat') );
num_files = length(dinfo);
filenames = fullfile( projectdir, {dinfo.name} );
Pr = cell(num_files, 1);
for K = 1 : num_files
  this_file = filenames{K};
  tmp = restore_idl(this_file);
  Pr{K} = tmp.DATA.Time;
end

stagelat = double(stage.DATA.LAT);
stagelon = double(stage.DATA.LON);

StagePr = cat(3,Pr{:,1});

PrInt1 = cell(152, 1);
PrInt2 = cell(152, 1);
PrInt3 = cell(152, 1);
PrInt4 = cell(152, 1);
PrInt5 = cell(152, 1);
PrInt6 = cell(152, 1);


for i = 1:152
    tmp = double(StagePr(:,:,i));
    tmpp=griddata(stagelat(:),stagelon(:),tmp(:),latM ,lonM);
    tmp1 = tmpp.*PrCrctF1;
    tmp1 = tmp1.*mask1MM;
    tmp2 = tmpp.*PrCrctF2;
    tmp2 = tmp2.*mask2MM;
      tmp3 = tmpp.*PrCrctF3;
      tmp3 = tmp3.*mask3MM;
%     tmp4 = tmpp.*PrCrctF4;
%     tmp4 = tmp4.*mask4MM;
%     tmp5 = tmpp.*PrCrctF5;
%     tmp5 = tmp5.*mask5MM;
%     tmp6 = tmpp.*PrCrctF6;
%     tmp6 = tmp6.*mask6MM;
    PrInt1{i} = tmp1;
    PrInt2{i} = tmp2;
      PrInt3{i} = tmp3;
%     PrInt4{i} = tmp4;
%     PrInt5{i} = tmp5;
%     PrInt6{i} = tmp6;
end    

StagePrMasked1 = cat(3,PrInt1{:,1});
StagePrMasked2 = cat(3,PrInt2{:,1});
StagePrMasked3 = cat(3,PrInt3{:,1});
StagePrMasked4 = cat(3,PrInt4{:,1});
StagePrMasked5 = cat(3,PrInt5{:,1});
StagePrMasked6 = cat(3,PrInt6{:,1});


Stage1Ts = nanmean(nanmean(StagePrMasked1,1),2);
Stage1AccTs = cumsum(Stage1Ts,'omitnan');
Stage1AccTs = Stage1AccTs - Stage1AccTs(1);

Stage2Ts = nanmean(nanmean(StagePrMasked2,1),2);
Stage2AccTs = cumsum(Stage2Ts,'omitnan');
Stage2AccTs = Stage2AccTs - Stage2AccTs(1);

Stage3Ts = nanmean(nanmean(StagePrMasked3,1),2);
Stage3AccTs = cumsum(Stage3Ts,'omitnan');
Stage3AccTs = Stage3AccTs - Stage3AccTs(1);

%%
z = sum(StagePr(:,:,71:133),3);
zz = griddata(stagelat(:),stagelon(:),z(:),latM ,lonM);
figure
h = pcolor(lonM(1,:),latM(:,1),zz);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
cmap=jet;
cmap=[1 1 1;cmap];
colormap(cmap);
colorbar;
set(h, 'EdgeColor', 'none');
caxis([0 300])

% fileID = fopen('hrapmask_hires_1121x881.txt','rt');
% C = textscan(fileID, '%f%f%f', 'MultipleDelimsAsOne',true, 'Delimiter','[;', 'HeaderLines',2);
% fclose(fileID);
% 
% formatSpec = '%i';
% A = fscanf(fileID,formatSpec);
