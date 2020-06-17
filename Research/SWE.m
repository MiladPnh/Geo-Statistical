projectdir = 'C:\Users\miladpanahi\Desktop\Master\Paper\SWE\UA_Snow_Data\2010-11S';
dinfo = dir( fullfile(projectdir, '*.tif') );
num_files = length(dinfo);
filenames = fullfile( projectdir, {dinfo.name} );
DF = cell(num_files, 1);
for K = 1 : num_files
  this_file = filenames{K};
  DF{K} = read(Tiff(this_file,'r'));
%   imshow(SWE{K});
end
SWEDF = double(cat(3,DF{:,1}));


[llat, llon] = meshgrid(v2,v1);llat = llat';llon = llon';
rr = nanmean(SWEDF(23:610,2:1393,76:120),3);
rr1=griddata(llat(:),llon(:),rr(:),latM ,lonM);
rrr = rr1.*PrCrctF4;
rrr = rrr.*mask4MM;
rf = nanmean(nanmean(rrr));

tmp1 = SWEDF(23:610,2:1393,62:92) - SWEDF(23:610,2:1393,52);
tmp2 = SWEDF(23:610,2:1393,97) - SWEDF(23:610,2:1393,31);
tmp3 = SWEDF(23:610,2:1393,105) - SWEDF(23:610,2:1393,63);
tmp4 = SWEDF(23:610,2:1393,97) - SWEDF(23:610,2:1393,60);
tmp5 = SWEDF(23:610,2:1393,87) - SWEDF(23:610,2:1393,18);
tmp6 = SWEDF(23:610,2:1393,98) - SWEDF(23:610,2:1393,67);

rr1=griddata(llat(:),llon(:),rr(:),latM ,lonM);
SWEDFInt2=griddata(llat(:),llon(:),tmp2(:),latM ,lonM);
SWEDFInt3=griddata(llat(:),llon(:),tmp3(:),latM ,lonM);
SWEDFInt4=griddata(llat(:),llon(:),tmp4(:),latM ,lonM);
SWEDFInt5=griddata(llat(:),llon(:),tmp5(:),latM ,lonM);
SWEDFInt6=griddata(llat(:),llon(:),tmp6(:),latM ,lonM);

SWEDFMasked1 = SWEDFInt1.*PrCrctF1;
SWEDFMasked1 = SWEDFMasked1.*mask1MM;

SWEDFMasked2 = SWEDFInt2.*PrCrctF2;
SWEDFMasked2 = SWEDFMasked2.*mask2MM;

SWEDFMasked3 = SWEDFInt3.*PrCrctF3;
SWEDFMasked3 = SWEDFMasked3.*mask3MM;

SWEDFMasked4 = SWEDFInt4.*PrCrctF4;
SWEDFMasked4 = SWEDFMasked4.*mask4MM;

SWEDFMasked5 = SWEDFInt5.*PrCrctF5;
SWEDFMasked5 = SWEDFMasked5.*mask5MM;

SWEDFMasked6 = SWEDFInt6.*PrCrctF6;
SWEDFMasked6 = SWEDFMasked6.*mask6MM;

DFMasked1 = cell(num_files, 1);
DFMasked2 = cell(num_files, 1);
DFMasked3 = cell(num_files, 1);
DFMasked4 = cell(num_files, 1);
DFMasked5 = cell(num_files, 1);
DFMasked6 = cell(num_files, 1);

for i = 43:60
    tmp = double(DF{i,1}(23:610,2:1393));
    tmpp = griddata(llat(:),llon(:),tmp(:),latM ,lonM);
    %tmpp1 = tmpp.*PrCrctF1;
    DFMasked4{i} = tmpp.*mask4MM;
%     tmpp2 = tmpp.*PrCrctF2;
%     DFMasked2{i} = tmpp2.*mask2MM;
%     tmpp3 = tmpp.*PrCrctF3;
%     DFMasked3{i} = tmpp3.*mask3MM;
%     tmpp4 = tmpp.*PrCrctF4;
%     DFMasked4{i} = tmpp4.*mask4MM;
%     tmpp5 = tmpp.*PrCrctF5;
%     DFMasked5{i} = tmpp5.*mask5MM;
%     tmpp6 = tmpp.*PrCrctF6;
%     DFMasked6{i} = tmpp6.*mask6MM;
end

SWEDFMaskedd1 = double(cat(3,DFMasked1{:,1}));
SWEDFMaskedd2 = double(cat(3,DFMasked2{:,1}));
SWEDFMaskedd3 = double(cat(3,DFMasked3{:,1}));
SWEDFMaskedd4 = double(cat(3,DFMasked4{:,1}));
SWEDFMaskedd5 = double(cat(3,DFMasked5{:,1}));
SWEDFMaskedd6 = double(cat(3,DFMasked6{:,1}));
% SWEDFMasked7 = double(cat(3,DFMasked7{:,1}));
% SWEDFMasked8 = double(cat(3,DFMasked8{:,1}));
% SWEDFMasked9 = double(cat(3,DFMasked9{:,1}));

diff = cell(32, 1);
for i = 1:32
    tmp = SWEDFMaskedd1(:,:,i+1) - SWEDFMaskedd1(:,:,i);
    diff{i} = tmp;
end
dif = double(cat(3,diff{:,1}));
di = dif;   
for i = 1:32
    di = min(di,0);
end
ddiiff = -nansum(di,3);

SWEDFMaskedtmp1 = SWEDFMasked1;
SWEDFMaskedtmp1 = ddiiff + SWEDFMaskedtmp1;
SWEDFMaskedt1 = SWEDFMasked1 + 9.3;

K = .15*ones(2);
Zsmooth1 = conv2(SWEDFMasked1,K,'same');

fsp = fspecial('gaussian',[3 3],.65);
SWEDFMaskedtmp1 = nanconv(SWEDFMaskedtmp1,fsp,'edge','nanout');

yourmatrix = yourmatrix +9.29;

SWEDFMaskedtemp5 = SWEDFMasked3 + 8.48;

figure
latlim = [40 50];
lonlim = [-80 -65];
imagesc(lonM(1,:),latM(:,1),SWEDFMaskedtemp5);
states = shaperead('usastatehi', 'UseGeoCoords', true);
geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
colormap(jet);
set(gca,'Ydir','Normal');
cmap=jet;
cmap=[1 1 1;cmap];
colormap(cmap);
colorbar;
caxis([0 80])
set(gca, 'XLim', lonlim, 'YLim',latlim);
xlabel('Lon [\circ]')
ylabel('Lat [\circ]')
title('UA SWE [mm]')


h = fspecial('gaussian');
y = filter2(h, SWEDFMaskedtmp1);


SWEDFMaskedtmp1 = SWEDFMasked1;
[B1, window1] = smoothdata(SWEDFMaskedtmp1,1,'gaussian','omitnan');
[B2, window2] = smoothdata(SWEDFMaskedtmp1,2,'gaussian','omitnan');
B = (B1 + B2)./2;


%%
SWEDFMasked1Avg = nanmean(nanmean(SWEDFMaskedd1,1),2);
SWEDFMasked1Avgg = SWEDFMasked1Avg(1,1,67:128);
SWEDFMasked1Avgg = SWEDFMasked1Avgg - SWEDFMasked1Avgg(1);

% DeltaSWE1 = zeros(1,44);
% for i = 1:44
%     DeltaSWE1(1,i) = SWEDFMasked1Avgg(1,1,i+1) - SWEDFMasked1Avgg(1,1,i);
% end   

% SWEAcc1 = double(DFMasked1{86}-DFMasked1{42});
% SWEDFMasked11 = SWEDFMasked1(:,:,133) - SWEDFMasked1(:,:,71);
% SWEDFMasked22 = SWEDFMasked2(:,:,111) - SWEDFMasked2(:,:,72);
% SWEDFMasked33 = SWEDFMasked3(:,:,93) - SWEDFMasked3(:,:,64);

%%%%%%%%%%%%%%%%%%
SWEDFMasked2Avg = nanmean(nanmean(SWEDFMaskedd2,1),2);
SWEDFMasked2Avgg = SWEDFMasked1Avg(1,1,67:111);
SWEDFMasked2Avgg = SWEDFMasked2Avgg - SWEDFMasked2Avgg(1);

% DeltaSWE2 = zeros(1,32);
% for i = 1:32
%     DeltaSWE2(1,i) = SWEDFMasked2Avgg(1,1,i+1) - SWEDFMasked2Avgg(1,1,i);
% end 

% SWEAcc2 = double(DFMasked2{74}-DFMasked2{42});

%%%%%%%%%%%%%%%%%%%
SWEDFMasked3Avg = nanmean(nanmean(SWEDFMaskedd3,1),2);
SWEDFMasked3Avgg = SWEDFMasked3Avg(1,1,63:109);
SWEDFMasked3Avgg = SWEDFMasked3Avgg - SWEDFMasked3Avgg(1);

% DeltaSWE3 = zeros(1,28);
% for i = 1:28
%     DeltaSWE3(1,i) = SWEDFMasked3Avgg(1,1,i+1) - SWEDFMasked3Avgg(1,1,i);
% end 

% SWEAcc3 = double(DFMasked3{73}-DFMasked3{45});
%%%%%%%%%%%%%%%%%%%%

SWEDFMasked4Avg = nanmean(nanmean(SWEDFMaskedd4,1),2);
SWEDFMasked4Avgg = SWEDFMasked4Avg(1,1,76:120);
SWEDFMasked4Avgg = SWEDFMasked4Avgg - SWEDFMasked4Avgg(1);

% DeltaSWE4 = zeros(1,38);
% for i = 1:38
%     DeltaSWE4(1,i) = SWEDFMasked4Avgg(1,1,i+1) - SWEDFMasked4Avgg(1,1,i);
% end 

% SWEAcc4 = double(DFMasked4{83}-DFMasked4{45});
%%%%%%%%%%%%%%%%%%%%
SWEDFMasked5Avg = nanmean(nanmean(SWEDFMasked5,1),2);
SWEDFMasked5Avgg = SWEDFMasked5Avg(1,1,93:126);
SWEDFMasked5Avgg = SWEDFMasked5Avgg - SWEDFMasked5Avgg(1);




% DeltaSWE5 = zeros(1,33);
% for i = 1:33
%     DeltaSWE5(1,i) = SWEDFMasked5Avgg(1,1,i+1) - SWEDFMasked5Avgg(1,1,i);
% end 

% SWEAcc5 = double(DFMasked5{88}-DFMasked5{55});
%%%%%%%%%%%%%%%%%%%%
SWEDFMasked6Avg = nanmean(nanmean(SWEDFMaskedd6,1),2);
SWEDFMasked6Avgg = SWEDFMasked6Avg(1,1,39:67);
SWEDFMasked6Avgg = SWEDFMasked6Avgg - SWEDFMasked6Avgg(1);

% DeltaSWE6 = zeros(1,49);
% for i = 1:49
%     DeltaSWE6(1,i) = SWEDFMasked6Avgg(1,1,i+1) - SWEDFMasked6Avgg(1,1,i);
% end 

% SWEAcc6 = double(DFMasked6{91}-DFMasked6{42});
%%%%%%%%%%%%%%%%%%%%

figure;
ts1 = timeseries(SWEDFMasked1Avgg);
ts1.Name = 'SWE [mm]';
ts1.TimeInfo.Units = 'days';
ts1.TimeInfo.StartDate = '10-Jan-2003';     % Set start date.
ts1.TimeInfo.Format = 'mmm dd, yy';       % Set format for display on x-axis.
ts1.Time = ts1.Time - ts1.Time(1);        % Express time relative to the start date.
subplot(2,2,1);
plot(ts1)
grid on
grid minor
title('Area-Averaged Snow Water Equivalent over Region1')