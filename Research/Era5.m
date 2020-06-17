ncvars =  {'latitude', 'longitude', 'tp'};
projectdir = 'C:\Users\miladpanahi\Desktop\Master\Paper\Era5\2010-11';
dinfo = dir( fullfile(projectdir, '*.nc') );
num_files = length(dinfo);
filenames = fullfile( projectdir, {dinfo.name} );
Pr = cell(num_files, 1);
%Sub = cell(num_files, 1);
Pr{1} = 1000*ncread('Era5-Pr-10-US.nc', 'tp');
Pr{2} = 1000*ncread('Era5-Pr-11-US.nc', 'tp');
%2160:2880%2881:3624
PrNov = Pr{1}(:,:,2137:2880);PrDec = Pr{1}(:,:,2881:3624);PrJan = Pr{2}(:,:,1:744);PrFeb = Pr{2}(:,:,745:1416);PrMarch = Pr{2}(:,:,1417:2160);

tpNov = cell(30, 1);tpDec = cell(31, 1);tpJan = cell(31, 1);tpFeb = cell(28, 1);tpMarch = cell(31, 1);
for i = 1:30
        tmp = transpose(sum(PrNov(:,:,(i-1)*24+1:i*24),3));
        tpNov{i} = tmp(6:102,2:232,:);
end
Era5NovPr = cat(3,tpNov{:,1});

for i = 1:31
        tmp = transpose(sum(PrDec(:,:,(i-1)*24+1:i*24),3));
        tpDec{i} = tmp(6:102,2:232,:);
end
Era5DecPr = cat(3,tpDec{:,1});

for i = 1:31
        tmp = transpose(sum(PrJan(:,:,(i-1)*24+1:i*24),3));
        tpJan{i} = tmp(6:102,2:232,:);
end
Era5JanPr = cat(3,tpJan{:,1});

for i = 1:28
        tmp = transpose(sum(PrFeb(:,:,(i-1)*24+1:i*24),3));
        tpFeb{i} = tmp(6:102,2:232,:);
end
Era5FebPr = cat(3,tpFeb{:,1});

for i = 1:31
        tmp = transpose(sum(PrMarch(:,:,(i-1)*24+1:i*24),3));
        tpMarch{i} = tmp(6:102,2:232,:);
end
Era5MarchPr = cat(3,tpMarch{:,1});

latEra5 = double(ncread("Era5-Pr-02-US.nc", 'latitude'));latEra5 = latEra5(6:102);
lonEra5 = double(ncread("Era5-Pr-02-US.nc", 'longitude')); lonEra5 = lonEra5(2:232);

Era5PrTot = zeros(97,231,151);
Era5PrTot(:,:,1:30) = Era5NovPr(:,:,:);
Era5PrTot(:,:,31:61) = Era5DecPr(:,:,:);
Era5PrTot(:,:,62:92) = Era5JanPr(:,:,:);
Era5PrTot(:,:,93:120) = Era5FebPr(:,:,:);
Era5PrTot(:,:,121:151) = Era5MarchPr(:,:,:);



PrInt1 = cell(152, 1);
PrInt2 = cell(152, 1);
PrInt3 = cell(152, 1);
PrInt4 = cell(152, 1);
PrInt5 = cell(152, 1);
PrInt6 = cell(152, 1);
[llat, llon] = meshgrid(latEra5,lonEra5);llat = llat';llon = llon';

for i = 1:151
    tmp = Era5PrTot(:,:,i);
    %tmp = transpose(tmp);
    tmpp=griddata(llat(:),llon(:),tmp(:),latM ,lonM);
    tmp1 = tmpp.*PrCrctF1;
    tmp1 = tmp1.*mask1MM;
    tmp2 = tmpp.*PrCrctF2;
    tmp2 = tmp2.*mask2MM;
      tmp3 = tmpp.*PrCrctF3;
      tmp3 = tmp3.*mask3MM;
    tmp4 = tmpp.*PrCrctF4;
    tmp4 = tmp4.*mask4MM;
    tmp5 = tmpp.*PrCrctF5;
    tmp5 = tmp5.*mask5MM;
    tmp6 = tmpp.*PrCrctF6;
    tmp6 = tmp6.*mask6MM;
    PrInt1{i} = tmp1;
    PrInt2{i} = tmp2;
    PrInt3{i} = tmp3;
    PrInt4{i} = tmp4;
    PrInt5{i} = tmp5;
    PrInt6{i} = tmp6;
end    
Era5PrMasked1 = cat(3,PrInt1{:,1});
Era5PrMasked2 = cat(3,PrInt2{:,1});
Era5PrMasked3 = cat(3,PrInt3{:,1});
Era5PrMasked4 = cat(3,PrInt4{:,1});
Era5PrMasked5 = cat(3,PrInt5{:,1});
Era5PrMasked6 = cat(3,PrInt6{:,1});

Era51Ts = nanmean(nanmean(Era5PrMasked1,1),2);
Era51AccTs = cumsum(Era51Ts);
Era51AccTs = Era51AccTs - Era51AccTs(1);

Era52Ts = nanmean(nanmean(Era5PrMasked2,1),2);
Era52AccTs = cumsum(Era52Ts);
Era52AccTs = Era52AccTs - Era52AccTs(1);

Era53Ts = nanmean(nanmean(Era5PrMasked3,1),2);
Era53AccTs = cumsum(Era53Ts);
Era53AccTs = Era53AccTs - Era53AccTs(1);

Era54Ts = nanmean(nanmean(Era5PrMasked4,1),2);
Era54AccTs = cumsum(Era54Ts);
Era54AccTs = Era54AccTs - Era54AccTs(1);

Era55Ts = nanmean(nanmean(Era5PrMasked5,1),2);
Era55AccTs = cumsum(Era55Ts);
Era55AccTs = Era55AccTs - Era55AccTs(1);
  
Era56Ts = nanmean(nanmean(Era5PrMasked6,1),2);
Era56AccTs = cumsum(Era56Ts);
Era56AccTs = Era56AccTs - Era56AccTs(1);

