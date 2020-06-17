ncdisp('1.nc');
ncvars =  {'latitude', 'longitude', 'tp'};
projectdir = 'C:\Users\miladpanahi\Desktop\Master\Paper\Era Int\2014-15';
dinfo = dir( fullfile(projectdir, '*.nc') );
num_files = length(dinfo);
filenames = fullfile( projectdir, {dinfo.name} );
Pr = cell(num_files, 1);
for K = 1 : num_files
  this_file = filenames{K};
  Pr{K} = 1000*ncread(this_file, ncvars{3});
end
 

Prr = cell(num_files, 1);
for K = 1: num_files
    tmp = Pr{K};
    b = cell(size( tmp , 3 )/2, 1);
    for i = 1:2:size( tmp , 3 )
        b{i} = tmp(:,:,i)+tmp(:,:,i+1);
    end
    bb = b(~cellfun(@isempty, b));
    Prr{K} = cat(3,bb{:,1});
end
EraPr = cat(3,Prr{:,1});


    
    
latEra = double(ncread("1.nc", 'latitude'));
latEra = latEra(41:81);
lonEra = double(ncread("1.nc", 'longitude'));
lonEra = lonEra(11:128);

EraPr = EraPr(11:128,41:81,:);
% EraSub = cat(3,Subb{:,1});
% EraSub = EraSub(11:128,41:81,:);
    


%%

PrInt1 = cell(152, 1);
PrInt2 = cell(152, 1);
PrInt3 = cell(152, 1);
PrInt4 = cell(152, 1);
PrInt5 = cell(152, 1);
PrInt6 = cell(152, 1);


[llat, llon] = meshgrid(latEra,lonEra);llat = llat';llon = llon';


for i = 1:151
    tmp = EraPr(:,:,i);
    tmp = permute(tmp,[2 1 3]);
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
%     tmp6 = tmpp.*PrCrctF6;
%     tmp6 = tmp6.*mask6MM;
    PrInt1{i} = tmp1;
    PrInt2{i} = tmp2;
     PrInt3{i} = tmp3;
    PrInt4{i} = tmp4;
    PrInt5{i} = tmp5;
%     PrInt6{i} = tmp6;
end    
EraPrMasked1 = cat(3,PrInt1{:,1});
EraPrMasked2 = cat(3,PrInt2{:,1});
EraPrMasked3 = cat(3,PrInt3{:,1});
EraPrMasked4 = cat(3,PrInt4{:,1});
EraPrMasked5 = cat(3,PrInt5{:,1});
EraPrMasked6 = cat(3,PrInt6{:,1});

% z = nanmean(EraPrMasked1,3);

Era1Ts = nanmean(nanmean(EraPrMasked1,1),2);
Era1AccTs = cumsum(Era1Ts,'omitnan');
Era1AccTs = Era1AccTs - Era1AccTs(1);

Era2Ts = nanmean(nanmean(EraPrMasked2,1),2);
Era2AccTs = cumsum(Era2Ts,'omitnan');
Era2AccTs = Era2AccTs - Era2AccTs(1);

Era3Ts = nanmean(nanmean(EraPrMasked3,1),2);
Era3AccTs = cumsum(Era3Ts,'omitnan');
Era3AccTs = Era3AccTs - Era3AccTs(1);

Era4Ts = nanmean(nanmean(EraPrMasked4,1),2);
Era4AccTs = cumsum(Era4Ts,'omitnan');
Era4AccTs = Era4AccTs - Era4AccTs(1);

Era5Ts = nanmean(nanmean(EraPrMasked5,1),2);
Era5AccTs = cumsum(Era5Ts,'omitnan');
Era5AccTs = Era5AccTs - Era5AccTs(1);
  
Era6Ts = nanmean(nanmean(EraPrMasked6,1),2);
Era6AccTs = cumsum(Era6Ts,'omitnan');
Era6AccTs = Era6AccTs - Era6AccTs(1);
%we should add the first value of SWE to it


    
