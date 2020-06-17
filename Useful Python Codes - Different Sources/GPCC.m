ncvars =  {'precip'};
projectdir = 'C:\Users\miladpanahi\Desktop\Master\Paper\GPCC\monitoring';
dinfo = dir( fullfile(projectdir, '*.nc') );
num_files = length(dinfo);
filenames = fullfile( projectdir, {dinfo.name} );
precipsInt = cell(num_files, 1);
for K = 1 : num_files
  this_file = filenames{K};
  precipsInt{K} = transpose(ncread(this_file, 'corr_fac'));
  %precipsInt{1,K}(isnan(mydata{1,K}))=0;
end
FuchsCorrFacs = cat(3,precipsInt{:,1});
FuchsCorrFacs = FuchsCorrFacs(41:67,55:114,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ncvars =  {'latitude', 'longitude', 'precip'};
projectdir = 'C:\Users\miladpanahi\Desktop\Master\Paper\GPCP\2015-16';
dinfo = dir( fullfile(projectdir, '*.nc') );
num_files = length(dinfo);
filenames = fullfile( projectdir, {dinfo.name} );
precipsInt = cell(num_files, 1);
%if 151 ==> 305:365 & 1:90
%if 152 ==> 306:366 & 1:91
for K = 1 : num_files
  tmp = ncread(filenames{K}, ncvars{3});
  tmp1 = tmp(:,:,305:365);
  tmp2 = tmp(:,:,1:90);
  
  %precipsInt{1,K}(isnan(mydata{1,K}))=0;
end