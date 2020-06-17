% time  = summer_2009.image_1.UTC;
% newtime = zeros(1,length(time));
% for i = 1: length(time)
%     tmpp = split(char(time(i)),'-');
%     newtime(1,i) = str2num(tmpp{1});
% end 

% for K = 1 : num_files
%   this_file = filenames{K};
%   TotalPr{K} = ncread(this_file, ncvars{3});
%   TotalPrC{K} = ncread(this_file, ncvars{4});
%   %precipsInt{1,K}(isnan(mydata{1,K}))=0;
% end
% latMerra = ncread('MERRA2_300.inst3_3d_asm_Np.20090617.SUB.nc', 'lat');
% lonMerra = ncread('MERRA2_300.inst3_3d_asm_Np.20090617.SUB.nc', 'lon');

g = summer_2009.image_4.UTC;
gg = summer_2009.image_4;

a = gg.min_lat;
b = fix(a);
c = a-b;
for i = 1:length(a)
    if c(i)>.5
        if round(a(i))>30
            a(i) = round(a(i))-.5;
        else
            a(i) = round(a(i));
        end    
    else
        a(i) = round(a(i));
    end    
end


a1 = gg.max_lat;
b = fix(a1);
c = a1-b;
for i = 1:length(a1)
    if c(i)>.5
        a1(i) = round(a1(i));
    else
        if round(a1(i))<50
            a1(i) = round(a1(i))+.5;
        else
            a1(i) = round(a1(i));
        end    
    end    
end


a2 = gg.min_lon;
b = fix(a2);
c = a2-b;
for i = 1:length(a2)
    if abs(c(i))>=.66
        if mod(fix(a2(i)), 2) == 1
            a2(i) = round(a2(i));
        else
            a2(i) = fix(a2(i))-.666666666722;
        end    
    elseif .33<abs(c(i)) && abs(c(i))<.66
        if mod(fix(a2(i)), 2) == 0
            a2(i) = fix(a2(i))-.666666666722;
        else
            a2(i) = fix(a2(i))-.333333333388;
        end    
    else
        if mod(fix(a2(i)), 2) == 1
            if fix(a2(i))<=-135
                a2(i) = -134.666666666712;
            else
                a2(i) = fix(a2(i))-.333333333388;
            end
        else
            a2(i) = round(a2(i));
        end    
    end    
end


a3 = gg.max_lon;
b = fix(a3);
c = a3-b;
for i = 1:length(a3)
    if abs(c(i))<=.33
        if mod(fix(a3(i)), 2) == 0
            a3(i) = round(a3(i));
        else
            a3(i) = fix(a3(i))-.333333333388;
        end
    elseif .33<abs(c(i)) && abs(c(i))<.66
        if mod(fix(a3(i)), 2) == 0
            a3(i) = fix(a3(i))-.666666666722;
        else
            a3(i) = fix(a3(i))-.333333333388;
        end    
    else
        if mod(fix(a3(i)), 2) == 0
            a3(i) = fix(a3(i))-.666666666722;
        else
            a3(i) = round(a3(i));
        end    
    end    
end

indexT = zeros(1,length(a1));
indexxT = zeros(1,length(a1));
indexxxT = zeros(1,length(a1));
indexxxxT = zeros(1,length(a1));

for i = 1 : length(a)
    if isnan(a2(i))==1 || a(i)==0
    else
        [tmp] = find(fix(a(i,1)*10000)==fix(latMerra*10000));
        indexT(1, i) = tmp;
        [tmp] = find(fix(a1(i,1)*10000)==fix(latMerra*10000));
        indexxT(1, i) = tmp;
        [tmp] = find(fix(a2(i,1)*10000)==fix(lonMerra*10000));
        indexxxT(1, i) = tmp;
        [tmp] = find(fix(a3(i,1)*10000)==fix(lonMerra*10000));
        indexxxxT(1, i) = tmp;
    end
end    


% ncvars =  {'V', 'U', 'T', 'SLP', 'RH', 'OMEGA'};
% projectdir1 = 'C:\Users\miladpanahi\Desktop\Master\Paper\Hossein Paper\Merra2\2018';
% % ncdisp('MERRA2_300.inst3_3d_asm_Np.20090617.SUB.nc');
% dinfo1 = dir( fullfile(projectdir1, '*.nc') );
% num_files = length(dinfo1);
% filenames = fullfile( projectdir1, {dinfo1.name} );
% projectdir2 = 'C:\Users\miladpanahi\Desktop\Master\Paper\Hossein Paper\Merra2\2018\Modis';
% % ncdisp('MERRA2_300.inst1_2d_asm_Nx.20090601.SUB.nc');
% dinfo2 = dir( fullfile(projectdir2, '*.nc') );
% filenames2 = fullfile( projectdir2, {dinfo2.name} );
% projectdir3 = 'C:\Users\miladpanahi\Desktop\Master\Paper\Hossein Paper\Merra2\2018\10m and 2m air Temperature and wind speed,sea level pressure,surface pressure';
% % ncdisp('MERRA2_300.inst1_2d_asm_Nx.20090617.SUB.nc');
% dinfo3 = dir( fullfile(projectdir3, '*.nc') );
% filenames3 = fullfile( projectdir3, {dinfo3.name} );
% ncvars1 =  {'U10M','V10M'};
% 
% 
% tt1 = hour(g)+round(minute(g)/60);
% tt = round((hour(g)+round(minute(g)/60))/3);
% Que = zeros(ii,12);
% for ii = 1:length(a)
%     if indexT(ii)==0
%     else    
%         T1 = ncread(filenames{ii}, ncvars{3});
%         T1at1000 = nanmean(nanmean(T1(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),1,tt(ii))));
%         T1at850 = nanmean(nanmean(T1(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),3,tt(ii))));
%         T1at700 = nanmean(nanmean(T1(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),4,tt(ii))));
%         U1 = ncread(filenames{ii}, ncvars{2});U1 = nanmean(nanmean(U1(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),5,tt(ii))));
%         SLP1 = ncread(filenames3{ii}, 'SLP');SLP1 = nanmean(nanmean(SLP1(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),tt1(ii))));
%         RH1 = ncread(filenames{ii}, ncvars{5});
%         RH1at700 = nanmean(nanmean(RH1(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),4,tt(ii))));
%         RH1at850 = nanmean(nanmean(RH1(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),3,tt(ii))));
%         RH1at950 = nanmean(nanmean(RH1(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),2,tt(ii))));
%         OMEGA1 = ncread(filenames{ii}, ncvars{6}); OMEGA1 = nanmean(nanmean(OMEGA1(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),4,tt(ii))));
%         Windat10U = ncread(filenames3{ii},ncvars1{1});Windat10U = nanmean(nanmean(Windat10U(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),tt1(ii))));
%         Windat10V = ncread(filenames3{ii},ncvars1{2});Windat10V = nanmean(nanmean(Windat10V(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),tt1(ii))));
%         CF = ncread(filenames2{ii},'MDSCLDFRCLO');CF = nanmean(nanmean(CF(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),tt1(ii))));
%         Que(ii,1) = T1at1000;Que(ii,2) = T1at850;Que(ii,3) = T1at700;Que(ii,4) = OMEGA1;Que(ii,5) = RH1at700;Que(ii,6) = RH1at850;Que(ii,7) = RH1at950;
%         Que(ii,8) = SLP1;Que(ii,9) = U1;Que(ii,10) = Windat10U;Que(ii,11) = Windat10V;Que(ii,12) = CF;
%     end
% end

% 
% tt = round((hour(g)+round(minute(g)/60))/3);
% Quee = zeros(length(a),1);
% projectdir4 = 'C:\Users\miladpanahi\Desktop\Master\Paper\Hossein Paper\Merra2\2018\AOD';
% % ncdisp('MERRA2_400.inst3_2d_gas_Nx.20120824.SUB.nc');
% dinfo4 = dir( fullfile(projectdir4, '*.nc') );
% filenames4 = fullfile( projectdir4, {dinfo4.name} );
% ncvars2 =  {'AODANA','AODINC'};
% 
% for ii = 1:length(a)
%     if indexT(ii)==0
%     else    
%         AOD = ncread(filenames4{ii}, ncvars2{1});
%         AOD = nanmean(nanmean(AOD(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),tt(ii))));
%         Quee(ii,1) = AOD;
%     end
% end



ncvars =  {'V', 'U', 'T', 'SLP', 'RH', 'OMEGA'};
projectdir1 = 'C:\Users\miladpanahi\Desktop\Master\Paper\Hossein Paper\Merra2\2018';
% ncdisp('MERRA2_300.inst3_3d_asm_Np.20090617.SUB.nc');
dinfo1 = dir( fullfile(projectdir1, '*.nc') );
num_files = length(dinfo1);
filenames = fullfile( projectdir1, {dinfo1.name} );
tt = round((hour(g)+round(minute(g)/60))/3);
Queee = zeros(length(a),5);

for ii = 1:length(a)
    if indexT(ii)==0
    else    
        V = ncread(filenames{ii}, ncvars{1});V1 = nanmean(nanmean(V(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),5,tt(ii))));
        V2 = nanmean(nanmean(V(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),3,tt(ii))));
        V3 = nanmean(nanmean(V(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),4,tt(ii))));
        U = ncread(filenames{ii}, ncvars{2});U1 = nanmean(nanmean(U(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),3,tt(ii))));
        U2 = nanmean(nanmean(U(indexxxT(ii):indexxxxT(ii),indexT(ii):indexxT(ii),4,tt(ii))));
        Queee(ii,1) = V1;Queee(ii,2) = U1;Queee(ii,3) = V2;Queee(ii,4) = U2;Queee(ii,5) = V3;
    end
end





x1 = summer_2018.image_1.UTC;
x2 = summer_2018.image_2.UTC;
x3 = summer_2018.image_3.UTC;

y1 = hour(x1)+round(minute(x1)/60);
y2 = hour(x2)+round(minute(x2)/60);
y3 = hour(x3)+round(minute(x3)/60);

yy1 = y2-y1;
yy2 = y3-y2;
yy3 = y3-y1;