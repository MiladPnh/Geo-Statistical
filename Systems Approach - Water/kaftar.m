% Importing Mean Daily Data 
clear; clc;
data = load("LeafRiverDaily.txt");
% Creating a for First Year of Data
period = 1:1095;
date = transpose(linspace(datetime(1948,10,1),datetime(1951,9,30),1095));
pcp = data(period,1); %Extractng Precipitation Data
pet = data(period,2); %Extractng Potential Evapotranspiration Data
str = data(period,3); %Extractng Streamflow Data

x1 = zeros(max(period),1);
px1 = zeros(max(period),1);
px2 = zeros(max(period),1);
rg = zeros(max(period),1);
of = zeros(max(period),1);
x2 = zeros(max(period),1);
x3 = zeros(max(period),1);
x4 = zeros(max(period),1);
q = zeros(max(period),1);

parmin = [.2, 10, .5, 0, .0001, .1]';
parmax = [.9, 300, 1.5, 1, .1, .9]';
par = [parmin, parmax, parmax-parmin];

tk = 0.5;
tc = 200;
tp = 1;
ta = 0.5;
tg = 0.5;
ts = 0.5;
t = 1095;
% rmse = fun5(t, pcp, pet, x1, tk, tc, tp, ta, tg, x2, x3, x4, ts, str);
rmse = zeros(1, 7);
ini_p = rand(6,7) .* repmat(par(:,3),1,7) + repmat(par(:,1),1,7);

for i=1:7
rmse(1,i) = fun5(period, pcp, pet, x1, ini_p(1,i), ini_p(2,i), ini_p(3,i), ini_p(4,i), ini_p(5,i), x2, x3, x4, ini_p(6,i), str);
end

i = 1;
copy_rmse = rmse;
copy_ini_p = ini_p;

par = zeros(6,1);
RMSE = 0;

while min(copy_rmse)>1.8 && i<30
disp(copy_rmse)

[max1, ind1] = max(copy_rmse);
wp = copy_ini_p(:,ind1);
[min1, ind3] = min(copy_rmse);
bp = copy_ini_p(:,ind3);
copy_ini_p(:,ind1) = zeros;
copy_rmse(ind1) = NaN(1);
[max2, ind2] = nanmax(copy_rmse);
wp2 = copy_ini_p(:,ind2);
% copy_rmse(ind1) = max1;
centroid = sum(copy_ini_p, 2)/6;
% copy_ini_p(:,ind1) = wp;
%projection
dis = wp - centroid;
np = centroid - dis;
rmse_np = fun5(t, pcp, pet, x1, np(1), np(2), np(3), np(4), np(5), x2, x3, x4, np(6), str);
if rmse_np < rmse(ind2)
    RMSE(i) = rmse_np;
    disp('projection');
    copy_rmse(ind1) = rmse_np;
    copy_ini_p(:,ind1) = np;
    ep = centroid - 2 * dis;
    rmse_e = fun5(t, pcp, pet, x1, ep(1), ep(2), ep(3), ep(4), ep(5), x2, x3, x4, ep(6), str);
    if rmse_e < rmse_np
        RMSE(i) = rmse_e;
        disp('extension');
        copy_ini_p(:,ind1) = ep;
        copy_rmse(ind1) = rmse_e;
    end
else 
    np = wp - dis/2;
    
    rmse_np = fun5(t, pcp, pet, x1, np(1), np(2), np(3), np(4), np(5), x2, x3, x4, np(6), str);
    if rmse_np < rmse(ind2)
        disp('contraction');
        RMSE(i) = rmse_np;
        copy_ini_p(:,ind1) = np;
        copy_rmse(ind1) = rmse_np;
    else
        disp('shrinkage');
        copy_ini_p(:,ind1) = wp;
        disc = copy_ini_p - copy_ini_p(:,ind3);
        copy_ini_p = copy_ini_p - disc/2;
        for j=1:7
            copy_rmse(1,j) = fun5(t, pcp, pet, x1, copy_ini_p(1,j), copy_ini_p(2,j), copy_ini_p(3,j), copy_ini_p(4,j), copy_ini_p(5,j), x2, x3, x4, copy_ini_p(6,j), str);
        end
        RMSE(i) = min(copy_rmse);
    end
end
i = i + 1;

end
[minf, indf] = min(copy_rmse); 
disp(min(copy_rmse));
disp(copy_ini_p(:, indf));
disp(min(RMSE))    
    
    
    
    