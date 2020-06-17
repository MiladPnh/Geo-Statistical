% Importing Mean Daily Data
data = load("LeafRiverDaily.txt");
% Creating a for First Year of Data

for z = 1:20
    for zz = 1:5000

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

tk = 0.2;
tc = 40;
tp = 0.5;
ta = 0.5;
tg = 0.0005;
ts = 0.6;
t = 1095;
% rmse = fun5(t, pcp, pet, x1, tk, tc, tp, ta, tg, x2, x3, x4, ts, str);
rmse = zeros(1, 7);
ini_p = rand(6,7) .* repmat(par(:,3),1,7) + repmat(par(:,1),1,7);

for i=1:7
rmse(1,i) = fun5(t, pcp, pet, x1, ini_p(1,i), ini_p(2,i), ini_p(3,i), ini_p(4,i), ini_p(5,i), x2, x3, x4, ini_p(6,i), str);
end

i = 1;
copy_rmse = rmse;
copy_ini_p = ini_p;

par = zeros(6,1);
RMSE = 0;

while min(copy_rmse)>0.5 && i<35
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
centroid = (sum(copy_ini_p, 2)-copy_ini_p(:,ind1))/6;
% copy_ini_p(:,ind1) = wp;
%projection
dis = wp - centroid;
np = centroid - dis;
[rmse_np,qq] = fun5(t, pcp, pet, x1, np(1), np(2), np(3), np(4), np(5), x2, x3, x4, np(6), str);
if rmse_np < rmse(ind2)
    RMSE(i) = rmse_np;
    qqq(i,:) = qq;
    npp(i,:)=np;
    disp('projection');
    copy_rmse(ind1) = rmse_np;
    copy_ini_p(:,ind1) = np;
    ep = centroid - 2 * dis;
    [rmse_e,qq] = fun5(t, pcp, pet, x1, ep(1), ep(2), ep(3), ep(4), ep(5), x2, x3, x4, ep(6), str);
    if rmse_e < rmse_np
        RMSE(i) = rmse_e;
        qqq(i,:) = qq;
        npp(i,:)=np;
        disp('extension');
        copy_ini_p(:,ind1) = ep;
        copy_rmse(ind1) = rmse_e;
    end
else 
    np = wp - dis/2;

    
    [rmse_np,qq] = fun5(t, pcp, pet, x1, np(1), np(2), np(3), np(4), np(5), x2, x3, x4, np(6), str);
    if rmse_np < rmse(ind2)
        disp('contraction');
        RMSE(i) = rmse_np;
        qqq(i,:)=qq;
        npp(i,:)=np;
        copy_ini_p(:,ind1) = np;
        copy_rmse(ind1) = rmse_np;
    else
        disp('shrinkage');
        copy_ini_p(:,ind1) = wp;
        disc = copy_ini_p - copy_ini_p(:,ind3);
        copy_ini_p = copy_ini_p - disc/2;

        for j=1:7
            [copy_rmse(1,j),qq] = fun5(t, pcp, pet, x1, copy_ini_p(1,j), copy_ini_p(2,j), copy_ini_p(3,j), copy_ini_p(4,j), copy_ini_p(5,j), x2, x3, x4, copy_ini_p(6,j), str);
        end
        [RMSE(i)] = min(copy_rmse);
        qqq(i,:)=qq;
        npp(i,:)=np;
    end
end
i = i + 1;
end
[minf, indf] = min(copy_rmse); 
disp(minf);
disp(copy_ini_p(:, indf));
[avalue, aindex] = min(RMSE);
disp(avalue)    
disp(aindex)
%qopt(zz,:) = qqq(aindex,:);
R(zz) = avalue;
nppp(zz,:) = npp(aindex,:);
    end
%qoptt(z) = qopt(zz,:);
RR(z,:) = R(:,:);
npppp(z,:,:) = nppp;
end

% tk = 0.2;
% tc = 40;
% tp = 0.5;
% ta = 0.5;
% tg = 0.0001;
% ts = 0.65;
% [err, qoptm] = fun5(t, pcp, pet, x1, 0.2, tc, tp, ta, tg, x2, x3, x4, ts, str);

% x = (qopt'-str);
% y = (qoptm-str);
% binRange = -10:0.5:10;
% hcx = histcounts(x,[binRange Inf]);
% hcy = histcounts(y,[binRange Inf]);
% bar(binRange,[hcx;hcy]');



% color = [0, 0.4470, 0.7410];
% subplot(4,2,[1, 2]);
% % title('Precipitation and Streamflow','FontWeight','bold');
% % yyaxis left;
% plot(RMSE,'k');
% ylabel('Objective Function - RMSE');
% %axis('ij');
% ax = gca;
% ax.YAxis(1).Color = 'k';
% % hold on;
% % yyaxis right;
% % plot(npp(:,1));
% % ylabel('Parameter  Variations');
% % hold off;
% %ax.YAxis(2).Color = color;
% subplot(4,2,3);
% plot(npp(:,1));
% ylabel('Parameter ?K_1_3 Variations');
% %ax.YAxis(2).Color = color;
% subplot(4,2,4);
% plot(npp(:,2));
% ylabel('Parameter ?c_1 Variations');
% %ax.YAxis(2).Color = color;
% subplot(4,2,5);
% plot(npp(:,3));
% ylabel('Parameter ?P_1 Variations');
% %ax.YAxis(2).Color = color;
% subplot(4,2,6);
% plot(npp(:,4));
% ylabel('Parameter ?? Variations');
% %ax.YAxis(2).Color = color;
% subplot(4,2,7);
% plot(npp(:,5));
% ylabel('Parameter ?K_2_Q Variations');
% xlabel('Number of Iterations in AutoCalibration');
% %ax.YAxis(2).Color = color;
% subplot(4,2,8);
% plot(npp(:,6));
% ylabel('Parameter ?K_3_Q Variations');
% xlabel('Number of Iterations in AutoCalibration');
% 
% 
% 
% 
% color = [0, 0.4470, 0.7410];
% subplot(4,1,1);
% title('Precipitation and Streamflow','FontWeight','bold');
% 
% yyaxis left;
% bar(period,pcp,1,'r');
% ylabel('Precipitation(mm)');
% axis('ij');
% ax = gca;
% ax.YAxis(1).Color = 'k';
% 
% hold on;
% yyaxis right;
% plot(period,qopt','b');
% ylabel('Streamflow (mm)');
% hold on;
% plot(period,str,'-k');
% hold off;
% legend('Precipitation (mm)', 'Modeled Streamflow (mm)','Observed Streamflow (mm)');
% set(legend,'Location','northeast');
% 
% 
% ax.YAxis(2).Color = color;

RRR = real(RR);

[RRRv, RRRidx] = min(RRR,[],2);

figure;
yyaxis left;
plot(1:20,RRRv);
hold on;
yyaxis left;
tmp = npppp(1:20,RRRidx(1:20),1);
plot(1:20,diag(npppp(1:20,RRRidx(1:20),1)));
%ylabel('Parameter ?K_1_3');
hold on;
yyaxis left;
plot(1:20,diag(npppp(1:20,RRRidx(1:20),3)));
%ylabel('Parameter ?P_1');
hold on;
yyaxis left;
plot(1:20,diag(npppp(1:20,RRRidx(1:20),4)));
%ylabel('Parameter ??');
hold on;
yyaxis left;
plot(1:20,diag(npppp(1:20,RRRidx(1:20),5)));
%ylabel('Parameter ?K_2_Q');
hold on;
yyaxis left;
plot(1:20,diag(npppp(1:20,RRRidx(1:20),6)));
yyaxis right;
plot(1:20,diag(npppp(1:20,RRRidx(1:20),2)));
ylabel('Parameter ?c_1');
%yyaxis([-1000 1000]);
ax = gca;
%ax.YAxis(1).Limit = [-1000 1000];
axis([1 20 -1000 1000]);
%ylabel('Parameter ?K_3_Q');
legend('RMSE','Parameter ?K_1_3', 'Parameter ?P_1','Parameter ??','Parameter ?K_2_Q','Parameter ?K_3_Q','Parameter ?c_1');