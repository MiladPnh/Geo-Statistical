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

tkk = 0.2:0.1:0.9;   %0.2, 0.9
tcc = 10:10:300;    %10, 300
tpp = 0.5:0.1:1.5;   %0.5, 1.5
taa = 0:0.1:1;   %0, 1
tgg = 0.0001:0.01:0.1;   %0.0001, 0.1
tss = 0.1:0.1:0.9;   %0.1, 0.9

rmse(tk,tc,tp,ta,k2Q,k3Q)
rmse(2,8,1,4,9,5)

z = rmse(2,8,:,4,9,5)
figure
subplot(3,2,1)
plot(tcc,rmse(2,:,1,4,9,5))
subplot(3,2,2)
z = rmse(2,8,1,:,9,5);
plot(taa(:),z(:))
subplot(3,2,3)
plot(tkk,rmse(:,8,1,4,9,5))
subplot(3,2,4)
z = rmse(2,8,:,4,9,5);
plot(tpp(:),z(:))
subplot(3,2,5)
z = rmse(2,8,1,4,:,5);
plot(tgg(:),z(:))
subplot(3,2,6)
z = rmse(2,8,1,4,9,:);
plot(tss(:),z(:))


figure
subplot(3,2,1)
plot(tcc,psnr(2,:,1,4,9,5))
xlabel('Parameter C1');
ylabel('Peak signal-to-noise ratio');
subplot(3,2,2)
z = psnr(2,8,1,:,9,5);
plot(taa(:),z(:))
xlabel('Parameter K12');
ylabel('Peak signal-to-noise ratio');
subplot(3,2,3)
plot(tkk,psnr(:,8,1,4,9,5))
xlabel('Parameter K13');
ylabel('Peak signal-to-noise ratio');
subplot(3,2,4)
z = psnr(2,8,:,4,9,5);
plot(tpp(:),z(:))
xlabel('Parameter P1');
ylabel('Peak signal-to-noise ratio');
subplot(3,2,5)
z = psnr(2,8,1,4,:,5);
plot(tgg(:),z(:))
xlabel('Parameter K2Q');
ylabel('Peak signal-to-noise ratio');
subplot(3,2,6)
z = psnr(2,8,1,4,9,:);
plot(tss(:),z(:))
xlabel('Parameter K3Q');
ylabel('Peak signal-to-noise ratio');


rmse = zeros(length(tkk), length(tcc), length(tpp), length(taa), length(tgg), length(tss));
psnr = zeros(length(tkk), length(tcc), length(tpp), length(taa), length(tgg), length(tss));
r = zeros(length(tkk), length(tcc), length(tpp), length(taa), length(tgg), length(tss));
nrmse = zeros(length(tkk), length(tcc), length(tpp), length(taa), length(tgg), length(tss));
mape = zeros(length(tkk), length(tcc), length(tpp), length(taa), length(tgg), length(tss));


for ts = tss
for tg = tgg
for ta = taa
for tp = tpp
for tc = tcc
for tk = tkk
for t=1:max(period)
[x1(t+1),px(t)] = fun1(pcp(t),pet(t),x1(t),tk,tc,tp);
[rg(t),of(t)] = fun2(px(t),ta);
[bf(t),x2(t+1)] = fun3(rg(t),tg,x2(t));
[sf(t),x3(t+1),x4(t+1)] = fun4(x3(t),x4(t),of(t),ts);
q(t) = sf(t) + bf(t);
end
s = CalcPerf(str,q);
rmse(find(tkk==tk),find(tcc==tc),find(tpp==tp),find(taa==ta),find(tgg==tg),find(tss==ts)) = s.RMSE;
psnr(find(tkk==tk),find(tcc==tc),find(tpp==tp),find(taa==ta),find(tgg==tg),find(tss==ts)) = s.PSNR;
r(find(tkk==tk),find(tcc==tc),find(tpp==tp),find(taa==ta),find(tgg==tg),find(tss==ts)) = s.Rvalue;
nrmse(find(tkk==tk),find(tcc==tc),find(tpp==tp),find(taa==ta),find(tgg==tg),find(tss==ts)) = s.NRMSE;
mape(find(tkk==tk),find(tcc==tc),find(tpp==tp),find(taa==ta),find(tgg==tg),find(tss==ts)) = s.Mape;
end
end
end
end
end
end

% Stats = cell2mat(rmse);
% s = {Stats.RMSE};
% s1 = cell2mat({Stats.RMSE});
[C,I] = min(rmse(:));
[I1,I2,I3,I4,I5,I6] = ind2sub(size(rmse),I);

[C,I] = max(psnr(:));
[I1,I2,I3,I4,I5,I6] = ind2sub(size(psnr),I);
psnr(I1,I2,I3,I4,I5,I6)
[C,I] = max(r(:));
[I1,I2,I3,I4,I5,I6] = ind2sub(size(r),I);

[C,I] = min(nrmse(:));
[I1,I2,I3,I4,I5,I6] = ind2sub(size(nrmse),I);

[C,I] = min(mape(:));
[I1,I2,I3,I4,I5,I6] = ind2sub(size(mape),I);

