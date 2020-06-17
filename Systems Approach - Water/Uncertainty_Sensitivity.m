% Importing Mean Daily Data 
%clear; clc;
% data = load("LeafRiverDaily.txt");
% % Creating a for First Year of Data
% t = 1:1095;
% date = transpose(linspace(datetime(1948,10,1),datetime(1951,9,30),1095));
% pcp = data(t,1); %Extractng Precipitation Data
% pet = data(t,2); %Extractng Potential Evapotranspiration Data
% str = data(t,3); %Extractng Streamflow Data
% 
% x1 = zeros(max(t),1);
% px1 = zeros(max(t),1);
% px2 = zeros(max(t),1);
% rg = zeros(max(t),1);
% of = zeros(max(t),1);
% x2 = zeros(max(t),1);
% x3 = zeros(max(t),1);
% x4 = zeros(max(t),1);

l = 9;


% tk_min = 0.2;
% tk_max = 0.9;
% tc_min = 10;
% tc_max = 300;
% tp_min = 0.5;
% tp_max = 1.5;
% ta_min = .1;
% ta_max = 1;
% tg_min = 0.0001;
% tg_max = 0.1;
% ts_min = 0.1;
% ts_max = 0.9;



rnd = 10000;
rng(42);
a = rand(rnd, 6);
tk = a(:,1) * (tk_max - tk_min) + tk_min;
tc = a(:,2) * (tc_max - tc_min) + tc_min;
tp = a(:,3) * (tp_max - tp_min) + tp_min;
ta = a(:,4) * (ta_max - ta_min) + ta_min;
tg = a(:,5) * (tg_max - tg_min) + tg_min;
ts = a(:,6) * (ts_max - ts_min) + ts_min;

q = zeros(max(t),rnd, 7);
for i = 1:rnd
    for t=1:max(t)
        [x1(t+1),px(t)] = fun1(pcp(t),pet(t),x1(t),tk(i),tc(i),tp(i));
        [rg(t),of(t)] = fun2(px(t),ta(i));
        [bf(t),x2(t+1)] = fun3(rg(t),tg(i),x2(t));
        [sf(t),x3(t+1),x4(t+1)] = fun4(x3(t),x4(t),of(t),ts(i));
        q(t, i, 1) = sf(t) + bf(t);
    end
    
end
rmse = sqrt(sum((q(:,:,1) - str).^2/1095));


tk = 0.2;
for i = 1:rnd
    for t=1:max(t)
        [x1(t+1),px(t)] = fun1(pcp(t),pet(t),x1(t),tk,tc(i),tp(i));
        [rg(t),of(t)] = fun2(px(t),ta(i));
        [bf(t),x2(t+1)] = fun3(rg(t),tg(i),x2(t));
        [sf(t),x3(t+1),x4(t+1)] = fun4(x3(t),x4(t),of(t),ts(i));
        q(t, i, 2) = sf(t) + bf(t);
    end
    
end

rmse1 = sqrt(sum((q(:,:,2) - str).^2/1095));
tk = a(:,1) * (tk_max - tk_min) + tk_min;

tc = 300;
for i = 1:rnd
    for t=1:max(t)
        [x1(t+1),px(t)] = fun1(pcp(t),pet(t),x1(t),tk(i),tc,tp(i));
        [rg(t),of(t)] = fun2(px(t),ta(i));
        [bf(t),x2(t+1)] = fun3(rg(t),tg(i),x2(t));
        [sf(t),x3(t+1),x4(t+1)] = fun4(x3(t),x4(t),of(t),ts(i));
        q(t, i, 3) = sf(t) + bf(t);
    end
    
end
rmse2 = sqrt(sum((q(:,:,3) - str).^2/1095));
tc = a(:,2) * (tc_max - tc_min) + tc_min;

tp = 1.5;
for i = 1:rnd
    for t=1:max(t)
        [x1(t+1),px(t)] = fun1(pcp(t),pet(t),x1(t),tk(i),tc(i),tp);
        [rg(t),of(t)] = fun2(px(t),ta(i));
        [bf(t),x2(t+1)] = fun3(rg(t),tg(i),x2(t));
        [sf(t),x3(t+1),x4(t+1)] = fun4(x3(t),x4(t),of(t),ts(i));
        q(t, i, 4) = sf(t) + bf(t);
    end
    
end
rmse3 = sqrt(sum((q(:,:,4) - str).^2/1095));
tp = a(:,3) * (tp_max - tp_min) + tp_min;

ta = 1;
for i = 1:rnd
    for t=1:max(t)
        [x1(t+1),px(t)] = fun1(pcp(t),pet(t),x1(t),tk(i),tc(i),tp(i));
        [rg(t),of(t)] = fun2(px(t),ta);
        [bf(t),x2(t+1)] = fun3(rg(t),tg(i),x2(t));
        [sf(t),x3(t+1),x4(t+1)] = fun4(x3(t),x4(t),of(t),ts(i));
        q(t, i, 5) = sf(t) + bf(t);
    end
    
end
rmse4 = sqrt(sum((q(:,:,5) - str).^2/1095));
ta = a(:,4) * (ta_max - ta_min) + ta_min;

tg = 0.1;
for i = 1:rnd
    for t=1:max(t)
        [x1(t+1),px(t)] = fun1(pcp(t),pet(t),x1(t),tk(i),tc(i),tp(i));
        [rg(t),of(t)] = fun2(px(t),ta(i));
        [bf(t),x2(t+1)] = fun3(rg(t),tg,x2(t));
        [sf(t),x3(t+1),x4(t+1)] = fun4(x3(t),x4(t),of(t),ts(i));
        q(t, i, 6) = sf(t) + bf(t);
    end
    
end
rmse5 = sqrt(sum((q(:,:,6) - str).^2/1095));
tg = a(:,5) * (tg_max - tg_min) + tg_min;

ts = 0.9;
for i = 1:rnd
    for t=1:max(t)
        [x1(t+1),px(t)] = fun1(pcp(t),pet(t),x1(t),tk(i),tc(i),tp(i));
        [rg(t),of(t)] = fun2(px(t),ta(i));
        [bf(t),x2(t+1)] = fun3(rg(t),tg(i),x2(t));
        [sf(t),x3(t+1),x4(t+1)] = fun4(x3(t),x4(t),of(t),ts);
        q(t, i, 7) = sf(t) + bf(t);
    end
end
rmse6 = sqrt(sum((q(:,:,7) - str).^2/1095));
ts = a(:,6) * (ts_max - ts_min) + ts_min;


% r = zeros(rnd, 7);
% r(:,1) = rmse';
% r(:,2) = rmse1';
% r(:,3) = rmse2';
% r(:,4) = rmse3';
% r(:,5) = rmse4';
% r(:,6) = rmse5';
% r(:,7) = rmse6';



q_sorted = zeros(max(t),rnd, 7);

for i = 1 : 7
    qq = q(:,:,i)';
    q_sorted(:,:,i) = sort(qq)';
end


%5_95 confidence intervals
q_sorted(:,1:500,:) = [];
q_sorted(:,9001:9500,:) = [];

Var = squeeze(var(q, 0, 2));
Mean = nanmean(q, 2);

Sens = (Var(:,1) - Var)./Var(:,1);
AvgSens = nanmean(Sens);
%AvgSensTot = zeros(10, 7);
AvgSensTot(l,:) = AvgSens;




AvgVar = mean(Var);
sqrt(AvgVar);
AvgMean = mean(Mean);
Rd = (AvgVar - AvgVar(1));

%%
%pp1 = AvgSensTot(:,2); [ .2 .3 .4 .5 .6 .7 .8 .9]
%pp2 = AvgSensTot(:,3); [10 50 100 150 200 250 300]
%pp3 = AvgSensTot(:,4); [0.5 0.7 0.85 1 1.15 1.3 1.5]
%pp4 = AvgSensTot(:,5); [.1 .25 .4 .55 .7 .8 .9 1]
%pp5 = AvgSensTot(:,6); [0.0001 0.01 .02 .03 .04 .05 .06 .07 .08 .09 .1]
%pp6 = AvgSensTot(1:9,7); [.1 .2 .3 .4 .5 .6 .7 .8 .9]

figure
subplot(2,3,1)
plot([ .2 .3 .4 .5 .6 .7 .8 .9], pp1)
title('?K_1_3 Varible Fixation', 'FontWeight','bold')
subplot(2,3,2)
plot([10 50 100 150 200 250 300], nonzeros(pp2))
title('?C_1 Varible Fixation', 'FontWeight','bold')
subplot(2,3,3)
plot([0.5 0.7 0.85 1 1.15 1.3 1.5], nonzeros(pp3))
title('?P_1 Varible Fixation', 'FontWeight','bold')
subplot(2,3,4)
plot([.1 .25 .4 .55 .7 .8 .9 1], nonzeros(pp4))
title('?? Varible Fixation', 'FontWeight','bold')
subplot(2,3,5)
plot([0.0001 0.01 .02 .03 .04 .05 .06 .07 .08 .09 .1], nonzeros(pp5))
title('?K_2_Q Varible Fixation', 'FontWeight','bold')
subplot(2,3,6)
plot([.1 .2 .3 .4 .5 .6 .7 .8 .9], nonzeros(pp6))
title('?K_3_Q Varible Fixation', 'FontWeight','bold')
ylabel('Sensitivity Indicator Factor')


% t = 1:1095;
% 
% 
% 
% 
% % [p,x] = hist(rmse); plot(x,p/sum(p)); %PDF
% 
% 
% [f,x] = ecdf(rmse); plot(x,f, 'LineWidth',2); %CDF
% hold on
% [f,x] = ecdf(rmse1); plot(x,f, 'LineWidth',2); %CDF
% hold on
% [f,x] = ecdf(rmse2); plot(x,f, 'LineWidth',2); %CDF
% hold on
% [f,x] = ecdf(rmse3); plot(x,f, 'LineWidth',2); %CDF
% hold on
% [f,x] = ecdf(rmse4); plot(x,f, 'LineWidth',2); %CDF
% hold on
% [f,x] = ecdf(rmse5); plot(x,f, 'LineWidth',2); %CDF
% hold on
% [f,x] = ecdf(rmse6); plot(x,f, 'LineWidth',2); %CDF
% ylabel('Cumulative Probability of RMSE');
% xlabel('RMSE');
% legend('All Params Random', '?K_1_3 Fixed', '?C_1 Fixed', '?P_1 Fixed', '?? Fixed', '?K_2_Q Fixed', '?K_3_Q Fixed');
% set(legend,'Location','northeast');
% 
% 
% 
% 
% 
% 
% color = [0, 0.4470, 0.7410];
% 
% figure;
% subplot(3,3,[1,3])
% yyaxis left;
% shade_plot(t',q_sorted(:,1,1),q_sorted(:,9000,1),nanmean(q_sorted(:,:,1),2))
% xlim([0 1100])
% ylabel('Monte Carlo Simulated Streamflow (cfs)');
% ax = gca;
% ax.YAxis(1).Color = 'k';
% yyaxis right;
% axis('ij');
% bar(t, Var(:,1))
% ylabel('Variance');
% %ax.YAxis(2).Color = color;
% legend('Confidence Intervals (5% - 95%) (cfs)', 'Streamflow (cfs)', 'Variance');
% set(legend,'Location','northeast');
% hold off;
% title('All Params Random', 'FontWeight','bold')
% % plot(1:1095, q_sorted(:,1,1));
% % hold on
% % plot(1:1095, q_sorted(:,9000,1));
% % hold on
% % plot(1:1095, nanmean(q_sorted(:,:,1),2));
% % hold on
% % plot(1:1095, str);
% % legend('5%' , '90%', 'mean', 'str')
% subplot(3,3,4)
% yyaxis left;
% shade_plot(t',q_sorted(:,1,2),q_sorted(:,9000,2),nanmean(q_sorted(:,:,2),2))
% xlim([0 1100])
% ylabel('Monte Carlo Simulated Streamflow (cfs)');
% ax = gca;
% ax.YAxis(1).Color = 'k';
% yyaxis right;
% axis('ij');
% bar(t, Var(:,2))
% ylabel('Variance');
% %ax.YAxis(2).Color = color;
% legend('Confidence Intervals (5% - 95%) (cfs)', 'Streamflow (cfs)', 'Variance');
% set(legend,'Location','northeast');
% hold off;
% title('?K_1_3 Fixed', 'FontWeight','bold')
% % plot(1:1095, q_sorted(:,1,2));
% % hold on
% % plot(1:1095, q_sorted(:,9000,2));
% % hold on
% % plot(1:1095, nanmean(q_sorted(:,:,2),2));
% % legend('5%' , '90%', 'mean')
% subplot(3,3,5)
% yyaxis left;
% shade_plot(t',q_sorted(:,1,3),q_sorted(:,9000,3),nanmean(q_sorted(:,:,3),2))
% xlim([0 1100])
% ylabel('Monte Carlo Simulated Streamflow (cfs)');
% ax = gca;
% ax.YAxis(1).Color = 'k';
% yyaxis right;
% axis('ij');
% bar(t, Var(:,3))
% ylabel('Variance');
% %ax.YAxis(2).Color = color;
% legend('Confidence Intervals (5% - 95%) (cfs)', 'Streamflow (cfs)', 'Variance');
% set(legend,'Location','northeast');
% hold off;
% title('?C_1 Fixed', 'FontWeight','bold')
% % plot(1:1095, q_sorted(:,1,3));
% % hold on
% % plot(1:1095, q_sorted(:,9000,3));
% % hold on
% % plot(1:1095, nanmean(q_sorted(:,:,3),2));
% % legend('5%' , '90%', 'mean')
% subplot(3,3,6)
% yyaxis left;
% shade_plot(t',q_sorted(:,1,4),q_sorted(:,9000,4),nanmean(q_sorted(:,:,4),2))
% xlim([0 1100])
% ylabel('Monte Carlo Simulated Streamflow (cfs)');
% ax = gca;
% ax.YAxis(1).Color = 'k';
% yyaxis right;
% axis('ij');
% bar(t, Var(:,4))
% ylabel('Variance');
% %ax.YAxis(2).Color = color;
% legend('Confidence Intervals (5% - 95%) (cfs)', 'Streamflow (cfs)', 'Variance');
% set(legend,'Location','northeast');
% hold off;
% title('?P_1 Fixed', 'FontWeight','bold')
% % plot(1:1095, q_sorted(:,1,4));
% % hold on
% % plot(1:1095, q_sorted(:,9000,4));
% % hold on
% % plot(1:1095, nanmean(q_sorted(:,:,4),2));
% % legend('5%' , '90%', 'mean')
% subplot(3,3,7)
% yyaxis left;
% shade_plot(t',q_sorted(:,1,5),q_sorted(:,9000,5),nanmean(q_sorted(:,:,5),2))
% xlim([0 1100])
% ylabel('Monte Carlo Simulated Streamflow (cfs)');
% ax = gca;
% ax.YAxis(1).Color = 'k';
% yyaxis right;
% axis('ij');
% bar(t, Var(:,5))
% ylabel('Variance');
% xlabel('Days');
% %ax.YAxis(2).Color = color;
% legend('Confidence Intervals (5% - 95%) (cfs)', 'Streamflow (cfs)', 'Variance');
% set(legend,'Location','northeast');
% hold off;
% title('?? Fixed', 'FontWeight','bold')
% % plot(1:1095, q_sorted(:,1,5));
% % hold on
% % plot(1:1095, q_sorted(:,9000,5));
% % hold on
% % plot(1:1095, nanmean(q_sorted(:,:,5),2));
% % legend('5%' , '90%', 'mean')
% subplot(3,3,8)
% yyaxis left;
% shade_plot(t',q_sorted(:,1,6),q_sorted(:,9000,6),nanmean(q_sorted(:,:,6),2))
% xlim([0 1100])
% ylabel('Monte Carlo Simulated Streamflow (cfs)');
% ax = gca;
% ax.YAxis(1).Color = 'k';
% yyaxis right;
% axis('ij');
% bar(t, Var(:,6))
% ylabel('Variance');
% xlabel('Days');
% %ax.YAxis(2).Color = color;
% legend('Confidence Intervals (5% - 95%) (cfs)', 'Streamflow (cfs)', 'Variance');
% set(legend,'Location','northeast');
% hold off;
% title('?K_2_Q Fixed', 'FontWeight','bold')
% % plot(1:1095, q_sorted(:,1,6));
% % hold on
% % plot(1:1095, q_sorted(:,9000,6));
% % hold on
% % plot(1:1095, nanmean(q_sorted(:,:,6),2));
% % legend('5%' , '90%', 'mean')
% subplot(3,3,9)
% yyaxis left;
% shade_plot(t',q_sorted(:,1,7),q_sorted(:,9000,7),nanmean(q_sorted(:,:,7),2))
% xlim([0 1100])
% ylabel('Monte Carlo Simulated Streamflow (cfs)');
% ax = gca;
% ax.YAxis(1).Color = 'k';
% yyaxis right;
% axis('ij');
% bar(t, Var(:,7))
% ylabel('Variance');
% xlabel('Days');
% %ax.YAxis(2).Color = color;
% legend('Confidence Intervals (5% - 95%) (cfs)', 'Streamflow (cfs)', 'Variance');
% set(legend,'Location','northeast');
% hold off;
% title('?K_3_Q Fixed', 'FontWeight','bold')
% % plot(1:1095, q_sorted(:,1,7));
% % hold on
% % plot(1:1095, q_sorted(:,9000,7));
% % hold on
% % plot(1:1095, nanmean(q_sorted(:,:,7),2));
% % legend('5%' , '90%', 'mean')
% 
% % c = repmat([1:rnd],rnd,1);
% % c = c/rnd;
% % lr = 0.05*rnd;
% % ur = 0.95*rnd;
% % %plot(mean(q, 2));
% % plot(mean(q, 2));
% % hold on;
% % %plot(str);
% % plot(b(:,lr));
% % plot(b(:,ur));
% % legend ("q_mean", '5', '95')
% % hold off;