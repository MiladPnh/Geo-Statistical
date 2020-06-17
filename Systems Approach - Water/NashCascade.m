clear; clc; close all;
%%Load Data
load LeafRiverDaily.txt;

fig = figure('Name', 'Figures');

period = 1:365; %Setting a Period
Precip = LeafRiverDaily(period,1);
ET = LeafRiverDaily(period,2);
Flow = LeafRiverDaily(period,3); % Plug all watershed data

Xsens = cell(101,1);
Xsens{1} = Flow;
n = 3;
x0 = 0;
dt = 1;
for K = 0.01:0.01:1
    for i = 1:n
        if i == 1
            U = Precip;
        else
            U = y(:,i-1);
        end    
        [x(:,i), y(:,i)] = ModLLR(U, x0, dt, K, period);
    end
    Xsens{int16(100*K+1),1} = y(:,end);
end
stat = cat(2,Xsens{:,1});


%%Test Score Analysis
for iseris = 50 : 74
    S = allstats(stat(:,1),stat(:,iseris));
    MYSTATS(iseris-48,:) = S(:,2); % We get stats versus referenc
end  %for iserie

MYSTATS(1,:) = S(:,1);

taylordiag(MYSTATS(:,2),MYSTATS(:,3),MYSTATS(:,4));


[x, y] = ModLLR(Precip, x0, dt, 0.14, period);
plot(period, y); hold on;
scatter(period, Flow);
xlim([0, 365]);
legend('Model', 'Obs');