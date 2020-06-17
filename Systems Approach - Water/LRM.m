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
for K = 0.01:0.01:1
    %K = 0.13; % Setting Time-Invariant Parameters
    x0 = 0;
    dt = 1; % Setting Time Intervals
    [x, y] = ModLLR(Precip, x0, dt, K, period);
    Xsens{int16(100*K+1),1} = y;
end

stat = cat(2,Xsens{:,1});


%%Test Score Analysis
for iserie = fix(size(stat,2)/4)+1 : fix(size(stat,2)/2)-1
    S = allstats(stat(:,1),stat(:,iserie));
    MYSTATS(iserie-24,:) = S(:,2); % We get stats versus referenc
end  %for iserie

MYSTATS(1,:) = S(:,1);

taylordiag(MYSTATS(:,2),MYSTATS(:,3),MYSTATS(:,4));


[x, y] = ModLLR(Precip, x0, dt, 0.14, period);
plot(period, y); hold on;
scatter(period, Flow);
xlim([0, 365]);
legend('Model', 'Obs');