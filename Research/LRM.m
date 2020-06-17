load LeafRiverDaily.txt;

Period = 1:365;
Precip = LeafRiverDaily(Period,1);
ET = LeafRiverDaily(Period,2);
Flow = LeafRiverDaily(Period,3);

%%
figure('Name', 'Figures');

subplot(3,1,1)
T = 365;
time = [1:T];

