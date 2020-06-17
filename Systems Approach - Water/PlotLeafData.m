load LeafRiverDaily.txt;

Period = 276:640;
Precip = LeafRiverDaily(Period,1);
ET = LeafRiverDaily(Period,2);
Flow = LeafRiverDaily(Period,3);

%%
figure('Name', 'Figures');

subplot(3,1,1)
T = 365;
time = (1:T)';
bar(time,Precip);
axis ij
xlim([0 370])
%title('Leaf River Daily Data')
ylabel('Pr (mm)')
legend Precipitation
%%
subplot(3,1,2)
T = 365;
time = (1:T)';
plot(time,ET, '-o', 'MarkerSize',5);
xlim([0 370])
legend Evapotranspiration
ylabel('ET (mm)')

%%
subplot(3,1,3)
T = 365;
time = (1:T)';
ar = area(time,Flow);
ar.FaceColor = 'c';ar.EdgeColor = 'k';
xlim([0 370])
legend Flowrate
ylabel('Q (mm)')
xlabel('Time (days)')

%%
print(gcf,'10.png','-dpng','-r800');