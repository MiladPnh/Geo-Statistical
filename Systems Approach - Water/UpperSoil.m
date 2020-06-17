% Importing Mean Daily Data 
clear; clc;
data = load("LeafRiverDaily.txt");
% Creating a for First Year of Data
t = 1:1095;
date = transpose(linspace(datetime(1948,10,1),datetime(1951,9,30),1095));
pcp = data(t,1); %Extractng Precipitation Data
pet = data(t,2); %Extractng Potential Evapotranspiration Data
str = data(t,3); %Extractng Streamflow Data

x1 = zeros(max(t),1);
px1 = zeros(max(t),1);
px2 = zeros(max(t),1);
rg = zeros(max(t),1);
of = zeros(max(t),1);
x2 = zeros(max(t),1);
x3 = zeros(max(t),1);
x4 = zeros(max(t),1);
q = zeros(max(t),1);

tk = 0.5;
tc = 200;
tp = 1;
ta = 0.5;
tg = 0.5;
ts = 0.5;

for t=1:max(t)
[x1(t+1),px(t)] = fun1(pcp(t),pet(t),x1(t),tk,tc,tp);
[rg(t),of(t)] = fun2(px(t),ta);
[bf(t),x2(t+1)] = fun3(rg(t),tg,x2(t));
[sf(t),x3(t+1),x4(t+1)] = fun4(x3(t),x4(t),of(t),ts);
q(t) = sf(t) + bf(t);
end

%Plot Number 1
subplot(4,1,1);
bar(date,pcp,1);
yyaxis left;
ylabel('Precipitation(mm)','FontWeight','bold');
title('Precipitation and Streamflow','FontWeight','bold');
axis('ij');

hold on;
yyaxis right;
plot(date,q);
ylabel('Streamflow (mm)','FontWeight','bold');

scatter(date,str,7.5,'k','filled');
hold off;
legend('Precipitation (mm)', 'Modeled Streamflow (mm)','Observed Streamflow (mm)');
set(legend,'Location','best');

% Upper soil zone
h=subplot(4,1,2);
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[],'xcolor','none','ycolor','none');
axesHandles = findall(h,'type','axes');
pos=get(axesHandles,'position');
pos_1=pos+[0 0 0 -pos(4)/2];
ay1 = axes('Position',pos_1);
 
p1=plot(ay1,period,PP,'b-'); hold on;
set(ay1,'box','off','color','none','xtick',[],'xticklabel',[],'xcolor','none');
ylabel({'Precipiation', '(mm)'});
axis([min(period) max(period) 0 1.2*max(PP)]); % Set the axis range
 
 
pos_2=pos+[0 pos(4)/2 0 -pos(4)/2];
ay2 = axes('Position',pos_2);
 
p2=plot(ay2,period,X1,'k-'); hold on;
set(ay2,'box','off','color','none','xtick',[],'xticklabel',[],'xcolor','none');
ylabel({'Water content', '(mm)'});
axis([min(period) max(period) 0 1.2*max(X1)]); % Set the axis range
 
ay3 = axes('Position',pos);
p3=plot(ay3,period,Y1); 
set(ay3,'box','off','color','none','xcolor',[0 0 0]);
set(ay3,'Ydir', 'reverse');
title('Upper Soil Zone');
ylabel('Output Flux (mm)');
axis([min(period) max(period) 0 8*max(Y1)]); % Set the axis range
set(ay3,'YAxisLocation','right'); % Move Yaxis to right
legend([p1 p2 p3],{'Input Flux (mm)','Water content (mm)','Output Flux (mm)'},'Location','northeast');
text(20*(min(period)+1),1.2*max([max(X1) max(Y1)]),'(b)','FontSize',14);
hold off;