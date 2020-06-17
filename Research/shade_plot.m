function shade_plot=shade_plot(x,y_min,y_max,y_mean)
% all inputs are column vectors (n*1)
x = x(~isnan(y_mean));
y_min = y_min(~isnan(y_min));
y_max = y_max(~isnan(y_max));
y_mean = y_mean(~isnan(y_mean));
X=[x;flip(x)];
Y=[y_min;flip(y_max)];
fill(X,Y,[.65 .65 .65],'LineStyle','none','Facealpha',1);
hold on
plot(x,y_mean,'k','LineWidth',0.1);
xlabel('Lat [\circ]')
ylabel('Correction Percentage [%]')
grid on
end




