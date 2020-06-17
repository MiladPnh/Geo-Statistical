clear; close all; clc;
fid = fopen('N80W091.hgt','r');
SRTM1 = fread(fid,[1201,inf],'int16','b');
fclose(fid);
fid = fopen('N80W092.hgt','r');
SRTM2 = fread(fid,[1201,inf],'int16','b');
fclose(fid);

SRTM1 = SRTM1'; SRTM1 = flipud(SRTM1);
SRTM2 = SRTM2'; SRTM2 = flipud(SRTM2);
SRTM1(find(SRTM1 == -32768)) = NaN;
SRTM2(find(SRTM2 == -32768)) = NaN;

SRTM1(:,1201) = [];
SRTM = horzcat(SRTM1,SRTM2);
clear SRTM1 SRTM2

for i = 2 : 1200
    for j = 2 : 2400
        if isnan(SRTM(i,j)) == 1
            SRTM(i,j) = nanmean(nanmean(...
            SRTM(i-1:i+1,j-1:j+1)));
        end
    end
end
clear i j

B = 1/25 * ones(5,5);
SRTM = filter2(B,SRTM);


[LON,LAT] = meshgrid(91:1/1200:93,79:1/1200:80);
v = [700 800 900 1000 1100 1200 ...
     1300 1500 2000 2500 3000];
 
 figure1 = figure('Color',[1 1 1],...
    'Position',[50 50 1200 600]);
axes1 = axes('Visible','off',...
    'Units','centimeters',...
    'FontSize',8);
hold(axes1,'all');
surf1 = surf(LON,LAT,SRTM,...
    'SpecularExponent',20,...
    'FaceLighting','phong',...
    'FaceColor','interp',...
    'EdgeColor','none');
light1 = light('Parent',axes1,...
    'Style','local',...
    'Position',[145 70 900000]);
set(gca,'View',[25 20])
daspect([1 1 20000])
colormap(flipud(hsv))
contour3(LON,LAT,SRTM,v,...
    'Color','w')