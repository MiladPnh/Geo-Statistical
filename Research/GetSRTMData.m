function out = GetSRTMData(p1, p2)
% function out = GetSRTMData(p1, [p2])
%
% General:
% The function retrieves SRTM-Data from the according SRTM-files (3 arcsecond resolution only !),
% which need to be supplied by the user. The path to the unzipped SRTM-files 
% (.hgt-format) needs to be specified in the variable 'path'.
% SRTM-files can be downloaded for free -> ftp://e0dps01u.ecs.nasa.gov/srtm/
% Missing data can be filled with the free tool SRTM-Fill  -> http://www.3dnature.com/srtmfill.html
%
% Input Argument(s):
% p1/2 need to be specified as decimal degree coordinates, 
% negative values for western longitudes resp. southern latitudes.
% p1  			-  lon/lat point or lon/lat point-list
% p2(optional)  -  lon/lat point or lon/lat point-list (list of same size as p1)
%
% Output Argument:
% out - structure containing Point(s)/profiles with [lon / lat / height].
% ...
% The following height data is retrived from the SRTM-files:
% p1 point or list / p2 empty: -> For specified point(s) 							-> Example1
% p1 point / p2 point: -> For points on line p1 <-> p2 								-> Example2
% p1 list / p2 list: -> For points on line p1(1) <-> p2(1) ... p1(n) <-> p2(n)		-> Example3
% 
% Examples:
% 1: out = GetSRTMData([100.5 40.1234]);
% 2: out = GetSRTMData([100.5 40.1234],[100.6 41.4321]);
% 3: out = GetSRTMData([100 40; 101 41];[100.1 40.8; 101.1 41.8]);
%
% Version 0.9:  Just finished it.
% Version 0.91: Added switch for files with southern latitudes / western longitudes.
% Version 1.0: Corrected bug mentioned by John Tanner for lat / lon <10° (3.12.2004).
%
% Author
% Sebastian Hölz, TU Berlin, Germany
% hoelz@geophysik.tu-berlin.de
% If you find any bugs, let me know ...


if nargin ==1; p2=[]; end

% 1. Checking for right dimension (nx2)
if size(p1,2) ~= 2; p1=p1'; end
if size(p2,2) ~= 2; p2=p2'; end
if nargin == 2 & size(p1) ~= size(p2)
    error('For height profiles the point lists need to have the same size')
    return
end

% 1. Loading SRTM-files
[SRTM, lon, lat] = LoadSRTM(p1,p2);

% 2. Generating the profile indizes
if isempty(p2)
    out = {GetIndizes(lon, lat, p1, p2)};
else
    for i = 1:size(p1,1)
        out{i} = GetIndizes(lon, lat, p1(i,:), p2(i,:));
    end
end

% 3. Extracting Altitude data
for i = 1:length(out)
    ind = (out{i}(:,1)-1)*size(SRTM,1)+out{i}(:,2);    
    out{i}(:,3)=SRTM(ind);
    out{i}(:,1)=(out{i}(:,1)-1)/1200+lon(1);
    out{i}(:,2)=(out{i}(:,2)-1)/1200+lat(1);
end

if length(out)==1; out=out{1}; end

% -----------------------------------------
function [SRTM, lon, lat] = LoadSRTM(p1,p2)

SRTM=[];
SRTM_tmp=[];
path = 'C:\Dokumente und Einstellungen\Sebi\Desktop\SRTM\';
tmp = [p1; p2];

lon(1) = floor(min(tmp(:,1)));
lon(2) = ceil(max(tmp(:,1)));
lat(1) = floor(min(tmp(:,2)));
lat(2) = ceil(max(tmp(:,2)));
if lon(2)==lon(1); lon(2)=lon(2)+1; end
if lat(2)==lat(1); lat(2)=lat(2)+1; end

for x = lon(1):lon(2)-1
    for y = lat(1):lat(2)-1
        if x<=-100; LON = 'W'
        elseif x>-100 & x<=-10; LON = 'W0'; 
        elseif x>-10 & x<0; LON = 'W00';
        elseif x>=0 & x<10; LON = 'E00';
        elseif x>10 & x<100; LON = 'E0';
		else; LON ='E'; 
		end
        if y<=-10; LAT = 'S'; 
        elseif y>-10 & y<0; LAT = 'S0';
        elseif y>=0 & y<10; LAT = 'N0'; 
        else; LAT = 'N';
        end
        file=[path LAT num2str(abs(y)) LON num2str(abs(x)) '.hgt'];
        fid=fopen(file,'r','b');
        if fid == -1; error([file ': not found !!!']); return; end
        
        if isempty(SRTM_tmp)
            SRTM_tmp = rot90(fread(fid,[1201 1201],'*uint16'));
        else
            SRTM_tmp = [SRTM_tmp(1:end-1,:); rot90(fread(fid,[1201 1201],'*uint16'))];
        end
        fclose(fid);
    end
    if isempty(SRTM)
        SRTM = SRTM_tmp;
    else
        SRTM = [SRTM(:,1:end-1) SRTM_tmp];
    end
    SRTM_tmp = [];
end

% ------------------------------
function out = GetIndizes(lon, lat, p1, p2)

if isempty(p2)                                  % Only single point or point list needed
    out = zeros(size(p1,1),3);
    out(:,1)=round((p1(:,1)-lon(1))*1200)+1;
    out(:,2)=round((p1(:,2)-lat(1))*1200)+1;
else                                            % Profile between two points needed
    indx = [round((p1(1)-lon(1))*1200)+1 round((p2(1)-lon(1))*1200)+1];
    indy = [round((p1(2)-lat(1))*1200)+1 round((p2(2)-lat(1))*1200)+1];
    diffx = diff(indx);
    diffy = diff(indy);
    len = max(abs([diffx diffy]))+1;
    
    out = zeros(len,2);
    if indx(1)==indx(2)
        out(:,1)=indx(1);
    else
        out(:,1)=round(indx(1):diffx/(len-1):indx(2))';
    end
    if indy(1)==indy(2)
        out(:,2)=indy(1);
    else
        out(:,2)=round(indy(1):diffy/(len-1):indy(2))';
    end
end




