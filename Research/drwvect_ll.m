function oh = drwvect_ll(vfilnam,clr)
% drwvect_ll -- plots coordinates on an existing image assuming
%	it is all ready for the overlay.
%	
%	by DKB 060922

hold on
vec = loadfn(vfilnam);

clrstr=1;
if nargin < 2
	clr = 'y';
else
	if (isstr(clr))
		clrstr=1;
	else
		clrstr=0;
	end
end

%nm = bounds(1);
%tm = bounds(2);
%nx = bounds(3);
%tx = bounds(4);

%ind = find((vec(:,1) >= nm & vec(:,2) >= tm & ...
%	    vec(:,1) <= nx & vec(:,2) <= tx) | isnan(vec(:,1)));
%vec = vec(ind,:);

%dy = tx - tm;
%dx = nx - nm;
%ypp = dy/dim(1);
%xpp = dx/dim(2);

%vec(:,1) = (vec(:,1) - nm)/xpp;
%vec(:,2) = (tx - vec(:,2))/ypp;

set(gca,'ydir','norm')
axis image
hold on

if nargout > 0
	if(clrstr)
		oh = plot(vec(:,1),vec(:,2),clr);
	else
		oh = plot(vec(:,1),vec(:,2));
		set(oh,'color',clr);
	end
else
	if(clrstr)
		plot(vec(:,1),vec(:,2),clr);
	else
		temph = plot(vec(:,1),vec(:,2));
		set(temph,'color',clr);
	end
end
