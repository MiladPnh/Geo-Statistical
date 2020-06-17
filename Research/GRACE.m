%fn='/data/home/abehrang/A/dataset/ERA_I/tp_1979-2000.nc'
 %ncdisp('C:\Users\ali\Documents\JPL_Laptop\2018_paper\Milad\GRACE_ROWDATA\GRCTellus.JPL.200204_201706.GLO.RL06M.MSCNv01CRIv01.nc');
 
 %%% MASCON
 fn='C:\Users\miladpanahi\Desktop\Master\Paper\GRACE\GRCTellus.JPL.200204_201706.GLO.RL06M.MSCNv01CRIv01.nc';
 time =ncread(fn,'time');
 lwe=ncread(fn,'lwe_thickness')*10;
 GraceMc = flipud(permute(lwe,[2 1 3]));
 t1=GraceMc(:,1:360,:);t2=GraceMc(:,361:720,:);GraceMscn=[t2 t1];
 GraceMscn = GraceMscn(81:132,111:226,:);  % mm
 unc=ncread(fn,'uncertainty');
 Unc1 = flipud(permute(unc,[2 1 3]));
 t1=Unc1(:,1:360,:);t2=Unc1(:,361:720,:);Unc=[t2 t1];
 
 lonGRACEM=ncread(fn,'lon');lonGRACEM = lonGRACEM-180; lonGRACEM = lonGRACEM(111:226);
 latGRACEM=flipud(ncread(fn,'lat')); latGRACEM = latGRACEM(81:132);
 
Time_N = datevec(datenum(double(time)+double(datenum('01-Jan-2002'))));
 
 %f=find(Time_N(:,1)==2016);Time_N(f,:)
% figure(10); imagesc(lonGRACEM,latGRACEM,GraceMscn(:,:,150)-GraceMscn(:,:,149)); caxis([-20 40]);colorbar;colormap(jet);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
% set(gca,'YDIR','Normal'); 

GraceTwsaMasFeb15Jan22 = GraceMscn(:,:,142)-GraceMscn(:,:,141);
GraceTwsaMasFeb15Mar16 = GraceMscn(:,:,143)-GraceMscn(:,:,142);
GraceTwsaMasFeb15Mar16 = GraceMscn(:,:,115)-GraceMscn(:,:,114);
GraceTwsaMasFeb15Mar16 = GraceMscn(:,:,93)-GraceMscn(:,:,92);


r = GraceMscn(:,:,103);


[llat, llon] = meshgrid(latGRACEM,lonGRACEM);llat = llat';llon = llon';
w=griddata(llat(:),llon(:),r(:),latM ,lonM);
w1 = w.*PrCrctF4; w1 = w1.*mask4MM;
w11 = nanmean(nanmean(w1));

ww=griddata(llat(:),llon(:),GraceTwsaMasFeb15Mar16(:),latM ,lonM);
www=griddata(llat(:),llon(:),GraceTwsaMasJan9Dec16(:),latM ,lonM);
wwww=griddata(llat(:),llon(:),GraceTwsaMasFeb15Mar16(:),latM ,lonM);

w1 = w.*PrCrctF1;w1 = w1.*mask1MM;w1=nanmean(nanmean(w1));
ww1 = ww.*PrCrctF1;ww1 = ww1.*mask1MM;ww1=nanmean(nanmean(ww1));
www1 = www.*PrCrctF1;www1 = www1.*mask1MM;www1=nanmean(nanmean(www1));
wwww1 = wwww.*PrCrctF1;wwww1 = wwww1.*mask1MM;wwww1=nanmean(nanmean(wwww1));

w2 = w.*PrCrctF2;w2 = w2.*mask2MM;w2=nanmean(nanmean(w2));
ww2 = ww.*PrCrctF2;ww2 = ww2.*mask2MM;ww2=nanmean(nanmean(ww2));
www2 = www.*PrCrctF2;www2 = www2.*mask2MM;www2=nanmean(nanmean(www2));
wwww2 = wwww.*PrCrctF2;wwww2 = wwww2.*mask2MM;wwww2=nanmean(nanmean(wwww2));


w3 = w.*PrCrctF3;w3 = w3.*mask3MM;w3=nanmean(nanmean(w3));
ww3 = ww.*PrCrctF3;ww3 = ww3.*mask3MM;ww3=nanmean(nanmean(ww3));
www3 = www.*PrCrctF3;www3 = www3.*mask3MM;www3=nanmean(nanmean(www3));
wwww3 = wwww.*PrCrctF3;wwww3 = wwww3.*mask3MM;wwww3=nanmean(nanmean(wwww3));


w4 = w.*PrCrctF4;w4 = w4.*mask4MM;w4=nanmean(nanmean(w4));
ww4 = ww.*PrCrctF4;ww4 = ww4.*mask4MM;ww4=nanmean(nanmean(ww4));
www4 = www.*PrCrctF4;www4 = www4.*mask4MM;www4=nanmean(nanmean(www4));
wwww4 = wwww.*PrCrctF4;wwww4 = wwww4.*mask4MM;wwww4=nanmean(nanmean(wwww4));

w5 = w.*PrCrctF5;w5 = w5.*mask5MM;w5=nanmean(nanmean(w5));
ww5 = ww.*PrCrctF5;ww5 = ww5.*mask5MM;ww5=nanmean(nanmean(ww5));
www5 = www.*PrCrctF5;www5 = www5.*mask5MM;www5=nanmean(nanmean(www5));
wwww5 = wwww.*PrCrctF5;wwww5 = wwww5.*mask5MM;wwww5=nanmean(nanmean(wwww5));

w6 = w.*PrCrctF6;w6 = w6.*mask6MM;w6=nanmean(nanmean(w6));
ww6 = ww.*PrCrctF6;ww6 = ww6.*mask6MM;ww6=nanmean(nanmean(ww6));
www6 = www.*PrCrctF6;www6 = www6.*mask6MM;www6=nanmean(nanmean(www6));
wwww6 = wwww.*PrCrctF6;wwww6 = wwww6.*mask6MM;wwww6=nanmean(nanmean(wwww6));


% SPHERICAL HARMONIC
fn2='C:\Users\miladpanahi\Desktop\Master\Paper\GRACE\CLM4.SCALE_FACTOR.DS.G300KM.RL05.DSTvSCS1409.nc';
fn3='C:\Users\miladpanahi\Desktop\Master\Paper\GRACE\GRCTellus.JPL.200204_201701.LND.RL05_1.DSTvSCS1411.nc';
timev5=ncread(fn3,'time');
lwe5=ncread(fn3,'lwe_thickness')*10; % mm,
sc=ncread(fn2,'SCALE_FACTOR');lwe5 = lwe5.*sc;
Time_Nv5=datevec(double(datenum(double(timev5)+double(datenum('01-Jan-2002')))));
GraceH = flipud(permute(lwe5,[2 1 3]));
t1=GraceH(:,1:180,:);t2=GraceH(:,181:360,:);GraceHar=[t2 t1];GraceHar = GraceHar(42:66,56:113,:);  % mm
lonGRACEH=ncread(fn3,'lon');lonGRACEH = lonGRACEH-180; lonGRACEH = lonGRACEH(56:113);
latGRACEH=flipud(ncread(fn3,'lat')); latGRACEH = latGRACEH(42:66);

% figure(10); imagesc(lonGRACEH,latGRACEH,GraceHar(:,:,150)-GraceHar(:,:,149)); caxis([-20 40]);colorbar;colormap(jet);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
% set(gca,'YDIR','Normal');

% f=find(Time_N(:,1)==2016);Time_N(f,:)
% f5=find(Time_Nv5(:,1)==2015);Time_Nv5(f5,:)

GraceTwsaHarDec23Jan17 = GraceHar(:,:,150)-GraceHar(:,:,149);
GraceTwsaHarJan17Feb14 = GraceHar(:,:,151)-GraceHar(:,:,150);

[llat, llon] = meshgrid(latGRACEH,lonGRACEH);llat = llat';llon = llon';
GraceHD23J17=griddata(llat(:),llon(:),GraceTwsaHarDec23Jan17(:),latM ,lonM);
GraceHJ17F14=griddata(llat(:),llon(:),GraceTwsaHarJan17Feb14(:),latM ,lonM);
GraceH1D23J17 = GraceHD23J17.*PrCrctF1;GraceH1D23J17 = GraceH1D23J17.*mask1MM;GraceH1D23J17=nanmean(nanmean(GraceH1D23J17));
GraceH1J17F14 = GraceHJ17F14.*PrCrctF1;GraceH1J17F14 = GraceH1J17F14.*mask1MM;GraceH1J17F14=nanmean(nanmean(GraceH1J17F14));
GraceH2D23J17 = GraceHD23J17.*PrCrctF2;GraceH2D23J17 = GraceH2D23J17.*mask2MM;GraceH2D23J17=nanmean(nanmean(GraceH2D23J17));
GraceH2J17F14 = GraceHJ17F14.*PrCrctF2;GraceH2J17F14 = GraceH2J17F14.*mask2MM;GraceH2J17F14=nanmean(nanmean(GraceH2J17F14));
GraceH3D23J17 = GraceHD23J17.*PrCrctF3;GraceH3D23J17 = GraceH3D23J17.*mask3MM;GraceH3D23J17=nanmean(nanmean(GraceH3D23J17));
GraceH3J17F14 = GraceHJ17F14.*PrCrctF3;GraceH3J17F14 = GraceH3J17F14.*mask3MM;GraceH3J17F14=nanmean(nanmean(GraceH3J17F14));
GraceH4D23J17 = GraceHD23J17.*PrCrctF4;GraceH4D23J17 = GraceH4D23J17.*mask4MM;GraceH4D23J17=nanmean(nanmean(GraceH4D23J17));
GraceH4J17F14 = GraceHJ17F14.*PrCrctF4;GraceH4J17F14 = GraceH4J17F14.*mask4MM;GraceH4J17F14=nanmean(nanmean(GraceH4J17F14));
GraceH5D23J17 = GraceHD23J17.*PrCrctF5;GraceH5D23J17 = GraceH5D23J17.*mask5MM;GraceH5D23J17=nanmean(nanmean(GraceH5D23J17));
GraceH5J17F14 = GraceHJ17F14.*PrCrctF5;GraceH5J17F14 = GraceH5J17F14.*mask5MM;GraceH5J17F14=nanmean(nanmean(GraceH5J17F14));
GraceH6D23J17 = GraceHD23J17.*PrCrctF6;GraceH6D23J17 = GraceH6D23J17.*mask6MM;GraceH6D23J17=nanmean(nanmean(GraceH6D23J17));
GraceH6J17F14 = GraceHJ17F14.*PrCrctF6;GraceH6J17F14 = GraceH6J17F14.*mask6MM;GraceH6J17F14=nanmean(nanmean(GraceH6J17F14));