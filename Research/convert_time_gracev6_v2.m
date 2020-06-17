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
 
%  Time_N = datevec(datenum(double(time)+double(datenum('01-Jan-2002'))));
 
 %f=find(Time_N(:,1)==2016);Time_N(f,:)
% figure(10); imagesc(lonGRACEM,latGRACEM,GraceMscn(:,:,150)-GraceMscn(:,:,149)); caxis([-20 40]);colorbar;colormap(jet);
% states = shaperead('usastatehi', 'UseGeoCoords', true);
% geoshow(states,'FaceColor',[1,1,1],'facealpha',0,'DefaultEdgeColor','black');
% set(gca,'YDIR','Normal'); 

GraceTwsaDec23Jan16 = GraceMscn(:,:,150)-GraceMscn(:,:,149);
GraceTwsaJan16Feb14 = GraceMscn(:,:,151)-GraceMscn(:,:,150);

[llat, llon] = meshgrid(latGRACEM,lonGRACEM);llat = llat';llon = llon';
GraceMD23J16=griddata(llat(:),llon(:),GraceTwsaDec23Jan16(:),latM ,lonM);
GraceMJ16F14=griddata(llat(:),llon(:),GraceTwsaJan16Feb14(:),latM ,lonM);
GraceM1D23J16 = GraceMD23J16.*PrCrctF1;GraceM1D23J16 = GraceM1D23J16.*mask1MM;GraceM1D23J16=nanmean(nanmean(GraceM1D23J16));
GraceM1J16F14 = GraceMJ16F14.*PrCrctF1;GraceM1J16F14 = GraceM1J16F14.*mask1MM;GraceM1J16F14=nanmean(nanmean(GraceM1J16F14));
GraceM2D23J16 = GraceMD23J16.*PrCrctF2;GraceM2D23J16 = GraceM2D23J16.*mask2MM;GraceM2D23J16=nanmean(nanmean(GraceM2D23J16));
GraceM2J16F14 = GraceMJ16F14.*PrCrctF2;GraceM2J16F14 = GraceM2J16F14.*mask2MM;GraceM2J16F14=nanmean(nanmean(GraceM2J16F14));
GraceM3D23J16 = GraceMD23J16.*PrCrctF3;GraceM3D23J16 = GraceM3D23J16.*mask3MM;GraceM3D23J16=nanmean(nanmean(GraceM3D23J16));
GraceM3J16F14 = GraceMJ16F14.*PrCrctF3;GraceM3J16F14 = GraceM3J16F14.*mask3MM;GraceM3J16F14=nanmean(nanmean(GraceM3J16F14));
GraceM4D23J16 = GraceMD23J16.*PrCrctF4;GraceM4D23J16 = GraceM4D23J16.*mask4MM;GraceM4D23J16=nanmean(nanmean(GraceM4D23J16));
GraceM4J16F14 = GraceMJ16F14.*PrCrctF4;GraceM4J16F14 = GraceM4J16F14.*mask4MM;GraceM4J16F14=nanmean(nanmean(GraceM4J16F14));
GraceM5D23J16 = GraceMD23J16.*PrCrctF5;GraceM5D23J16 = GraceM5D23J16.*mask5MM;GraceM5D23J16=nanmean(nanmean(GraceM5D23J16));
GraceM5J16F14 = GraceMJ16F14.*PrCrctF5;GraceM5J16F14 = GraceM5J16F14.*mask5MM;GraceM5J16F14=nanmean(nanmean(GraceM5J16F14));
GraceM6D23J16 = GraceMD23J16.*PrCrctF6;GraceM6D23J16 = GraceM6D23J16.*mask6MM;GraceM6D23J16=nanmean(nanmean(GraceM6D23J16));
GraceM6J16F14 = GraceMJ16F14.*PrCrctF6;GraceM6J16F14 = GraceM6J16F14.*mask6MM;GraceM6J16F14=nanmean(nanmean(GraceM6J16F14));








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

GraceTwsaDec23Jan17 = GraceHar(:,:,150)-GraceHar(:,:,149);
GraceTwsaJan17Feb14 = GraceHar(:,:,151)-GraceHar(:,:,150);

[llat, llon] = meshgrid(latGRACEH,lonGRACEH);llat = llat';llon = llon';
GraceHD23J17=griddata(llat(:),llon(:),GraceTwsaDec23Jan17(:),latM ,lonM);
GraceHJ17F14=griddata(llat(:),llon(:),GraceTwsaJan17Feb14(:),latM ,lonM);
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