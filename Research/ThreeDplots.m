RelDann(:,:,1) = nanmean(FuchsCorrFacsDJFss,3);
RelDann(:,:,2) = nanmean(FuchsCorrFacsMAMss,3);
RelDann(:,:,3) = nanmean(FuchsCorrFacsJJAss,3);
RelDann(:,:,4) = nanmean(FuchsCorrFacsSONss,3);
RelDann1 = nanmean(RelDann,3);

% RelDann(:,:,1) = DJFLeg;
% RelDann(:,:,2) = MAMLeg;
% RelDann(:,:,3) = JJALeg;
% RelDann(:,:,4) = SONLeg;
% RelDann1 = nanmean(RelDann,3);
% RelDann1 = RelDann1.*maskk;

PhaseAnn(:,:,1) = phasepercentageDJFint;PhaseAnn(:,:,2) = phasepercentageMAMint;PhaseAnn(:,:,3) = phasepercentageJJAint;PhaseAnn(:,:,4) = phasepercentageSONint;
PhaseAnn1 = nanmean(PhaseAnn,3);


maskk = RelDann1;maskk(maskk>0)=1;maskk(maskk<=0)=1;
PhaseAnn11 = PhaseAnn1.*maskk;
eraZintt1 = -eraZintt.*maskk;

[Aphase,Iphase] = sort(PhaseAnn11(:));
[Aalt,Ialt] = sort(eraZintt1(:));
[Ard,Ird] = sort(RelDann1(:));

sampleNum = 3200;
size = 259200/3200;

AphaseMapped = sepblockfun(Aphase(:), [sampleNum,1], @nanmean);
AaltMapped = sepblockfun(Aalt(:), [sampleNum,1], @nanmean); 
ArdMapped = sepblockfun(Ard(:), [sampleNum,1], @nanmean); 

realSize = length(ArdMapped(~isnan(ArdMapped)));


whos ArdMapped

latcell = cell(realSize,1);
loncell = cell(realSize,1);

for i = 1:realSize
    loncell{i} = Ialt(sampleNum*(i-1)+1:sampleNum*i);
end
loncellc = cat(2,loncell{:,1});

for i = 1:realSize
    latcell{i} = Iphase(sampleNum*(i-1)+1:sampleNum*i);
end
latcellc = cat(2,latcell{:,1});

G = zeros(sampleNum,realSize);


for i = 1:sampleNum
    for j = 1:realSize
        [row, col] = find(latcellc==loncellc(i,j));
        G(i,j) = col;
    end
end

for i = 1:realSize
    for j = 1:realSize
        GG(i,j).data = [];
    end
end

tmp = RelDann1(:);
for i = 1:sampleNum
    for j = 1:realSize
        GG(G(i,j),j).data = [GG(G(i,j),j).data tmp(loncellc(i,j))];
    end
end

% figure
% [h,ch]=plot3c(eraZintt1(:),phasepercentageDJFint1(:),RelDdjf(:),[-100, -75, -50, -25, 0, 25, 50, 75, 100]);

GGG = zeros(realSize,realSize);
for i = 1:realSize
    for j = 1:realSize
        GGG(i,j) = nanmean(GG(i,j).data);
    end
end

GGGcnt = zeros(realSize,realSize);
for i = 1:realSize
    for j = 1:realSize
        GGGcnt(i,j) = length(GG(i,j).data);
    end
end

xlon = AaltMapped(1:realSize);
ylat = AphaseMapped(1:realSize);

figure
imagesc(xlon, ylat, GGG);
colormap jet
colorbar
set(gca,'Ydir','Normal');

% figure
% imagesc(xlon, ylat, GGGcnt);
% colormap jet
% colorbar
% %set(gca,'Ydir','Normal');