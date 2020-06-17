avgFuchsDJF1
avgFuchsMAM1
avgFuchsJJA1
avgFuchsSON1

DJFLeg
MAMLeg
JJALeg
SONLeg

RelDiffDJFs = cell(size(FuchsCorrFacsDJFss,3),1);
for i = 1:size(FuchsCorrFacsDJFss,3)
    RelDiffDJFs{i} = 200*(FuchsCorrFacsDJFss(:,:,i) - DJFLeg)./(FuchsCorrFacsDJFss(:,:,i) + DJFLeg);
end
RelDiffDJFss = cat(3,RelDiffDJFs{:,1});
RelDiffDJF = nanmean(RelDiffDJFss,3);
    
RelDiffMAMs = cell(size(FuchsCorrFacsMAMss,3),1);
for i = 1:size(FuchsCorrFacsDJFss,3)
    RelDiffMAMs{i} = 200*(FuchsCorrFacsMAMss(:,:,i) - MAMLeg)./(FuchsCorrFacsMAMss(:,:,i) + MAMLeg);
end
RelDiffMAMss = cat(3,RelDiffMAMs{:,1});
RelDiffMAM = nanmean(RelDiffMAMss,3);

    

RelDiffJJAs = cell(size(FuchsCorrFacsJJAss,3),1);
for i = 1:size(FuchsCorrFacsDJFss,3)
    RelDiffJJAs{i} = 200*(FuchsCorrFacsJJAss(:,:,i) - JJALeg)./(FuchsCorrFacsJJAss(:,:,i) + JJALeg);
end
RelDiffJJAss = cat(3,RelDiffJJAs{:,1});
RelDiffJJA = nanmean(RelDiffJJAss,3);



RelDiffSONs = cell(size(FuchsCorrFacsSONss,3),1);
for i = 1:size(FuchsCorrFacsDJFss,3)
    RelDiffSONs{i} = 200*(FuchsCorrFacsSONss(:,:,i) - SONLeg)./(FuchsCorrFacsSONss(:,:,i) + SONLeg);
end
RelDiffSONss = cat(3,RelDiffSONs{:,1});
RelDiffSON = nanmean(RelDiffSONss,3);



RelDiffDJF = 200*(avgFuchsDJF - DJFLeg)./(avgFuchsDJF + DJFLeg);
RelDiffMAM = 200*(avgFuchsMAM - MAMLeg)./(avgFuchsMAM + MAMLeg);
RelDiffJJA = 200*(avgFuchsJJA - JJALeg)./(avgFuchsJJA + JJALeg);
RelDiffSON = 200*(avgFuchsSON - SONLeg)./(avgFuchsSON + SONLeg);

[h,ch]=plot3c(WindEra5InttSONN(:),TEra5InttSONN(:),RelDiffSON(:),10)
[h,ch]=plot3c(WindEra5InttSONN(:),RhSON(:),RelDiffSON(:),10)
[h,ch]=plot3c(RhSON(:),TEra5InttSONN(:),RelDiffSON(:),10)


ChelsaCF_DJF_int
ChelsaCF_MAM_int
ChelsaCF_JJA_int
ChelsaCF_SON_int
ChpCF_DJF_int
ChpCF_MAM_int
ChpCF_JJA_int
ChpCF_SON_int 
WcCF_DJF_int 
WcCF_MAM_int
WcCF_JJA_int
WcCF_SON_int


GPCC_CF_EnsBeck_DJF 
GPCC_CF_EnsBeck_MAM
GPCC_CF_EnsBeck_JJA 
GPCC_CF_EnsBeck_SON


ChelsaCF
ChpCF
WcCF


GPCCrawDJF1
GPCCrawMAM1
GPCCrawJJA1
GPCCrawSON1

GPCCrawDJF_L
GPCCrawMAM_L
GPCCrawJJA_L
GPCCrawSON_L

GPCCrawDJF_F
GPCCrawMAM_F
GPCCrawJJA_F
GPCCrawSON_F

CHELSA_V12_Corr_DJF_int
CHELSA_V12_Corr_MAM_int
CHELSA_V12_Corr_JJA_int
CHELSA_V12_Corr_SON_int
CHPclim_V1_Corr_DJF_int 
CHPclim_V1_Corr_MAM_int 
CHPclim_V1_Corr_JJA_int 
CHPclim_V1_Corr_SON_int 
WorldClim_V2_Corr_DJF_int
WorldClim_V2_Corr_MAM_int 
WorldClim_V2_Corr_JJA_int
WorldClim_V2_Corr_SON_int

EnsembleBeck_DJF
EnsembleBeck_MAM
EnsembleBeck_JJA
EnsembleBeck_SON


nanmean(GPCCrawDJF1,3)
nanmean(GPCCrawMAM1,3)
nanmean(GPCCrawJJA1,3)
nanmean(GPCCrawSON1,3)



