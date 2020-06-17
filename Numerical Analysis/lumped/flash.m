%----- ( ss method ) ------------------------------------------------------
clear,clc
format shortg
%----read data-------------------------------------------------------------
dd=.1;
hh=.2;
vc=pi*dd^2*hh/4;

noc=3;%tedat mavad 
K=zeros(noc);
R=.000082053;
counter=0;
nv=.5;
equation=2;
P=.1/.101325;%atm
T=40.08+273.15;%k
z=[60 30 10]';
z=z/sum(z);
Pc=[3.381 3.025 2.11 ]'/.101325;%mpa
Tc=[460.43 507.6 617.7]';%k
w=[.2275 .3013 .4923]';
mw=[72.15 86.177 142.285]';
k=wilson(P,T,Pc,Tc,w,noc);
k_wilson=k;
%--------------------------------------------------------------------------
knew=k;
ssss=1;
tic;
while ssss>10^(-9)
    k=knew;
    
    %----mole fractions--------------------------------------------------------
    x=molefraction_L(z,nv,k,noc);
    y=molefraction_V(z,nv,k,noc);
    
    %----equation select-------------------------------------------------------
    [C,delta_1,delta_2,omega_a,omega_b,Zc,m]=equationselect(w,noc,equation,mw);
    %----parameter-------------------------------------------------------------
    [ai,bi,ac,Tr,alpha]=parameter(Pc,Tc,T,omega_a,omega_b,R,m,noc);
    %----parameter_mixture-----------------------------------------------------
    [A_L,A_V,B_L,B_V,a_L,a_V,b_L,b_V]=parameter_mixture(x,y,ai,bi,P,K,R,T,noc);
    %----z calculation---------------------------------------------------------
    Z_L=Z_L_cal(A_L,B_L,C);
    Z_V=Z_V_cal(A_V,B_V,C);
    %----s calculation---------------------------------------------------------
    [s_L,s_V]=s_cal(ai,x,y,K,noc);
    %----fugacity coefficient--------------------------------------------------
    phi_L=fugacitycoefficient_L(bi,b_L,a_L,Z_L,A_L,B_L,delta_1,delta_2,s_L,noc);
    phi_V=fugacitycoefficient_V(bi,b_V,a_V,Z_V,A_V,B_V,delta_1,delta_2,s_V,noc);
    
    %----fugacity--------------------------------------------------------------
    knew=k_cal(phi_L,phi_V,noc);
    [f_L,f_V]=fugacity(phi_L,phi_V,P,x,y,noc);
    %--------------------------------------------------------------------------
    ssss=0;
    for i=1:noc
        ssss=(1-f_L(i)/f_V(i))^2+ssss;
    end
    
    f_nv=0;
    diff_fnv=0;
    for i=1:noc
        f_nv=((z(i)*(k(i)-1))/(1+nv*(k(i)-1)))+f_nv;
        diff_fnv=((-z(i)*(k(i)-1).^2)/((1+nv*(k(i)-1)).^2))+diff_fnv;
    end
    nv_new=nv-f_nv/diff_fnv;
    nv=nv_new;
    
    
    
    counter=counter+1;
end
time=toc;
%----outputs---------------------------------------------------------------

fprintf(' P = %g atm\n T = %g K\n equation = %g (1:PR76 2:PR78 3:SRK 4:SRKG&D)\n nv = %g\n Z_L = %g \n Z_V = %g\n ste1ps = %g\n time = %g\n',P,T,equation,nv,Z_L,Z_V,counter,toc)

vvl=Z_L*R*T/(P);
vvv=Z_V*R*T/(P);
nvk=.8*vc/vvv;
nlk=.2*vc/vvl;
molcol=nvk+nlk;




meyar1=(nv*vvv)/(nv*vvv+(1-nv)*vvl)