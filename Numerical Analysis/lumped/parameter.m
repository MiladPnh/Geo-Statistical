function [ai,bi,ac,Tr,alpha]=parameter(Pc,Tc,T,omega_a,omega_b,R,m,noc)
for i=1:noc
    Tr(i,1)=T/Tc(i);
    alpha(i,1)=(1+m(i)*(1-(Tr(i)).^.5)).^2;
    ac(i,1)=omega_a.*((R*Tc(i)).^2)/Pc(i);
    ai(i,1)=ac(i).*alpha(i);
    bi(i,1)=omega_b.*(R*Tc(i)/Pc(i));
end
end