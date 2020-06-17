function [f_L,f_V]=fugacity(phi_L,phi_V,P,x,y,noc)
for i=1:noc
    f_L(i,1)=x(i)*phi_L(i)*P;
    f_V(i,1)=y(i)*phi_V(i)*P;
end
end