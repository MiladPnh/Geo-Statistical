function k=k_cal(phi_L,phi_V,noc)
for i=1:noc
    k(i,1)=phi_L(i)/phi_V(i);
end