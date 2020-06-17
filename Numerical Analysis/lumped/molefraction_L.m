function x=molefraction_L(z,nv,k,noc)
for i=1:noc
    x(i,1)=z(i)/(1+nv*(k(i)-1));
end
end