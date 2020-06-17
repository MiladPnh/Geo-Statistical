function y=molefraction_V(z,nv,k,noc)
for i=1:noc
    y(i,1)=(z(i)*k(i))/(1+nv*(k(i)-1));
end
end