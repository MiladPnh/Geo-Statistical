function k=wilson(P,T,Pc,Tc,w,noc)
for i=1:noc
    k(i,1)=(Pc(i)/P)*exp(5.37*(1+w(i))*(1-(Tc(i)/T)));
end
end