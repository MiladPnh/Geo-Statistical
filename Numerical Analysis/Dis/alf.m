function alfa=alf(T)
alfa=zeros(3,1);
ww=[0.251 0.296 0.49]';
Tc=[469.6 507.4 617]';
kk=0.37464*[1 1 1]'+1.54226*ww+0.26992*ww.*ww;
for i=1:3
    alfa(i)=(1+kk(i)*(1-(T/Tc(i))^0.5))^2;
end
end