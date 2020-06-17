function [A_L,A_V,B_L,B_V,a_L,a_V,b_L,b_V]=parameter_mixture(x,y,ai,bi,P,K,R,T,noc)
sum2=0;
for i=1:noc
    sum1=0;
    for j=1:noc
        sum1=x(i)*x(j)*(ai(i)*ai(j))^.5*(1-K(i,j))+sum1;
    end
    sum2=sum1+sum2;
end
a_L=sum2;
sum=0;
for i=1:noc
    sum=x(i).*bi(i)+sum;
end
b_L=sum;
%--------------------------------------------------------------------------
sum22=0;
for i=1:noc
    sum11=0;
    for j=1:noc
        sum11=y(i)*y(j)*(ai(i)*ai(j))^.5*(1-K(i,j))+sum11;
    end
    sum22=sum11+sum22;
end
a_V=sum22;
sum0=0;
for i=1:noc
    sum0=y(i)*bi(i)+sum0;
end
b_V=sum0;
%--------------------------------------------------------------------------
A_L=a_L*P/(R*T)^2;
A_V=a_V*P/(R*T)^2;
B_L=b_L*P/(R*T);
B_V=b_V*P/(R*T);
end
