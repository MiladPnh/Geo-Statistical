function Z_L=Z_L_cal(A_L,B_L,C)
% z^3-(1-C*B_L)*z^2+(A_L-B_L*(1+C)-B_L^2*(1+2*C))*z-(A_L*B_L-C*(B_L^3+B_L^2));
a=[1 -(1-C*B_L) (A_L-B_L*(1+C)-B_L^2*(1+2*C)) -(A_L*B_L-C*(B_L^3+B_L^2))];
r=roots(a);
num=find(imag(r)==0);
z=r(num);
if min(z)>0
    Z_L=min(z);
elseif min(z)<0
    Z_L=max(z);
end
end
    

