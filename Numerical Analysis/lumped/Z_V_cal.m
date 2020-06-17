function Z_V=Z_V_cal(A_V,B_V,C)
% z^3-(1-C*B_V)*z^2+(A_V-B_V*(1+C)-B_V^2*(1+2*C))*z-(A_V*B_V-C*(B_V^3+B_V^2))
a=[1 -(1-C*B_V) (A_V-B_V*(1+C)-B_V^2*(1+2*C)) -(A_V*B_V-C*(B_V^3+B_V^2))];
r=roots(a);
n=find(imag(r)==0);
z = r(n);
Z_V=max(z);
end

