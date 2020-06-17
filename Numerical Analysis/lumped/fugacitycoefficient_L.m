function phi_L=fugacitycoefficient_L(bi,b_L,a_L,Z_L,A_L,B_L,delta_1,delta_2,s_L,noc,c,P,R,T)
phi_L=zeros(noc,1);
for i=1:noc
    phi_L(i,1)=exp(bi(i)/b_L*(Z_L-1)-log(Z_L-B_L)-(A_L/(B_L*(delta_2-delta_1))*(2*s_L(i)/a_L-bi(i)/b_L)*log((Z_L+delta_2*B_L)/(Z_L+delta_1*B_L))));

end
end