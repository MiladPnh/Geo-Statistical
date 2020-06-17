function phi_V=fugacitycoefficient_V(bi,b_V,a_V,Z_V,A_V,B_V,delta_1,delta_2,s_V,noc)
for i=1:noc
    phi_V(i,1)=exp(bi(i)/b_V*(Z_V-1)-log(Z_V-B_V)-(A_V/(B_V*(delta_2-delta_1))*(2*s_V(i)/a_V-bi(i)/b_V)*log((Z_V+delta_2*B_V)/(Z_V+delta_1*B_V))));
end
end