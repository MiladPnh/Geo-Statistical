function [s_L,s_V]=s_cal(ai,x,y,K,noc)
for i=1:noc
    pp=0;
    for j=1:noc
        pp=x(j)*(1-K(i,j))*(ai(j)*ai(i)).^.5+pp;
    end
    s_L(i,1)=pp;
end
for i=1:noc
    pp=0;
    for j=1:noc
        pp=y(j)*(1-K(i,j))*(ai(j)*ai(i)).^.5+pp;
    end
    s_V(i,1)=pp;
end
end