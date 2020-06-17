function x=masstomole(w,M)
x=zeros(3,1);
t=w(1)/M(1)+w(2)/M(2)+w(3)/M(3);
for i=1:3
x(i)=w(i)/M(i)/t;
end
end