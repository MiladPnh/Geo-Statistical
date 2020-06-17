% movazene koli : deltat(q2*Ab-h*As*(T-Tinf)=deltaU
function Y=kolmoaedle(x,C,nl1,ng1,hrl1,hrg1,nt,P1,T1,xx,yy)











R=8314;
d=.1;
h=.2;
Vc=pi*d^2/4*h;
% y(1)=(nl2*hrl2)+(ng2*hrg2)-(nl1*hrl1)+(ng1*hrg1)+f1-V*(P2-P1)-C;

Y(1)=x(1)*x(2)+x(3)*x(4)-(nl1*hrl1)+(ng1*hrg1)+nt*(8.5284*(x(5)-T1))+7.255*10^(-5)*(x(5)^2-T1^2)...
    +4.57*10^-5*(x(5)^3-T1^3)-4.3763*10^(-8)*(x(5)^4-T1^4)+1.3548*10^-11*((x(5)^5-T1^5))-Vc*(x(6)-P1)-C;
%-----------------------------------------------------------------------
zigg=0;
w=[.2275 .3013 .4923];
m=.480+1.574.*w-0.176.*w.^2;
Tc=[460.43 507.6 617.7];%k
Pc=[3.381 3.025 2.11]*10;%pa
mw=[72.15 86.177 142.285];

Tr=x(5)./Tc;
alfa=(1+m.*(1-sqrt(Tr))).^2;
ac=0.457235*R^2.*Tc.^2./Pc;
a=ac.*alfa;
for i=1:3
    for j=1:3
        
        zigg=zigg+x(7+i)*x(7+j)*sqrt(a(i)*a(j))*(1/2*(-m(i)*sqrt(Tr(i)/alfa(i)))-1/2*(-m(j)*(Tr(i)/alfa(j))));
    end
end

b=.08664*R.*Tc./Pc;
bg=x(8)*b(1)+x(9)*b(2)+x(10)*b(3);
bl=x(12)*b(1)+x(13)*b(2)+x(14)*b(3);
Bg=(x(6).*bg)/R*x(5);
Bl=(x(6).*bl)/R*x(5);
    %moadele 2

Y(2)=x(4)-R*x(5)*(x(7)-1)-(zigg/(2*sqrt(2).*bg)).*real(log((x(7)+2.414*Bg)./(x(7)-.414*Bg)));
zigl=0;
for i=1:3
    for j=1:3
        
        zigl=zigl+x(11+i)*x(11+j)*sqrt(a(i)*a(j))*(1/2*(-m(i)*sqrt(Tr(i)/alfa(i)))-1/2*(-m(j)*(Tr(i)/alfa(j))));
    end
end
%moadele 3


Y(3)=x(2)-R*x(5)*(x(11)-1)-(zigl/(2*sqrt(2).*bl)).*real(log((x(11)+2.414*Bl)./(x(11)-.414*Bl)));
al=0;
ag=0;

for i=1:3
    for j=1:3
        al=al+x(i+11)*x(j+11)*sqrt(a(i)*a(j));
        ag=ag+x(i+7)*x(j+7)*sqrt(a(i)*a(j));
    end
end

Al=(x(6).*al)/(R*x(5))^2;
Ag=(x(6).*ag)/(R*x(5))^2;
        


Y(4)=x(11)^3-(1-Bl)*x(11)^2+(Al-3*Bl^2-2*Bl^2).*x(11)-(Al.*Bl-Bl.^2-Bl.^3);
Y(5)=x(7)^3-(1-Bg)*x(7)^2+(Ag-3*Bg^2-2*Bg^2).*x(7)-(Ag.*Bg-Bg.^2-Bg.^3);

slu1=0;
slu2=0;
slu3=0;
ssl1=0;
sgu1=0;
sgu2=0;
sgu3=0;
ssg1=0;
for i=1:3
  slu1=slu1+x(11+i)*sqrt(a(1)*a(j));
  slu2=slu2+x(11+i)*sqrt(a(2)*a(j));
  slu3=slu3+x(11+i)*sqrt(a(3)*a(j));
  
  ssl1=ssl1+x(11+1)*a(i);
   
   sgu1=sgu1+x(7+i)*sqrt(a(1)*a(j));
   sgu2=sgu2+x(7+i)*sqrt(a(2)*a(j));
   sgu3=sgu3+x(7+i)*sqrt(a(3)*a(j));

   ssg1=ssg1+x(7+1)*a(i);

end
Y(6)=real(log(x(12)))+(b(1)/bl*(x(11)-1)-real(log(x(11)-Bl))-Al/(2*sqrt(2)*Bl)*(slu1/ssl1-b(1)/bl)-real(log(((x(11)+2.414*Bl)/(x(11)-.414*Bl))))...
    -real(log(x(8)))- b(1)/bg*(x(7)-1)-real(log(x(7)-Bg))-Ag/(2*sqrt(2)*Bg)*(sgu1/ssg1-b(1)/bg)-real(log((x(7)+2.414*Bg)/(x(7)-.414*Bg)))) ;

Y(7)=real(log(x(13)))+b(2)/bl*(x(11)-1)-real(log(x(11)-Bl))-Al/(2*sqrt(2)*Bl)*(slu2/ssl1-b(2)/bl)-real(log((x(11)+2.414*Bl)/(x(11)-.414*Bl)))...
    -real(log(x(9)))- b(2)/bg*(x(7)-1)-real(log(x(7)-Bg))-Ag/(2*sqrt(2)*Bg)*(sgu2/ssg1-b(2)/bg)-real(log((x(7)+2.414*Bg)/(x(7)-.414*Bg))) ;

Y(8)=real(log(x(14)))+b(3)/bl*(x(11)-1)-real(log(x(11)-Bl))-Al/(2*sqrt(2)*Bl)*(slu3/ssl1-b(3)/bl)-real(log((x(11)+2.414*Bl)/(x(11)-.414*Bl)))...
    -real(log(x(10)))-b(3)/bg*(x(7)-1)-real(log(x(7)-Bg))-Ag/(2*sqrt(2)*Bg)*(sgu3/ssg1-b(3)/bg)-real(log((x(7)+2.414*Bg)/(x(7)-.414*Bg))) ;


Y(9)=x(12)+x(13)+x(14)-1;
Y(10)=x(8)+x(9)+x(10)-1;
Y(11)=x(11)-(x(6)*x(15))/(R*x(5));
Y(12)=x(7)-(x(6)*x(16))/(R*x(5));
Y(13)=x(1)*x(15)+x(3)*x(16)-Vc;
Y(14)=x(8)*x(3)+x(12)*x(1)-yy(1)*ng1-xx(1)*nl1;
Y(15)=x(9)*x(3)+x(13)*x(1)-yy(2)*ng1-xx(2)*nl1;
Y(16)=x(10)*x(3)+x(14)*x(1)-yy(3)*ng1-xx(3)*nl1;







end