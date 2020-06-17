clc
clear all

x=rand(1,16)*10;
% mohasebe az flash

deltat=1;

d=.1;
h=.2;
Vc=pi*d^2/4*h;
Ab=pi*d^2/4;
Aside=pi*d*h;
H=10;
qzz=1366;
C=deltat*Ab*qzz;
R=8314;
%C=?
nl1=2.4173;
ng1=0.05;
nt=nl1+ng1;

Tc=[460.43 507.6 617.7];%k
Pc=[3.381 3.025 2.11]*10;%pa
mw=[72.15 86.177 142.285];
%hrl1=??

%hrg1=?
P1=10^(5);
T1=313.23;
xx=[
       0.5938
      0.30401
      0.10219];
  
  yy=[0.88167
      0.11771
   0.00061167];

Zl1=.00499;
Zg1=0.9635;
% mohasebe hr avalie


w=[.2275 .3013 .4923];
m=.480+1.574.*w-0.176.*w.^2;


Tr=T1./Tc;
alfa=(1+m.*(1-sqrt(Tr))).^2;
ac=0.457235*R^2.*Tc.^2./Pc;
a=ac.*alfa;

b=.08664*R.*Tc./Pc;
bl=xx(1)*b(1)+x(2)*b(2)+xx(3)*b(3);
Bl=(P1.*bl)/R*T1;
zigl=0;
for i=1:3
    for j=1:3
        
        zigl=zigl+xx(i)*xx(j)*sqrt(a(i)*a(j))*(1/2*(-m(i)*sqrt(Tr(i)/alfa(i)))-1/2*(-m(j)*(Tr(i)/alfa(j))));
    end
end


hrl1=R*T1*(Zl1-1)-(zigl/(2*sqrt(2).*bl)).*log(Zl1+2.414*Bl)./(Zl1-.414*Bl);




  %%------------------------------
  % mohasebe hg avalie
  
  zigg=0;
  for i=1:3
    for j=1:3
        
        zigg=zigg+yy(i)*yy(j)*sqrt(a(i)*a(j))*(1/2*(-m(i)*sqrt(Tr(i)/alfa(i)))-1/2*(-m(j)*(Tr(i)/alfa(j))));
    end
end

b=.08664*R.*Tc./Pc;
bg=yy(1)*b(1)+yy(2)*b(2)+yy(3)*b(3);

Bg=(P1.*bg)/R*T1;


  hrg1=R*T1*(Zg1-1)-(zigl/(2*sqrt(2).*bg)).*log(Zg1+2.414*Bg)./(Zg1-.414*Bg);

  
  
  %hads avlie 
  xi=rand(1,16).*10;
  ev=kolmoaedle(xi,C,nl1,ng1,hrl1,hrg1,nt,P1,T1,xx,yy);
 
m=0;
% mohasebe norm be ezaye hads avalie
for iii=1:16
    m=m+ev(iii)^2;
end

nnorm=sqrt(m);




%%-------------------
% baze zamani

for t=1:2
deltat=deltat+1;

while nnorm>10^(-6)
    

yy1=kolmoaedle(xi,C,nl1,ng1,hrl1,hrg1,nt,P1,T1,xx,yy);
% jacobian ----------
for i=1:16
    for ii=1:16
        xxi=xi;
        xxi(ii)=1.001*xi(ii);
        yy2=kolmoaedle(xxi,C,nl1,ng1,hrl1,hrg1,nt,P1,T1,xx,yy);
        
        j(i,ii)=(yy2(i)-yy1(i))/(.001*x(ii));
    end
end
  %-----------------------------------------  
  % hal moadele be manzoor taiin dx
%dx=gaussel(j,-yy1');
dx=j\-yy1';
%---------------------------------
xi=xi+dx';

ev=kolmoaedle(xi,C,nl1,ng1,hrl1,hrg1,nt,P1,T1,xx,yy);

m=0;
% taiin norm be ezaye x jadid
for iiii=1:16
    m=m+ev(iiii)^2;
end

nnorm=sqrt(m);

%-----------------------------




end
T(t)=x(5);
P(t)=x(6);
y1(t)=x(8);
y2(t)=x(9);
y3(t)=x(10);
yy=[y1(t) y2(t) y3(t)];

x1(t)=x(12);
x2(t)=x(13);
x3(t)=x(14);
xx=[x1(t) x2(t) x3(t)];
C=deltat*(Ab*qzz+(x(5)-T)*h*Aside);

end