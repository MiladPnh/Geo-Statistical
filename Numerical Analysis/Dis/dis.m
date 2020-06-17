function f=dis(x,xo,H)
h=10;
R=8.314;
sum=0;
M=[79.149 86.175 142.282]';%g/mol
ww=[0.251 0.296 0.49]';
mm=0.37464*[1 1 1]'+1.54226*ww+0.26992*ww.*ww;
Tc=[469.6 507.4 617]';
Pc=10^5*[33.74 29.69 21.08]';
ac=0.47524*R^2*Tc.*Tc./Pc./Pc';
b=0.007780*R*Tc./Pc;
b=b';
kl=113.52*10^-3;
kg=24.03*10^-3;
mul=233*10^-6;
mug=12.4*10^-3;
cvl=190.379;
cvg=129.792;%cpg=138.106;
dt=1;
Dml=[5.31*10^-6 4.7*10^-6 4*10^-6]';
Dmg=[0.276*10^-4 0.241*10^-4 0.205*10^-4]';
dz=10^-2;% 20 ta mesh dar rastaye z darim
Z=0.2/dz;% Z tedad mesh ha dar jahat z
nz=Z+1;% nz tedad noghat dar jahat z
dr=10^-2;% 5 ta mesh dar rastaye r darim
RR=0.05/dr;%RR tedad meshha dar jahat r
nr=RR+1;%nr tedad noghat dar jahat r ast
nl=floor(H/dz)+1;%nl tedad noghat dar rastaye z dar phaze maye ast.
%ng=nz-nl;% ng tedad noghat dar phase gas
n=length(x);
f=zeros(n);
EquationIndex=1;
A=zeros(nr,nz,:);
B=zeros(nr,nz,:);
for i=1:n/7
    q1=xo((i-1)*7+1:i*7);
    s=floor(i/(R+1));
    B(i,s,:)=q1;
end
for i=1:n/7
    q22=x((i-1)*7+1:i*7);
    s=floor(i/(R+1));
    A(i,s,:)=q22;
end
for z=2:nz-1;% r shomarandeye noghat dar jahat shoaaa(r) ast.
    if z<nl
        Dm=Dml;
        cv=cvl;
        k=kl;
        mu=mul;
    elseif z>nl
        Dm=Dmg;
        cv=cvg;
        mu=mug;
        k=kg;
    end
    for r=2:nr-1;
        % firt continuities
        for i=1:3
            a1=(A(r,z,6)+B(r,z,6))/2;
            a2=(A(r,z,i)-B(r,z,i))/dt;
            a3=(A(r,z,4)+B(r,z,4))/2*(A(r+1,z,i)-A(r,z,i))/dr;
            a4=((A(r,z,5)+B(r,z,5))/2)*((A(r,z+1,i)-A(r,z,i))/dz);
            D1=a1*(a2+a3+a4);
            a5=((A(r+1,z,i)-2*A(r,z,i)+A(r-1,z,i))/dr^2);
            a6=1/(dr*(r-1))+((A(r+1,z,i)-A(r,z,i))/dr);
            a7=((A(r,z+1,i)-2*A(r,z,i)+A(r,z-1,i))/dz^2);
            D2=-Dm(i)*a1*(a5+a6+a7);
            a8=(A(r+1,z,6)-A(r,z,6))/(dr)*(A(r+1,z,i)-A(r,z,i))/(dr);
            a9=(A(r,z+1,6)-A(r,z,6))/(dz)*(A(r,z+1,i)-A(r,z,i))/(dz);
            D3=-Dm(i)*(a8+a9);
            f(EquationIndex)=D1+D2+D3;
            EquationIndex=EquationIndex+1;
        end
        %equation of energy
        a10=(A(r,z,6)+B(r,z,6))/2;
        a11=(A(r,z,7)-B(r,z,7))/dt;
        a12=((A(r,z,4)+B(r,z,4))/2)*((A(r+1,z,7)-A(r,z,7))/dr);
        a13=((A(r,2,5)+B(r,z,5))/2)*((A(r,z+1,7)-A(r,z,7))/dz);
        D4=cv*a10*(a11+a12+a13);
        a14=(A(r+1,z,7)-2*A(r,z,7)+A(r-1,z,7))/dr^2;
        a15=1/((r-1)*dr)*((A(r+1,z,7)-A(r,z,7))/dr);
        a16=((A(r,z+1,7)-2*A(r,z,7)+A(r-1,z,7))/dz^2);
        D5=-k*(a14+a15+a16);
        a17=8.314*((A(r,z,7)+B(r,z,7))/2);
        w=zeros(3,1);
        w(1)=(A(r,z,1)+B(r,z,1))/2;
        w(2)=(A(r,z,2)+B(r,z,2))/2;
        w(3)=(A(r,z,3)+B(r,z,3))/2;
        xx=masstomole(w);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        MM=xx(1)*M(1)+xx(3)*M(3)+xx(2)*M(2);
        ro=((A(r,z,6)+B(r,z,6))/2);
        bb=xx(1)*b(1)+xx(2)*b(2)+xx(3)*b(3);
        a19=MM^2/ro^2+2*bb*MM/ro-bb^2;
        alfa=alf(T);
        for i=1:3
            for j=1:3
                ee=xx(i)*xx(j)*sqrt(ac(i)*ac(j))*sqrt(alfa(i)*alfa(j));
                ff=1/2*(-mm(i)*(T/Tc(i)/alfa(i))-mm(j)*(T/Tc(j)/alfa(j)));
                pp=ee+ff;
                sum=sum+pp;
            end
        end
        D6=a17/(a18-bb)-sum/a19;
        for i=1:3
            a20=10*((A(r,z,6)+B(r,z,6))/2)*((A(r,z,i)+B(r,z,i))/2)*((A(r,z,5)+B(r,z,5))/2);
            a21=10*((A(r,z,6)+B(r,z,6))/2)*Dm(i)*((A(r,z+1,i)-A(r,z,i))/dz);
            qq=a20+a21;
            sum1=sum1+qq;
        end
        D7=sum1;
        f(EquationIndex)=D4+D5+D6+D7;
        EquationIndex=EquationIndex+1;
        %equation of motion
        a22=(A(r,z,6)+B(r,z,6))/2;
        a23=(A(r,z,4)-B(r,z,4))/dt;
        a24=((A(r,z,4)+B(r,z,4))/2)*((A(r+1,z,4)-A(r,z,4))/dr);
        a25=((A(r,z,5)+B(r,z,5))/2)*((A(r,z+1,4)-A(r,z,4))/dz);
        D8=a22*(a23+a24+a25);
        a26=4/3*(A(r+1,z,4)-2*A(r,z,4)+A(r-1,z,4));
        a27=+1/3*((A(r+1,z+1,5)-A(r+1,z,5))-(A(r,z+1,5)-A(r,z,5)))/dr/dz;
        a28=4/3/((r-1)*dr)*((A(r+1,z,4)-A(r,z,4))/dr)-4/3/((r-1)*dr)*(A(r,z+1,5)-A(r,z,5))/dz;
        a29=4/3/((r-1)*dr)^2*(A(r,z,4)+B(r,z,4))/2;
        a30=(A(r,z+1,4)-2*A(r,z,4)+A(r,z-1,4))/dr/dz;
        D9=mu*(a26+a27+a28+a29+a30);
        f(EquationIndex)=D8+D9;
        EquationIndex=EquationIndex+1;
        a31=(A(r,z,6)+B(r,z,6))/2;
        a32=((A(r,z,5)-B(r,z,5))/dt);
        a33=((A(r,z,4)+B(r,z,4))/2)*((A(r+1,z,5)-A(r,z,5))/dr);
        a34=((A(r,z,5)+B(r,z,5))/2)*((A(r,z+1,5)-A(r,z,5))/dz);
        D10=a31*(a32+a33+a34);
        a35=(1/((r-1)*dr)*((A(r+1,z,5)-A(r,z,5))/dr));
        a36=1/3/((r-1)*dr)*((A(r,z+1,4)-A(r,z,4))/dz);
        a37=((A(r+1,z,5)-2*A(r,z,5)+A(r-1,z,5))/dr^2);
        a38=1/3*(((A(r+1,z+1,4)-A(r+1,z,4))-(A(r,z+1,4)-A(r,z,4)))/dr/dz);
        a39=4/3*((A(r,z+1,5)-2*A(r,z,5)+A(r,z-1,5))/dz^2);
        D11=mu*(a35+a36+a37+a38+a39);
        f(EquationIndex)=D11+D10;
        EquationIndex=EquationIndex+1;
        % sumation of mole fractions
        f(EquationIndex)=(A(r,z,1)+B(r,z,1))/2+(A(r,z,2)+B(r,z,2))/2+(A(r,z,2)+B(r,z,3))/2-1;
        EquationIndex=EquationIndex+1;
    end
end
%boundary cond s
for r=2,nr-1
    for j=1:3
        f(EquationIndex)= A(r,1,j)-A(r,2,j);
        EquationIndex=EquationIndex+1;
        f(EquationIndex)= A(r,nz,j)-A(r,nz-1,j);%g
        EquationIndex=EquationIndex+1;
    end
    f(EquationIndex)=A(r,1,1)+A(r,1,2)+A(r,1,3)-1;
    EquationIndex=EquationIndex+1;
    f(EquationIndex)=A(r,nz,1)+A(r,nz,2)+A(r,nz,3)-1;%g
    EquationIndex=EquationIndex+1;
    f(EquationIndex)=1666*dz/kl+(A(r,2,7)-A(r,1,7));
    EquationIndex=EquationIndex+1;
    f(EquationIndex)=(A(r,nz,7)-A(r,nz-1,7));%g
    EquationIndex=EquationIndex+1;
    f(EquationIndex)=A(r,1,4);
    EquationIndex=EquationIndex+1;
    f(EquationIndex)=A(r,1,5);
    EquationIndex=EquationIndex+1;
    f(EquationIndex)=A(r,nz,4);
    EquationIndex=EquationIndex+1;%g
    f(EquationIndex)=A(r,nz,5);
    EquationIndex=EquationIndex+1;%g
end
for z=2:nz-1
    for i=1:3
        f(EquationIndex)=A(nr,z,i)-A(nr,z,i);
        EquationIndex=EquationIndex+1;
        f(EquationIndex)=A(1,z,i)-A(1,z,i);
        EquationIndex=EquationIndex+1;
    end
    f(EquationIndex)=A(nr,z,1)+A(nr,z,2)+A(nr,z,3)-1;
    EquationIndex=EquationIndex+1;
    f(EquationIndex)=A(1,z,1)+A(1,z,2)+A(1,z,3)-1;
    EquationIndex=EquationIndex+1;
    f(EquationIndex)=A(2,z,7)-A(1,z,7);
    EquationIndex=EquationIndex+1;
    f(EquationIndex)=A(2,z,5)-A(2,z,5);
    EquationIndex=EquationIndex+1;
    f(EquationIndex)=A(2,z,4)-A(1,z,4);
    EquationIndex=EquationIndex+1;
    f(EquationIndex)=A(nr,z,4);
    EquationIndex=EquationIndex+1;
    f(EquationIndex)=A(nr,z,5);
    EquationIndex=EquationIndex+1;
    if z<nl
        k=kl;
    elseif z>nl
        k=kg;
    end
f(EquationIndex)=h*(A(nr,z,7)-313)+k*(A(nr,z,7)-A(nr-1,z,7))/dz;
EquationIndex=EquationIndex+1;
end
%boundary conditions for tips
for i=1:3
f(EquationIndex)=A(nr,nz,i)-A(nr-1,nz,i);
EquationIndex=EquationIndex+1;
f(EquationIndex)=A(nr,nz,i)-A(nr,nz-1,i);
EquationIndex=EquationIndex+1;
end
f(EquationIndex)=kg(A(nr,nz,7)-A(nr-1,nz,7))/dr+0.5*h(A(nr,nz,7)-313);
EquationIndex=EquationIndex+1;
f(EquationIndex)=A(nr,nz,1)+A(nr,nz,2)+A(nr,nz,3)-1;
EquationIndex=EquationIndex+1;
f(EquationIndex)=A(nr,nz,4);
EquationIndex=EquationIndex+1;
f(EquationIndex)=A(nr,nz,5);
EquationIndex=EquationIndex+1;
%-----------------------------------------------------------------------
f(EquationIndex)=A(nr,1,5);
EquationIndex=EquationIndex+1;
%-----------------------------------------------------------------------
f(EquationIndex)=A(1,1,5);
EquationIndex=EquationIndex+1;
%-----------------------------------------------------------------------
f(EquationIndex)=A(1,nz,5);
EquationIndex=EquationIndex+1;
end