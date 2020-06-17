function [C,delta_1,delta_2,omega_a,omega_b,Zc,m]=equationselect(w,noc,equation,mw)
m=zeros(noc,1);
switch equation
    case 1
        C=1;
        delta_1=1+(2)^.5;
        delta_2=1-(2^.5);
        omega_a=.457235;
        omega_b=.077796;
        Zc=.3074;
        for i=1:noc
            m(i,1)=.37469+1.54226*w(i)-.26992*(w(i))^2;
        end
    case 2
        C=1;
        delta_1=1+2^.5;
        delta_2=1-2^.5;
        omega_a=.457235;
        omega_b=.077796;
        Zc=.3074;
        
        for i=1:noc
            if mw(i)>144
                m(i,1)=.379642+1.48503*w(i)-.164423*(w(i))^2+.016666*(w(i))^3;
            else
                m(i,1)=.37469+1.54226*w(i)-.26992*(w(i))^2;
            end
        end
    case 3
        C=0;
        delta_1=1;
        delta_2=0;
        omega_a=.42747;
        omega_b=.08664;
        Zc=1/3;
        for i=1:noc
            m(i,1)=.48+1.574*w(i)-.176*(w(i))^2;
        end
    case 4
        C=0;
        delta_1=1;
        delta_2=0;
        omega_a=.42747;
        omega_b=.08664;
        Zc=1/3;
        for i=1:noc
            m(i,1)=.48508+1.55171*w(i)-.15613*(w(i))^2;
        end
end
end
