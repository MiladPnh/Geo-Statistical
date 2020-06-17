%load var.mat

%%%Tropics%%%
%X-Momentum%
for i=1:46
    x_mom_005(i,1)=((u_005(3,3,2,i+2)-u_005(3,3,2,i))/(12*60*60))+(u_005(3,3,2,i+1)/(a*cosd(u_005(3,1,2,i+1))))*((u_005(3,4,2,i+1)-u_005(3,2,2,i+1))/((pi/180)*(u_005(1,4,2,i+1)-u_005(1,2,2,i+1))))+(v_005(3,3,2,i+1)/a)*((u_005(4,3,2,i+1)-u_005(2,3,2,i+1))/((pi/180)*(u_005(2,1,2,i+1)-u_005(4,1,2,i+1))))+w_005(3,3,2,i+1)*((u_005(3,3,3,i+1)-u_005(3,3,1,i+1))/7500);
end
    for i=1:46
    x_mom_005(i,2)=(u_005(3,3,2,i+1)*v_005(3,3,2,i+1))*(tand(u_005(3,1,2,i+1)/a));
    end
    for i=1:46
    x_mom_005(i,3)=(R*T_005(3,3,2,i+1)/(70000*g))*(u_005(3,3,2,i+1)*w_005(3,3,2,i+1)/a);
    end
    for i=1:46
    x_mom_005(i,4)=(1/(a*cosd(Z_005(3,1,2,i+1))))*((g*Z_005(3,4,2,i+1)-g*Z_005(3,2,2,i+1))/((pi/180)*(Z_005(1,4,2,i+1)-Z_005(1,2,2,i+1))));
    end
for i=1:46
    x_mom_005(i,5)=(2*earth_r*sind(v_005(3,1,2,i+1)))*v_005(3,3,2,i+1);
end
    for i=1:46
    x_mom_005(i,6)=(2*earth_r*cosd(w_005(3,1,2,i+1)))*(R*T_005(3,3,2,i+1)/(70000*g))*w_005(3,3,2,i+1);
    end
   for i=1:46
       x_mom_005(i,7)=((70000*g)/(R*T_005(3,3,2,i+1)))^2*(0.5*(KM_005(3,3,3,i+1)+KM_005(3,3,2,i+1))*((u_005(3,3,3,i+1)-u_005(3,3,2,i+1))/(2500))-0.5*(KM_005(3,3,2,i+1)+KM_005(3,3,1,i+1))*((u_005(3,3,2,i+1)-u_005(3,3,1,i+1))/(5000)))/(71250-67500);
   end
   for i=1:46
       x_mom_005(i,8)=(u_variance_005(i+1,3)*w_variance_005(i+1,3)-u_variance_005(i+1,1)*w_variance_005(i+1,1))/(7500);
   end
  x_mom_005 = abs(x_mom_005);
   
%Y-Momentum%
for i=1:46
    y_mom_005(i,1)=((v_005(3,3,2,i+2)-v_005(3,3,2,i))/(12*60*60))+(u_005(3,3,2,i+1)/(a*cosd(v_005(3,1,2,i+1))))*((v_005(3,4,2,i+1)-v_005(3,2,2,i+1))/((pi/180)*(v_005(1,4,2,i+1)-v_005(1,2,2,i+1))))+(v_005(3,3,2,i+1)/a)*((v_005(4,3,2,i+1)-v_005(2,3,2,i+1))/((pi/180)*(v_005(2,1,2,i+1)-v_005(4,1,2,i+1))))+w_005(3,3,2,i+1)*((v_005(3,3,3,i+1)-v_005(3,3,1,i+1))/7500);
end  
 for i=1:46
    y_mom_005(i,2)=(u_005(3,3,2,i+1)*u_005(3,3,2,i+1))*(tand(u_005(3,1,2,i+1)/a));
 end
for i=1:46
    y_mom_005(i,3)=(R*T_005(3,3,2,i+1)/(70000*g))*(v_005(3,3,2,i+1)*w_005(3,3,2,i+1)/a);
end    
     for i=1:46
    y_mom_005(i,4)=(1/a)*((g*Z_005(2,3,2,i+1)-g*Z_005(4,3,2,i+1))/((pi/180)*(Z_005(4,1,2,i+1)-Z_005(2,1,2,i+1))));
     end
    
     for i=1:46
    y_mom_005(i,5)=(2*earth_r*sind(u_005(3,1,2,i+1)))*u_005(3,3,2,i+1);
end
   
   for i=1:46
       y_mom_005(i,6)=(((70000*g)/(R*T_005(3,3,2,i+1)))^2)*(0.5*(KM_005(3,3,3,i+1)+KM_005(3,3,2,i+1))*((v_005(3,3,3,i+1)-v_005(3,3,2,i+1))/(2500))-0.5*(KM_005(3,3,2,i+1)+KM_005(3,3,1,i+1))*((v_005(3,3,2,i+1)-v_005(3,3,1,i+1))/(5000)))/(71250-67500);
   end
   for i=1:46
       y_mom_005(i,7)=(v_variance_005(i+1,3)*w_variance_005(i+1,3)-v_variance_005(i+1,1)*w_variance_005(i+1,1))/(7500);
   end
   y_mom_005 = abs(y_mom_005);
   
%Vertical-Momentum%
   for i=1:46
    vert_mom_005(i,1)=(R*T_005(3,3,2,i+1)/(70000*g))*(((w_005(3,3,2,i+2)-w_005(3,3,2,i))/(12*60*60))+(u_005(3,3,2,i+1)/(a*cosd(w_005(3,1,2,i+1))))*((w_005(3,4,2,i+1)-w_005(3,2,2,i+1))/((pi/180)*(w_005(1,4,2,i+1)-w_005(1,2,2,i+1))))+(v_005(3,3,2,i+1)/a)*((w_005(4,3,2,i+1)-w_005(2,3,2,i+1))/((pi/180)*(w_005(2,1,2,i+1)-w_005(4,1,2,i+1))))+w_005(3,3,2,i+1)*((w_005(3,3,3,i+1)-w_005(3,3,1,i+1))/7500));
   end 
 for i=1:46
    vert_mom_005(i,2)=((u_005(3,3,2,i+1)*u_005(3,3,2,i+1))+(v_005(3,3,2,i+1)*v_005(3,3,2,i+1)))/a;
 end
     for i=1:46
        vert_mom_005(i,3)=-(R*T_005(3,3,2,i+1)/(70000))*((72500-65000)/(Z_005(3,3,3,i+1)-Z_005(3,3,1,i+1)));
     end
         for i=1:46
    vert_mom_005(i,4)=(2*earth_r*cosd(u_005(3,1,2,i+1)))*u_005(3,3,2,i+1);
         end
for i=1:46
       vert_mom_005(i,5)=((70000*g)/(R*T_005(3,3,2,i+1)))*(0.5*(KM_005(3,3,3,i+1)+KM_005(3,3,2,i+1))*((w_005(3,3,3,i+1)-w_005(3,3,2,i+1))/(2500))-0.5*(KM_005(3,3,2,i+1)+KM_005(3,3,1,i+1))*((w_005(3,3,2,i+1)-w_005(3,3,1,i+1))/(5000)))/(71250-67500);
end
   for i=1:46
       vert_mom_005(i,6)=(R*T_005(3,3,2,i+1)/(70000))*((w_variance_005(i+1,3)*w_variance_005(i+1,3)-w_variance_005(i+1,1)*w_variance_005(i+1,1))/(7500));
   end
   
vert_mom_005 = abs(vert_mom_005);   
   
 %Thermo%
 
   for i=1:46
    thermo_005(i,1)=((T_005(3,3,2,i+2)-T_005(3,3,2,i))/(12*60*60));
   end
for i=1:46
  thermo_005(i,2)=(u_005(3,3,2,i+1)/(a*cosd(T_005(3,1,2,i+1))))*((T_005(3,4,2,i+1)-T_005(3,2,2,i+1))/((pi/180)*(T_005(1,4,2,i+1)-T_005(1,2,2,i+1))));
end 
for i=1:46
  thermo_005(i,3)=(v_005(3,3,2,i+1)/a)*((T_005(4,3,2,i+1)-T_005(2,3,2,i+1))/((pi/180)*(T_005(2,1,2,i+1)-T_005(4,1,2,i+1))));
end
for i=1:46
thermo_005(i,4)=-w_005(3,3,2,i+1)*((R*T_005(3,3,2,i+1)/(70000*1003))-(T_005(3,3,3,i+1)-T_005(3,3,1,i+1))/7500);
end
for i=1:46
       thermo_005(i,5)=(theta_variance_005(i+1,3)*w_variance_005(i+1,3)-theta_variance_005(i+1,1)*w_variance_005(i+1,1))/(7500);
end
   for i=1:46
       thermo_005(i,6)=(((70000*g)/(R*T_005(3,3,2,i+1)))^2)*(0.5*(KH_005(3,3,3,i+1)+KH_005(3,3,2,i+1))*((T_005(3,3,3,i+1)-T_005(3,3,2,i+1))/(2500))-0.5*(KH_005(3,3,2,i+1)+KH_005(3,3,1,i+1))*((T_005(3,3,2,i+1)-T_005(3,3,1,i+1))/(5000)))/(71250-67500);
   end
   
thermo_005 = abs(thermo_005);
   
   
%Continuity%
   for i=1:46
  cont_005(i,1)=(1/(a*cosd(u_005(3,1,2,i+1))))*((u_005(3,4,2,i+1)-u_005(3,2,2,i+1))/((pi/180)*(u_005(1,4,2,i+1)-u_005(1,2,2,i+1))));
   end
   for i=1:46
    cont_005(i,2)=(1/a)*((v_005(4,3,2,i+1)-v_005(2,3,2,i+1))/((pi/180)*(v_005(2,1,2,i+1)-v_005(4,1,2,i+1))));
   end
for i=1:46
    cont_005(i,3)=((w_005(3,3,3,i+1)-w_005(3,3,1,i+1))/7500);
end
   
cont_005 = abs(cont_005);

%%%%Mid-Latitude%%%%
%X-Momentum%
for i=1:46
    x_mom_050(i,1)=((u_050(3,3,2,i+2)-u_050(3,3,2,i))/(12*60*60))+(u_050(3,3,2,i+1)/(a*cosd(u_050(3,1,2,i+1))))*((u_050(3,4,2,i+1)-u_050(3,2,2,i+1))/((pi/180)*(u_050(1,4,2,i+1)-u_050(1,2,2,i+1))))+(v_050(3,3,2,i+1)/a)*((u_050(4,3,2,i+1)-u_050(2,3,2,i+1))/((pi/180)*(u_050(2,1,2,i+1)-u_050(4,1,2,i+1))))+w_050(3,3,2,i+1)*((u_050(3,3,3,i+1)-u_050(3,3,1,i+1))/7500);
end
    for i=1:46
    x_mom_050(i,2)=(u_050(3,3,2,i+1)*v_050(3,3,2,i+1))*(tand(u_050(3,1,2,i+1)/a));
    end
    for i=1:46
    x_mom_050(i,3)=(R*T_050(3,3,2,i+1)/(70000*g))*(u_050(3,3,2,i+1)*w_050(3,3,2,i+1)/a);
    end
    for i=1:46
    x_mom_050(i,4)=(1/(a*cosd(Z_050(3,1,2,i+1))))*((g*Z_050(3,4,2,i+1)-g*Z_050(3,2,2,i+1))/((pi/180)*(Z_050(1,4,2,i+1)-Z_050(1,2,2,i+1))));
    end
for i=1:46
    x_mom_050(i,5)=(2*earth_r*sind(v_050(3,1,2,i+1)))*v_050(3,3,2,i+1);
end
    for i=1:46
    x_mom_050(i,6)=(2*earth_r*cosd(w_050(3,1,2,i+1)))*(R*T_050(3,3,2,i+1)/(70000*g))*w_050(3,3,2,i+1);
    end
   for i=1:46
       x_mom_050(i,7)=((70000*g)/(R*T_050(3,3,2,i+1)))^2*(0.5*(KM_050(3,3,3,i+1)+KM_050(3,3,2,i+1))*((u_050(3,3,3,i+1)-u_050(3,3,2,i+1))/(2500))-0.5*(KM_050(3,3,2,i+1)+KM_050(3,3,1,i+1))*((u_050(3,3,2,i+1)-u_050(3,3,1,i+1))/(5000)))/(71250-67500);
   end
   for i=1:46
       x_mom_050(i,8)=(u_variance_050(i+1,3)*w_variance_050(i+1,3)-u_variance_050(i+1,1)*w_variance_050(i+1,1))/(7500);
   end
   
x_mom_050 = abs(x_mom_050);   
   
%Y-Momentum%
for i=1:46
    y_mom_050(i,1)=((v_050(3,3,2,i+2)-v_050(3,3,2,i))/(12*60*60))+(u_050(3,3,2,i+1)/(a*cosd(v_050(3,1,2,i+1))))*((v_050(3,4,2,i+1)-v_050(3,2,2,i+1))/((pi/180)*(v_050(1,4,2,i+1)-v_050(1,2,2,i+1))))+(v_050(3,3,2,i+1)/a)*((v_050(4,3,2,i+1)-v_050(2,3,2,i+1))/((pi/180)*(v_050(2,1,2,i+1)-v_050(4,1,2,i+1))))+w_050(3,3,2,i+1)*((v_050(3,3,3,i+1)-v_050(3,3,1,i+1))/7500);
end  
 for i=1:46
    y_mom_050(i,2)=(u_050(3,3,2,i+1)*u_050(3,3,2,i+1))*(tand(u_050(3,1,2,i+1)/a));
 end
for i=1:46
    y_mom_050(i,3)=(R*T_050(3,3,2,i+1)/(70000*g))*(v_050(3,3,2,i+1)*w_050(3,3,2,i+1)/a);
end    
     for i=1:46
    y_mom_050(i,4)=(1/a)*((g*Z_050(2,3,2,i+1)-g*Z_050(4,3,2,i+1))/((pi/180)*(Z_050(4,1,2,i+1)-Z_050(2,1,2,i+1))));
     end
    
     for i=1:46
    y_mom_050(i,5)=(2*earth_r*sind(u_050(3,1,2,i+1)))*u_050(3,3,2,i+1);
end
   
   for i=1:46
       y_mom_050(i,6)=(((70000*g)/(R*T_050(3,3,2,i+1)))^2)*(0.5*(KM_050(3,3,3,i+1)+KM_050(3,3,2,i+1))*((v_050(3,3,3,i+1)-v_050(3,3,2,i+1))/(2500))-0.5*(KM_050(3,3,2,i+1)+KM_050(3,3,1,i+1))*((v_050(3,3,2,i+1)-v_050(3,3,1,i+1))/(5000)))/(71250-67500);
   end
   for i=1:46
       y_mom_050(i,7)=(v_variance_050(i+1,3)*w_variance_050(i+1,3)-v_variance_050(i+1,1)*w_variance_050(i+1,1))/(7500);
   end
   
y_mom_050 = abs(y_mom_050);   
   
 %Vertical-Momentum%
   for i=1:46
    vert_mom_050(i,1)=(R*T_050(3,3,2,i+1)/(70000*g))*(((w_050(3,3,2,i+2)-w_050(3,3,2,i))/(12*60*60))+(u_050(3,3,2,i+1)/(a*cosd(w_050(3,1,2,i+1))))*((w_050(3,4,2,i+1)-w_050(3,2,2,i+1))/((pi/180)*(w_050(1,4,2,i+1)-w_050(1,2,2,i+1))))+(v_050(3,3,2,i+1)/a)*((w_050(4,3,2,i+1)-w_050(2,3,2,i+1))/((pi/180)*(w_050(2,1,2,i+1)-w_050(4,1,2,i+1))))+w_050(3,3,2,i+1)*((w_050(3,3,3,i+1)-w_050(3,3,1,i+1))/7500));
   end 
 for i=1:46
    vert_mom_050(i,2)=((u_050(3,3,2,i+1)*u_050(3,3,2,i+1))+(v_050(3,3,2,i+1)*v_050(3,3,2,i+1)))/a;
 end
     for i=1:46
        vert_mom_050(i,3)=-(R*T_050(3,3,2,i+1)/(70000))*((72500-65000)/(Z_050(3,3,3,i+1)-Z_050(3,3,1,i+1)));
     end
         for i=1:46
    vert_mom_050(i,4)=(2*earth_r*cosd(u_050(3,1,2,i+1)))*u_050(3,3,2,i+1);
         end
for i=1:46
       vert_mom_050(i,5)=((70000*g)/(R*T_050(3,3,2,i+1)))*(0.5*(KM_050(3,3,3,i+1)+KM_050(3,3,2,i+1))*((w_050(3,3,3,i+1)-w_050(3,3,2,i+1))/(2500))-0.5*(KM_050(3,3,2,i+1)+KM_050(3,3,1,i+1))*((w_050(3,3,2,i+1)-w_050(3,3,1,i+1))/(5000)))/(71250-67500);
end
   for i=1:46
       vert_mom_050(i,6)=(R*T_050(3,3,2,i+1)/(70000))*((w_variance_050(i+1,3)*w_variance_050(i+1,3)-w_variance_050(i+1,1)*w_variance_050(i+1,1))/(7500));
   end
   
 vert_mom_050 = abs(vert_mom_050); 
   
 %Thermo%
   for i=1:46
    thermo_050(i,1)=((T_050(3,3,2,i+2)-T_050(3,3,2,i))/(12*60*60));
   end
for i=1:46
  thermo_050(i,2)=(u_050(3,3,2,i+1)/(a*cosd(T_050(3,1,2,i+1))))*((T_050(3,4,2,i+1)-T_050(3,2,2,i+1))/((pi/180)*(T_050(1,4,2,i+1)-T_050(1,2,2,i+1))));
end 
for i=1:46
  thermo_050(i,3)=(v_050(3,3,2,i+1)/a)*((T_050(4,3,2,i+1)-T_050(2,3,2,i+1))/((pi/180)*(T_050(2,1,2,i+1)-T_050(4,1,2,i+1))));
end
for i=1:46
thermo_050(i,4)=-w_050(3,3,2,i+1)*((R*T_050(3,3,2,i+1)/(70000*1003))-(T_050(3,3,3,i+1)-T_050(3,3,1,i+1))/7500);
end
for i=1:46
       thermo_050(i,5)=(theta_variance_050(i+1,3)*w_variance_050(i+1,3)-theta_variance_050(i+1,1)*w_variance_050(i+1,1))/(7500);
end
   for i=1:46
       thermo_050(i,6)=(((70000*g)/(R*T_050(3,3,2,i+1)))^2)*(0.5*(KH_050(3,3,3,i+1)+KH_050(3,3,2,i+1))*((T_050(3,3,3,i+1)-T_050(3,3,2,i+1))/(2500))-0.5*(KH_050(3,3,2,i+1)+KH_050(3,3,1,i+1))*((T_050(3,3,2,i+1)-T_050(3,3,1,i+1))/(5000)))/(71250-67500);
   end
      
thermo_050 = abs(thermo_050);
   
  %Continuity%
   for i=1:46
  cont_050(i,1)=(1/(a*cosd(u_050(3,1,2,i+1))))*((u_050(3,4,2,i+1)-u_050(3,2,2,i+1))/((pi/180)*(u_050(1,4,2,i+1)-u_050(1,2,2,i+1))));
   end
   for i=1:46
    cont_050(i,2)=(1/a)*((v_050(4,3,2,i+1)-v_050(2,3,2,i+1))/((pi/180)*(v_050(2,1,2,i+1)-v_050(4,1,2,i+1))));
   end
for i=1:46
    cont_050(i,3)=((w_050(3,3,3,i+1)-w_050(3,3,1,i+1))/7500);
   end

cont_050 = abs(cont_050);   
   
   
   