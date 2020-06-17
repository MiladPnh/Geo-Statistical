%importing data file to seperate arrays
fname=dir('C:\Users\miladpanahi\Desktop\Master\1st Sem\Dynamic Meteorology\Project\ATMO541A2017proj');
fname=extractfield(fname,'name');
P=[65000,70000,72500,75000];
R=287;
cp=1003;
cv=717;
g=9.8;
a=6371000;
earth_r=7.292*10^-5;
%KH_005
for j=1:4
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
KH_005(:,:,j,i)=importdata(s);
end
end
%KH_050
for j=5:8
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
KH_050(:,:,j-4,i)=importdata(s);
end
end
%KM_005
for j=9:12
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
KM_005(:,:,j-8,i)=importdata(s);
end
end
%KM_050
for j=13:16
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
KM_050(:,:,j-12,i)=importdata(s);
end
end
%T_005
for j=17:20
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
T_005(:,:,j-16,i)=importdata(s);
end
end
%T_050
for j=21:24
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
T_050(:,:,j-20,i)=importdata(s);
end
end
%Z_005
for j=25:28
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
Z_005(:,:,j-24,i)=importdata(s);
end
end
%Z_050
for j=29:32
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
Z_050(:,:,j-28,i)=importdata(s);
end
end
%w_005
for j=33:36
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
w_005(:,:,j-32,i)=importdata(s);
end
end
%w_050
for j=37:40
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
w_050(:,:,j-36,i)=importdata(s);
end
end
%ps_005
for j=41
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
ps_005(:,:,j-40,i)=importdata(s);
end
end
%ps_050
for j=42
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
ps_050(:,:,j-41,i)=importdata(s);
end
end
%slp_005
for j=43
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
slp_005(:,:,j-42,i)=importdata(s);
end
end
%slp_050
for j=44
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
slp_050(:,:,j-43,i)=importdata(s);
end
end
%u_005
for j=45:48
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
u_005(:,:,j-44,i)=importdata(s);
end
end
%u_050
for j=49:52
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
u_050(:,:,j-48,i)=importdata(s);
end
end
%v_005
for j=53:56
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
v_005(:,:,j-52,i)=importdata(s);
end
end
%v_050
for j=57:60
for i=1:48
s=char(fname{1,i+(j-1)*48+2});
v_050(:,:,j-56,i)=importdata(s);
end
end
%Variance__005
for j=1:4
   for i=1:48
    u_variance_005(i,j)=u_005(3,3,j,i)-mean(u_005(3,3,j,:));
   end
end
for j=1:4
   for i=1:48
    v_variance_005(i,j)=v_005(3,3,j,i)-mean(v_005(3,3,j,:));
   end
end
for j=1:4
   for i=1:48
    w_variance_005(i,j)=w_005(3,3,j,i)-mean(w_005(3,3,j,:));
   end
end
for j=1:4
   for i=1:48
    theta_variance_005(i,j)=T_005(3,3,j,i).*(slp_005(3,3,1,i)./P(1,j))^(R/cp)-mean((slp_005(3,3,1,:)./P(1,j)).^(R/cp).*T_005(3,3,j,:));
   end
end

%Variance__050
for j=1:4
   for i=1:48
    u_variance_050(i,j)=u_050(3,3,j,i)-mean(u_050(3,3,j,:));
   end
end
for j=1:4
   for i=1:48
    v_variance_050(i,j)=v_050(3,3,j,i)-mean(v_050(3,3,j,:));
   end
end
for j=1:4
   for i=1:48
    w_variance_050(i,j)=w_050(3,3,j,i)-mean(w_050(3,3,j,:));
   end
end
for j=1:4
   for i=1:48
    theta_variance_050(i,j)=T_050(3,3,j,i).*(slp_050(3,3,1,i)./P(1,j))^(R/cp)-mean((slp_050(3,3,1,:)./P(1,j)).^(R/cp).*T_050(3,3,j,:));
   end
end

%save var;
