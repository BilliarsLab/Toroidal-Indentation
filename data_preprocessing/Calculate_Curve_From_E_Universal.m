clear
clc
close all

%% User Input
Sample_id='1';

Probe_R1=1; %Smaller Radius um
Probe_R2=11; %Larger Radius (outer) um

%Only allow: ASP3.25 Probe: R1=1, R2=4.25; ASP7 Probe: R1=1; R2=8; ASP 10
%Probe:R1=1; R2=11

E_Para=[]; % Parallel Direction Young's modulus in kPa
E_Perp=[]; % Perpendicular Direction Young's modulus in kPa

% Storage_Path=cwd;

%%

f2=1; %choose 1 if ASP=<10

E_eff_0=E_Para*1.33; % Effective Young's modulus for incompressible Mat. 
E_eff_90=E_Perp*1.33; % kPa 

Re=sqrt(Probe_R1*Probe_R2);
Max_D=0.5*Probe_R1; %Max Indentation Depth is half of probe small radii.

D=0:(Max_D/100):Max_D;

F0=(4/3)*E_eff_0*D.^1.5*sqrt(Re)/(f2^1.5); %nN if R in um; mN if R in mm
F90=(4/3)*E_eff_90*D.^1.5*sqrt(Re)/(f2^1.5); %
    
Fit_Data_0=[D',F0'];
Fit_Data_90=[D',F90'];

xlswrite(['Demo_',Sample_id,'_0deg.xls'],Fit_Data_0)
xlswrite(['Demo_',Sample_id,'_90deg.xls'],Fit_Data_90)

