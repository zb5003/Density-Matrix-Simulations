% Fourth order Runge-Kutta density matrix solver for N level atom
%
% This program numerically finds the equations of motion of the density
% matrix of an N level atom using a fourth order Runge-Kutta method.
%{
% In general this is how the Runge-Kutta method works for 2 coupled equations:
%
% The differential Equations to be solved are
% da1/dt=F(t,a1,a2)
% da2/dt=G(t,a1,a2)
%
% To approximate each function a1(t) and a2(t) we use the following formulas:
%
%                 (K1+2*K2+2*K3+K4)
% a1_(n+1)=a1_n + -----------------
%                        6
%
%
%                 (L1+2*L2+2*L3+L4)
% a2_(n+1)=a2_n + -----------------
%                        6
%
% The K and L coefficients are found using these formulas:
%
% h = time step size
%
% K1 = h*F(t1, a1(n), a2(n));
% L1 = h*G(t1, a1(n), a2(n));
%
% K2 = h*F(t1+h/2, a1(n)+K1/2  a2(n)+L1/2);
% L2 = h*G(t1+h/2, a1(n)+K1/2, a2(n)+L1/2);
%
% K3 = h*F(t1+h/2, a1(n)+K2/2  a2(n)+L2/2);
% L3 = h*G(t1+h/2, a1(n)+K2/2, a2(n)+L2/2);
%
% K4 = h*F(t1+h, a1(n)+K3, a2(n)+L3);
% L4 = h*G(t1+h, a1(n)+K3, a3(n)+L3);
%
% Because Matlab does matrix operations more efficiently than 'for' loops,
% as much as possible will be written as matrices.  The K and L
% coefficients will be put into one K matrix.  For the two level problem it
% looks like this:
%
% K=[K1 K2 K3 K4]
%   [L1 L2 L3 L4]
%
% In general, there will be as many rows in the K matrix as there are
% differential equations.
%
% Also since the equations we are interested in are linear we can write the
% rate equations as da/dt=F*a, where F is a matrix of coefficients and 'a' is
% column vector:
%
%   da/dt  =      F       a
%
% [da1/dt] = [t1 p11 p12][t ]
% [da2/dt] = [t2 p21 p22][a1]
%                        [a2]
%
% The pij's are the coefficients in the rate equations. In general, there
% might be explicit dependence on time in the rate equations, so it should
% be included as a variable in 'a' even though it's coefficient is zero
% most of the time in the cases we are looking at.
%
%}
%
% The equation of motion of a density matrix is
%
% d(rho)    1
% ------ = ----[H,rho] - 1/2{G,rho}
%   dt      ih
%
%
% What needs to be changed for each new problem?
% 1) The Hamiltonian - All of the parameters need to be defined first.
%
% 2) The decay matrix - All of the parameters need to be defined first.
%
% 3) Whichever variables you want to store for reference later.  For
%    example you might want to look at how rho_12 changes with detuning.
%

tic
clear
% close all
clc

% This should be changed to the appropriate folder
% cd('C:\Users\YavuzLab\Dropbox\7LevelSim');

% This automatically docks any figures that are created
% set(0,'DefaultFigureWindowStyle','docked');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Physical Constants %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
ep_0 = 8.85e-12;     % Permitivity of free space (N*A^-2)  %%%
mu_0 = 4*pi*10^(-7); % Permeability of free space (F*m^-1) %%%
c = 2.99792e8;    % Speed of light (m/s)                %%%
hbar = 1.054571e-34; % Plank's constant (J*s)              %%%
mu_B = 9.274e-24;    % Bohr Magneton (J/T)                 %%%
n = 1.8;          % Index of refraction                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
% Formula for Rabi frequency
%
%         E_ij*mu_ij
%  W_ij= ------------
%            hbar
%
%}
%{
% These are the labels used in the problem:
%
%                  |7>___________________
%
%                                         d1
%                  |6>___________________
%                         dcm1
%                            ^
%                            |
%                            |            d2
%                  |5>_______|___________
%                            |
%                            |
%                            |
%                            |
%                            |
%   |4>____________          |
%           5D0              |
%                            |wc
%                            |
%                            |
%                            |
%                            |
%                            |
%                  |3>_______|___________
%                            |
%                            |           d3
%                  |2>_______|___________
%                            |  
%                            | dcm2 
%                            |
%
%                                        d4
%                  |1>___________________
%
%
%
%}

% Decay rate
Gamma1 = 1/(33e-6); % rad/s This is the decay rate from the 5D1 lines to the 5D0 line
Gamma2 = 1/(1.61e-3); % rad/s This is the decay rate from the 5D0 line to the 7F0 ground states

% Detunings
% FWHM = 1600e6;
% N_A = 40000; % Number of atoms
% stdev = FWHM/(2*sqrt(2*log(2)));
Broadening=2*pi*(-.5e9:25e3:.5e9);%(2*pi*(normrnd(0,stdev,1,N_A)));
PointSaver1=1000; % Number of points to save in each rho
N_A = length(Broadening);
% display(num2str(max(abs(Broadening))/1e9));
TimeStepFactor=1;
h_t=.5e-10;
Isos=[1 2];
tau=9e-9; % pulse rise time is actually tau*(log(1/.1-1)-log(1/.9-1))=4.3944*tau
tau0=10;

% Probe beam
P_p = 277e-3; % Probe power
w_p = 112e-6/2; % Waist of probe
B = sqrt(4*mu_0*n*P_p/(c*pi*w_p^2));

% gamma matrix from lauritzen paper
g=sqrt([0.03,0.22,0.75;
        0.12,0.68,0.20;
        0.85,0.10,0.05]);


mu_guess=0.063;
mu=mu_guess*mu_B*[  0 ,   0 ,    0 ,   0, g(3,1),g(3,2),g(3,3);
                    0 ,   0 ,    0 ,   0, g(2,1),g(2,2),g(2,3);
                    0 ,   0 ,    0 ,   0, g(1,1),g(1,2),g(1,3);
                    0 ,   0 ,    0 ,   0,   0 ,   0 ,    0 ;
                  g(3,1),g(2,1),g(1,1),0,   0 ,   0 ,    0 ;
                  g(3,2),g(2,2),g(1,2),0,   0 ,   0 ,    0 ;
                  g(3,3),g(2,3),g(1,3),0,   0 ,   0 ,    0 ];

W=B.*mu./hbar;
%%
% Vector of rabi frequencies involved
Freqs=[Broadening,reshape(W,1,49)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Time step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% h_t = 1/(TimeStepFactor*max(abs(Freqs)));  % Time step size in seconds
T = 2.1e-6;                      % Total time to run simulations
N_t = round(T/h_t);               % Number of time steps
t_scale = 0:h_t:h_t*(N_t-1);          % This is used to properly scale the time axis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

PointSaver=round((N_t-1)/PointSaver1);
TimesToSave=t_scale(1:PointSaver:end);

% These are used to estimate the time it will take to run a program based
% on past times
TimeEstimates=[0.251394/397 8.685901/16332 6.055934/11219]; % Array of elapsed time/time steps
display(['Time steps = ',num2str(N_t)])
display(['Time steps saved = ',num2str(length(TimesToSave))])
display(['Estimated Time per atom = ',num2str(N_t*mean(TimeEstimates)),' s'])
display(['Estimated Time = ',num2str(2*N_A*N_t*mean(TimeEstimates)/3600),' hours'])

% Number of Levels
N=7;

%{
% This preallocates memory for the K matrix according to the number of
% equations
%}
K = zeros(N^2,4);

%{
% This preallocates memory for the rho matrix according to the number of
% equations and number of time steps. The first row is for the time
% variable, even though it is zero in most cases we see, it should be there
% in general. One column of the rho matrix will look like this:
%
% rho = [t rho_11 rho_12 ... rho_1N rho_21 rho_22 ... rho_2N ... rho_N1 rho_N2 ... rho_NN]^T
%}
rho = zeros(N^2+1,1);

%{
% This is the initial state of the system.  Remember that the first row of
% the rho matrix is for the time variable.
%}
rho(RhoRowFinder(N,[1 1])) = 1/3;
rho(RhoRowFinder(N,[2 2])) = 1/3;
rho(RhoRowFinder(N,[3 3])) = 1/3;

%{
% This is where to preallocate memory for any variables we want to store in
the loops below.  For example in the four level probe/stark beam system we
are interested in rho_12.
%}

SavedRho1=[TimesToSave;zeros(N^2,length(TimesToSave))];

%{
if exist([FileName],'file')==2
    display('Writing to existing file...');
    fid=fopen([FileLocation,FileName,'.txt'],'wt+');
else
    display('Writing to new file...')
    fid=fopen([FileLocation,FileName,'.txt'],'wt+');
    fprintf(fid,['Number of Atoms = %d\n',...
        'FWHM = %d GHz\n',...
        'Dipole moment = %d mu_B\n',...
        'Beam waist = %d microns\n',...
        'Beam power = %d mW\n',...
        'Beam intensity = %d W/cm^2\n',...
        'Decay Rate from 5D1 = %d kHz\n',...
        'Decay Rate from 5D0 = %d kHz\n',...
        'Total simulation time = %d s\n',...
        'Number of time points saved = %d\n'],...
        [N_A, FWHM/1e9, mu_guess, w_p/1e-6, P_p/1e-3, 2*P_p/(pi*w_p^2)/(1e2^2), Gamma1/1e3, Gamma2/1e3, T,length(TimesToSave)]);
    
    fprintf(fid,['%d',repmat('\t%d',1,length(Broadening)-1)],Broadening);
    fprintf(fid,'\nTime Step\trho11\trho12\trho13\trho14\trho15\trho16\trho17\trho22\trho23\trho24\trho25\trho26\trho27\trho33\trho34\trho35\trho36\trho37\trho44\trho45\trho46\trho47\trho55\trho56\trho57\trho66\trho67\trho77\n');
    
end
fclose(fid);
%}

Broadening=sortrows(Broadening')';
for D1=Broadening
    for isotope=Isos
        if isotope==1;
            d1 = 2*pi*114e6;
            d2 = 2*pi*183e6;
            dcm1=d2-(d1+d2+d2)/3;
            d3 = 2*pi*76e6;
            d4 = 2*pi*148e6;
            dcm2=d4-(d3+d4+d4)/3; 
        elseif isotope==2;
            d1 = 2*pi*43e6;
            d2 = 2*pi*71e6;
            dcm1=d2-(d1+d2+d2)/3;
            d3 = 2*pi*29.5e6;
            d4 = 2*pi*57.3e6;
            dcm2=d4-(d3+d4+d4)/3;
        end

        rho = zeros(N^2+1,1);
        %{
% This is the initial state of the system.  Remember that the first row of
% the rho matrix is for the time variable.
        %}
        rho(RhoRowFinder(N,[1 1])) = 1/3;
        rho(RhoRowFinder(N,[2 2])) = 1/3;
        rho(RhoRowFinder(N,[3 3])) = 1/3;
        SaveRhoMarker=1;
        for n_t=1:N_t % This steps through each time step and finds the K matrix
            %     for D2=Broadening
            %         fprintf(fid,'\n');
            % Hamiltonian, must have N rows and N columns
            H = hbar*[    0,0,0,0,  -W(1,5)/2*exp(-1i*((d2-dcm1)-(  d4-dcm2 )+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))), -W(1,6)/2*exp(-1i*((-dcm1)-(  d4-dcm2 )+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))), -W(1,7)/2*exp(-1i*(-(d1+dcm1)-(  d4-dcm2 )+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau)));
                          0,0,0,0,  -W(2,5)/2*exp(-1i*((d2-dcm1)-(    -dcm2 )+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))), -W(2,6)/2*exp(-1i*((-dcm1)-(    -dcm2 )+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))), -W(2,7)/2*exp(-1i*(-(d1+dcm1)-(    -dcm2 )+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau)));
                          0,0,0,0,  -W(3,5)/2*exp(-1i*((d2-dcm1)-(-(d3+dcm2))+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))), -W(3,6)/2*exp(-1i*((-dcm1)-(-(d3+dcm2))+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))), -W(3,7)/2*exp(-1i*(-(d1+dcm1)-(-(d3+dcm2))+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau)));
                          0,0,0,0,0,0,0;
                                    -W(5,1)/2*exp(1i*( (d2-dcm1)-(d4-dcm2)+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))),    -W(5,2)/2*exp(1i*( (d2-dcm1)-(-dcm2)+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))),    -W(5,3)/2*exp(1i*( (d2-dcm1)-(-(d3+dcm2))+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))), 0,0,0,0;
                                    -W(6,1)/2*exp(1i*( (  -dcm1)-(d4-dcm2)+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))),    -W(6,2)/2*exp(1i*( (  -dcm1)-(-dcm2)+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))),    -W(6,3)/2*exp(1i*( (  -dcm1)-(-(d3+dcm2))+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))), 0,0,0,0;
                                    -W(7,1)/2*exp(1i*(-(d1+dcm1)-(d4-dcm2)+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))),    -W(7,2)/2*exp(1i*(-(d1+dcm1)-(-dcm2)+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))),    -W(7,3)/2*exp(1i*(-(d1+dcm1)-(-(d3+dcm2))+D1)*h_t*n_t)*(1./(1+exp(-(h_t*n_t-tau0*tau)./tau))), 0,0,0,0];
            
            % Decay Matrix, must have N rows and N columns
            G = [0, 0, 0,   0,     0,     0,     0;
                 0, 0, 0,   0,     0,     0,     0;
                 0, 0, 0,   0,     0,     0,     0;
                 0, 0, 0, Gamma2,  0,     0,     0;
                 0, 0, 0,   0,   Gamma1,  0,     0;
                 0, 0, 0,   0,     0,   Gamma1,  0;
                 0, 0, 0,   0,     0,     0,   Gamma1];
            
            % This initializes the F matrix for each detuning step
            F=zeros(N^2,N^2+1);
            
            m=1; % This counts the row number of F inside these loops
            
            for i=1:N
                for j=1:N
                    p=zeros(N,N); % Initialize p matrix for each new ij
                    for k=1:N
                        %{
                % These equations calculate the entries in the F matrix.
                % They are set up to add on to the previous matrix element
                % in case of kj=ik and more than one term is in the
                % coefficient. I think is possible to vectorize this loop
                % but I cannot tell a difference in computation time and I
                % think it is easier to see what is going on when it is
                % written in loops.
                        %}
                        p(k,j)=p(k,j)+1/(1i*hbar)*( H(i,k))-G(i,k)/2;
                        p(i,k)=p(i,k)+1/(1i*hbar)*(-H(k,j))-G(k,j)/2;
                        %{
                % These next if statements add population back to levels
                % that higher levels decay to so that population remains
                % constant.
                        %}
                        %
                        % All the excited states decay first to the 5D0 level
                        if i==4 && j==4
                            p(5,5)=p(5,5)+G(5,5)/7;
                            p(6,6)=p(6,6)+G(6,6)/7;
                            p(7,7)=p(7,7)+G(7,7)/7;
                        end
                        %
                        % Then the 5D0 decays to the three ground states
                        if (i==1 && j==1) || (i==2 && j==2) || (i==3 && j==3)
                            p(4,4)=p(4,4)+G(4,4)/21;
                        end
                        %}
                    end
                    %{
            % This reshapes the p matrix into a 1 by N^2 and stores it in
            % the corresponding row of F
                    %}
                    F(m,2:end)=reshape(conj(p)',1,N^2);
                    m=m+1;
                end
            end
            
            % This saves the number of points specified by 'TimesToSave'
            % Each coherence needs to have its phase corrected by the
            % detunings.  
            if any(TimesToSave==t_scale(n_t))==1
                    SavedRho1(RhoRowFinder(N,[1 1]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[1 1]),SaveRhoMarker) + rho(RhoRowFinder(N,[1 1]));
                    SavedRho1(RhoRowFinder(N,[2 2]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[2 2]),SaveRhoMarker) + rho(RhoRowFinder(N,[2 2]));
                    SavedRho1(RhoRowFinder(N,[3 3]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[3 3]),SaveRhoMarker) + rho(RhoRowFinder(N,[3 3]));
                    SavedRho1(RhoRowFinder(N,[4 4]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[4 4]),SaveRhoMarker) + rho(RhoRowFinder(N,[4 4]));
                    SavedRho1(RhoRowFinder(N,[5 5]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[5 5]),SaveRhoMarker) + rho(RhoRowFinder(N,[5 5]));
                    SavedRho1(RhoRowFinder(N,[6 6]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[6 6]),SaveRhoMarker) + rho(RhoRowFinder(N,[6 6]));
                    SavedRho1(RhoRowFinder(N,[7 7]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[7 7]),SaveRhoMarker) + rho(RhoRowFinder(N,[7 7]));
                    
                    SavedRho1(RhoRowFinder(N,[5 1]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[5 1]),SaveRhoMarker) + rho(RhoRowFinder(N,[5 1])).*exp(-1i.*( (d2-dcm1)-(  d4-dcm2 )+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[6 1]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[6 1]),SaveRhoMarker) + rho(RhoRowFinder(N,[6 1])).*exp(-1i.*( (  -dcm1)-(  d4-dcm2 )+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[7 1]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[7 1]),SaveRhoMarker) + rho(RhoRowFinder(N,[7 1])).*exp(-1i.*(-(d1+dcm1)-(  d4-dcm2 )+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[5 2]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[5 2]),SaveRhoMarker) + rho(RhoRowFinder(N,[5 2])).*exp(-1i.*( (d2-dcm1)-(    -dcm2 )+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[6 2]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[6 2]),SaveRhoMarker) + rho(RhoRowFinder(N,[6 2])).*exp(-1i.*( (  -dcm1)-(    -dcm2 )+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[7 2]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[7 2]),SaveRhoMarker) + rho(RhoRowFinder(N,[7 2])).*exp(-1i.*(-(d1+dcm1)-(    -dcm2 )+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[5 3]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[5 3]),SaveRhoMarker) + rho(RhoRowFinder(N,[5 3])).*exp(-1i.*( (d2-dcm1)-(-(d3+dcm2))+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[6 3]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[6 3]),SaveRhoMarker) + rho(RhoRowFinder(N,[6 3])).*exp(-1i.*( (  -dcm1)-(-(d3+dcm2))+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[7 3]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[7 3]),SaveRhoMarker) + rho(RhoRowFinder(N,[7 3])).*exp(-1i.*(-(d1+dcm1)-(-(d3+dcm2))+D1)*h_t*n_t);
                    
                    SavedRho1(RhoRowFinder(N,[1 5]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[1 5]),SaveRhoMarker) + rho(RhoRowFinder(N,[1 5])).*exp( 1i.*( (d2-dcm1)-(  d4-dcm2 )+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[1 6]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[1 6]),SaveRhoMarker) + rho(RhoRowFinder(N,[1 6])).*exp( 1i.*( (  -dcm1)-(  d4-dcm2 )+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[1 7]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[1 7]),SaveRhoMarker) + rho(RhoRowFinder(N,[1 7])).*exp( 1i.*(-(d1+dcm1)-(  d4-dcm2 )+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[2 5]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[2 5]),SaveRhoMarker) + rho(RhoRowFinder(N,[2 5])).*exp( 1i.*( (d2-dcm1)-(    -dcm2 )+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[2 6]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[2 6]),SaveRhoMarker) + rho(RhoRowFinder(N,[2 6])).*exp( 1i.*( (  -dcm1)-(    -dcm2 )+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[2 7]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[2 7]),SaveRhoMarker) + rho(RhoRowFinder(N,[2 7])).*exp( 1i.*(-(d1+dcm1)-(    -dcm2 )+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[3 5]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[3 5]),SaveRhoMarker) + rho(RhoRowFinder(N,[3 5])).*exp( 1i.*( (d2-dcm1)-(-(d3+dcm2))+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[3 6]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[3 6]),SaveRhoMarker) + rho(RhoRowFinder(N,[3 6])).*exp( 1i.*( (  -dcm1)-(-(d3+dcm2))+D1)*h_t*n_t);
                    SavedRho1(RhoRowFinder(N,[3 7]),SaveRhoMarker) = SavedRho1(RhoRowFinder(N,[3 7]),SaveRhoMarker) + rho(RhoRowFinder(N,[3 7])).*exp( 1i.*(-(d1+dcm1)-(-(d3+dcm2))+D1)*h_t*n_t);
           
                    SaveRhoMarker=SaveRhoMarker+1;

            end
            
            
            % This finds the K matrix.  See the comments at the beginning for more info.
            K(:,1)=h_t.*(F*rho(:));
            
            K(:,2)=h_t.*(F*[rho(1)+h_t/2;...
                rho(2:end)+K(:,1)/2]);
            
            K(:,3)=h_t.*(F*[rho(1)+h_t/2;...
                rho(2:end)+K(:,2)/2]);
            
            K(:,4)=h_t.*(F*[rho(1)+h_t;...
                rho(2:end)+K(:,3)]);
            
            % This calculates the n+1 column in the 'a' matrix using the K matrix
            
            rho(2:end) = rho(2:end)+(K(:,1)+2*K(:,2)+2*K(:,3)+K(:,4))/6;
            
            %{
            rho=rho';
        Total_Population(n_t) = rho(RhoRowFinder(N,[1 1]),n_t)+rho(RhoRowFinder(N,[2 2]),n_t)+rho(RhoRowFinder(N,[3 3]),n_t)+...
                             rho(RhoRowFinder(N,[4 4]),n_t)+rho(RhoRowFinder(N,[5 5]),n_t)+rho(RhoRowFinder(N,[6 6]),n_t)+...
                             rho(RhoRowFinder(N,[7 7]),n_t);

        if round(real(Total_Population(n_t)),10)-1 ~= 0
            display(n_t)
           display('Population not conserved')
        end
        rho=rho';
            %}
        end
    end
end


toc;
display('Done!')
%%
clear
load('/Users/Nick/Dropbox/7LevelSim/Results/Atoms40001_277mW_20161121174005.mat')

% This adds all the populations
Pop_g = (SavedRho1(RhoRowFinder(N,[1 1]),:)+SavedRho1(RhoRowFinder(N,[2 2]),:)+SavedRho1(RhoRowFinder(N,[3 3]),:))/(2*N_A);

Pop_e = (SavedRho1(RhoRowFinder(N,[5 5]),:)+SavedRho1(RhoRowFinder(N,[6 6]),:)+SavedRho1(RhoRowFinder(N,[7 7]),:))/(2*N_A);

% This adds all the coherences
Coherences = sum([SavedRho1(RhoRowFinder(N,[5 1]),:);SavedRho1(RhoRowFinder(N,[5 2]),:);SavedRho1(RhoRowFinder(N,[5 3]),:);...
                   SavedRho1(RhoRowFinder(N,[6 1]),:);SavedRho1(RhoRowFinder(N,[6 2]),:);SavedRho1(RhoRowFinder(N,[6 3]),:);...
                   SavedRho1(RhoRowFinder(N,[7 1]),:);SavedRho1(RhoRowFinder(N,[7 2]),:);SavedRho1(RhoRowFinder(N,[7 3]),:)])/(2*N_A);
% Coherences = SavedRho1(RhoRowFinder(N,[6 2]),:);
display(['Max Coherence = ',num2str(max(abs(Coherences)))])

% This cuts out the first few points before the laser turns on
cut=20;
Pop_g(1:cut)=[];
Pop_e(1:cut)=[];
Coherences(1:cut)=[];
TimesToSave=TimesToSave(1:length(Pop_g));

figure
subplot(2,1,1)
plot(TimesToSave,real(Pop_e-Pop_g))
title({'Population',['Isotope ',num2str(Isos)]})

zoom on
% ylim([-1 0])

subplot(2,1,2)
hold all
plot(TimesToSave,abs(Coherences))
plot(TimesToSave,real(Coherences),'-r')
% plot(TimesToSave,abs(aa));%.*conj(Coherences))
title Coherences
zoom on



% cd('C:\Users\YavuzLab\Dropbox\7LevelSim\Results')
% save(['Atoms',num2str(N_A),'_',[num2str(round(P_p*1000)),'mW_',datestr(now,'yyyymmddHHMMSS')]]);

% display(['p15 = ',num2str(((d2-dcm1)-(d4-dcm2)+D1)/(2*pi*1e6)),' MHz'])
% display(['p16 = ',num2str(((-dcm1)-(d4-dcm2)+D1)/(2*pi*1e6)),' MHz'])
% display(['p17 = ',num2str((-(d1+dcm1)-(d4-dcm2)+D1)/(2*pi*1e6)),' MHz'])
% display(['p25 = ',num2str(((d2-dcm1)-(-dcm2)+D1)/(2*pi*1e6)),' MHz'])
% display(['p26 = ',num2str(((-dcm1)-(-dcm2)+D1)/(2*pi*1e6)),' MHz'])
% display(['p27 = ',num2str((-(d1+dcm1)-(-dcm2)+D1)/(2*pi*1e6)),' MHz'])
% display(['p35 = ',num2str(((d2-dcm1)-(-(d3+dcm2))+D1)/(2*pi*1e6)),' MHz'])
% display(['p36 = ',num2str(((-dcm1)-(-(d3+dcm2))+D1)/(2*pi*1e6)),' MHz'])
% display(['p37 = ',num2str((-(d1+dcm1)-(-(d3+dcm2))+D1)/(2*pi*1e6)),' MHz'])
%% Histogram of detunings
figure
hold all
hist=histogram(Broadening/(2*pi),100,'Normalization','count');
line([-FWHM/2 -FWHM/2],[0 max(hist.Values)],'color','k')
line([FWHM/2 FWHM/2],[0 max(hist.Values)],'color','k')
zoom on



