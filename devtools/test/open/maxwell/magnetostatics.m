%{
- 3D Laplace's equation
- Neumann boundary condition
- Torus boundary
- Single and Double layer potential representation
- mth mode (non-axisymmetric B.C.)
%}

clearvars; 
close all;
format long e;

%% geometry
% target is where we evaluate the solution
[chnkr,target,charge1,charge2] = get_torus_geometry();

npts = chnkr.npt; % total number of points in discretization
src = chnkr.r(:,:); % generating curve
n_src = chnkr.n(:,:); % normals

% plot geometry
%plot(chnkr);

p_modes = 3; % number of positive fourier modes
n_modes = 2*p_modes + 1; % number of fourier modes (must be odd for pos/0/neg)
n_angles = n_modes; % number of angles/rotations
modes = -p_modes:p_modes;
strength = 1.0;

%% === PRECOMPUTE for magnetostatics setup ===
% the final system should be:
% 
%
% [ A  q ] [ sigma ] = [ g ]
% [ f  c ] [ alpha ]   [ b ]
%
% sizes:
% A is 11x128x128
% q is 11x128
% g is 11x128
% f should be 1x128
% c is 1x1
% b is 1x1
%
% sigma is 11x128
% alpha is 1x1
%
%
% H^inc = curl S[L]
% H = grad S[sigma] + alpha * curl S[h]
% h = 1/r * e_theta
% 

%% g = -n*H^inc
% incident field from a ring L
% Ring L is a circle around z-axis at (R0,z0) with current tangent e_theta'
R0 = mean(src(1,:));   % choose ring radius
z0 = 0.0;              % choose ring height

origin = [0,0];
all_modes = false;

% We need Fourier mode n=1
mA = 2;

% Targets are the generating curve points (r,z)
targ = src;            % 2 x npts
srcL = [R0; z0];       % 2 x 1 (single ring in meridian)

% val  = A_theta(r,z)
% grad(:,:,1) = d/dr (target), grad(:,:,3) = d/dz (target)
[val, grad] = chnk.axissymlap2d.green_modal(srcL, targ, origin, mA, all_modes);

Atheta = val;           % npts x 1
dAr    = grad(:,1,1);        % d/dr Atheta
dAz    = grad(:,1,3);        % d/dz Atheta

r = targ(1,:).';             % npts x 1

Hr = -dAz;                   % H_r = -d/dz Atheta
Hz = dAr + Atheta./r;        % H_z = d/dr Atheta + Atheta/r

% n*H^inc on the surface (independent of azimuth angle for this symmetric ring)
g0 = -n_src(1,:)'.*Hr + n_src(2,:)'.*Hz;   % npts x 1

% should be concentrated in the 0th Fourier mode
%fprintf('max |g_m| per mode: \n');
%disp(max(abs(g_m),[],2).');

%% b = -int_A H^inc * n_A dA = -int_C S[J]*t dl = -2pi Rin * Atheta(Rin,0)
% used stokes since H^inc = curl S[J] using Biot-Savart Law
Rin = min(src(1,:));      % radius of the inner hole cap disk

% target point on the boundary circle C in meridian coords (r,z)=(Rin,0)
targC = [Rin; 0.0];
srcL = [R0; z0];

[valC_inc, ~] = chnk.axissymlap2d.green_modal(srcL, targC, origin, mA, all_modes);
AthetaC_inc = valC_inc(1,1);

b = -2*pi*Rin*AthetaC_inc;

%fprintf('b (Stokes) = %.16e  (for nA = +zhat)\n', b);

%% q = alpha * n * curl S[h]
% Make a copy of the chunker
chnk_r = chnkr;

% Set n = e_r everywhere
fake_n = zeros(size(chnk_r.n));
fake_n(1,:,:) = 1;   % n_r = 1
fake_n(2,:,:) = 0;   % n_z = 0
chnk_r.n = fake_n;

chnk_z = chnkr;

% Set n = e_z everywhere
fake_n = zeros(size(chnk_z.n));
fake_n(1,:,:) = 0;
fake_n(2,:,:) = 1;
chnk_z.n = fake_n;

opts = [];
opts.rcip = false;
opts.forcesmooth = false;
opts.l2scale = false;

Spker = kernel('axissymlap','sprime',mA,all_modes);

% IMPORTANT: no -0.5*I here (we want raw d/drS and d/dzS, not the Neumann jump)
Sp_r = chunkermat_normal(chnk_r, Spker, opts);   % gives -(d/dr S)
Sp_z = chunkermat_normal(chnk_z, Spker, opts);   % gives -(d/dz S)

% Scalar single-layer matrix
Sker  = kernel('axissymlap','s',mA,all_modes);
Smat  = chunkermat_normal(chnkr, Sker, opts);

dens_h = 1 ./ src(1,:).';
Atheta = Smat * dens_h;
dAr    = -(Sp_r * dens_h);
dAz    = -(Sp_z * dens_h);

r  = src(1,:).';
Hr = -dAz;
Hz = dAr + Atheta./r;

q0 = n_src(1,:)'.*Hr + n_src(2,:)'.*Hz;

%fprintf('q0 range: [%g, %g]\n', min(q0), max(q0));
%fprintf('max |q_m| per mode:\n'); disp(max(abs(q_m),[],2).');

%% c = int_A curl S[h] * n_A dA = 2pi Rin * Atheta_h(Rin,0)
% using stokes
srcG = src;
targC = [Rin; 0.0];
[valC_h, ~] = chnk.axissymlap2d.green_modal(srcG, targC, origin, mA, all_modes);

w    = chnkr.wts(:);                            % ds weights
AthetaC_h = valC_h * (w .* dens_h);             % already has pi r' in kernel

c = 2*pi*Rin*AthetaC_h;                         % flux of curl S[h] through A

%fprintf('c (Stokes, nA=+zhat) = %.16e\n', c);

%% f = int_A grad S[sigma] * n_A dA
% compute: f*sigma = 2*pi * ∫_0^{Rin} (d/dz S[sigma])(r,0) * r dr
% Build a chunker for the generating curve of the disk A: (r, z=0), r in [0,Rin]

eps0 = 1e-12;                 % avoid exactly r=0
chnkA = get_disk_curve(Rin, eps0);

targA = chnkA.r(:,:);         % 2 x nptA targets on segment
wA    = chnkA.wts(:);         % nptA x 1, ds weights (here ds = dr)
rA    = targA(1,:).';         % nptA x 1

% axisymmetric surface measure for revolving the segment:
wSurf = (2*pi) * (rA .* wA);  % nptA x 1

% Sources: torus generating curve nodes
srcG = src;                   % 2 x npts
wsrc = chnkr.wts(:);           % npts x 1 (ds weights on torus generating curve)

% Mode 0 for scalar potential phi = S[sigma]
m0 = 1;
origin = [0,0];
all_modes = false;

% Evaluate d/dz (target) of the modal Green kernel at disk targets
[~, gradA] = chnk.axissymlap2d.green_modal(srcG, targA, origin, m0, all_modes);
dGdz = gradA(:,:,3);           % (nptA x npts)

% Build row functional:
f = (wSurf.' * (dGdz .* (wsrc.')));   % 1 x npts


%% --- Build mode-0 A operator ---
opts = [];
opts.rcip = false;
opts.forcesmooth = false;
opts.l2scale = false;

% scalar mode index for your mapping: mode=0 -> m=1
m0 = 1;
Sp0 = kernel('axissymlap','sprime',m0,all_modes);

A0 = chunkermat_normal(chnkr, Sp0, opts) - 0.5*eye(npts);
A0 = A0 + onesmat(chnkr);

%% Assemble and solve the block system
% g0: your boundary data for mode 0 (npts x 1)

M = [A0, q0; f, c];
rhs = [g0; b];

sol = M \ rhs;
sigma0 = sol(1:npts);
alpha  = sol(end);

%% ================================
%  OPTION 3: DISCRETE MANUFACTURED SOLUTION TEST (MODE 0)
%  Choose sigma_star and alpha_star, manufacture RHS using your discrete operators,
%  solve, and verify you recover sigma_star and alpha_star.
%% ================================

do_mms = false;   % <-- set false to go back to the physical ring test

if do_mms
    % --- Ensure A0, q0, f, c already exist in workspace ---
    % A0 : (npts x npts) Neumann operator for scalar single-layer, mode 0
    % q0 : (npts x 1)   boundary trace n·curl S[h], mode 0
    % f  : (1 x npts)   flux functional row
    % c  : scalar       flux contribution from curl S[h]
    %
    % You already built these above.

    % --- Pick a smooth manufactured sigma_star on the generating curve ---
    % (Any smooth vector is fine; avoid something nearly-constant if you want
    % a stronger test.)
    r = src(1,:).';
    z = src(2,:).';

    % Example: smooth combination of r,z (axisymmetric)
    sigma_star = 0.7*cos(2.0*z) + 0.4*sin(3.0*z) + 0.2*cos(1.5*r);

    % Optional: remove mean (not required if you already stabilized A0 with onesmat)
    sigma_star = sigma_star - mean(sigma_star);

    % --- Choose alpha_star ---
    alpha_star = 1.23456789;   % any nonzero scalar

    % --- Manufacture RHS by applying the SAME discrete operators ---
    rhs1_star = A0*sigma_star + q0*alpha_star;      % npts x 1
    rhs2_star = f*sigma_star + c*alpha_star;        % scalar (1x1)

    % --- Solve the coupled block system ---
    M   = [A0, q0; f, c];
    rhs = [rhs1_star; rhs2_star];

    sol     = M \ rhs;
    sigma0  = sol(1:npts);
    alpha   = sol(end);

    % --- Compare to manufactured truth ---
    err_sigma = sigma0 - sigma_star;
    err_alpha = alpha  - alpha_star;

    fprintf('\n=== MMS (Option 3) recovery errors ===\n');
    fprintf('alpha_star = %.16e\n', alpha_star);
    fprintf('alpha_rec  = %.16e\n', alpha);
    fprintf('|alpha_err| = %.3e\n', abs(err_alpha));

    fprintf('\nSigma errors:\n');
    fprintf('||err_sigma||_inf = %.3e\n', norm(err_sigma, inf));
    fprintf('||err_sigma||_2   = %.3e\n', norm(err_sigma, 2));
    fprintf('rel ||err_sigma||2 / ||sigma_star||2 = %.3e\n', norm(err_sigma,2)/max(norm(sigma_star,2),1e-300));

    % --- Residual check (should also be roundoff) ---
    rBC   = A0*sigma0 + q0*alpha - rhs1_star;
    rFlux = f*sigma0  + c*alpha  - rhs2_star;

    fprintf('\nResiduals (MMS):\n');
    fprintf('||rBC||_inf  = %.3e\n', norm(rBC, inf));
    fprintf('||rBC||_2    = %.3e\n', norm(rBC, 2));
    fprintf('rFlux        = %.16e\n', rFlux);
else
    % --- Your original physical solve goes here ---
    % (keep your existing ring-based rhs1=-g0, rhs2=b_rhs etc.)
    % Rebuild A0 (mode 0 scalar Neumann operator) exactly as used in the solve        
    % mode-0 scalar Laplace corresponds to m = abs(0)+1 = 1 in your convention
    m0 = 1;
    Sp0 = kernel('axissymlap','sprime',m0,all_modes);
    
    A0 = chunkermat_normal(chnkr, Sp0, opts) - 0.5*eye(npts);
    A0 = A0 + onesmat(chnkr);   % only for mode 0
    
    % --- 1) Boundary condition residual ---
    % If enforcing n·Htot = 0: A0*sigma0 + alpha*q0 = -g0
    rBC = A0*sigma0 + alpha*q0 - g0;
    
    fprintf('\n=== Boundary residual (mode 0) ===\n');
    fprintf('||rBC||_inf  = %.3e\n', norm(rBC, inf));
    fprintf('||rBC||_2    = %.3e\n', norm(rBC, 2));
    fprintf('rel ||rBC||2 / ||g0||2 = %.3e\n', norm(rBC,2)/max(norm(g0,2),1e-300));
end

%% ============================================================
%  Evaluate H, Hinc, Htot at an off-surface point (rt,zt)
%  (meridian components Hr,Hz; optional Cartesian at angle theta)
%% ============================================================

rt = R0 + 0.35;     % choose a point in the exterior (NOT on boundary)
zt = 0.25;
targT = [rt; zt];

wsrc = chnkr.wts(:);          % ds weights on generating curve
rsrc = src(1,:).';            % source r' values
dens_h = 1 ./ rsrc;           % h = 1/r' (source-side)

%% --- 1) Incident field Hinc from ring at (R0,z0) ---
% Use mode n=1 -> mA=2 to get A_theta and its derivatives
[val_inc, grad_inc] = chnk.axissymlap2d.green_modal(srcL, targT, origin, mA, all_modes);

Atheta_inc = val_inc(1,1);
dAr_inc    = grad_inc(1,1,1);       % d/dr at target
dAz_inc    = grad_inc(1,1,3);       % d/dz at target

Hr_inc = -dAz_inc;
Hz_inc =  dAr_inc + Atheta_inc/rt;

%% --- 2) Scattered grad S[sigma] (mode 0) ---
m0 = 1;
[~, grad0] = chnk.axissymlap2d.green_modal(src, targT, origin, m0, all_modes);

wsigma = wsrc .* sigma0;            % kernel already includes π r' convention -> integrate with ds weights
dphidr = grad0(:,:,1) * wsigma;     % scalar
dphidz = grad0(:,:,3) * wsigma;

Hr_phi = dphidr;
Hz_phi = dphidz;

%% --- 3) Scattered curl S[h] via A_theta = S[h] (mode 1 / mA=2) ---
[val_h, grad_h] = chnk.axissymlap2d.green_modal(src, targT, origin, mA, all_modes);

wh = wsrc .* dens_h;

Atheta_h = val_h * wh;
dAr_h    = grad_h(:,:,1) * wh;
dAz_h    = grad_h(:,:,3) * wh;

Hr_curl = -dAz_h;
Hz_curl =  dAr_h + Atheta_h/rt;

Hr_scat = Hr_phi + alpha * Hr_curl;
Hz_scat = Hz_phi + alpha * Hz_curl;

%% --- 4) Totals ---
Hr_tot = Hr_scat + Hr_inc;
Hz_tot = Hz_scat + Hz_inc;

fprintf('\n=== Field eval at (r=%.6g, z=%.6g) ===\n', rt, zt);
fprintf('Hinc  [Hr,Hz] = [% .6e, % .6e]\n', Hr_inc,  Hz_inc);
fprintf('Hscat [Hr,Hz] = [% .6e, % .6e]\n', Hr_scat, Hz_scat);
fprintf('Htot  [Hr,Hz] = [% .6e, % .6e]\n', Hr_tot,  Hz_tot);
fprintf('|Hinc|  = %.6e\n', hypot(Hr_inc,  Hz_inc));
fprintf('|Hscat| = %.6e\n', hypot(Hr_scat, Hz_scat));
fprintf('|Htot|  = %.6e\n', hypot(Hr_tot,  Hz_tot));



%% geometry functions
function [chnkobj,target,charge1,charge2] = get_torus_geometry()
    pref = [];
    pref.k = 16; % points per chunk
    %pref.nchmax = 2;

    cparams = [];
    %cparams.eps = 1.0e-10;
    %cparams.nover = 1;
    cparams.ifclosed = true;
    cparams.ta = 0;
    cparams.tb = 2*pi;
    cparams.maxchunklen = 2;
    cparams.nchmin = 8;

    ctr = [3 0];
    narms = 0;
    amp = 0.25;

    chnkobj = chunkerfunc(@(t) starfish(t, narms, amp, ctr), cparams, pref); 
    chnkobj = sort(chnkobj);

    target = [3;0.0;-0.7];
    charge1 = [1.0;0.0;3.0];
    charge2 = [1.0;0.0;-3.0];
end

function chnkA = get_disk_curve(Rin, eps0)
    if nargin < 2
        eps0 = 0.0;
    end
    a = max(eps0, 0.0);
    b = Rin;

    pref = [];
    pref.k = 16;               % order per chunk (match your torus pref if you want)

    cparams = [];
    cparams.ta = 0;
    cparams.tb = 1;
    cparams.nchmin = 8;        % increase if you want more radial resolution

    % chunkgraph expects verts as 2 x nv
    verts = [a b; 0 0];        % two vertices: (a,0) and (b,0)
    edge2verts = [1; 2];
    fchnks = [];

    chnkA = chunkgraph(verts, edge2verts, fchnks, cparams, pref);
    chnkA = balance(chnkA);
end
