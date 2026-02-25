%{
- 3D Laplace's equation
- Neumann boundary condition
- sphere boundary
- Single and Double layer potential representation
- mth mode (non-axisymmetric B.C.)
%}

clearvars; 
close all;
format long e;

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
% H^true = curl S[L]
% H = grad S[sigma] + alpha * curl S[h]
% h = 1/r * e_theta
% 

%% geometry
[chnkr] = get_torus_geometry();
%plot(chnkr);

npts  = chnkr.npt;
src   = chnkr.r(:,:);      % generating curve points [r; z]
n_src = chnkr.n(:,:);      % generating curve normals [nr; nz]
wsrc  = chnkr.wts(:);      % ds weights on generating curve
Rin = min(src(1,:));

origin = [0,0];
all_modes = false;
m0 = 1;                    % Fourier mode 0

%% ============================================================
% Manufactured Htrue EXACTLY like sphere case:
% Htrue = grad(phi_true),  phi_true = G0(srcQ1) - G0(srcQ2)
% (two mode-0 ring sources of opposite strength)
% ============================================================
srcQ1 = [3.0;  0.20];   % [rq; zq]-
srcQ2 = [3.0; -0.20];

[~, gradQ1] = chnk.axissymlap2d.green_modal(srcQ1, src, origin, m0, all_modes);
[~, gradQ2] = chnk.axissymlap2d.green_modal(srcQ2, src, origin, m0, all_modes);

% Htrue on boundary
Hr_true_bdy = gradQ1(:,1,1) - gradQ2(:,1,1);
Hz_true_bdy = gradQ1(:,1,3) - gradQ2(:,1,3);

% Neumann data g = n·Htrue on boundary
g0 = -(n_src(1,:)'.*Hr_true_bdy + n_src(2,:)'.*Hz_true_bdy);

% (optional) compatibility check
wsurf = 2*pi * src(1,:)' .* wsrc;
fprintf('Compatibility integral int g dS = %.6e\n', wsurf.'*g0);

%% ============================================================
% b = flux of Htrue through spanning disk A (direct computation)
% Htrue = grad(G0(srcQ1) - G0(srcQ2))
% Use cutoff + extrapolation because the disk touches the torus at the rim
% ============================================================

eps_list_b = [1e-1, 7e-2, 5e-2, 3e-2, 2e-2, 1e-2];
b_eps = zeros(numel(eps_list_b),1);

for ie = 1:numel(eps_list_b)
    epsr = eps_list_b(ie);
    rcap = Rin - epsr;
    if rcap <= 0
        error('Rin-eps <= 0 in b computation. Reduce eps_list_b.');
    end

    % Truncated spanning disk curve (z=0, r in [0, rcap])
    chnkA_b = get_disk_curve(rcap, 0.0);

    targA = chnkA_b.r(:,:);      % 2 x nA
    wA    = chnkA_b.wts(:);      % dr weights
    rA    = targA(1,:).';
    wSurfA = 2*pi * (rA .* wA);  % revolved disk weights
    [~, gradQ1] = chnk.axissymlap2d.green_modal(srcQ1, targA, origin, m0, all_modes);
    [~, gradQ2] = chnk.axissymlap2d.green_modal(srcQ2, targA, origin, m0, all_modes);
    
    Hz_true_A = gradQ1(:,1,3) - gradQ2(:,1,3);

    b_eps(ie) = wSurfA.' * Hz_true_A;
end

% Extrapolate eps -> 0
deg_b = min(2, numel(eps_list_b)-1);
pb = polyfit(eps_list_b(:), b_eps(:), deg_b);
b = polyval(pb, 0);

fprintf('b (direct disk flux, extrapolated) = %.16e\n', b);

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

mA = 2;
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

q0 = -(n_src(1,:)'.*Hr + n_src(2,:)'.*Hz);

%fprintf('q0 range: [%g, %g]\n', min(q0), max(q0));
%fprintf('max |q_m| per mode:\n'); disp(max(abs(q_m),[],2).');

%% ============================================================
% Robust genus-1 flux row for torus: compute f and c by cutoff+extrapolation
% ============================================================
w = chnkr.wts(:);
dens_h = 1 ./ src(1,:).';

eps_list = [1e-1, 7e-2, 5e-2, 3e-2, 2e-2, 1e-2];
neps = numel(eps_list);

f_eps = zeros(neps, npts);
c_eps = zeros(neps, 1);

for ie = 1:neps
    epsr = eps_list(ie);
    rcap = Rin - epsr;
    if rcap <= 0
        error('Rin-eps <= 0; reduce eps_list.');
    end

    % ---- c(eps): flux of curl S[h] through disk of radius rcap via Stokes ----
    targC = [rcap; 0.0];
    [valC_h, ~] = chnk.axissymlap2d.green_modal(src, targC, origin, mA, all_modes);
    AthetaC_h = valC_h * (w .* dens_h);
    c_eps(ie) = 2*pi*rcap * AthetaC_h;

    % ---- f(eps): flux row for grad S[sigma] over truncated spanning disk ----
    chnkA = get_disk_curve(rcap, 0.0);   % your existing helper (0..rcap)

    targA = chnkA.r(:,:);         % 2 x nA
    wA    = chnkA.wts(:);         % dr weights
    rA    = targA(1,:).';
    wSurf = 2*pi * (rA .* wA);    % revolved surface weights

    % mode-0 Green derivative wrt target z for grad S · n_A (with n_A = +z)
    [~, gradA] = chnk.axissymlap2d.green_modal(src, targA, origin, 1, all_modes);
    dGdz = gradA(:,:,3);          % nA = +e_z convention

    % row action on sigma: integral_A (grad S[sigma] · nA) dA
    % gradA is nA-by-npts, w is source ds weights
    f_eps(ie,:) = wSurf.' * (dGdz .* (w.'));
end

% Extrapolate eps -> 0
deg = min(2, neps-1);

pc = polyfit(eps_list(:), c_eps(:), deg);
c = polyval(pc, 0);

f = zeros(1,npts);
for j = 1:npts
    pj = polyfit(eps_list(:), f_eps(:,j), deg);
    f(j) = polyval(pj, 0);
end


%% --- Build mode-0 A operator ---
opts = [];
opts.rcip = false;
opts.forcesmooth = false;
opts.l2scale = false;

Sp0 = kernel('axissymlap','sprime',m0,all_modes);
A0 = chunkermat_normal(chnkr, Sp0, opts) + 0.5*eye(npts);
A0 = A0 + onesmat(chnkr);

%% Assemble and solve the block system

M = [A0, q0; f, c];
rhs = [g0; b];

sol = M \ rhs;
sigma0 = sol(1:npts);
alpha  = sol(end);

bc_res = A0*sigma0 + q0*alpha - g0;
fprintf('BC residual (block row) = %.6e\n', norm(bc_res,inf));

flux_res = f*sigma0 + c*alpha - b;
fprintf('Flux residual (extrapolated row) = %.6e\n', flux_res);

fprintf('alpha = %.16e\n', alpha);

%% ============================================================
%  Evaluate H and Htrue at an off-surface point (rt,zt)
%  (meridian components Hr,Hz; optional Cartesian at angle theta)
% ============================================================

rt = 6.0;
zt = 1.0;
targT = [rt; zt];

[~, gradQ1] = chnk.axissymlap2d.green_modal(srcQ1, targT, origin, m0, all_modes);
[~, gradQ2] = chnk.axissymlap2d.green_modal(srcQ2, targT, origin, m0, all_modes);

% Htrue on boundary
Hr_true = gradQ1(1,1,1) - gradQ2(1,1,1);
Hz_true = gradQ1(1,1,3) - gradQ2(1,1,3);

% --- grad S[sigma] (mode 0) ---
opts_eval = [];
opts_eval.forcesmooth = false;
opts_eval.verb = false;
opts_eval.quadkgparams = {'RelTol',1e-10,'AbsTol',1e-10};
opts_eval.sing = 'log';

Sp0 = kernel('axissymlap','sprime',m0,all_modes);

% target as ptinfo with fake normal = e_r
tinfo_r = [];
tinfo_r.r = targT;
tinfo_r.n = [1;0];
Hr_phi = -chunkerkerneval(chnkr, Sp0, sigma0, tinfo_r, opts_eval);

% target as ptinfo with fake normal = e_z
tinfo_z = [];
tinfo_z.r = targT;
tinfo_z.n = [0;1];
Hz_phi = -chunkerkerneval(chnkr, Sp0, sigma0, tinfo_z, opts_eval);

%% --- curl S[h] via A_theta = S[h] (mode 1 / mA=2) ---
[val_h, grad_h] = chnk.axissymlap2d.green_modal(src, targT, origin, mA, all_modes);

dens_h = 1 ./ src(1,:).';

SkerA = kernel('axissymlap','s',mA,all_modes);
SpA = kernel('axissymlap','sprime',mA,all_modes);

tinfo = [];
tinfo.r = targT;     % [rt; zt]
Atheta_h = chunkerkerneval(chnkr, SkerA, dens_h, tinfo, opts_eval);

% d/dr A_theta (remember sprime = -n·grad)
tinfo_r = [];
tinfo_r.r = targT;
tinfo_r.n = [1;0];
dAr_h = -chunkerkerneval(chnkr, SpA, dens_h, tinfo_r, opts_eval);

% d/dz A_theta
tinfo_z = [];
tinfo_z.r = targT;
tinfo_z.n = [0;1];
dAz_h = -chunkerkerneval(chnkr, SpA, dens_h, tinfo_z, opts_eval);

Hr_curl = -dAz_h;
Hz_curl =  dAr_h + Atheta_h/rt;

Hr = Hr_phi + alpha * Hr_curl;
Hz = Hz_phi + alpha * Hz_curl;

fprintf('\n=== Field eval at (r=%.6g, z=%.6g) ===\n', rt, zt);
fprintf('Htrue  [Hr,Hz] = [% .6e, % .6e]\n', Hr_true,  Hz_true);
fprintf('H [Hr,Hz] = [% .6e, % .6e]\n', Hr, Hz);


%% geometry functions
function [chnkobj] = get_torus_geometry()
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
end

function [chnkobj] = get_sphere_geometry()
    pref = [];
    pref.k = 16; % points per chunk

    cparams = [];
    cparams.eps = 1.0e-10;
    cparams.nover = 1;
    cparams.ifclosed = false;
    cparams.ta = -pi/2;
    cparams.tb = pi/2;
    cparams.maxchunklen = 2;
    %cparams.nchmin = 8;

    narms = 0;
    amp = 0.0;

    chnkobj = chunkerfunc(@(t) starfish(t, narms, amp), cparams, pref); 
    chnkobj = sort(chnkobj);
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


