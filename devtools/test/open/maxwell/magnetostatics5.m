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
Rin = 1.75; %Rin = min(src(1,:));
R0 = 3.0; %R0 = mean(src(1,:));
z0 = 0.0;

origin = [0,0];
all_modes = false;
m0 = 1;
mA = 2;

rp  = src(1,:).';

targC_want = [Rin; 0.0];
targ_idx = 1;
bestdist = inf;
for i = 1:size(src,2)
    dist = norm(src(:,i) - targC_want);
    if dist < bestdist
        bestdist = dist;
        targ_idx = i;
    end
end
targC = src(:,targ_idx);

%% ============================================================
% g = n dot Htrue on the boundary
% ============================================================

% Targets are the generating curve points (r,z)
targ = src;            % 2 x npts
srcL = [R0; z0];       % 2 x 1 (single ring in meridian)

[val, grad] = chnk.axissymlap2d.green_modal(srcL, targ, origin, mA, all_modes);

Atheta = val;           % npts x 1
dAr    = grad(:,1,1);        % d/dr Atheta
dAz    = grad(:,1,3);        % d/dz Atheta

Hr = -dAz;                   % H_r = -d/dz Atheta
Hz = dAr + Atheta./rp;        % H_z = d/dr Atheta + Atheta/r

% n*H^inc on the surface (independent of azimuth angle for this symmetric ring)
g0 = -(n_src(1,:)'.*Hr + n_src(2,:)'.*Hz);   % npts x 1

% compatibility check
wsurf = 2*pi * rp .* wsrc;
fprintf('Compatibility integral int g dS = %.6e\n', wsurf.'*g0);

%% ============================================================
% b = flux of Htrue through spanning disk A
% ============================================================
srcL = [R0; z0];
[valC, ~] = chnk.axissymlap2d.green_modal(srcL, targC, origin, mA, all_modes);
AthetaC = valC(1,1);
b = 2*pi*Rin*AthetaC;

%% q = alpha * n * curl A[L]
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
%opts.rcip = false;
opts.forcesmooth = false;
opts.l2scale = false;
opts.sing = 'log';

Sp1 = kernel('axissymlap','sprime',mA,all_modes);
S1  = kernel('axissymlap','s',mA,all_modes);

% IMPORTANT: no 0.5*I here (we want raw d/drS and d/dzS, not the Neumann jump)
Sp_r = chunkermat_normal(chnk_r, Sp1, opts);
Sp_z = chunkermat_normal(chnk_z, Sp1, opts);
Smat  = chunkermat_normal(chnkr, S1, opts);

dens_h = 1 ./ rp;
Atheta = Smat * dens_h;
dAr    = -Sp_r * dens_h;
dAz    = -Sp_z * dens_h;

Hr = -dAz;
Hz = dAr + Atheta./rp;

q0 = -(n_src(1,:)'.*Hr + n_src(2,:)'.*Hz);

%% ============================================================
% Robust genus-1 flux row for torus: compute f
% ============================================================
w = chnkr.wts(:);
chnkA = get_disk_curve(Rin, 0.0);   % your existing helper (0..rcap)

targA = chnkA.r(:,:);         % 2 x nA
wA    = chnkA.wts(:);         % dr weights
rA    = targA(1,:).';
wSurf = 2*pi * (rA .* wA);    % revolved surface weights

% mode-0 Green derivative wrt target z for grad S · n_A (with n_A = +z)
[~, gradA] = chnk.axissymlap2d.green_modal(src, targA, origin, m0, all_modes);
dGdz = gradA(:,:,3);          % nA = +e_z convention

% row action on sigma: integral_A (grad S[sigma] · nA) dA
% gradA is nA-by-npts, w is source ds weights
f = wSurf.' * (dGdz .* (w.'));

%% --- c = flux of H_L through spanning disk (Stokes) ---
%opts.sing = 'log';
%AthetaC_h = Smat(targ_idx,:) * dens_h;
%c = 2*pi*src(1,targ_idx)*AthetaC_h;


%% ============================================================
% c = 2*pi * int_0^Rin Hz(r,0) * r dr
% where Hz = d/dr Atheta + Atheta/r
% ============================================================

% radial chunk from 0 to Rcut along z=0
chnkAeps = get_disk_curve(Rin, 0.0);
targAeps = chnkAeps.r(:,:);      % 2 x nA
rAeps    = targAeps(1,:).';      % nA x 1
wAeps    = chnkAeps.wts(:);      % dr weights

% evaluate d/dr Atheta on disk targets
tinfo_r = [];
tinfo_r.r = targAeps;
tinfo_r.n = repmat([1;0], 1, size(targAeps,2));   % e_r at each target

dArA = -chunkerkerneval(chnkr, Sp1, dens_h, tinfo_r, opts);  % nA x 1

% evaluate Atheta on disk targets
AthetaA = chunkerkerneval(chnkr, S1, dens_h, targAeps, opts); % nA x 1

% H_z on the disk
HzA = dArA + AthetaA ./ rAeps;

% direct flux integral over truncated disk
%c = 2*pi * sum(HzA .* rAeps .* wAeps);
c = 2*pi * sum(HzA .* wAeps)


%% --- Build mode-0 A operator ---
Sp0 = kernel('axissymlap','sprime',m0,all_modes);
A0 = chunkermat_normal(chnkr, Sp0, opts) + 0.5*eye(npts);

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

rt = 5.0;
zt = 2.0;
targT = [rt; zt];

[val_true, grad_true] = chnk.axissymlap2d.green_modal(srcL, targT, origin, mA, all_modes);

Atheta_true = val_true(1,1);
dAr_true    = grad_true(1,1,1);
dAz_true    = grad_true(1,1,3);

Hr_true = -dAz_true;
Hz_true =  dAr_true + Atheta_true/rt;

% --- grad S[sigma] (mode 0) ---
opts_eval = [];
opts_eval.forcesmooth = false;
opts_eval.verb = false;
opts_eval.quadkgparams = {'RelTol',1e-10,'AbsTol',1e-10};
opts_eval.sing = 'log';

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

%% --- curl S[h] via A_theta = S[h] (mode 1) ---
dAr = -chunkerkerneval(chnkr, Sp1, dens_h, tinfo_r, opts_eval);
dAz = -chunkerkerneval(chnkr, Sp1, dens_h, tinfo_z, opts_eval);
Atheta = chunkerkerneval(chnkr, S1, dens_h, targT, opts_eval);

Hr_curl = -dAz;
Hz_curl = dAr + Atheta/rt;



%% full solution
Hr = Hr_phi + alpha * Hr_curl;
Hz = Hz_phi + alpha * Hz_curl;

fprintf('\n=== Field eval at (r=%.6g, z=%.6g) ===\n', rt, zt);
fprintf('Htrue  [Hr,Hz] = [% .6e, % .6e]\n', Hr_true,  Hz_true);
fprintf('H [Hr,Hz] = [% .6e, % .6e]\n', Hr, Hz);

Htrue_vec = [Hr_true; Hz_true];
H_vec     = [Hr; Hz];
relerr = norm(H_vec - Htrue_vec, 2) / norm(Htrue_vec, 2);
fprintf('Relative error : %.16e\n', relerr);





%% geometry functions
function [chnkobj] = get_torus_geometry()
    pref = [];
    pref.k = 16; % points per chunk
    %pref.nchmax = 4;

    cparams = [];
    %cparams.eps = 1.0e-10;
    %cparams.nover = 1;
    cparams.ifclosed = true;
    cparams.ta = 0;
    cparams.tb = 2*pi;
    cparams.maxchunklen = 2;
    cparams.nchmin = 16;

    ctr = [3 0];
    narms = 0;
    amp = 0.25;

    chnkobj = chunkerfunc(@(t) starfish(t, narms, amp, ctr), cparams, pref); 
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
    cparams.nchmin = 16;        % increase if you want more radial resolution

    % chunkgraph expects verts as 2 x nv
    verts = [a b; 0 0];        % two vertices: (a,0) and (b,0)
    edge2verts = [1; 2];
    fchnks = [];

    chnkA = chunkgraph(verts, edge2verts, fchnks, cparams, pref);
    chnkA = balance(chnkA);
end


