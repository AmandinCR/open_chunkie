clearvars;
close all;
format long e;

%% ============================================================
%  Axisymmetric open annulus solver with D'S scalar formulation
%
%  Unknowns are interleaved as
%       x = [sigma_1; mu_1; sigma_2; mu_2; ...; sigma_n; mu_n; alpha]
%
%  The scalar part is represented using a temporary density mu:
%       mu + c*S[sigma] = 0              % sign convention chosen to match SD' example
%       scalar field phi = c*D[mu]
%
%  Therefore the boundary normal derivative uses D' applied to mu:
%       sigma + c*D'[mu] + q*alpha = g
%
%  Eliminating mu gives
%       sigma - c^2*D'*S[sigma] + q*alpha = g.
%
%  The flux row is the flux of the actual scalar field c*D[mu],
%  so it acts on the mu component, not directly on sigma.
% ============================================================

%% geometry: open annulus in the meridian plane
Rin_ann  = 2.0;
Rout_ann = 4.0;
zin = 0.0;
[chnkr] = get_annulus_geometry(Rin_ann, Rout_ann, zin);

npts  = chnkr.npt;
src   = chnkr.r(:,:);      % generating curve points [r; z]
n_src = chnkr.n(:,:);      % generating curve normals [nr; nz]
wsrc  = chnkr.wts(:);      % ds weights on generating curve

[Rin, minIdx] = min(src(1,:));
zin = src(2,minIdx);
origin = [0,0];
m0 = 1;
mA = 2;

rp     = src(1,:).';
dens_h = 1 ./ rp;
targC  = [Rin; zin];

%% kernels
S0 = kernel('axissymlap','s',m0,false);
S1  = kernel('axissymlap','s',mA,false);
Sp1 = kernel('axissymlap','sprime',mA,false);
Dp0 = kernel('axissymlap','dprime',m0,false);
Scurl = kernel('axissymlap','scurl',mA,false);

%% g
% Choose a ring source away from the annulus surface. Adjust these for your test.
R0 = 3.0;
z0 = 1.0;
srcL = [R0; z0];

[val, grad] = chnk.axissymlap2d.green_modal(srcL, src, origin, mA, false);

Atheta = val;
dAr    = grad(:,1,1);
dAz    = grad(:,1,3);

Hr_true_bdry = -dAz;
Hz_true_bdry =  dAr + Atheta./rp;

rhs = -(n_src(1,:)'.*Hr_true_bdry + n_src(2,:)'.*Hz_true_bdry);

wsurf = 2*pi * rp .* wsrc;
fprintf('Compatibility integral int g dS over annulus = %.6e\n', wsurf.'*rhs);

g0 = zeros(2*npts,1);
g0(1:2:end) = rhs;

%% c
opts = [];
Smat = chunkermat(chnkr, S1, opts);
AthetaC_h = Smat(minIdx,:) * dens_h;
c0 = 2*pi*Rin*AthetaC_h;

%% b
[valC, ~] = chnk.axissymlap2d.green_modal(srcL, targC, origin, mA, false);
b0 = 2*pi*Rin*valC;

%% q
opts = [];

q = chunkermat(chnkr, Scurl, opts) * dens_h;

% alpha contributes only to the boundary-condition row, i.e. the sigma rows.
q0 = zeros(2*npts,1);
q0(1:2:end) = q;

%% A
p = 2.0;
Z  = kernel.zeros();

% Interleaved two-density operator:
%   sigma + p*D'[mu] = g
%   mu    + p*S[sigma] = 0
%
% Eliminating mu gives sigma - p^2 D'S[sigma] = g.
K = [ Z        p*Dp0;
      p*S0   Z      ];
K = kernel(K);

opts = [];
opts.l2scale = false;
opts.rcip = true;
opts.nsub_or_tol = 30;

A = chunkermat(chnkr, K, opts) + eye(2*npts);

%% f
eps = 0.01; % THIS WILL CAUSE BAD ERROR!!
chnkA = get_disk_curve(Rin-eps, zin);
targA = chnkA.r(:,:);
wA    = chnkA.wts(:);
rA    = targA(1,:).';
wSurf = 2*pi*(rA .* wA);

Hz_mat = -chunkerkernevalmat(chnkr, Dp0, chnkA);
f = -wSurf.' * Hz_mat;          % maps mu -> flux of grad D[mu]

f0 = zeros(1,2*npts);
f0(2:2:end) = p * f;        % phi = p*D'[mu]

%% ============================================================
% Assemble and solve augmented system
% ============================================================
M = [A, q0;
     f0, c0];
rhs = [g0; b0];

sol = gmres(M, rhs, [], 1e-12, 200);
sol_density = sol(1:2*npts);
alpha  = sol(end);
sigma0 = sol_density(1:2:end);
mu0    = sol_density(2:2:end);

bc_res = A*sol_density + q0*alpha - g0;
fprintf('BC residual = %.6e\n', norm(bc_res,inf));

flux_res = f0*sol_density + c0*alpha - b0;
fprintf('Flux residual = %.6e\n', flux_res);

%% ============================================================
% Evaluate H and Htrue at an off-surface point
% ============================================================
rt = 1.0;
zt = -1.0;
targT = [rt; zt];

[val_true, grad_true] = chnk.axissymlap2d.green_modal(srcL, targT, origin, mA, false);
Atheta_true = val_true(1,1);
dAr_true    = grad_true(1,1,1);
dAz_true    = grad_true(1,1,3);

Hr_true = -dAz_true;
Hz_true =  dAr_true + Atheta_true/rt;

opts_eval = [];
opts_eval.forcesmooth = false;
opts_eval.verb = false;
opts_eval.quadkgparams = {'RelTol',1e-10,'AbsTol',1e-10};
opts_eval.sing = 'log';

% Scalar field is phi = p*D[mu0].  Its gradient is evaluated by D'
% with fake target normals e_r and e_z.

tinfo_r = [];
tinfo_r.r = targT;
tinfo_r.n = [1;0];
Hr_phi = -p * chunkerkerneval(chnkr, Dp0, mu0, tinfo_r, opts_eval);

tinfo_z = [];
tinfo_z.r = targT;
tinfo_z.n = [0;1];
Hz_phi = -p * chunkerkerneval(chnkr, Dp0, mu0, tinfo_z, opts_eval);

% Harmonic/vector-potential basis part alpha*curl S[h]
dAr = -chunkerkerneval(chnkr, Sp1, dens_h, tinfo_r, opts_eval);
dAz = -chunkerkerneval(chnkr, Sp1, dens_h, tinfo_z, opts_eval);
Atheta_h = chunkerkerneval(chnkr, S1, dens_h, targT, opts_eval);

Hr_curl = -dAz;
Hz_curl =  dAr + Atheta_h/rt;

Hr = Hr_phi + alpha * Hr_curl;
Hz = Hz_phi + alpha * Hz_curl;

fprintf('H_phi  [Hr_phi,Hz_phi] = [% .6e, % .6e]\n', Hr_phi,  Hz_phi);
fprintf('H_curl [Hr_curl,Hz_curl] = [% .6e, % .6e]\n', Hr_curl, Hz_curl);

fprintf('\n=== Field eval at (r=%.6g, z=%.6g) ===\n', rt, zt);
fprintf('Htrue [Hr,Hz] = [% .6e, % .6e]\n', Hr_true, Hz_true);
fprintf('H     [Hr,Hz] = [% .6e, % .6e]\n', Hr, Hz);

Htrue_vec = [Hr_true; Hz_true];
H_vec     = [Hr; Hz];
relerr = norm(H_vec - Htrue_vec, 2) / norm(Htrue_vec, 2);
fprintf('Relative error : %.16e\n', relerr);

%% geometry functions
function chnkobj = get_annulus_geometry(Rin, Rout, zin)
    pref = [];
    pref.k = 16;

    cparams = [];
    cparams.ta = 0;
    cparams.tb = 1;
    cparams.nchmin = 8;
    cparams.ifclosed = false;

    chnkobj = chunkerfunc(@(t) annulus_segment(t, Rin, Rout, zin), ...
                          cparams, pref);

    chnkobj = sort(chnkobj);
end

function [r, d, d2] = annulus_segment(t, Rin, Rout, zin)
    t = t(:).';   % force row vector

    r = [Rin + (Rout - Rin).*t;
         zin*ones(size(t))];

    d = [(Rout - Rin)*ones(size(t));
         zeros(size(t))];

    d2 = [zeros(size(t));
          zeros(size(t))];
end

function chnkA = get_disk_curve(Rin, zin)
    pref = [];
    pref.k = 16;

    cparams = [];
    cparams.ta = 0;
    cparams.tb = 1;
    cparams.nchmin = 4;
    cparams.ifclosed = false;

    chnkA = chunkerfunc(@(t) disk_segment(t, Rin, zin), cparams, pref);
    chnkA = sort(chnkA);
end

function [r, d, d2] = disk_segment(t, Rin, zin)
    t = t(:).';   % force row vector

    r = [Rin.*t;
         zin*ones(size(t))];

    d = [Rin*ones(size(t));
         zeros(size(t))];

    d2 = [zeros(size(t));
          zeros(size(t))];
end
