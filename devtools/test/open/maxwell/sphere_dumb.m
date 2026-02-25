%{
Genus-0 (sphere) magnetostatics sanity test
Solve exterior harmonic Neumann problem on a sphere-like axisymmetric surface:

    div H = 0,  curl H = 0   in exterior
    n · H = g                 on boundary

Representation:
    H = grad phi,   phi = S[sigma]

Manufactured truth:
    Htrue = grad G(x, xq) from a point source xq inside the sphere (not in exterior)
%}

clearvars;
close all;
format long e;

%% Geometry (sphere)
[chnkr, target] = get_sphere_geometry();

npts  = chnkr.npt;
src   = chnkr.r(:,:);      % generating curve points [r; z]
n_src = chnkr.n(:,:);      % generating curve normals [nr; nz]
wsrc  = chnkr.wts(:);      % ds weights on generating curve

origin = [0,0];
all_modes = false;
m0 = 1;                    % Fourier mode 0

%% ============================================================
% Manufactured true field Htrue = grad G from TWO interior ring sources
% with opposite strengths (compatibility condition satisfied)
%% ============================================================

srcQ1 = [0.1;  0.20];   % [rq; zq]-
srcQ2 = [0.1; -0.20];

[~, gradQ1] = chnk.axissymlap2d.green_modal(srcQ1, src, origin, m0, all_modes);
[~, gradQ2] = chnk.axissymlap2d.green_modal(srcQ2, src, origin, m0, all_modes);

% Htrue on boundary = grad G1 - grad G2
Hr_true_bdy = gradQ1(:,1,1) - gradQ2(:,1,1);
Hz_true_bdy = gradQ1(:,1,3) - gradQ2(:,1,3);

% Neumann data g = n·Htrue on boundary
g0 = -(n_src(1,:)'.*Hr_true_bdy + n_src(2,:)'.*Hz_true_bdy);

% (optional) compatibility check
wsurf = 2*pi * src(1,:)' .* wsrc;
fprintf('Compatibility integral int g dS = %.6e\n', wsurf.'*g0);

%% ============================================================
% Build and solve the scalar Neumann BIE:
% (S' - 1/2 I + onesmat) sigma = g
%% ============================================================

opts = [];
opts.rcip = false;
opts.forcesmooth = false;
opts.l2scale = false;

Sp0 = kernel('axissymlap','sprime',m0,all_modes);
A0 = chunkermat_normal(chnkr, Sp0, opts) + 0.5*eye(npts);
A0 = A0 + onesmat(chnkr);   % same nullspace-fix convention you used before

sigma0 = A0 \ g0;

%% ============================================================
% Evaluate reconstructed H and Htrue at an off-surface test point
% (CONSISTENT version: use chunkerkerneval + kernel('axissymlap','sprime'))
%% ============================================================

rt = 2.0;
zt = -2.5;
targT = [rt; zt];

% --- Htrue at target (compatible manufactured field) ---
[~, gradQ1_T] = chnk.axissymlap2d.green_modal(srcQ1, targT, origin, m0, all_modes);
[~, gradQ2_T] = chnk.axissymlap2d.green_modal(srcQ2, targT, origin, m0, all_modes);

Hr_true = gradQ1_T(1,1,1) - gradQ2_T(1,1,1);
Hz_true = gradQ1_T(1,1,3) - gradQ2_T(1,1,3);

% --- Reconstructed H = grad S[sigma] at target ---
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
Hr = -chunkerkerneval(chnkr, Sp0, sigma0, tinfo_r, opts_eval);

% target as ptinfo with fake normal = e_z
tinfo_z = [];
tinfo_z.r = targT;
tinfo_z.n = [0;1];
Hz = -chunkerkerneval(chnkr, Sp0, sigma0, tinfo_z, opts_eval);

fprintf('\n=== Field eval at (r=%.6g, z=%.6g) ===\n', rt, zt);
fprintf('Htrue [Hr,Hz] = [% .6e, % .6e]\n', Hr_true, Hz_true);
fprintf('H     [Hr,Hz] = [% .6e, % .6e]\n', Hr, Hz);

%% Geometry functions
function [chnkobj,target] = get_sphere_geometry()
    pref = [];
    pref.k = 16;

    cparams = [];
    cparams.eps = 1.0e-10;
    cparams.nover = 1;
    cparams.ifclosed = false;     % generating curve is an open semicircle
    cparams.ta = -pi/2;
    cparams.tb =  pi/2;
    cparams.maxchunklen = 2;

    % starfish(t, narms, amp) with narms=0, amp=0 gives a circle of radius 1
    narms = 0;
    amp = 0.0;

    chnkobj = chunkerfunc(@(t) starfish(t, narms, amp), cparams, pref);
    chnkobj = sort(chnkobj);

    target = [2.0; 0.0; 0.0];
end