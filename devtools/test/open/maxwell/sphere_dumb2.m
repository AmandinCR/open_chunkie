clearvars; 
close all;
format long e;

%% === PRECOMPUTE for magnetostatics setup ===
% the final system should be:
% [ A ] [ sigma ] = [ g ]
%
% sizes:
% A is 11x128x128
% sigma is 11x128
%
% H = grad S[sigma]
% 

%% geometry
[chnkr] = get_sphere_geometry();
%plot(chnkr);

npts  = chnkr.npt;
src   = chnkr.r(:,:);      % generating curve points [r; z]
n_src = chnkr.n(:,:);      % generating curve normals [nr; nz]
wsrc  = chnkr.wts(:);      % ds weights on generating curve

origin = [0,0];
all_modes = false;
m0 = 1;                    % Fourier mode 0

%% ============================================================
% Manufactured Htrue EXACTLY like sphere case:
% Htrue = grad(phi_true),  phi_true = G0(srcQ1) - G0(srcQ2)
% (two mode-0 ring sources of opposite strength)
% ============================================================
rmid = 0.5;

srcQ1p = [0.3; +0.1];
srcQ1m = [0.3; -0.1];
srcQ2p = [0.4; +0.1];
srcQ2m = [0.4; -0.1];

% amplitudes chosen to satisfy compatibility (ring strength scales with radius)
a1 = 1.0;
a2 = 0.3/0.4;

% Evaluate gradients of mode-0 Green from each source ring
[~, g1p] = chnk.axissymlap2d.green_modal(srcQ1p, src, origin, m0, all_modes);
[~, g1m] = chnk.axissymlap2d.green_modal(srcQ1m, src, origin, m0, all_modes);
[~, g2p] = chnk.axissymlap2d.green_modal(srcQ2p, src, origin, m0, all_modes);
[~, g2m] = chnk.axissymlap2d.green_modal(srcQ2m, src, origin, m0, all_modes);

% Htrue = grad( a1*(G1+ + G1-) - a2*(G2+ + G2-) )
Hr_true_bdy = a1*(g1p(:,1,1) + g1m(:,1,1)) - a2*(g2p(:,1,1) + g2m(:,1,1));
Hz_true_bdy = a1*(g1p(:,1,3) + g1m(:,1,3)) - a2*(g2p(:,1,3) + g2m(:,1,3));

% physical Neumann data
g0 = -(n_src(1,:)'.*Hr_true_bdy + n_src(2,:)'.*Hz_true_bdy);

% compatibility checks
wsurf = 2*pi * (src(1,:)' .* wsrc);
fprintf('Compatibility integral int g0 dS    = %.6e\n', wsurf.'*g0);

%% --- Build mode-0 A operator ---
opts = [];
opts.rcip = false;
opts.forcesmooth = false;
opts.l2scale = false;

Sp0 = kernel('axissymlap','sprime',m0,all_modes);
A0 = chunkermat_normal(chnkr, Sp0, opts) + 0.5*eye(npts);
A0 = A0 + onesmat(chnkr);

sigma0 = A0 \ g0;

%% ============================================================
%  Evaluate H and Htrue at an off-surface point (rt,zt)
%  (meridian components Hr,Hz; optional Cartesian at angle theta)
% ============================================================

rt = 2.0;
zt = -2.5;
targT = [rt; zt];

[~, GT1p] = chnk.axissymlap2d.green_modal(srcQ1p, targT, origin, m0, all_modes);
[~, GT1m] = chnk.axissymlap2d.green_modal(srcQ1m, targT, origin, m0, all_modes);
[~, GT2p] = chnk.axissymlap2d.green_modal(srcQ2p, targT, origin, m0, all_modes);
[~, GT2m] = chnk.axissymlap2d.green_modal(srcQ2m, targT, origin, m0, all_modes);

Hr_true = a1*(GT1p(1,1,1) + GT1m(1,1,1)) - a2*(GT2p(1,1,1) + GT2m(1,1,1));
Hz_true = a1*(GT1p(1,1,3) + GT1m(1,1,3)) - a2*(GT2p(1,1,3) + GT2m(1,1,3));

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
Hr = -chunkerkerneval(chnkr, Sp0, sigma0, tinfo_r, opts_eval);

% target as ptinfo with fake normal = e_z
tinfo_z = [];
tinfo_z.r = targT;
tinfo_z.n = [0;1];
Hz = -chunkerkerneval(chnkr, Sp0, sigma0, tinfo_z, opts_eval);

fprintf('\n=== Field eval at (r=%.6g, z=%.6g) ===\n', rt, zt);
fprintf('Htrue  [Hr,Hz] = [% .6e, % .6e]\n', Hr_true,  Hz_true);
fprintf('H [Hr,Hz] = [% .6e, % .6e]\n', Hr, Hz);

%% geometry functions
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