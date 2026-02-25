[chnkr] = get_torus_geometry();

npts  = chnkr.npt;
src   = chnkr.r(:,:);      % generating curve points [r; z]
n_src = chnkr.n(:,:);      % generating curve normals [nr; nz]
wsrc  = chnkr.wts(:);      % ds weights on generating curve

origin = [0,0];
all_modes = false;
m0 = 1;                    % Fourier mode 0

%% ============================================================
% Manufactured true field Htrue from FOUR interior ring charges
% (compatible + zero hole-flux setup)
%
% Pattern:
%   +a1 [ G(r1,+z0) + G(r1,-z0) ]  - a2 [ G(r2,+z0) + G(r2,-z0) ]
% with a2 = a1 * (r1/r2)
%
% This gives:
%   - Neumann compatibility satisfied (weighted net ring charge cancels)
%   - Flux through z=0 hole disk is zero by symmetry
% ============================================================

% ---- choose ring locations inside the torus body ----
r1 = 3.00;
r2 = 2.80;      % different radius, still inside torus body
z0 = 0.20;

a1 = 1.0;
a2 = a1 * (r1/r2);   % compatibility fix for axisymmetric modal weighting

srcQ1p = [r1; +z0];
srcQ1m = [r1; -z0];
srcQ2p = [r2; +z0];
srcQ2m = [r2; -z0];

% ---- boundary truth field (gradient of manufactured potential) ----
[~, gradQ1p] = chnk.axissymlap2d.green_modal(srcQ1p, src, origin, m0, all_modes);
[~, gradQ1m] = chnk.axissymlap2d.green_modal(srcQ1m, src, origin, m0, all_modes);
[~, gradQ2p] = chnk.axissymlap2d.green_modal(srcQ2p, src, origin, m0, all_modes);
[~, gradQ2m] = chnk.axissymlap2d.green_modal(srcQ2m, src, origin, m0, all_modes);

% grad(phi_true)
Hr_grad_bdy = a1*(gradQ1p(:,1,1) + gradQ1m(:,1,1)) - a2*(gradQ2p(:,1,1) + gradQ2m(:,1,1));
Hz_grad_bdy = a1*(gradQ1p(:,1,3) + gradQ1m(:,1,3)) - a2*(gradQ2p(:,1,3) + gradQ2m(:,1,3));

% Physical field convention in your script: H_true = -grad(phi_true)
Hr_true_bdy = -Hr_grad_bdy;
Hz_true_bdy = -Hz_grad_bdy;

% Neumann data g = n·H_true on boundary
g0 = n_src(1,:)'.*Hr_true_bdy + n_src(2,:)'.*Hz_true_bdy;

% Compatibility check (should be ~ machine precision)
wsurf = 2*pi * src(1,:)' .* wsrc;
fprintf('Compatibility integral int g dS = %.6e\n', wsurf.'*g0);

opts = [];
opts.rcip = false;
opts.forcesmooth = false;
opts.l2scale = false;

Sp0 = kernel('axissymlap','sprime',m0,all_modes);
A0 = chunkermat_normal(chnkr, Sp0, opts) + 0.5*eye(npts);
A0 = A0 + onesmat(chnkr);   % same nullspace-fix convention you used before

sigma0 = A0 \ g0;

%% ============================================================
% Flux of TRUE field through torus hole (disk z=0, n_A=+zhat)
% Should be ~ 0 by symmetry for this 4-ring construction
% ============================================================

Nr_flux = 100;
Rin = min(src(1,:));                     % hole rim radius from generating curve
rflux = linspace(1e-8, Rin, Nr_flux+1).'; % avoid r=0 axis issue
dr = rflux(2)-rflux(1);

Hz_true_phys = zeros(size(rflux));

for ii = 1:numel(rflux)
    targF = [rflux(ii); 0.0];

    [~, g1p] = chnk.axissymlap2d.green_modal(srcQ1p, targF, origin, m0, all_modes);
    [~, g1m] = chnk.axissymlap2d.green_modal(srcQ1m, targF, origin, m0, all_modes);
    [~, g2p] = chnk.axissymlap2d.green_modal(srcQ2p, targF, origin, m0, all_modes);
    [~, g2m] = chnk.axissymlap2d.green_modal(srcQ2m, targF, origin, m0, all_modes);

    % H_true,z = - d(phi_true)/dz
    Hz_true_phys(ii) = -( a1*(g1p(1,1,3) + g1m(1,1,3)) - a2*(g2p(1,1,3) + g2m(1,1,3)) );
end

integrand_flux = Hz_true_phys .* rflux;
flux_true_hole = 2*pi * dr * (0.5*integrand_flux(1) + sum(integrand_flux(2:end-1)) + 0.5*integrand_flux(end));

fprintf('\n=== True flux through torus hole (disk z=0, n_A=+zhat) ===\n');
fprintf('Flux_true = %.16e\n', flux_true_hole);

%% ============================================================
% (Optional) Flux of NUMERICAL solved field H = -grad S[sigma0]
% Compare against flux_true_hole (should also be ~0 if alpha=0 test is consistent)
% ============================================================

Nr_flux_num = 100;
rflux_num = linspace(1e-8, Rin, Nr_flux_num+1).';
dr_num = rflux_num(2)-rflux_num(1);

Hz_num = zeros(size(rflux_num));

opts_eval_flux = [];
opts_eval_flux.forcesmooth = false;
opts_eval_flux.verb = false;
opts_eval_flux.quadkgparams = {'RelTol',1e-10,'AbsTol',1e-10};
opts_eval_flux.sing = 'log';

Sp0_flux = kernel('axissymlap','sprime',m0,all_modes);

for ii = 1:numel(rflux_num)
    targF = [rflux_num(ii); 0.0];
    tinfo_zf = [];
    tinfo_zf.r = targF;
    tinfo_zf.n = [0;1];   % extract d/dz via fake normal e_z

    Hz_num(ii) = -chunkerkerneval(chnkr, Sp0_flux, sigma0, tinfo_zf, opts_eval_flux);
end

integrand_num = Hz_num .* rflux_num;
flux_num_hole = 2*pi * dr_num * (0.5*integrand_num(1) + sum(integrand_num(2:end-1)) + 0.5*integrand_num(end));

fprintf('\n=== Numerical flux through torus hole (disk z=0, n_A=+zhat) ===\n');
fprintf('Flux_num = %.16e\n', flux_num_hole);






rt = 5.0;
zt = -2.5;
targT = [rt; zt];


% ============================================================
% Off-surface truth field at target (same manufactured 4-ring setup)
% ============================================================

rt = 5.0;
zt = -2.5;
targT = [rt; zt];

% ---- boundary truth field (gradient of manufactured potential) ----
[~, gradQ1p] = chnk.axissymlap2d.green_modal(srcQ1p, targT, origin, m0, all_modes);
[~, gradQ1m] = chnk.axissymlap2d.green_modal(srcQ1m, targT, origin, m0, all_modes);
[~, gradQ2p] = chnk.axissymlap2d.green_modal(srcQ2p, targT, origin, m0, all_modes);
[~, gradQ2m] = chnk.axissymlap2d.green_modal(srcQ2m, targT, origin, m0, all_modes);

% grad(phi_true)
Hr_grad_bdy = a1*(gradQ1p(:,1,1) + gradQ1m(:,1,1)) - a2*(gradQ2p(:,1,1) + gradQ2m(:,1,1));
Hz_grad_bdy = a1*(gradQ1p(:,1,3) + gradQ1m(:,1,3)) - a2*(gradQ2p(:,1,3) + gradQ2m(:,1,3));

% Physical field convention in your script: H_true = -grad(phi_true)
Hr_true = -Hr_grad_bdy;
Hz_true = -Hz_grad_bdy;

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
