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
plot(chnkr,'b-x')
axis equal

npts  = chnkr.npt;
src   = chnkr.r(:,:);      % generating curve points [r; z]
n_src = chnkr.n(:,:);      % generating curve normals [nr; nz]
wsrc  = chnkr.wts(:);      % ds weights on generating curve

[Rin, minIdx] = min(src(1,:)); % radius of inner disk
R0 = mean(src(1,:)); % radius of true solution ring
Ralpha = 3.3;        % radius of augmented solution ring
zin = src(2,minIdx);
z0 = 0.2;
zalpha = -0.1;

origin = [0,0];
all_modes = false;
m0 = 1;
mA = 2;

%% ============================================================
% g = boundary condition on the torus of Htrue
% ============================================================

% Targets are the generating curve points (r,z)
targ = src;            % 2 x npts
srcL = [R0; z0];       % 2 x 1 (single ring in meridian)

[val, grad] = chnk.axissymlap2d.green_modal(srcL, targ, origin, mA, all_modes);

Atheta = val;           % npts x 1
dAr    = grad(:,1,1);        % d/dr Atheta
dAz    = grad(:,1,3);        % d/dz Atheta

r = src(1,:).';             % npts x 1
Hr = -dAz;                   % H_r = -d/dz Atheta
Hz = dAr + Atheta./r;        % H_z = d/dr Atheta + Atheta/r

% n*H^inc on the surface (independent of azimuth angle for this symmetric ring)
g0 = -(n_src(1,:)'.*Hr + n_src(2,:)'.*Hz);   % npts x 1

% compatibility check
wsurf = 2*pi * src(1,:)' .* wsrc;
fprintf('Compatibility condition: int g dS = %.6e\n', wsurf.'*g0);

%% ============================================================
% b = flux of Htrue through spanning disk A
% ============================================================

% target point on the boundary circle C in meridian coords
targC = [Rin; zin];
srcL = [R0; z0];

[valC, ~] = chnk.axissymlap2d.green_modal(srcL, targC, origin, mA, all_modes);
AthetaC = valC(1,1);

b = 2*pi*Rin*AthetaC;

%% ============================================================
% q = alpha * n * curl A[L]
% ============================================================
srcLalpha = [Ralpha; zalpha];

[valL, gradL] = chnk.axissymlap2d.green_modal(srcLalpha, src, origin, mA, all_modes);

AthetaL = valL;       % npts x 1
dArL = gradL(:,1,1);  % d/dr A_theta
dAzL = gradL(:,1,3);  % d/dz A_theta

r = src(1,:).';
HrL = -dAzL;
HzL = dArL + AthetaL./r;

q0 = -(n_src(1,:)'.*HrL + n_src(2,:)'.*HzL);

%% ============================================================
% f = extra flux condition row
% ============================================================
w = chnkr.wts(:);
chnkA = get_disk_curve(Rin, zin);

targA = chnkA.r(:,:);         % 2 x nA
wA    = chnkA.wts(:);         % dr weights
rA    = targA(1,:).';
wSurf = 2*pi*(rA .* wA);    % revolved surface weights

% mode-0 Green derivative wrt target z for grad S · n_A (with n_A = +z)
[~, gradA] = chnk.axissymlap2d.green_modal(src, targA, origin, m0, all_modes);
dGdz = gradA(:,:,3);          % nA = +e_z convention

% row action on sigma: integral_A (grad S[sigma] · nA) dA
% gradA is nA-by-npts, w is source ds weights
f = wSurf.' * (dGdz .* (w.'));



% CHECK F WITH TEST SIGMA
ds = chnkr.wts(:);
s_nodes = [0; cumsum(ds(1:end-1))];
L = sum(ds);

phi = 0.1;
sigma_test = cos(4*pi*s_nodes/L + phi);
sigma_test = sigma_test / norm(sigma_test);

opts = [];
Hz = zeros(size(targA,2),1);
Sp0 = kernel('axissymlap','sprime',m0,all_modes);
for k = 1:size(targA,2)
    tinfo = [];
    tinfo.r = targA(:,k);
    tinfo.n = [0;1];
    Hz(k) = -chunkerkerneval(chnkr, Sp0, sigma_test, tinfo, opts);
end
flux_direct = sum(wSurf .* Hz);
flux_row = f * sigma_test;
fprintf('test sigma: difference in f = %.3e\n', abs(flux_direct - flux_row));
%{
f_basis = zeros(1,npts);
for j = 1:npts
    ej = zeros(npts,1);
    ej(j) = 1;
    fluxj = 0;
    for k = 1:size(targA,2)
        tinfo = [];
        tinfo.r = targA(:,k);
        tinfo.n = [0;1];   % n_A = +e_z
        Hzk = -chunkerkerneval(chnkr, Sp0, ej, tinfo, opts);
        fluxj = fluxj + wSurf(k) * Hzk;
    end
    f_basis(j) = fluxj;
end
flux_basis = f_basis * sigma_test;
fprintf('test sigma: difference in f = %.3e\n', abs(flux_basis - flux_row));
fprintf('test sigma: difference in f = %.3e\n', abs(flux_direct - flux_basis));
%f = f_basis;
%}
%% --- c = flux of H_L through spanning disk (Stokes) ---
targC = [Rin; zin];
[valC_L, ~] = chnk.axissymlap2d.green_modal(srcLalpha, targC, origin, mA, all_modes);
AthetaC_L = valC_L(1,1);
c = 2*pi*Rin*AthetaC_L;   % same sign as b

%% --- Build mode-0 A operator ---
opts = [];
opts.rcip = false;
opts.forcesmooth = false;
opts.l2scale = false;

Sp0 = kernel('axissymlap','sprime',m0,all_modes);
A0 = chunkermat_normal(chnkr, Sp0, opts) + 0.5*eye(npts);
%A0 = A0 + onesmat(chnkr);

%% Assemble and solve the block system

M = [A0, q0; f, c];
rhs = [g0; b];

sol = M \ rhs;
sigma0 = sol(1:npts);
alpha  = sol(end);

bc_res = A0*sigma0 + q0*alpha - g0;
fprintf('BC residual = %.6e\n', norm(bc_res,inf));

flux_res = f*sigma0 + c*alpha - b;
fprintf('Flux residual = %.6e\n', flux_res);

%% ============================================================
%  Evaluate H and Htrue at an off-surface point
% ============================================================

rt = 1.0;
zt = -1.0;
targT = [rt; zt];

[val_true, grad_true] = chnk.axissymlap2d.green_modal(srcL, targT, origin, mA, all_modes);

Atheta_true = val_true(1,1);
dAr_true    = grad_true(1,1,1);       % d/dr at target
dAz_true    = grad_true(1,1,3);       % d/dz at target

Hr_true = -dAz_true;
Hz_true =  dAr_true + Atheta_true/rt;

% --- grad S[sigma] (mode 0) ---
opts_eval = [];
opts_eval.forcesmooth = false;
opts_eval.verb = false;
opts_eval.quadkgparams = {'RelTol',1e-16,'AbsTol',1e-16};
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
[valL_t, gradL_t] = chnk.axissymlap2d.green_modal(srcLalpha, targT, origin, mA, all_modes);

AthetaL_t = valL_t(1,1);
dArL_t    = gradL_t(1,1,1);
dAzL_t    = gradL_t(1,1,3);

Hr_curl = -dAzL_t;
Hz_curl = dArL_t + AthetaL_t/rt;

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
    %pref.nchmax = 2;

    cparams = [];
    %cparams.eps = 1.0e-14;
    %cparams.nover = 1;
    cparams.ifclosed = true;
    cparams.ta = 0;
    cparams.tb = 2*pi;
    cparams.maxchunklen = 2;
    cparams.nchmin = 8;

    %ctr = [3 0];
    %narms = 0;
    %amp = 0.25;
    %chnkobj = chunkerfunc(@(t) starfish(t, narms, amp, ctr), cparams, pref);
    chnkobj = chunkerfunc(@(t) ellipse2(t, 1,2,3,0), cparams, pref); 

    %beta = 0.99;
    %chnkobj = chunkerfunc(@(u) ellipse2_clustered(u,1,2,3,0,beta), cparams, pref);
    
    %chnkobj = sort(chnkobj);
end

function chnkA = get_disk_curve(Rin, zin)
    pref = [];
    pref.k = 16;               % order per chunk (match your torus pref if you want)
 
    cparams = [];
    cparams.ta = 0;
    cparams.tb = 1;
    cparams.nchmin = 4;

    % chunkgraph expects verts as 2 x nv
    verts = [0.0 Rin; zin zin];        % two vertices: (a,0) and (b,0)
    edge2verts = [1; 2];
    fchnks = [];

    chnkA = chunkgraph(verts, edge2verts, fchnks, cparams, pref);
    %chnkA = balance(chnkA);
end

function [r, d, d2] = ellipse2_clustered(u,a,b,c,dcen,beta)
    % smooth periodic map clustering near t = pi
    t   = u - beta*sin(u - pi);
    tp  = 1 - beta*cos(u - pi);
    tpp = beta*sin(u - pi);

    % force row vectors so sizes match ellipse2 output (2 x N)
    t = t(:).';
    tp = tp(:).';
    tpp = tpp(:).';

    % original ellipse evaluated at t
    [r0, d0t, d20t] = ellipse2(t, a, b, c, dcen);

    % chain rule
    r  = r0;
    d  = d0t .* repmat(tp,  2, 1);
    d2 = d20t .* repmat(tp.^2, 2, 1) + d0t .* repmat(tpp, 2, 1);
end