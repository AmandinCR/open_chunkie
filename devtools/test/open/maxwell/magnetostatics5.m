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
zin = src(2,minIdx);
Ralpha = 3.3;        % radius of augmented solution ring
zalpha = -0.1;
R0 = 3; % radius of true solution ring
z0 = 0.2;

origin = [0,0];
all_modes = false;
m0 = 1;
mA = 2;

rp  = src(1,:).';
dens_h = 1 ./ rp;
targC = [Rin;zin];

%% ============================================================
% g = n dot Htrue on the boundary
% ============================================================

% Targets are the generating curve points (r,z)
targ = src;            % 2 x npts
srcL = [R0; z0];       % 2 x 1 (single ring in meridian)

[val, grad] = chnk.axissymlap2d.green_modal(srcL, targ, origin, mA, all_modes);

Atheta = val;                % npts x 1
dAr    = grad(:,1,1);        % d/dr Atheta
dAz    = grad(:,1,3);        % d/dz Atheta

Hr = -dAz;                    % H_r = -d/dz Atheta
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
b = 2*pi*Rin*valC;

%% q = alpha * n * curl S[h]
%{
% Set n = e_r everywhere
chnk_r = chnkr;
fake_n = zeros(size(chnk_r.n));
fake_n(1,:,:) = 1;   % n_r = 1
fake_n(2,:,:) = 0;   % n_z = 0
chnk_r.n = fake_n;

% Set n = e_z everywhere
chnk_z = chnkr;
fake_n = zeros(size(chnk_z.n));
fake_n(1,:,:) = 0;
fake_n(2,:,:) = 1;
chnk_z.n = fake_n;

opts = [];
Sp1 = kernel('axissymlap','sprime',mA,all_modes);
S1  = kernel('axissymlap','s',mA,all_modes);

Sp_r = -chunkermat_normal(chnk_r, Sp1, opts);
Sp_z = -chunkermat_normal(chnk_z, Sp1, opts);
Smat = chunkermat_normal(chnkr, S1, opts);

Atheta = Smat * dens_h;
dAr = Sp_r * dens_h;
dAz = Sp_z * dens_h;

Hr = -dAz;
Hz = dAr + Atheta./rp;

q0 = -(n_src(1,:)'.*Hr + n_src(2,:)'.*Hz);
%}

%% ============================================================
% q = - n dot curl S[h]
% build q0 using chunkerkerneval off-surface
% ============================================================
Sp1 = kernel('axissymlap','sprime',mA,all_modes);
S1  = kernel('axissymlap','s',mA,all_modes);

rp = src(1,:).';
zp = src(2,:).';
nr = n_src(1,:).';
nz = n_src(2,:).';

epsq = 1e-3;
opts = [];
opts.quadkgparams = {'RelTol',1e-8,'AbsTol',1e-8};

% off-surface targets on both sides
targ_plus  = src + epsq * n_src;   % 2 x npts
targ_minus = src - epsq * n_src;   % 2 x npts

% ---- + side ----
tinfo_r_plus = [];
tinfo_r_plus.r = targ_plus;
tinfo_r_plus.n = repmat([1;0], 1, npts);
tinfo_z_plus = [];
tinfo_z_plus.r = targ_plus;
tinfo_z_plus.n = repmat([0;1], 1, npts);

dAr_plus = -chunkerkerneval(chnkr, Sp1, dens_h, tinfo_r_plus, opts);
dAz_plus = -chunkerkerneval(chnkr, Sp1, dens_h, tinfo_z_plus, opts);
Atheta_plus = chunkerkerneval(chnkr, S1, dens_h, targ_plus, opts);

Hr_plus = -dAz_plus(:);
Hz_plus =  dAr_plus(:) + Atheta_plus(:)./targ_plus(1,:).';
q0_plus = -( nr.*Hr_plus + nz.*Hz_plus );

% ---- - side ----
tinfo_r_minus = [];
tinfo_r_minus.r = targ_minus;
tinfo_r_minus.n = repmat([1;0], 1, npts);
tinfo_z_minus = [];
tinfo_z_minus.r = targ_minus;
tinfo_z_minus.n = repmat([0;1], 1, npts);

dAr_minus = -chunkerkerneval(chnkr, Sp1, dens_h, tinfo_r_minus, opts);
dAz_minus = -chunkerkerneval(chnkr, Sp1, dens_h, tinfo_z_minus, opts);
Atheta_minus = chunkerkerneval(chnkr, S1, dens_h, targ_minus, opts);

Hr_minus = -dAz_minus(:);
Hz_minus =  dAr_minus(:) + Atheta_minus(:)./targ_minus(1,:).';
q0_minus = -( nr.*Hr_minus + nz.*Hz_minus );

% average the two traces
q0 = 0.5*(q0_plus + q0_minus);

fprintf('||q+ - q-||inf / ||q0_eval||inf = %.6e\n', ...
    norm(q0_plus - q0_minus, inf) / max(norm(q0,inf),1));
%% ============================================================
% compute f
% ============================================================
chnkA = get_disk_curve(Rin, zin);
targA = chnkA.r(:,:);               % 2 x 128
wA    = chnkA.wts(:);               % 128 x 1
rA    = targA(1,:).';               % 128 x 1

% mode-0 Green derivative wrt target z for grad S · n_A (with n_A = +z)
[~, gradA] = chnk.axissymlap2d.green_modal(src, targA, origin, m0, all_modes);
dGdz = gradA(:,:,3);                % 128 x 128

% row action on sigma: integral_A (grad S[sigma] · nA) dA
% gradA is nA-by-npts, w is source ds weights
wSurf = 2*pi * (rA .* wA);          % 128 x 1
f = wSurf.' * (dGdz .* (wsrc.'));

%{
tinfoA = [];
tinfoA.r = targA;
tinfoA.n = repmat([0;1], 1, size(targA,2));
opts = [];
Sp0 = kernel('axissymlap','sprime',m0,all_modes);
Hzmat = -chunkerkernevalmat(chnkr, Sp0, tinfoA, opts);
f = wSurf.' * Hzmat;    % 1 x npts
%}

% CHECK F WITH TEST SIGMA
ds = chnkr.wts(:);
s_nodes = [0; cumsum(ds(1:end-1))];
L = sum(ds);
sigma_test = cos(4*pi*s_nodes/L + 0.1);
sigma_test = sigma_test / norm(sigma_test);

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


opts = [];
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
f = f_basis;

%% --- c = flux of H_L through spanning disk (Stokes) ---
opts = [];
Smat = chunkermat_normal(chnkr, S1, opts);
AthetaC_h = Smat(minIdx,:) * dens_h;
c = 2*pi*Rin*AthetaC_h;

% idk if i can just divide Smat by 1/r' once chunkermat has already
% been used. try to remove the r' from the chunkermat entirely for this?

% ============================================================
% c = 2*pi * int_0^Rin Hz(r,0) * r dr
% ============================================================
%
% radial chunk from 0 to Rcut along z=0
chnkAeps = get_disk_curve(Rin, zin);
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
c2 = 2*pi * sum(HzA .* rAeps .* wAeps);
fprintf('difference in c = %.3e\n', abs(c - c2));

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
fprintf('BC residual = %.6e\n', norm(bc_res,inf));

flux_res = f*sigma0 + c*alpha - b;
fprintf('Flux residual = %.6e\n', flux_res);

%% ============================================================
%  Evaluate H and Htrue at an off-surface point (rt,zt)
%  (meridian components Hr,Hz; optional Cartesian at angle theta)
% ============================================================

rt = 1.0;
zt = -1.0;
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

% --- curl S[h] ---
dAr = -chunkerkerneval(chnkr, Sp1, dens_h, tinfo_r, opts_eval);
dAz = -chunkerkerneval(chnkr, Sp1, dens_h, tinfo_z, opts_eval);
Atheta = chunkerkerneval(chnkr, S1, dens_h, targT, opts_eval);

Hr_curl = -dAz;
Hz_curl = dAr + Atheta/rt;



%% full solution
Hr = Hr_phi + alpha * Hr_curl;
Hz = Hz_phi + alpha * Hz_curl;

fprintf('H_phi  [Hr_phi,Hz_phi] = [% .6e, % .6e]\n', Hr_phi,  Hz_phi);
fprintf('H_curl  [Hr_curl,Hz_curl] = [% .6e, % .6e]\n', Hr_curl,  Hz_curl);

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
    %chnkobj = chunkerfunc(@(u) ellipse2_clustered(u,1,1,3,0,beta), cparams, pref);
    
    chnkobj = sort(chnkobj);
end

function chnkA = get_disk_curve(Rin, zin)
    pref = [];
    pref.k = 16;               % order per chunk (match your torus pref if you want)
 
    cparams = [];
    cparams.ta = 0;
    cparams.tb = 1;
    cparams.nchmin = 4;
    cparams.ifclosed = false;

    % chunkgraph expects verts as 2 x nv
    verts = [0.0 Rin; zin zin];        % two vertices: (a,0) and (b,0)
    edge2verts = [1; 2];
    fchnks = [];

    chnkA = chunkgraph(verts, edge2verts, fchnks, cparams, pref);
    chnkA = balance(chnkA);
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

