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
%plot(chnkr,'b-x')
%axis equal

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
m0 = 1;
mA = 2;

rp  = src(1,:).';
dens_h = 1 ./ rp;
targC = [Rin;zin];
all_modes = false;

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
opts = [];
Qker = kernel('axissymlap','scurl',mA, all_modes);
q0 = chunkermat_normal(chnkr, Qker, opts) * dens_h;

%% ============================================================
% compute f
% ============================================================
chnkA = get_disk_curve(Rin, zin);
targA = chnkA.r(:,:);
wA    = chnkA.wts(:);
rA    = targA(1,:).';
wSurf = 2*pi*(rA .* wA);

%Sp0 = kernel('axissymlaplace','sprime',m0);
Sp0 = @(s,t) chnk.axissymlap2d.kern_0th_mode(s,t,origin,'sprime');
% chunkerkernevalmat is geniunely broken and doesn't accept any opts
Hzmat = -chunkerkernevalmat(chnkr, Sp0, chnkA);
f = -wSurf.' * Hzmat;



% CHECK F WITH TEST SIGMA
ds = chnkr.wts(:);
s_nodes = [0; cumsum(ds(1:end-1))];
L = sum(ds);
sigma_test = cos(4*pi*s_nodes/L + 0.1);
sigma_test = sigma_test / norm(sigma_test);

Hz = zeros(size(targA,2),1);
Sp0 = kernel('axissymlap','sprime',m0, all_modes);
for k = 1:size(targA,2)
    tinfo = [];
    tinfo.r = targA(:,k);
    tinfo.n = [0;1];
    Hz(k) = -chunkerkerneval(chnkr, Sp0, sigma_test, tinfo, opts);
end
flux_direct = sum(wSurf .* Hz);
flux_row = f * sigma_test;
fprintf('test sigma: difference in f = %.3e\n', abs(flux_direct - flux_row));

%% --- c = flux of H_L through spanning disk (Stokes) ---
opts = [];
Sp1 = kernel('axissymlap','sprime',mA, all_modes);
S1  = kernel('axissymlap','s',mA, all_modes);
Smat = chunkermat_normal(chnkr, S1, opts);
AthetaC_h = Smat(minIdx,:) * dens_h;
c = 2*pi*Rin*AthetaC_h;

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
Sp0 = kernel('axissymlap','sprime',m0, all_modes);
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