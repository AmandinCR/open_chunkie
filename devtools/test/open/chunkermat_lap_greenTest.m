% WARNING: only works in chunkies chunkermat file instead of the one
% updated by Shidong.
% Notes: This code should work with standalone new version of chunkie (given
% chnk.axissymlap2d code).

% Test values and gradient of Green's third identity for Laplace eq. when 
% the target is on the boundary: 
% 1/2 u = D u - S du/dn
% and
% 1/2 u' = D' u - S' du/dn

% using chunkies hypersingular GGQ because chunkerkerneval only works for
% off/near boundary evaluation

clearvars; close all; format long e;
%addpaths_loc();

%% geometry
[chnkr] = get_sphere_geometry();
npts = chnkr.npt;
%plot(chnkr);

% pick a point on the torus (theta = 0 for convenience)
charge = [0.0;0.0;-2.0]; % source (axisymm data)

% convert to cylindrical coordinates
charge_cyl = [sqrt(charge(1)^2 + charge(2)^2);atan2(charge(2),charge(1));charge(3)];

src = chnkr.r(:,:); % generating curve
n_src = chnkr.n(:,:); % normals
gl_wts = chnkr.wts(:); % G-L weights

%% compute u and du/dn for densities

% get cartesian coordinates
x=src(1,:);
y=src(1,:)*0;
z=src(2,:);
pts=[x;y;z];

n_x=n_src(1,:);
n_y=n_src(1,:)*0;
n_z=n_src(2,:);
n_pts = [n_x;n_y;n_z];

% Laplace fundamental solution and normal derivative
rvec = pts - charge;
r = vecnorm(rvec);
r3 = vecnorm(rvec).^3;
dudn = sum(rvec .* n_pts, 1) ./ (4*pi*r3);
u = 1./(4*pi*r);

% usually take fft here but only using mode=0
u_m = u;
dudn_m = dudn;

%% evaluate operators
opts = [];
opts.rcip = false;
opts.forcesmooth = false;
%opts.eps = 1e-3;

% compute: grad u + grad S du/dny = grad D u
origin = [0,0];

% define kernels
S = @(s,t) 1/(4*pi^2) * kern_single_mode(s,t,origin,'s',1);
D = @(s,t) -1/(4*pi^2) * kern_single_mode(s,t,origin,'d',1);
Sprime = @(s,t) -1/(4*pi^2) * kern_single_mode(s,t,origin,'sprime',1);
Dprime = @(s,t) 1/(4*pi^2) * kern_single_mode(s,t,origin,'dprime',1);

% convolve and evaluate at target
Dmat = chunkermat(chnkr, D, opts);
Smat = chunkermat(chnkr, S, opts);
Sprimemat = chunkermat(chnkr, Sprime, opts);
opts.sing = 'hs';
Dprimemat = chunkermat(chnkr, Dprime, opts);

Du = Dmat*u';
Sdudn = Smat*dudn';
Sprimedudn = Sprimemat*dudn';
Dprimeu = Dprimemat*u';

%% compute error

val_error = 0.5*u' + Sdudn - Du
der_error = 0.5*dudn' + Sprimedudn - Dprimeu

%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [chnkobj] = get_torus_geometry()
    pref = [];
    pref.k = 16; % points per chunk
    %pref.nchmax = 6;

    cparams = [];
    cparams.eps = 1.0e-10;
    %cparams.nover = 1;
    cparams.ifclosed = true;
    cparams.ta = 0;
    cparams.tb = 2*pi;
    cparams.maxchunklen = 2;
    %cparams.nchmin = 4;

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