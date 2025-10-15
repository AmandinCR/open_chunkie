%{
- 3D Laplace's equation
- Neumann boundary condition
- Axisymmetric boundary
- Double layer potential representation
- Non-vectorized chunkermat for modes

Notes: This code should work with standalone new version of chunkie (given
chnk.axissymlap2d code).
%}

clearvars; 
close all;
format long e;

%% geometry
% target is where we evaluate the solution
[chnkr,target,charge1,charge2] = get_torus_geometry();
npts = chnkr.npt; % total number of points in discretization

src = chnkr.r(:,:); % coordinates of points on the generating curve [2,64]
n_src = chnkr.n(:,:); % normals of all the points on the generating curve [2,64]
gl_wts = chnkr.wts(:); % Gauss-Legendre weights for src points [64,1]

% plot geometry
%plot(chnkr, 'b.');

p_modes = 0; % number of positive fourier modes
n_modes = 2*p_modes + 1; % number of fourier modes (must be odd for pos/0/neg)
n_angles = n_modes; % number of angles/rotations
strength = 100.0;

%% discretization
% compute f (boundary condition)
f = zeros(n_angles,npts);
for i=1:n_angles
    % get polar/cartesian coordinates
    theta = (i-1)*2*pi/n_angles;
    
    x=src(1,:).*cos(theta);
    y=src(1,:).*sin(theta);
    z=src(2,:);
    pts=[x;y;z];

    n_x=n_src(1,:).*cos(theta);
    n_y=n_src(1,:).*sin(theta);
    n_z=n_src(2,:);
    n_pts = [n_x;n_y;n_z];

    % set up neumann boundary data (with both charges)
    rvec1 = pts - charge1;
    r3_1 = vecnorm(rvec1).^3;
    fn1 = sum(rvec1 .* n_pts, 1) ./ (4*pi*r3_1);

    rvec2 = pts - charge2;
    r3_2 = vecnorm(rvec2).^3;
    fn2 = -sum(rvec2 .* n_pts, 1) ./ (4*pi*r3_2);

    f(i,:) = strength*(fn1 + fn2);
end

% Reorder FFT output to match
modes = -p_modes:p_modes;
f_fft = fft(f, n_modes, 1) / n_modes; % FFT (normalized)
f_m = fftshift(f_fft, 1);  % puts negative freqs first

%% solve
% solve the integral equation for each fourier mode
opts = [];
opts.rcip = false;
opts.forcesmooth = false;
opts.l2scale = false;
opts.sing = 'hs';

% kern_all_modes returns both negative and positive modes
sigma_m = zeros(n_modes,npts);
for i=1:n_modes
    G = @(s,t) kern_single_mode(s,t,[0,0],'dprime',abs(modes(i))+1);
    A = chunkermat(chnkr, G, opts);

    % Build the system matrix
    A_m = -1/(4*pi^2) * A;

    % Enforce zero-mean constraint for compatability condition
    A_m = A_m + onesmat(chnkr);

    % Solve the linear system
    sigma_m(i,:) = gmres(A_m, f_m(i,:)', [], 1e-12, npts);
end

%% solution building 2.0f
opts.forcesmooth = false;
opts.verb = false;
opts.quadkgparams = {'RelTol', 1e-8, 'AbsTol', 1.0e-8};
opts.sing = 'smooth';

% target in cylindrical coordinates (r,theta,z)
target_cyl = [sqrt(target(1)^2 + target(2)^2);atan2(target(2),target(1));target(3)];
target_new = [target_cyl(1);target_cyl(3)];

u_sol = 0;
for i=1:n_modes
    G = @(s,t) kern_single_mode(s,t,[0,0],'d',abs(modes(i))+1);
    G_eval = 1/(4*pi^2) * kernel(G);

    u_m = chunkerkerneval(chnkr, G_eval, sigma_m(i,:), target_new, opts);
    u_sol = u_sol + real(u_m * exp(1i * modes(i) * target_cyl(2)));
end

% compute the exact solution explicitly
r1 = norm(target - charge1);
r2 = norm(target - charge2);
u_true = strength*1/(4*pi)*(1/r1 - 1/r2);

% compute the error
err = norm(u_sol-u_true)




%% gemotry functions
function [chnkobj,target,charge1,charge2] = get_torus_geometry()
    pref = [];
    pref.k = 16; % points per chunk
    pref.nchmax = 2;

    cparams = [];
    cparams.eps = 1.0e-10;
    %cparams.nover = 1;
    cparams.ifclosed = true;
    cparams.ta = 0;
    cparams.tb = 2*pi;
    cparams.maxchunklen = 2;
    %cparams.nchmin = 16;

    ctr = [3 0];
    narms = 0;
    amp = 0.25;

    chnkobj = chunkerfunc(@(t) starfish(t, narms, amp, ctr), cparams, pref); 
    chnkobj = sort(chnkobj);

    target = [3;0.0;-0.7];
    charge1 = [0.0;0.0;3.0];
    charge2 = [0.0;0.0;-3.0];
end
