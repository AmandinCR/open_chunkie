%{
- 3D Laplace's equation
- Neumann boundary condition
- Torus boundary
- Double layer potential representation
- 0th mode (axisymmetric B.C.)

Notes: This code should work with standalone version of chunkie (given
chnk.axissymlap2d code).
%}

clearvars; 
close all;
format long e;

%% geometry
% target is where we evaluate the solution
[chnkr,target,charge1,charge2] = get_torus_geometry();
npts = chnkr.npt; % total number of points in discretization

src = chnkr.r(:,:); % coordinates of points on the generating curve
n_src = chnkr.n(:,:); % normals of all the points on the generating curve

% plot geometry
%plot(chnkr, 'b.');
strength = 100.0;

% compute f (boundary condition)
x=src(1,:);
y=src(1,:).*0;
z=src(2,:);
pts=[x;y;z];

n_x=n_src(1,:);
n_y=n_src(1,:).*0;
n_z=n_src(2,:);
n_pts = [n_x;n_y;n_z];

% set up neumann boundary data
% (one positive and one negative charge for compatibility condition)
rvec1 = pts - charge1;
r3_1 = vecnorm(rvec1).^3;
fn1 = sum(rvec1 .* n_pts, 1) ./ (4*pi*r3_1);

rvec2 = pts - charge2;
r3_2 = vecnorm(rvec2).^3;
fn2 = -sum(rvec2 .* n_pts, 1) ./ (4*pi*r3_2);

f = strength*(fn1 + fn2);

% quadrature options
opts = [];
opts.rcip = false;
opts.forcesmooth = false;
opts.l2scale = false;
opts.sing = 'hs';

% discretize integral equation
% (chunkermat_normal is just the chunkermat that shidong has not edited for
% RCIP)
G = @(s,t) kern_0th_mode(s,t,[0,0],'dprime');
A = 1/(4*pi^2) * chunkermat_normal(chnkr, G, opts);

% enforce zero-mean constraint for compatability condition
A = A + onesmat(chnkr);

% solve the linear system
sigma = gmres(A, f', [], 1e-12, npts);

% evaluation quadrature options
opts.forcesmooth = false;
opts.verb = false;
opts.quadkgparams = {'RelTol', 1e-8, 'AbsTol', 1.0e-8};
opts.sing = 'smooth';

% target in cylindrical coordinates (r,z)
target_cyl = [sqrt(target(1)^2 + target(2)^2);target(3)];

% define solution representation
G = @(s,t) kern_0th_mode(s,t,[0,0],'d');
G_eval = -1/(4*pi^2) * kernel(G);

% evaluate at target
u_sol = chunkerkerneval(chnkr, G_eval, sigma, target_cyl, opts);

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
