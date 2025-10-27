% Solve the Axi-symmetric boundary and dirichlet boundary condition 
% Laplace problem (aka just the 0th mode modal greens function)

clearvars; 
close all;
format long e;

% geometry
[chnkr,~,~] = get_disk_geometry();
npts = chnkr.npt; % total number of points in discretization
src = chnkr.r(:,:); % coordinates of points on the generating curve [2,64]
%plot(chnkr, 'b.');

% setup quadrature options
opts = [];
opts.rcip = true;
opts.l2scale = false;
opts.forcesmooth = false;
opts.nsub_or_tol = 30;

%nsys = 2*npts;
origin = [0,0];

% define kernels
c = 2.0;
Z = kernel.zeros();
S = @(s,t) 1/(4*pi^2) * kern_0th_mode(s,t,origin,'s');
Dprime = @(s,t) 1/(4*pi^2) * kern_0th_mode(s,t,origin,'dprime');

opts.sing = 'log';
Smat = chunkermat_normal(chnkr, S, opts);
opts.sing = 'hs';
Dpmat = chunkermat_normal(chnkr, Dprime, opts);
Kmat = Smat*Dpmat;

% discretize system
%K = [Z, c * kernel(S);
%     c * kernel(Dprime), Z];
%K = kernel(K);

% chunkermat_normal gives actual accuracy unlike the shidong chunkermat
%Kmat = chunkermat_normal(chnkr, K, opts) + eye(nsys);

% compute boundary condition
f = ones(1,npts);
%rhs = zeros(nsys,1);
%rhs(1:2:end) = f';
rhs = f';

% solve
sigma = gmres(Kmat, rhs, [], 1e-12, npts);

sigma2 = gmres(Smat, rhs, [], 1e-12, npts);

% compute error
sigma_exact = 4./(pi*sqrt(1-src(1,:).^2))';
%error = sigma(2:2:end) - sigma_exact;
error1 = sigma2 - sigma_exact;
error2 = Dpmat*sigma - sigma_exact;



%% geometry functions
function [chnkobj,target,charge] = get_disk_geometry()
    pref = [];
    pref.k = 16;
    %pref.nchmax = 4;

    fchnks = [];
    cparams = [];
    %cparams.nover = 2;
    %cparams.maxchunklen = 2;
    cparams.ta = 0;
    cparams.tb = 1;
    cparams.nchmin = 8;

    verts = [0 1;0 0];
    edge2verts = [1;2];

    chnkobj = chunkgraph(verts, edge2verts, fchnks, cparams, pref);
    chnkobj = balance(chnkobj);

    target = [0.3;0.3;0.0];
    charge = [0.0;0.0;0.5];
end