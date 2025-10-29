% Solve the Axi-symmetric boundary and dirichlet boundary condition 
% Laplace problem (aka just the 0th mode modal greens function)

clearvars; 
close all;
format long e;

% geometry
[chnkr,~,~] = get_disk_geometry();
src = chnkr.r(:,:); % coordinates of points on the generating curve [2,64]

%plot(chnkr, 'b.');


% setup quadrature options
opts = [];

%nsys = 2*npts;
origin = [0,0];

% define kernels
c = 2.0;
Z = kernel.zeros();
S = kernel('axissymlap','s');
Dp = kernel('axissymlap','dprime');

K = [ Z       c*S;
      c*Dp   Z ];
K = kernel(K);
Keval = kernel([Z c*S]);

npts = chnkr.npt;
nsys = K.opdims(1)*npts;
rhs = zeros(nsys, 1);
rhs(1:K.opdims(1):end) = 1;

% Build the system matrix
opts = [];opts.l2scale = false;opts.rcip = true;
opts.nsub_or_tol = 30;
start = tic;
A = chunkermat(chnkr, K, opts) + eye(nsys);
t1 = toc(start);
fprintf('%5.2e s : time to build the system matrix\n', t1)

% Solve the linear system
start = tic;
sol = gmres(A, rhs, [], 1e-12, 200);
t1 = toc(start);

% Compute the numerical solution
opts.forcesmooth = false;
opts.verb = false;
opts.quadkgparams = {'RelTol', 1e-12, 'AbsTol', 1.0e-12};

if isa(chnkr, 'chunkgraph')
    chnkrs = chnkr.echnks;
    chnkrtotal = merge(chnkrs);
else
    chnkrtotal = chnkr;
end

ntarg = 100;
targets = rand(2,ntarg);targets(2,:)=targets(2,:)+0.5;
start = tic;
unum = chunkerkerneval(chnkrtotal, Keval, sol, targets, opts);
t2 = toc(start);
fprintf('%5.2e s : time to eval at targs (slow, adaptive routine)\n', t2)

% Reference solution 
rho = targets(1,:);rho=rho(:);
z = targets(2,:);z=z(:);
z2= z.^2;
r2 = rho.^2+z2;

uref = 2/pi*acot(sqrt(0.5*((r2-1)+sqrt((r2-1).^2+4*z2))));
relerr  = norm(unum-uref) / norm(uref)





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
