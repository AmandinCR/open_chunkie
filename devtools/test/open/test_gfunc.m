

%
% verify that the modeal Green's functions for Laplace are working properly
%

h = 0.5;

r = 1.0;
rp = r + h;
dr = -h; % r - r'

z = 0.5;
zp = z-h;
dz = h; % z - z'

maxm = 0;

exact = zeros(maxm+1,1);
exact_gdz = zeros(maxm+1,1);
exact_gdr = zeros(maxm+1,1);
exact_gdrp = zeros(maxm+1,1);

exact_gdzz = zeros(maxm+1,1);
exact_gdrz = zeros(maxm+1,1);
exact_gdrpz = zeros(maxm+1,1);
exact_gdrpr = zeros(maxm+1,1);

zk = 0;
nq = 2000;
hhh = 0.0001;

% evaluate the kernel using brute force trapezoidal integration, and then
% compare with the recurrence relation code. Note: The "exact" derivatives are
% estimated using finite difference, so it is expected that the error is O(h^2)

for m = 1:(maxm+1)
    dint = chnk.axissymlap2d.gkm_brute(r, rp, z, zp, zk, m-1, nq);
    exact(m) = dint*4*pi*pi*rp;

    % calculate z derivative using finite differences
    dint2 = chnk.axissymlap2d.gkm_brute(r, rp, z+hhh, zp, zk, m-1, nq);
    dint1 = chnk.axissymlap2d.gkm_brute(r, rp, z-hhh, zp, zk, m-1, nq);
    exact_gdz(m) = (dint2-dint1)/(2*hhh)*4*pi*pi*rp;
    
    % gdzz
    exact_gdzz(m) = (dint2 - 2*dint + dint1)/(hhh^2)*4*pi*pi*rp;

    % calculate r derivative using finite differences
    dint2 = chnk.axissymlap2d.gkm_brute(r+hhh, rp, z, zp, zk, m-1, nq);
    dint1 = chnk.axissymlap2d.gkm_brute(r-hhh, rp, z, zp, zk, m-1, nq);
    exact_gdr(m) = (dint2-dint1)/(2*hhh)*4*pi*pi*rp;

    % calculate rp derivative using finite differences
    dint2 = chnk.axissymlap2d.gkm_brute(r, rp+hhh, z, zp, zk, m-1, nq);
    dint1 = chnk.axissymlap2d.gkm_brute(r, rp-hhh, z, zp, zk, m-1, nq);
    exact_gdrp(m) = (dint2-dint1)/(2*hhh)*4*pi*pi*rp;
    
    % gdrz
    dpp = chnk.axissymlap2d.gkm_brute(r+hhh, rp, z+hhh, zp, zk, m-1, nq);  % +r, +z
    dpm = chnk.axissymlap2d.gkm_brute(r+hhh, rp, z-hhh, zp, zk, m-1, nq);  % +r, -z
    dmp = chnk.axissymlap2d.gkm_brute(r-hhh, rp, z+hhh, zp, zk, m-1, nq);  % -r, +z
    dmm = chnk.axissymlap2d.gkm_brute(r-hhh, rp, z-hhh, zp, zk, m-1, nq);  % -r, -z
    exact_gdrz(m) = (dpp - dpm - dmp + dmm) / (4 * hhh^2)*4*pi*pi*rp;

    % gdrpz
    dpp = chnk.axissymlap2d.gkm_brute(r, rp+hhh, z+hhh, zp, zk, m-1, nq);  % +r, +z
    dpm = chnk.axissymlap2d.gkm_brute(r, rp+hhh, z-hhh, zp, zk, m-1, nq);  % +r, -z
    dmp = chnk.axissymlap2d.gkm_brute(r, rp-hhh, z+hhh, zp, zk, m-1, nq);  % -r, +z
    dmm = chnk.axissymlap2d.gkm_brute(r, rp-hhh, z-hhh, zp, zk, m-1, nq);  % -r, -z
    exact_gdrpz(m) = (dpp - dpm - dmp + dmm) / (4 * hhh^2)*4*pi*pi*rp;

    %gdrpr
    dpp = chnk.axissymlap2d.gkm_brute(r+hhh, rp+hhh, z, zp, zk, m-1, nq);  % +r, +z
    dpm = chnk.axissymlap2d.gkm_brute(r+hhh, rp-hhh, z, zp, zk, m-1, nq);  % +r, -z
    dmp = chnk.axissymlap2d.gkm_brute(r-hhh, rp+hhh, z, zp, zk, m-1, nq);  % -r, +z
    dmm = chnk.axissymlap2d.gkm_brute(r-hhh, rp-hhh, z, zp, zk, m-1, nq);  % -r, -z
    exact_gdrpr(m) = (dpp - dpm - dmp + dmm) / (4 * hhh^2)*4*pi*pi*rp;
end


%
% compare with routine for just zero mode
%
[gval, gdz, gdr, gdrp, gdrpr, gdzz, gdrz, gdrpz] = chnk.axissymlap2d.gfunc(r, rp, dr, z, zp, dz);


disp(['from 0 mode gfunc, gval = ' num2str(gval) ' error = ' num2str(abs(gval-exact(1)))]);
disp(['from 0 mode gfunc, gdz = ' num2str(gdz) ' error = ' num2str(abs(gdz-exact_gdz(1)))]);
disp(['from 0 mode gfunc, gdr = ' num2str(gdr) ' error = ' num2str(abs(gdr-exact_gdr(1)))]);
disp(['from 0 mode gfunc, gdrp = ' num2str(gdrp) ' error = ' num2str(abs(gdrp-exact_gdrp(1)))]);
disp(['from 0 mode gfunc, gdzz = ' num2str(gdzz) ' error = ' num2str(abs(gdzz-exact_gdzz(1)))]);
disp(['from 0 mode gfunc, gdrz = ' num2str(gdrz) ' error = ' num2str(abs(gdrz-exact_gdrz(1)))]);
disp(['from 0 mode gfunc, gdrpz = ' num2str(gdrpz) ' error = ' num2str(abs(gdrpz-exact_gdrpz(1)))]);
disp(['from 0 mode gfunc, gdrpr = ' num2str(gdrpr) ' error = ' num2str(abs(gdrpr-exact_gdrpr(1)))]);