function [gval, gdz, gdr, gdrp, gdrpr, gdzz, gdrz, gdrpz] = gfuncall_amandin(r, rp, dr, z, zp, dz, m)

%
% chnk.axissymlap2d.gfunc evaluates the zeroth order axisymmetric Laplace
% Green's funcion, defined by the expression:
%
%     gfunc = pi * rp * \int_0^{2\pi} 1/|x - x'| d\theta'
%
% The extra factor of rp out front makes subsequent interfacing with RCIP
% slightly easier
%
% 

    t = (dz.^2+dr.^2)./(2.*r.*rp);
    chi = t+1;
    
    [qm, qmd, qmdd] = chnk.axissymlap2d.qleg_half_miller_vec(t,m);

    r = reshape(r,[1,size(r,1),size(r,2)]);
    rp = reshape(rp,[1,size(rp,1),size(rp,2)]);
    dr = reshape(dr,[1,size(dr,1),size(dr,2)]);
    z = reshape(z,[1,size(z,1),size(z,2)]);
    zp = reshape(zp,[1,size(zp,1),size(zp,2)]);
    dz = reshape(dz,[1,size(dz,1),size(dz,2)]);

    t = reshape(t,[1,size(t,1),size(t,2)]);
    chi = reshape(chi,[1,size(chi,1),size(chi,2)]);
    
    gval = 2*pi*sqrt(rp./r).*qm;
    gdz  = 2*pi*sqrt(rp./r).*qmd ...
          ./(rp.*r).*dz;
    
    rfac = -r/2.*qm+(-(1+t).*r+rp).*qmd;
    gdrp  = 2*pi*sqrt(rp./r)./(rp.*r) ...
            .*rfac;
    
    rfac = -rp/2.*qm+(-(1+t).*rp+r).*qmd;
    gdr  = 2*pi*sqrt(rp./r)./(rp.*r) ...
           .*rfac;

    rfac = 1./(rp.*r).*qmd + (dz./(rp.*r)).^2.*qmdd;
    gdzz = 2*pi*sqrt(rp./r).*rfac;
    
    rfac = -3./(2*r.^2.*rp).*qmd + (-chi./(r.^2.*rp) + 1./(r.*rp.^2)).*qmdd;
    gdrz = 2*pi*sqrt(rp./r).*dz.*rfac;

    rfac = -3./(2*rp.^2.*r).*qmd + (-chi./(rp.^2.*r) + 1./(rp.*r.^2)).*qmdd;
    gdrpz = 2*pi*sqrt(rp./r).*dz.*rfac;

    rfac = 1./(4*r.*rp).*qm ...
        + (2*chi./(rp.*r) - 3./(2*r.^2) - 3./(2*rp.^2)).*qmd ...
        + (-chi./r+1./rp).*(-chi./rp + 1./r).*qmdd;
    gdrpr = 2*pi*sqrt(rp./r).*rfac;
end