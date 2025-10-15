function [gvals, gdzs, gdrs, gdrps] = g0funcall_vec(r, rp, dr, z, zp, dz, maxm, mask_iter)
    % chunkermat gives a negative radius sometimes which makes no sense
    % maybe its testing a random point or something
    r0 = rp;
    rzero = sqrt(r.*r + r0.*r0 + dz.*dz);
    alpha = 2*r.*r0./rzero.^2;
    x = 1./alpha;
    
    % use forward reccurence
    if (maxm > 12307)
        mask1 = 1.00000005d0 <= (x < 1.005);
    elseif (maxm > 4380)
        mask1 = 1.0000005d0 <= (x < 1.005);
    elseif (maxm > 1438)
        mask1 = 1.000005d0 <= (x < 1.005);
    elseif (maxm > 503)
        mask1 = 1.00005d0 <= (x < 1.005);
    elseif (maxm > 163)
        mask1 = 1.0005d0 <= (x < 1.005);
    else
        mask1 = x < 1.005; 
    end
    mask2_stable = ~mask1; % use backward reccurence

    gvals = zeros(maxm+1,size(r,1),size(r,2));
    gdzs = zeros(maxm+1,size(r,1),size(r,2));
    gdrs = zeros(maxm+1,size(r,1),size(r,2));
    gdrps = zeros(maxm+1,size(r,1),size(r,2));
    
    [gvals(:,mask1),gdzs(:,mask1),gdrs(:,mask1),gdrps(:,mask1)] = forward_rec(r(mask1),rp(mask1),dr(mask1),z(mask1),zp(mask1),dz(mask1),maxm);
    %[gvals(:,mask2_stable),gdzs(:,mask2_stable),gdrs(:,mask2_stable),gdrps(:,mask2_stable)] = backward_rec(r(mask2_stable),rp(mask2_stable),dr(mask2_stable),z(mask2_stable),zp(mask2_stable),dz(mask2_stable),maxm);
    %
    xd = gvals(:,mask1);
    if any(isnan(xd), 'all')
        %size(xd)
        %sum(isnan(xd(:)))
    end

    for i=1:mask_iter
        % check if all green evaluations have beed made
        if nnz(mask2_stable) == 0
            break;
        elseif (i == mask_iter)
            need_more_masks = 1
        end

        [mask_blowup_vals,mask_stable_vals,nterms] = get_masks(r(mask2_stable),rp(mask2_stable),dz(mask2_stable),maxm);
        
        mask2_blowup = false(size(mask1));
        mask2_blowup(mask2_stable) = mask_blowup_vals;

        mask2_stable_temp = false(size(mask1));
        mask2_stable_temp(mask2_stable) = mask_stable_vals;
        mask2_stable = mask2_stable_temp;

        [gvals(:,mask2_blowup),gdzs(:,mask2_blowup),gdrs(:,mask2_blowup),gdrps(:,mask2_blowup)] = backward_rec(r(mask2_blowup),rp(mask2_blowup),dr(mask2_blowup),z(mask2_blowup),zp(mask2_blowup),dz(mask2_blowup),maxm,nterms);
        
        % check if anything blew up
        if any(isnan(gvals(:,mask2_blowup)), 'all')
            something_blew_up = 1
        end  
    end
    %
end

function [mask_blowup, mask_stable, nterms] = get_masks(r, rp, dz, maxm)
    r0 = rp;
    rzero = sqrt(r.*r + r0.*r0 + dz.*dz);
    alpha = 2*r.*r0./rzero.^2;
    x = 1./alpha;
    done = 1;
    half = done/2;
    f = ones(size(r,1),size(r,2));
    fprev = zeros(size(r,1),size(r,2));
    der = ones(size(r,1),size(r,2));
    derprev = zeros(size(r,1),size(r,2));
    maxiter = 10000;
    ubound = 1.0e20;
    lbound = 1.0e17;
    nterms = maxiter;
    for i = maxm:maxiter
        fnext = (2*i*x.*f - (i-half)*fprev)/(i+half);
        dernext = (2*i*(x.*der+f) - (i-half)*derprev)/(i+half);
        if (max(abs(fnext),[],"all") >= ubound)
            if (max(abs(dernext),[],"all") >= ubound)
                nterms = i+1;
                break
            end
        end
        fprev = f;
        f = fnext;
        derprev = der;
        der = dernext;
    end

    if (nterms < 10)
        nterms = 10;
    end

    mask_blowup = (abs(fnext) >= lbound) & (abs(dernext) >= lbound);
    mask_stable = ~mask_blowup;
end


function [gvals, gdzs, gdrs, gdrps] = forward_rec(r, rp, dr, z, zp, dz, maxm)
    done = 1.0;
    r0 = rp;
    z0 = zp;
    rzero = sqrt(r.*r + r0.*r0 + dz.*dz);
    alpha = 2*r.*r0./rzero.^2;
    x = 1./alpha;
    xminus = (dr.*dr + dz.*dz)/2./r./r0;

    dxdr = (r.^2 - r0.^2 - (dz).^2)/2./r0./r.^2;
    dxdz = 2*(dz)/2./r./r0;
    dxdr0 = (r0.^2 - r.^2 - (dz).^2)/2./r./r0.^2;
    dxdz0 = -dxdz;

    gvals = zeros(maxm+1,size(r,1),size(r,2));
    gdzs = zeros(maxm+1,size(r,1),size(r,2));
    gdrs = zeros(maxm+1,size(r,1),size(r,2));
    gdrps = zeros(maxm+1,size(r,1),size(r,2));

    [q0, q1, dq0] = chnk.axissymlap2d.qleg_half(xminus);
    dq1 = (-q0 + x.*q1)/2./(x+1)./xminus;
    
    half = done/2;

    fac = 2*pi*sqrt(rp./r);
    gvals(1,:,:) = fac.*q0;
    gvals(2,:,:) = fac.*q1;

    derprev = fac.*dq0;
    der = fac.*dq1;
    
    % the z derivatives
    gdzs(1,:,:) = derprev.*dxdz;
    gdzs(2,:,:) = der.*dxdz;
    
    % the r derivatives
    gdrs(1,:,:) = fac.*(dq0.*dxdr - q0/2./r);
    gdrs(2,:,:) = fac.*(dq1.*dxdr - q1/2./r);

    % the rp derivatives
    gdrps(1,:,:) = fac.*(dq0.*dxdr0 - q0/2./r0);
    gdrps(2,:,:) = fac.*(dq1.*dxdr0 - q1/2./r0);

    % don't compute the zp derivatives, just minus the z derivatives

    % run upward recursion for the Q's to calculate them things
    x_reshaped = reshape(x, [1, size(x,1), size(x,2)]);
    r_reshaped = reshape(r, [1, size(r,1), size(r,2)]);
    r0_reshaped = reshape(r0, [1, size(r0,1), size(r0,2)]);
    dxdz_reshaped = reshape(dxdz, [1, size(dxdz,1), size(dxdz,2)]);
    dxdr_reshaped = reshape(dxdr, [1, size(dxdr,1), size(dxdr,2)]);
    dxdr0_reshaped = reshape(dxdr0, [1, size(dxdr0,1), size(dxdr0,2)]);

    der_reshaped = reshape(der, [1, size(der,1), size(der,2)]);
    derprev_reshaped = reshape(derprev, [1, size(derprev,1), size(derprev,2)]);
    for i = 1:(maxm-1)
        gvals(i+2,:,:) = (2*i*x_reshaped.*gvals(i+1,:,:) - (i-half)*gvals(i,:,:))/(i+half);
        dernext = (2*i*(gvals(i+1,:,:)+x_reshaped.*der_reshaped) - (i-half)*derprev_reshaped)/(i+half);
        
        gdrs(i+2,:,:) = (dernext.*dxdr_reshaped - gvals(i+2,:,:)/2./r_reshaped);
        gdzs(i+2,:,:) = dernext.*dxdz_reshaped;
        gdrps(i+2,:,:) = (dernext.*dxdr0_reshaped - gvals(i+2,:,:)/2./r0_reshaped);
        derprev_reshaped = der_reshaped;
        der_reshaped = dernext;
    end
end

function [gvals, gdzs, gdrs, gdrps] = backward_rec(r, rp, dr, z, zp, dz, maxm, nterms)
    done = 1;
    half = done/2;
    r0 = rp;
    z0 = zp;
    rzero = sqrt(r.*r + r0.*r0 + dz.*dz);
    alpha = 2*r.*r0./rzero.^2;
    x = 1./alpha;
    xminus = (dr.*dr + dz.*dz)/2./r./r0;

    dxdr = (r.^2 - r0.^2 - (dz).^2)/2./r0./r.^2;
    dxdz = 2*(dz)/2./r./r0;
    dxdr0 = (r0.^2 - r.^2 - (dz).^2)/2./r./r0.^2;
    dxdz0 = -dxdz;

    gvals = zeros(maxm+1,size(r,1),size(r,2));
    gdzs = zeros(maxm+1,size(r,1),size(r,2));
    gdrs = zeros(maxm+1,size(r,1),size(r,2));
    gdrps = zeros(maxm+1,size(r,1),size(r,2));
    
    %!
    %! now start at nterms and recurse down to maxm
    %!    
    f = ones(size(r,1),size(r,2));
    fnext = zeros(size(r,1),size(r,2));
    der = ones(size(r,1),size(r,2));
    dernext = zeros(size(r,1),size(r,2)); 

    % run the downward recurrence
    for j = 1:(nterms-maxm+1)
        i = nterms-j+1;
        fprev = (2*i*x.*f - (i+half)*fnext)/(i-half);
        fnext = f;
        f = fprev;
        derprev = (2*i*(x.*der+f) - (i+half)*dernext)/(i-half);
        dernext = der;
        der = derprev;
    end

    gvals(maxm,:,:) = f;
    gvals(maxm+1,:,:) = fnext;

    ders = zeros(maxm+1,size(r,1),size(r,2));
    ders(maxm,:,:) = der;
    ders(maxm+1,:,:) = dernext;
    
    x_reshaped = reshape(x, [1, size(x,1), size(x,2)]);
    for j = 1:(maxm-1)
        i = maxm-1-j+1;
        gvals(i,:,:) = (2*i*x_reshaped.*gvals(i+1,:,:) - (i+half)*gvals(i+2,:,:))/(i-half);
        ders(i,:,:) = (2*i*(x_reshaped.*ders(i+1,:,:)+gvals(i+1,:,:)) - (i+half)*ders(i+2,:,:))/(i-half);
    end

    % !
    % ! normalize the values, and use a formula for the derivatives
    % !
    [q0, q1, dq0] = chnk.axissymlap2d.qleg_half(xminus);
    dq1 = (-q0 + x.*q1)/2./(x+1)./xminus;
    % call axi_q2lege01(x, xminus, q0, q1, dq0, dq1)

    sqrt_thing = sqrt(rp./r);
    sqrt_thing = reshape(sqrt_thing, [1, size(sqrt_thing,1), size(sqrt_thing,2)]);
    q0_reshaped = reshape(q0, [1, size(q0,1), size(q0,2)]);
    ratio = 2*pi*q0_reshaped./gvals(1,:,:).*sqrt_thing;
    
    for i = 1:(maxm+1)
        gvals(i,:,:) = gvals(i,:,:).*ratio;
    end

    ders(1,:,:) = 2*pi*dq0.*sqrt(rp./r);
    ders(2,:,:) = 2*pi*dq1.*sqrt(rp./r);
    xminus_reshaped = reshape(xminus, [1, size(xminus,1), size(xminus,2)]);
    for i = 2:maxm
       ders(i+1,:,:) = -(i-.5d0).*(gvals(i,:,:) - x_reshaped.*gvals(i+1,:,:))./(1+x_reshaped)./xminus_reshaped;
    end

    %
    % and scale the gradients properly everyone...
    %
    r_reshaped = reshape(r, [1, size(r,1), size(r,2)]);
    r0_reshaped = reshape(r0, [1, size(r0,1), size(r0,2)]);
    dxdz_reshaped = reshape(dxdz, [1, size(dxdz,1), size(dxdz,2)]);
    dxdr_reshaped = reshape(dxdr, [1, size(dxdr,1), size(dxdr,2)]);
    dxdr0_reshaped = reshape(dxdr0, [1, size(dxdr0,1), size(dxdr0,2)]);
    
    gdzs = ders.*dxdz_reshaped;
    gdrs = ders.*dxdr_reshaped - gvals/2./r_reshaped;
    gdrps = ders.*dxdr0_reshaped - gvals/2./r0_reshaped;
end