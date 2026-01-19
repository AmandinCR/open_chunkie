function [val, grad, hess] = green_modal(src, targ, origin, m, all_modes)
%
% CHNK.AXISSYMHELM2D.GREEN evaluate the Laplace modal green's function
% for the given sources and targets. 
%
% Note: that the first coordinate is r, and the second z.
% The code is not efficient in the sense that val, grad, hess 
% are always internally computed independent of nargout
%
% Returns for gradient are:
% grad = d_{r}, d_{r'}, d_{z}, d_{z'}
%
% Returns for hess are:
% hess = d_{rr'}, d_{zz'}, d_{rz'}, d_{r'z}
%
% m is the mode we return
%
% all_modes = true: means we return size [m*nt, ns] instead of [nt, ns]

[~, ns] = size(src);
[~, nt] = size(targ);

rt = repmat(targ(1,:).',1,ns); % r
rs = repmat(src(1,:),nt,1); % r'
r  = (rt + origin(1));
rp = (rs + origin(1));
dr = rt-rs; % r - r'
z  = repmat(targ(2,:).',1,ns);
zp = repmat(src(2,:),nt,1);
dz = z-zp; % z - z'

[gs,gdzs,gdrs,gdrps,gdrprs,gdzzs,gdrzs,gdrpzs] = chnk.axissymlap2d.g0funcall_vec(r,rp,dr,z,zp,dz,m);

% remove the last mode
sz = size(gs);
idx = repmat({':'}, 1, ndims(gs));
idx{1} = 1:sz(1)-1;
gs = gs(idx{:});
gdzs = gdzs(idx{:});
gdrs = gdrs(idx{:});
gdrps = gdrps(idx{:});
gdrprs = gdrprs(idx{:});
gdzzs = gdzzs(idx{:});
gdrzs = gdrzs(idx{:});
gdrpzs = gdrpzs(idx{:});


const = 1/(4*pi^2);

if all_modes == true
    val = reshape(gs,[m*nt,ns]);
    val = const*val;
    
    grad = zeros(m*nt, ns, 4);
    grad(:,:,1) = reshape(gdrs, [m*nt,ns]);
    grad(:,:,2) = reshape(gdrps, [m*nt,ns]);
    grad(:,:,3) = reshape(gdzs, [m*nt,ns]);
    grad(:,:,4) = reshape(-gdzs, [m*nt,ns]);
    grad = const*grad;
    
    hess = zeros(m*nt, ns, 4);
    hess(:,:,1) = reshape(gdrprs, [m*nt,ns]);
    hess(:,:,2) = reshape(-gdzzs, [m*nt,ns]);
    hess(:,:,3) = reshape(-gdrzs, [m*nt,ns]);
    hess(:,:,4) = reshape(gdrpzs, [m*nt,ns]);
    hess = const*hess;
else
    val = gs(m,:,:);
    val = reshape(val,[nt,ns]);
    val = const*val;
    
    grad = zeros(nt, ns, 4);
    grad(:,:,1) = gdrs(m,:,:);
    grad(:,:,2) = gdrps(m,:,:);
    grad(:,:,3) = gdzs(m,:,:);
    grad(:,:,4) = -gdzs(m,:,:);
    grad = const*grad;
    
    hess = zeros(nt, ns, 4);
    hess(:,:,1) = gdrprs(m,:,:);
    hess(:,:,2) = -gdzzs(m,:,:);
    hess(:,:,3) = -gdrzs(m,:,:);
    hess(:,:,4) = gdrpzs(m,:,:);
    hess = const*hess;
end




end
