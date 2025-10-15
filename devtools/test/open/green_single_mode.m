function [val, grad] = green_single_mode(src, targ, origin, mode)
%
% CHNK.AXISSYMHELM2D.GREEN evaluate the Laplace green's function
% for the given sources and targets. 
%
% Note: that the first coordinate is r, and the second z.
% The code relies on precomputed tables and hence loops are required for 
% computing various pairwise interactions.
% Finally, the code is not efficient in the sense that val, grad, hess 
% are always internally computed independent of nargout
%
% Returns for gradient are:
% grad = d_{r}, d_{r'}, d_{z}, d_{z'}

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

%{
gs = zeros(mode+1,size(r,1),size(r,2));
gdrs = zeros(mode+1,size(r,1),size(r,2));
gdrps = zeros(mode+1,size(r,1),size(r,2));
gdzs = zeros(mode+1,size(r,1),size(r,2));
for i=1:size(r,1)
    for j=1:size(r,2)
        [gs(:,i,j),gdzs(:,i,j),gdrs(:,i,j),gdrps(:,i,j)] = chnk.axissymlap2d.g0funcall(r(i,j),rp(i,j),dr(i,j),z(i,j),zp(i,j),dz(i,j),mode);
    end
end
%}
[gs,gdzs,gdrs,gdrps] = chnk.axissymlap2d.g0funcall_vec(r,rp,dr,z,zp,dz,mode,30);

val = gs(mode,:,:);
val = reshape(val, [nt, ns]);
gtmp = zeros(nt, ns, 4);
gtmp(:,:,1) = gdrs(mode,:,:);
gtmp(:,:,2) = gdrps(mode,:,:);
gtmp(:,:,3) = gdzs(mode,:,:);
gtmp(:,:,4) = -gdzs(mode,:,:);
grad = gtmp;

end
