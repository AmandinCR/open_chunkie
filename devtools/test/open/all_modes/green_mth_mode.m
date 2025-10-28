function [val, grad, hess] = green_mth_mode(src, targ, origin, m)
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
%
% Returns for hess are:
% hess = d_{rr'}, d_{zz'}, d_{rz'}, d_{r'z}
%
% m is the mode we return

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


%[gs,gdzs,gdrs,gdrps,gdrpr,gdzz,gdrz,gdrpz] = chnk.axissymlap2d.gfuncall_amandin(r,rp,dr,z,zp,dz,m);

gs = zeros(m+1,size(r,1),size(r,2));
gdrs = zeros(m+1,size(r,1),size(r,2));
gdrps = zeros(m+1,size(r,1),size(r,2));
gdzs = zeros(m+1,size(r,1),size(r,2));

gdzzs = zeros(m+1,size(r,1),size(r,2));
gdrprs = zeros(m+1,size(r,1),size(r,2));
gdrzs = zeros(m+1,size(r,1),size(r,2));
gdrpzs = zeros(m+1,size(r,1),size(r,2));
for i=1:size(r,1)
    for j=1:size(r,2)
        [gs(:,i,j),gdzs(:,i,j),gdrs(:,i,j),gdrps(:,i,j),gdrprs(:,i,j),gdzzs(:,i,j),gdrzs(:,i,j),gdrpzs(:,i,j)] = chnk.axissymlap2d.gfuncall_amandin(r(i,j),rp(i,j),dr(i,j),z(i,j),zp(i,j),dz(i,j),m);
    end
end

val = gs(m,:,:);
val = reshape(val,[nt,ns]);
grad = zeros(nt, ns, 4);
grad(:,:,1) = gdrs(m,:,:);
grad(:,:,2) = gdrps(m,:,:);
grad(:,:,3) = gdzs(m,:,:);
grad(:,:,4) = -gdzs(m,:,:);

hess = zeros(nt, ns, 4);
hess(:,:,1) = gdrprs(m,:,:);
hess(:,:,2) = -gdzzs(m,:,:);
hess(:,:,3) = -gdrzs(m,:,:);
hess(:,:,4) = gdrpzs(m,:,:);

end
