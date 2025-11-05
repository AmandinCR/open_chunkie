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




[gs,gdzs,gdrs,gdrps,gdrprs,gdzzs,gdrzs,gdrpzs] = chnk.axissymlap2d.gfuncall_amandin(r,rp,dr,z,zp,dz,m);

%{
gs2 = zeros(m+1,size(r,1),size(r,2));
gdrs2 = zeros(m+1,size(r,1),size(r,2));
gdrps2 = zeros(m+1,size(r,1),size(r,2));
gdzs2 = zeros(m+1,size(r,1),size(r,2));
for i=1:size(r,1)
    for j=1:size(r,2)
        [gs2(:,i,j),gdzs2(:,i,j),gdrs2(:,i,j),gdrps2(:,i,j)] = chnk.axissymlap2d.g0funcall(r(i,j),rp(i,j),dr(i,j),z(i,j),zp(i,j),dz(i,j),m);
    end
end

M1 = max(abs(gs2(m,:,:)-gs(m,:,:)), [], 'all');
if M1 > 1e-6
    xd = sum(abs(gs2(m,:,:)-gs) > 1e-6, 'all');
    disp(['error = ' num2str(xd)]);
end
%}



const = 1/(4*pi^2);

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
