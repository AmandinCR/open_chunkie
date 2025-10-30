function obj = axissymlap2d(type, coefs)
%KERNEL.AXISSYMLAP2D   Construct the axissymmetric Laplace kernel.

if ( nargin < 1 )
    error('Missing Laplace kernel type.');
end

obj = kernel();
obj.name = 'axissymlaplace';
obj.opdims = [1 1];

switch lower(type)

    case {'s', 'single'}
        obj.type = 's';
        obj.eval = @(s,t) chnk.axissymlap2d.kern(s, t, [0,0], 's');
        obj.shifted_eval = @(s,t,o) chnk.axissymlap2d.kern(s, t, o, 's');
        obj.fmm = [];
        %obj.sing = 'log';

    case {'dp', 'dprime'}
        obj.type = 'dp';
        obj.eval = @(s,t) chnk.axissymlap2d.kern(s, t, [0,0], 'dprime');
        obj.shifted_eval = @(s,t,o) chnk.axissymlap2d.kern(s, t, o, 'dprime');
        obj.fmm = [];
        %obj.sing = 'hs';

    otherwise
        error('Unknown axissym Laplace kernel type ''%s''.', type);

end

end
