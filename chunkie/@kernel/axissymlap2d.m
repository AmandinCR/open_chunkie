function obj = axissymlap2d(type, m, all_modes)
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
        obj.eval = @(s,t) chnk.axissymlap2d.kern_modal(s, t, [0,0], 's',m, all_modes);
        obj.shifted_eval = @(s,t,o) chnk.axissymlap2d.kern_modal(s, t, o, 's',m, all_modes);
        obj.fmm = [];
        obj.sing = 'log';

    case {'d', 'double'}
        obj.type = 'd';
        obj.eval = @(s,t) chnk.axissymlap2d.kern_modal(s, t, [0,0], 'd',m, all_modes);
        obj.shifted_eval = @(s,t,o) chnk.axissymlap2d.kern_modal(s, t, o, 'd',m, all_modes);
        obj.fmm = [];
        obj.sing = 'smooth';

    case {'sp', 'sprime'}
        obj.type = 'sp';
        obj.eval = @(s,t) chnk.axissymlap2d.kern_modal(s, t, [0,0], 'sprime',m, all_modes);
        obj.shifted_eval = @(s,t,o) chnk.axissymlap2d.kern_modal(s, t, o, 'sprime',m, all_modes);
        obj.fmm = [];
        obj.sing = 'log';

    case {'dp', 'dprime'}
        obj.type = 'dp';
        obj.eval = @(s,t) chnk.axissymlap2d.kern_modal(s, t, [0,0], 'dprime',m, all_modes);
        obj.shifted_eval = @(s,t,o) chnk.axissymlap2d.kern_modal(s, t, o, 'dprime',m, all_modes);
        obj.fmm = [];
        obj.sing = 'hs';

    case {'sc', 'scurl'}
        obj.type = 'sc';
        obj.eval = @(s,t) chnk.axissymlap2d.kern_modal(s, t, [0,0], 'scurl',m, all_modes);
        obj.shifted_eval = @(s,t,o) chnk.axissymlap2d.kern_modal(s, t, o, 'scurl',m, all_modes);
        obj.fmm = [];
        obj.sing = 'pv';

    otherwise
        error('Unknown axissym Laplace kernel type ''%s''.', type);

end

end
