function [r, d, d2] = ellipse2(t,varargin) 
%ELLIPSE return position, first and second derivatives of an ellipse 
% with the parameterization 
%
% x(t) = a*cos(t) + c
% y(t) = b*sin(t) + d
%
% Syntax: [r,d,d2] = ellipse(t,a,b) 
%
% Input:
%   t - array of points (in [0,2pi])
%
% Optional input:
%   a - xscaling
%   b - xscaling
%
% Output:
%   r - 2 x numel(t) array of positions, r(:,i) = [x(t(i)); y(t(i))]
%   d - 2 x numel(t) array of t derivative of r 
%   d2 - 2 x numel(t) array of second t derivative of r 
%
% Examples:
%   [r,d,d2] = ellipse(t); % circle parameterization
%   [r,d,d2] = ellipse(t,a,b); % stretch circle into ellipse
%
a = 1;
b = 1;
c = 0;
d = 0;
if nargin > 1 && ~isempty(varargin{1})
    a = varargin{1};
end
if nargin > 2 && ~isempty(varargin{2})
    b = varargin{2};
end
if nargin > 3 && ~isempty(varargin{3})
    c = varargin{3};
end
if nargin > 4 && ~isempty(varargin{4})
    d = varargin{4};
end

r =  [ a*cos(t(:).') + c; b*sin(t(:).') + d];
d =  [-a*sin(t(:).'); b*cos(t(:).')];
d2 = [-a*cos(t(:).');-b*sin(t(:).')];



end