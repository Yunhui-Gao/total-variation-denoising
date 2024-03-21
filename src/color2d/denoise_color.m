function [x,runtime] = denoise_color(y,lam,n_iters,varargin)
% *************************************************************************
% * This function applies the fast gradient projection (FGP) algorithm to
%   solve the following color image denoising problem: 
%
%           min { J(x) = 1/2 * || x - y ||_2^2 + lambda * TV(x) },
%            x
%   
%   where y denotes the noisy observation, TV(x) stands for the total 
%   variation (TV) regularizer of x.
%
%   See the README.md file for details.
% 
% * References:
%   [1] A. Beck and M. Teboulle, "Fast Gradient-Based Algorithms for 
%       Constrained Total Variation Image Denoising and Deblurring 
%       Problems," IEEE Transactions on Image Processing 18, 2419-2434 
%       (2009).
%
% *************************************************************************
% * Author : Yunhui Gao
% * Date   : 2021/11/20
% *************************************************************************
%
%   ===== Required inputs =================================================
%
%	- y       : 3D array of shape (n1, n2, 3)
%               Noisy image.
%
%   - lam     : float or 1D array of length 3
%               Regularization parameter.
%
%   - n_iters : int
%               Number of iterations.
%
%   ===== Optional inputs =================================================
%
%   - 'tv_type'  : string, default = 'anisotropic'
%                  Type of TV regularizer, should be either 'isotropic' or
%                  'anisotropic'.
%
%   ===== Outputs =========================================================
%
%   - x       : 3D array of shape (n1, n2, 3)
%               Denoised color image.
%
%   - runtime : float
%               Runtime of the algorithm.
%
% *************************************************************************
%% settings
% add path
addpath(genpath('utils'));  % path for helper functions

% assign default values
tv_type = 'anisotropic';

%% parse input arguments
if (nargin-length(varargin)) ~= 3
    error('Wrong number of required inputs');
elseif rem(length(varargin),2) == 1
    error('Optional inputs should always go by pairs')
end
for i = 1:2:length(varargin)-1
    switch lower(varargin{i})
        case 'tv_type'
            tv_type = varargin{i+1};
        otherwise
            error(['Invalid parameter: ',varargin{i}]);
    end
end

if ~strcmp(tv_type,'isotropic') && ~strcmp(tv_type,'anisotropic')
    error('Unknown tv_type (should be either ''isotropic'' or ''anisotropic'')')
end

if length(lam) == 1
    lam = [lam, lam, lam];
elseif length(lam) ~= 3
    error('The length of lambda should be 1 or 3')
end
if min(lam) <= 0
    error('The input lambda should take positive values')
end

%% main loop
[n1,n2,~] = size(y);
grad_next = zeros(n1,n2,3,2);
grad_prev = zeros(n1,n2,3,2);
u = zeros(n1,n2,3,2);

t_prev = 1;

timer = tic;
if strcmp(tv_type,'anisotropic')
    for i = 1:n_iters
        grad_next = u + 1/8*D(y - DT(u));
        deno = zeros(n1,n2,3,2);
        deno(:,:,1,1) = 1/lam(1)*max(lam(1),abs(grad_next(:,:,1,1)));
        deno(:,:,1,2) = 1/lam(1)*max(lam(1),abs(grad_next(:,:,1,2)));
        deno(:,:,2,1) = 1/lam(2)*max(lam(2),abs(grad_next(:,:,2,1)));
        deno(:,:,2,2) = 1/lam(2)*max(lam(2),abs(grad_next(:,:,2,2)));
        deno(:,:,3,1) = 1/lam(3)*max(lam(3),abs(grad_next(:,:,3,1)));
        deno(:,:,3,2) = 1/lam(3)*max(lam(3),abs(grad_next(:,:,3,2)));

        grad_next = grad_next./deno;
        t_next = (1+sqrt(1+4*t_prev^2))/2;
        u = grad_next + (t_prev-1)/t_next*(grad_next-grad_prev);
        grad_prev = grad_next;
        t_prev = t_next;
    end
else
    for i = 1:n_iters
        grad_next = u + 1/8*D(y - DT(u));
        deno = zeros(n1,n2,3,2);
        deno(:,:,1,1) = 1/lam(1)*max(lam(1),sqrt(grad_next(:,:,1,1).^2 + grad_next(:,:,1,2).^2));
        deno(:,:,1,2) = deno(:,:,1,1);
        deno(:,:,2,1) = 1/lam(2)*max(lam(2),sqrt(grad_next(:,:,2,1).^2 + grad_next(:,:,2,2).^2));
        deno(:,:,2,2) = deno(:,:,2,1);
        deno(:,:,3,1) = 1/lam(3)*max(lam(3),sqrt(grad_next(:,:,3,1).^2 + grad_next(:,:,3,2).^2));
        deno(:,:,3,2) = deno(:,:,3,1);

        grad_next = grad_next./deno;
        t_next = (1+sqrt(1+4*t_prev^2))/2;
        u = grad_next + (t_prev-1)/t_next*(grad_next-grad_prev);
        grad_prev = grad_next;
        t_prev = t_next;
    end
end

x = y - DT(grad_next);  % convert to the primal optimal
runtime = toc(timer);

end

