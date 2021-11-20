function [x,runtime] = ADMM_gray2d(y,lambda,n_iters,varargin)
% *************************************************************************
% * This function applies the alternating direction method of multipliers
%   (ADMM) algorithm to solve the following denoising problem:
%
%           min { J(x) = 1/2 || x - y ||_2^2 + lambda * TV(x) },
%            x
%
%   where y denotes the noisy observation, TV(x) stands for the total 
%   variation (TV) regularizer of x.
%
%   See the Readme.md file for details.
% 
% * References:
%   [1] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, 
%       "Distributed Optimization and Statistical Learning via the 
%       Alternating Direction Method of Multipliers,¡± Foundations and 
%       Trends? in Machine Learning 3, 1-122 (2011).
%
% *************************************************************************
% * Author : Yunhui Gao
% * Date   : 2021/11/20
% *************************************************************************
%
%   ===== Required inputs =================================================
%
%	- y       : 2D array
%               The noisy observation.
%
% 	- lambda  : float
%               Regularization paramter.
%
%   - n_iters : int
%               Number of iterations for solving the denoising problem.
%
%   ===== Optional inputs =================================================
%
%   - 'tv_type'  : string, default = 'anisotropic'
%                  Type of TV regularizer, should be either 'isotropic' or
%                  'anisotropic'.
%
%   - 'x_init'   : 2D array, default = zeros(size(y))
%                  Initial guess of the image x.
%
%   - 'rho_init' : float, default = 1
%                  Initial value for the penalty parameter rho.
% 
%   - 'mu'       : float, default = 10
%                  Updating parameter for the penalty rho. See the document
%                  for details.
%
%   - 'tau'      : float, default = 2
%                  Updating parameter for the penalty rho. See the document
%                  for details.
%
%   ===== Outputs =========================================================
%
%   - x       : 2D array
%               The solution.
%
%   - runtime :  float
%                Runtime of the algorithm.
%
% *************************************************************************

%% settings
% add path
addpath(genpath('utils'));  % path for helper functions

% assign default values
tv_type = 'anisotropic';
x = zeros(size(y));
rho = 1;
mu = 10;
tau = 2;

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
        case 'x_init'
            x = varargin{i+1};
        case 'rho_init'
            rho = varargin{i+1};
        case 'mu'
            mu = varargin{i+1};
        case 'tau'
            tau = varargin{i+1};
        otherwise
            error(['Invalid parameter: ',varargin{i}]);
    end
end

if ~strcmp(tv_type,'isotropic') && ~strcmp(tv_type,'anisotropic')
    error('Unknown tv_type (should be either ''isotropic'' or ''anisotropic'')')
end

% auxiliary variables
mask = ones(size(D(x)));    % mask (to deal with the circular boundary condition)
mask(end,:,1) = 0;
mask(:,end,2) = 0;
[m,n] = size(y);
dh = [0,0,0;-1,1,0;0,0,0];  % finite difference operator
dh_pad = zeros(m,n);
dh_pad(m/2:m/2+2,n/2:n/2+2) = dh;
dv = [0,-1,0;0,1,0;0,0,0];
dv_pad = zeros(m,n);
dv_pad(m/2:m/2+2,n/2:n/2+2) = dv;
fdh = fft2(dh_pad);
fdv = fft2(dv_pad);
deno = 1 + rho*abs(fdh).^2 + rho*abs(fdv).^2;


%% 
% =========================================================================
%                         auxilary functions
% =========================================================================
% calculate the 2-norm of a vector
function val = norm2(x)
    val = sqrt(dot(x(:),x(:)));
end

% solution to the z-subproblem
function z = z_solver(x,u,lambda,rho)
    if strcmp(tv_type,'anisotropic')
        w = mask.*(D(x)+1/rho*u);
        z = soft_threshold(w,lambda/rho);
    else
        w = mask.*(D(x)+1/rho*u);
        w_v = w(:,:,1);
        w_h = w(:,:,2);
        t = sqrt(w_v.^2+w_h.^2);
        z = soft_threshold(t,lambda/rho)./(t+eps).*w;
    end
end

% soft-thresholding function
function v = soft_threshold(x,kappa)
    v = max(x-kappa,0) - max(-x-kappa,0);
end

% solution to the x-subproblem
function x = x_solver(z,u,y,rho)
    x = ifft2(fft2(y + rho*DT(z) - DT(u))./deno);
end

%% 
% =========================================================================
%                               main loop
% =========================================================================
% initialization
z = D(x);
u = zeros(size(z));
timer = tic;
for i = 1:n_iters
    
    % x update
    x_next = x_solver(z,u,y,rho);
    
    % z update
    z_next = z_solver(x_next,u,lambda,rho);
    
    % u update
    u_next = u + rho*(D(x_next) - z_next);
    
    % update rho
    s = -rho*DT(z_next - z);
    r = D(x) - z;
    s_norm = norm2(s);
    r_norm = norm2(r);
    if r_norm > mu*s_norm
        rho = rho*tau;
    elseif s_norm > mu*r_norm
        rho = rho/tau;
    end
    
    % update the variables
    x = x_next;
    z = z_next;
    u = u_next;
    
end

runtime = toc(timer);

end

