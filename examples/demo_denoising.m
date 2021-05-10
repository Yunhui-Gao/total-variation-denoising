% *************************************************************************
% * This code applies the fast projected gradient algorithm to solve the
%   image denoising problem:
%
%           min { 0.5*|| x - b ||_2^2 + lambda*|| x ||_TV }.
%            x
% 
%   where x and b are two-dimensional arrays representing the estimate for
%   the denoised image and the observed noisy image, respectively. lambda 
%   is the regularization parameter.
% *************************************************************************
% * Author : Yunhui Gao
% * Date   : 2021/04/20
% *************************************************************************

%% generate data
clear;clc;
close all;

% load source functions and test image
addpath(genpath('../src'))
img = im2double(imread('../data/cameraman.tif'));

% Gaussian noise
b = img + normrnd(0, 1e-1, size(img));

% display the image
figure
subplot(1,2,1),imshow(img,[])
title('Ground truth','interpreter','latex','fontsize',16)
subplot(1,2,2),imshow(b,[])
title('Observation','interpreter','latex','fontsize',16)
set(gcf,'unit','normalized','position',[0.25,0.3,0.5,0.4])

%% run the algorithm
rng(0)  % random seed, for reproducibility

lambda = 1e-1;      % regularization parameter
n_iters = 50;       % number of iterations

% run the denoising algorithm, you may try proxTVi (denoising with 
% isotropic TV) or proxTVa (denoising with anisotropic TV) to see how it
% works
x = proxTVi(b,lambda,n_iters);

figure,imshow(x,[])     
title(['Reconstruction after ',num2str(n_iters),' iterations'],'interpreter','latex')
