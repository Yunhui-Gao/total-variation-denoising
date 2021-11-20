% *************************************************************************
% * This code applies the fast projected gradient algorithm to solve the
%   color image denoising problem:
%
%           min { J(x) = 1/2 || x - y ||_2^2 + lambda * TV(x) },
%            x
% 
%   where y denotes the noisy observation, TV(x) stands for the total 
%   variation (TV) regularizer of x.
%
% *************************************************************************
% * Author : Yunhui Gao
% * Date   : 2021/11/20
% *************************************************************************

%% generate data
clear;clc;
close all;

% load source functions and test image
addpath(genpath('../src'))
img = im2double(imread('../data/lighthouse.png'));
% img = imresize(img,[256,256]);

% Gaussian noise
sigma = [0.2, 0.2, 0.2];    % noise levels for R/G/B channels, respectively
noise = zeros(size(img));
noise(:,:,1) = normrnd(0,sigma(1),[size(img,1),size(img,2)]);
noise(:,:,2) = normrnd(0,sigma(2),[size(img,1),size(img,2)]);
noise(:,:,3) = normrnd(0,sigma(3),[size(img,1),size(img,2)]);
y = img + noise;

% display the image
figure
subplot(1,2,1),imshow(img,[])
title('Ground truth','interpreter','latex','fontsize',16)
subplot(1,2,2),imshow(y,[])
title('Observation','interpreter','latex','fontsize',16)
set(gcf,'unit','normalized','position',[0.25,0.3,0.5,0.4])

%% run the algorithm
rng(0)  % random seed, for reproducibility

lambda = 2e-1;      % regularization parameter
n_iters = 50;       % number of iterations

[x,runtime] = FGP_color2d(y,lambda,n_iters);  % FPG
disp(['runtime: ',num2str(runtime),' s'])
figure,imshow(x,[])     
title(['Reconstruction using the FGP algorithm after ',num2str(n_iters),' iterations'],'interpreter','latex')
