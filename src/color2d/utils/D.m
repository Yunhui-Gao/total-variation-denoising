function grad = D(x)
% *************************************************************************
% * This function calculates the finite differences of a color image x.
% *************************************************************************
% * Author : Yunhui Gao
% * Date   : 2021/08/21
% *************************************************************************
%
%   ===== Required inputs =================================================
%
%	- x     : 3D array of shape (n1, n2, 3)
%             The input color image. 
%
%   ===== Outputs =========================================================
%
%   - grad : 4D array of shape (n1, n2, 3, 2)
%            The finite differences of x.
%
% *************************************************************************

[n1,n2,~] = size(x);
grad = zeros(n1,n2,3,2);

r = squeeze(x(:,:,1));
g = squeeze(x(:,:,2));
b = squeeze(x(:,:,3));

grad(:,:,1,1) = r - circshift(r,[-1,0]);
grad(n1,:,1,1) = 0;
grad(:,:,1,2) = r - circshift(r,[0,-1]);
grad(:,n2,1,2) = 0;

grad(:,:,2,1) = g - circshift(g,[-1,0]);
grad(n1,:,2,1) = 0;
grad(:,:,2,2) = g - circshift(g,[0,-1]);
grad(:,n2,2,2) = 0;

grad(:,:,3,1) = b - circshift(b,[-1,0]);
grad(n1,:,3,1) = 0;
grad(:,:,3,2) = b - circshift(b,[0,-1]);
grad(:,n2,3,2) = 0;

end

