function div = DT(grad)
% *************************************************************************
% * This function calculates adjoint operator of D.
% *************************************************************************
% * Author : Yunhui Gao
% * Date   : 2021/08/21
% *************************************************************************
%
%   ===== Required inputs =================================================
%
%	- grad  : 4D array of shape (n1, n2, 3, 2)
%             The input finite differences.
%
%   ===== Outputs =========================================================
%
%   - div  : 3D array of shape (n1, n2, 3)
%            The divergence of grad.
%
% *************************************************************************

[n1,n2,~,~] = size(grad);

div = zeros(n1,n2,3);

shift = circshift(squeeze(grad(:,:,1,1)),[1,0]);
div1 = squeeze(grad(:,:,1,1)) - shift;
div1(1,:) = squeeze(grad(1,:,1,1));
div1(n1,:) = -shift(n1,:);

shift = circshift(squeeze(grad(:,:,1,2)),[0,1]);
div2 = squeeze(grad(:,:,1,2)) - shift;
div2(:,1) = squeeze(grad(:,1,1,2));
div2(:,n2) = -shift(:,n2);

div(:,:,1) = div1 + div2;

shift = circshift(squeeze(grad(:,:,2,1)),[1,0]);
div1 = squeeze(grad(:,:,2,1)) - shift;
div1(1,:) = squeeze(grad(1,:,2,1));
div1(n1,:) = -shift(n1,:);

shift = circshift(squeeze(grad(:,:,2,2)),[0,1]);
div2 = squeeze(grad(:,:,2,2)) - shift;
div2(:,1) = squeeze(grad(:,1,2,2));
div2(:,n2) = -shift(:,n2);

div(:,:,2) = div1 + div2;

shift = circshift(squeeze(grad(:,:,3,1)),[1,0]);
div1 = squeeze(grad(:,:,3,1)) - shift;
div1(1,:) = squeeze(grad(1,:,3,1));
div1(n1,:) = -shift(n1,:);

shift = circshift(squeeze(grad(:,:,3,2)),[0,1]);
div2 = squeeze(grad(:,:,3,2)) - shift;
div2(:,1) = squeeze(grad(:,1,3,2));
div2(:,n2) = -shift(:,n2);

div(:,:,3) = div1 + div2;

end

