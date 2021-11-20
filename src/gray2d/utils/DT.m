function div = DT(grad)
% *************************************************************************
% * This function calculates the adjoint operator of D.
% *************************************************************************
% * Author : Yunhui Gao
% * Date   : 2021/04/20
% *************************************************************************
%
%   ===== Required inputs =================================================
%
%	- grad  : 3D array
%             The input finite differences.
%
%   ===== Outputs =========================================================
%
%   - grad : 2D array
%            The output of the adjoint operator.
%
% *************************************************************************

[n1,n2,~] = size(grad);

shift = circshift(grad(:,:,1),[1,0,0]);
div1 = grad(:,:,1) - shift;
div1(1,:) = grad(1,:,1);
div1(n1,:) = -shift(n1,:);

shift = circshift(grad(:,:,2),[0,1,0]);
div2 = grad(:,:,2) - shift;
div2(:,1) = grad(:,1,2);
div2(:,n2) = -shift(:,n2);

div = div1 + div2;

end

