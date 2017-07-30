function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

row = size(z,1);
col = size(z,2);

for i = 1:row,
	for j = 1:col,
		g(i,j) = 1/(1+e^(-z(i,j)));
	end;
end;

for i = 1:row,
	for j = 1:col,
		g(i,j) = g(i,j)*(1 - g(i,j));
	end;
end;

% =============================================================
end
