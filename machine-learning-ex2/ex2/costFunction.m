function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
d = zeros(length(y));
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
nn = size(theta,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
d = sigmoid(X*theta);
summer = 0;

for i = 1:m,
	summer = summer - (y(i,1)*log(d(i,1)) + (1-y(i,1))*log(1-d(i,1)) );
end;

J = (1/m)*summer;


for j = 1:nn,
	summer2 = 0;
	for i = 1:m,
		summer2 = summer2 + (d(i,1) - y(i,1))*X(i,j);
	end
	temp = (1/m)*summer2;
	grad(j,1) = temp;
end;
% =============================================================

end
