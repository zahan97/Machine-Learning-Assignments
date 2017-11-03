function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

d = sigmoid(X*theta);
summer = 0;

for i = 1:m,
	summer = summer - (y(i,1)*log(d(i,1)) + (1-y(i,1))*log(1-d(i,1)) );
end;

sumation = 0;
for i = 2:nn,
	sumation = sumation + theta(i,1)*theta(i,1);
end;

J = (1/m)*summer + (lambda/(2*m))*sumation;


for j = 1:nn,
	summer2 = 0;
	for i = 1:m,
		if j == 1
		summer2 = summer2 + (d(i,1) - y(i,1))*X(i,j);
	else
		summer2 = summer2 + (d(i,1) - y(i,1))*X(i,j) + (lambda/m)*theta(j,1);
	end
	end
	temp = (1/m)*summer2;
	grad(j,1) = temp;
end;


% =============================================================

end
