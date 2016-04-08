function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ===*g===================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%hypot = sigmoid(X * theta);
%
%costsum = zeros(m, 1);
%for i=1:m
%    costsum(i) = -y(i) * log(hypot(i)) - (1 - y(i)) * log(1 - hypot(i));
%end
%J = sum(costsum, 1) / m;
%
%for j=1:size(grad)(1)
%    temp=0;
%    for i=1:m
%        temp = temp + (hypot(i)-y(i)) * X(i,j);
%    end
%    grad(j) = temp / m;
%end

hx = sigmoid(X * theta);
%m = length(X);

J = sum(-y' * log(hx) - (1 - y')*log(1 - hx)) / m;
grad = X' * (hx - y) / m;
% =============================================================

end
