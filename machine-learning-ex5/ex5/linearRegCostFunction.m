function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% X = [ones(m,1), X];
%J = sum((X * theta - y) .^ 2) / (2 * m) + lambda / (2 * m) * sum(theta(2:end) .^2); 
J = 1 / (2 * m) * sum((X * theta - y) .^ 2) + lambda / (2 * m) * sum(theta(2:end) .^ 2);

% NOTE: FOLLOWING IMPLEMENTATION IS WRONG SOMEHOW
%grad = (1 / m) * sum((X * theta - y) .* X) + (lambda / m) * [0, theta(2:end)'];  % Note the [0, theta(2:end)] here, very important
% And note that here the dimension of the regularized term is 1*2 not 2*1, so use , instead of ;

grad = (1 / m) * X' * (X * theta - y) + (lambda / m) * [0; theta(2:end)]; 

%htheta = X * theta;
%n = size(theta);
%J = 1 / (2 * m) * sum((htheta - y) .^ 2) + lambda / (2 * m) * sum(theta(2:n) .^ 2);
%
%grad = 1 / m * X' * (htheta - y);
%grad(2:n) = grad(2:n) + lambda / m * theta(2:n);
% =========================================================================

grad = grad(:);

end
