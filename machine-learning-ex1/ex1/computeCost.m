function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
%h = transpose(theta)*transpose(X);
%h=transpose(h);
% Below 1 line is equavilent to the above two lines
h=X*theta;
hy=h-y;
hy2=hy.^2;
S = sum(hy2, 1);

J=S/m/2;
% =========================================================================

end
