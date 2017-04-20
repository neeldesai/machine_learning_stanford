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
sq_theta = theta.^2;
% Make sq_theta(1) = 0 since we do not want to regularize the first element
% i.e the bias value
sq_theta(1) = 0;

J = (1/(2*m))*(sum(((X*theta)- y).^2)) + (lambda/(2*m))*(sum(sq_theta));

temp_theta = theta;
% Make the first element as 0 so we can use the generic equation to find 
% the gradient and not have to use two separate equation for gradient
% calculation
temp_theta(1) = 0;
regularization = (lambda/m).*temp_theta;
grad = ((sum(((X*theta) - y).*X))/m) + regularization';


% =========================================================================

grad = grad(:);

end
