function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
m = length(y);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = sigmoid(X*theta);

term_1 = y.*log(h);
term_2 = (1 - y).*log(1 - h);
num_elements = size(y,1);

sq_theta = theta.^2;
% make the first element of sq_theta as 0 since it is not to be added
% with the other elements for regularization.
sq_theta(1) = 0;

J = (sum(-term_1 - term_2))/num_elements + (lambda*sum(sq_theta))/(2*num_elements);

temp_theta = theta;
% Make the first element as 0 so we can use the generic equation to find 
% the gradient and not have to use two separate equation for gradient
% calculation
temp_theta(1) = 0;
regularization = (lambda/num_elements).*temp_theta;
grad = ((sum((h - y).*X))/num_elements) + regularization';

% =============================================================

end
