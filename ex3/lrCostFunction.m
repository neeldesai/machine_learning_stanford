function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


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

grad = grad(:);

end
