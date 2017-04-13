function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];

z_2 = X*Theta1';
layer_2 = sigmoid(z_2);
layer_2_rows = size(layer_2, 1);
layer_2 = [ones(layer_2_rows, 1) layer_2];

z_3 = layer_2*Theta2';
h = sigmoid(z_3);

for i=1:m
    temp_y = zeros(10,1);
    temp_y(y(i,1),1) = 1;
    Y = temp_y;
    for k=1:num_labels
        term_1 = Y(k,1)*log(h(i,k));
        term_2 = (1 - Y(k,1))*(log(1 - h(i,k)));
        J = J - ((term_1 + term_2)/m);
    end
end


for i=1:hidden_layer_size
% Start from index 2 since the 1st row will be bias values
    for j=2:input_layer_size+1
        J = J + (lambda/(2*m))*(Theta1(i,j))*(Theta1(i,j));
    end
end

for i=1:num_labels
% Start from index 2 since the 1st row will be bias values
    for j=2:hidden_layer_size+1
        J = J + (lambda/(2*m))*(Theta2(i,j))*(Theta2(i,j));
    end
end


% Part 2:

% We can re-use values of a_1, z_2, a_2 (layer_2), z_3 and a_3 (h)
% from the calculations above

%   temp_D3 contains values of delta_3 of all 5000 samples
%   temp_D3 = [ delta_3 for sample 1;
%               delta_3 for sample 2;
%               delta_3 for sample 3;
%               ...
%               ...
%               ...
%               delta_3 for sample 5000]
%   Similarlt temp_D2 contains values of delta_2 for all 5000 samples        
temp_D3 = zeros(m, num_labels);
temp_D2 = zeros(m, hidden_layer_size);

for t=1:m
    temp_y = zeros(1,num_labels);
    temp_y(1,y(t,1)) = 1;
    Y = temp_y;
    delta_3 = h(t, :) - Y;
    
    % Fill row t of the larger matrix with the current value of delta_3
    temp_D3(t, :) = delta_3;
    
    temp_delta_2 = (delta_3*Theta2);
    % Discard the 1st element since we do not want to add correction to the
    % bias unit as well as to get the matrix dimensions to match with that
    % of z_2
    delta_2 = temp_delta_2(2:end).*sigmoidGradient(z_2(t, :));

    % Fill the row t of the larger matrix with the current value of delta_2
    temp_D2(t, :) = delta_2;
end

Theta1_grad = (temp_D2'*X)./m;
Theta2_grad = (temp_D3'*layer_2)./m;

% Part 3
% Adding regularization
% Do not add regularization to the bias units i.e column 1. So start from
% column 2
for i=2:input_layer_size+1
    for j=1:hidden_layer_size
        Theta1_grad(j, i) = Theta1_grad(j, i) + (Theta1(j, i)*(lambda/m));
    end
end

% Do not add regularization to the bias units i.e column 1. So start from
% column 2
for i=2:hidden_layer_size+1
    for j=1:num_labels
        Theta2_grad(j, i) = Theta2_grad(j, i) + (Theta2(j, i)*(lambda/m));
    end
end

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
