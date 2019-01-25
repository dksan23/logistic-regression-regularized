function [J, grad] = costFunctionReg(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h=sigmoid((theta'*X')');
theta1=theta(2:size(theta));
J=1/m*(sum(-y.*log(h)-(1-y).*log(1-h)))+lambda*(1/(2*m))*sum(theta1.^2);
thetareg=theta;
thetareg(1,:)=0; 
grad=(1/m)*((h-y)'*X)'+(1/m)*lambda.*thetareg;






% =============================================================

end
