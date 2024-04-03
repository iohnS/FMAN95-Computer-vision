%Use LinearizeReprojErr, update_solution and ComputeReprojectionError to
%solve the problem using 
% Import data from computer exercise 2. 

load("ce2data.mat");

P = {K*P1, K*P2};
U = X;
u = {hx1, hx2};


gammak = 10^-10;
lambda = 0.1;

% Computes the reprejection error and the values of all the residuals
% for the current solution P ,U, u .
[err, res] = ComputeReprojectionError(P, U, u);

%Computes the r and J matrices for the appoximate linear least squares problem .
[r, J] = LinearizeReprojErr(P, U, u);  

% Computes the LM update .
deltav = -gammak *J' * r ;
n = 10;

% Update the variables.
[Pnew, Unew] = update_solution(deltav, P, U);
hold on
for i=1:n
    [err, res] = ComputeReprojectionError(Pnew, Unew, u);
    [r, J]=LinearizeReprojErr(Pnew,Unew,u);
    C = J'* J + lambda * speye ( size (J ,2));
    c = J'* r ;
    deltav = -C \ c ;
    [Pnew,Unew]=update_solution(deltav,Pnew,Unew);
    plot(i, sum(res), 'b*')
end
RMS = sqrt(err/size(res,2));
disp(RMS)
