%Use LinearizeReprojErr, update_solution and ComputeReprojectionError to
%solve the problem using 
% Import data from computer exercise 2. 

load("ce2data.mat");

P = {K*P1, K*P2};
U = X;
u = {hx1, hx2};


gammak = 10^-10;

[err, res] = ComputeReprojectionError(P, U, u);
[r, J]=LinearizeReprojErr(P,U,u);
deltav = -gammak *J' * r ;
[Pnew, Unew] = update_solution(deltav, P, U);

n = 10;
plot(0, sum(res), 'r*')
hold on

for i=1:n
    [err, res] = ComputeReprojectionError(Pnew, Unew, u);
    [r, J]=LinearizeReprojErr(Pnew,Unew,u);
    deltav = -gammak *J' * r ;
    [Pnew,Unew]=update_solution(deltav,Pnew,Unew);
    plot(i, sum(res), 'r*')
end
RMS = sqrt(err/size(res,2));
disp(RMS)