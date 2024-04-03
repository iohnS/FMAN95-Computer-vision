%Solve using given fivepoint_solver from 5 random points to get the
%essential matrix. Check for inliers using ransac and this its 5 pixels
%inside. 

% Load data

data = load('../assignment4data/compEx2data.mat');
im1 = imread('../assignment4data/im1.jpg');
im2 = imread('../assignment4data/im2.jpg');

K = data.K;
x1 = data.x{1};
x2 = data.x{2};

hx1 = [x1; ones(1, size(x1, 2))];
hx2 = [x2; ones(1, size(x2, 2))];

normx1 = K\hx1;
normx2 = K\hx2;

nx1t = normx1';
nx2t = normx2';

N = 100; % Number of iterations
e = 5; % Acceptable error
n_best_inliers = 0;
best_E = [];
for i = 1:N
    indices = randperm(size(nx1t, 1), 5);
    rx1 = nx1t(indices, :);
    rx2 = nx2t(indices, :);
    E = fivepoint_solver(rx1', rx2');
    for j = 1:length(E)
        currentE = E{j};
        F = (K^-1)'*currentE*K^-1;

        ep1 = pflat(F' * hx2);
        ep1 = ep1 ./ sqrt(repmat(ep1(1, :).^2 + ep1(2, :).^2 ,[3 1]));
        ep2 = pflat(F * hx1);
        ep2 = ep2 ./ sqrt(repmat(ep2(1, :).^2 + ep2(2, :).^2 ,[3 1]));

        d1 = abs(sum(ep1.*hx1));
        d2 = abs(sum(ep2.*hx2));
        
        inliers = (d1 < 5) & (d2 < 5);
        
        nbr_of_inliers = sum(inliers(:));
        if nbr_of_inliers > n_best_inliers
            best_E = E{j};
            n_best_inliers = nbr_of_inliers;
            best_inlier = inliers;
        end 
    end
end

W = [0 -1 0; 1 0 0; 0 0 1];
Z = [0 1 0; -1 0 0; 0 0 0];
[U, ~, V] = svd(best_E);
if det(U*V') < 0
    V = -V;
end
newE = U * diag([1 1 0]) * V';
[U, S, V] = svd(newE);
u3 = U(:, 3);
mu3 = -u3;
P1 = [eye(3) zeros(3, 1)];
P2a = [U * W * V' u3];
P2b = [U * W * V' mu3];
P2c = [U * W' * V' u3];
P2d = [U * W' * V' mu3];

X1 = pflat(triangulate(P1, P2a, nx1t, nx2t))';
X2 = pflat(triangulate(P1, P2b, nx1t, nx2t))';
X3 = pflat(triangulate(P1, P2c, nx1t, nx2t))';
X4 = pflat(triangulate(P1, P2d, nx1t, nx2t))';

P2s = {P2a, P2b, P2c, P2d};
Xs = {X1, X2, X3, X4};
[P2, X] = check_in_front(P1, P2s, Xs);
hx1 = hx1(:,best_inlier==1);
hx2 = hx2(:,best_inlier==1);
X = X(:,best_inlier==1);
reprx1 = P1*X;
reprx2 = P2*X;
distances1 = pdist2(reprx1(1:2, :)', normx1(1:2,:)');
distances1 = distances1(:,1);
distances2 = pdist2(reprx2(1:2, :)', normx2(1:2,:)');
distances2 = distances2(:,1);

histogram(distances2)
%plot3(X(1,:), X(2,:), X(3,:), ".")

save('ce2data.mat','P1', 'P2', 'X', 'hx1', 'hx2', 'K');
