addpath("../../../vlfeat-0.9.21/toolbox");
vl_setup;

A = imread('../assignment4data/a.jpg');
B = imread('../assignment4data/b.jpg');

best_H = [0.61 0.80 -67; -0.80 0.6 167.7; 0 0 1];

Htform = projective2d(best_H');

% Rout = imref2d(size(A), [-200 1600], [-400 1800]);
Rout = imref2d(size(A), [-200 800], [-400 600]);

[Atransf] = imwarp(A, Htform, 'OutputView', Rout);

Idtform = projective2d (eye(3));
[Btransf] = imwarp(B, Idtform, 'OutputView', Rout);

AB = Btransf;
AB(Btransf < Atransf) = Atransf(Btransf < Atransf);

imagesc(Rout.XWorldLimits, Rout.YWorldLimits ,AB);
