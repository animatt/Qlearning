clear, clc, close all

% Agent learns to navigate across a gridworld with obstacles including a
% large gap traversed by a bridge using Qlearning.

% initialize env
Im = double(imread('gridworld-bridge.png'));
Im = imresize(rgb2gray(Im), [32 24]);

GR = imquantize(Im, multithresh(Im, 1));
[m, n] = size(GR);

start = sub2ind(size(GR), 31 * ones(1, 5), 6 : 10);
finish = sub2ind(size(GR), 2 * ones(1, 5), 15 : 19);

GR([start; finish]) = 1.5;

imagesc(GR), colormap(gray), axis equal

% initialize agent (row, col[, action])
Q = zeros(m, n, 4);
target_policy = ones(m, n);
Action_Set = [1 0; 0 -1; -1 0; 0 1]';
alpha = 0.1;
gamma = 0.95;

learner_converging = true;
while learner_converging
    S = datasample(start, 1);
    A = randi(4);
    step = Action_Set * (1 : 4 == A)';
    
    reward = 0;
    
    % get subscripts
    [row, col] = ind2sub([m n], S);
    SA = sub2ind(size(Q), row, col, A);
    
    [~, S2] = bound(GR, [row; col], step + [row; col]);
    
    % do initial backup
    Q(SA) = Q(SA) + alpha * (reward + max(Q(S2(1), S2(2), :)));
    
    episode_in_progress = true;
    while episode_in_progress
        A = randi(4);
        rstep = -round(sin((A - 1) * pi / 2));
        cstep = round(cos((A - 1) * pi / 2));
    end
end