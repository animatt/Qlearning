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
Qsave = Q;
target_policy = zeros(m, n);

Action_Set = [1 0; 0 -1; -1 0; 0 1]';
alpha = 0.1;
gamma = 0.95;

tolerance = 1 / 100;
count = 0;
learner_converging = true;
while learner_converging
    [row, col] = ind2sub(size(GR), datasample(start, 1));
    A = randi(4);
    step = Action_Set * (1 : 4 == A)';
    
    SA = sub2ind(size(Q), row, col, A);
    
    [~, S2] = bound(GR, [row; col], step + [row; col]);
    
    reward = 0;
    
    % do initial backup
    Q(SA) = Q(SA) + alpha * (reward + gamma * max(Q(S2(1), S2(2), :)) - Q(SA));
    
    episode_in_progress = true;
    while episode_in_progress
        row = S2(1);
        col = S2(2);
        
        A = randi(4);
        step = Action_Set * (1 : 4 == A)';
        
        SA = sub2ind(size(Q), row, col, A);
        
        % get S' and R
        [episode_in_progress, S2] = bound(GR, [row; col], [row; col] + step);
        reward = ~episode_in_progress;
        
        % backup
        Q(SA) = Q(SA) + alpha * (reward + gamma * max(Q(S2(1), S2(2), :)) ...
            - Q(SA));
        
        % improve target policy
        [~, target_policy(row, col)] = max(Q(row, col, :));
    end

    count = count + 1;
    if mod(count, 1000) == 0
        ee = max(max(max(Q - Qsave)));
        if ee <= tolerance
            fprintf('count %d\n', count)
            learner_converging = false;
        end
        Qsave = Q;
    end
end