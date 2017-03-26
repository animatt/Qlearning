function [state_not_terminal, resultant_position] = bound(image, pos, pnext)
% check the next state and ensure it is allowed before accepting it. If it
% is not permitted, return the previous state. If the next state is on the
% finish line, permit it note that the state is terminal.
if image(pnext(1), pnext(2)) == 1.5 % reached finish line
    resultant_position = pnext;
    state_not_terminal = false;
elseif GR(pnext(1), pnext(2)) == 2
    resultant_position = pnext;
    state_not_terminal = true;
else
    resultant_position = pos;
    state_not_terminal = true;
end
end