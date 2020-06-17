function [x,y] = ModLLR(P, X0, dt, K, Period) % Creating a Subroutine
    x = zeros(max(Period),1);
    y = zeros(max(Period),1); % Initialization
    x(1) = X0;
    for t = 2:max(Period)
        y(t-1) = x(t-1)*K;
        x(t) = x(t-1) + (P(t-1) - y(t-1))*dt;
    end
    y(t) = x(t)*K;
end    