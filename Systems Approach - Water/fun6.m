function[RMSE] = fun5(t, pcp, pet, x1, tk, tc, tp, ta, tg, x2, x3, x4, ts, str)
for t=1:t
[x1(t+1),px(t)] = fun1(pcp(t),pet(t),x1(t),tk,tc,tp);
[rg(t),of(t)] = fun2(px(t),ta);
[bf(t),x2(t+1)] = fun3(rg(t),tg,x2(t));
[sf(t),x3(t+1),x4(t+1)] = fun4(x3(t),x4(t),of(t),ts);
q(t) = sf(t) + bf(t);
end
obs = q';
RMSE = sqrt(mean((obs - str).^2));
end