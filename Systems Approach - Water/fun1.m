function[x1,px] = fun1(pp,pet,x10,tk,tc,tp)
   px1 = max(0, pp+x10-tc);
   px2 = tk*x10;
   px = px1 + px2;
   x10 = x10 - px2;
   ae = min(pet*(x10/tc)^tp,x10);
   x10 = x10 - ae;
   x1 = x10 - px1 + pp;
end





