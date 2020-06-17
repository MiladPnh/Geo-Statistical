function[bf,x2] = fun3(rg,tg,x20)
   bf = tg*x20;
   x2 = x20 - bf + rg;
end