function[sf,x3,x4] = fun4(x30,x40,of,ts)
   y = x30 * ts;
   x3 = x30 - y + of;
   sf = x40 * ts;
   x4 = x40 + y - sf;
end