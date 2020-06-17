from SALib.test_functions import Ishigami
import numpy as np
import SALib

problem = {
'num_vars': 3,
'names': ['x1', 'x2', 'x3'],
'bounds': [[-3.14159265359, 3.14159265359],
[-3.14159265359, 3.14159265359],
[-3.14159265359, 3.14159265359]]
}

X = latin.sample(problem, 1000)
Y = Ishigami.evaluate(X)
Si = rbd_fast.analyze(problem, X, Y, print_to_console=True)

print("x1-x2:", Si['S2'][0,1])
print("x1-x3:", Si['S2'][0,2])
print("x2-x3:", Si['S2'][1,2])

X = SALib.sample.morris.sample(problem, 1000, num_levels=4)
Y = Ishigami.evaluate(X)
Si = SALib.analyze.morris.analyze(problem, X, Y, conf_level=0.95, print_to_console=True, num_levels=4)