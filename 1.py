import sympy as sp

x = sp.symbols('x')

def f(x):
    return 1 / x

slope_sym = sp.diff(f(x), x)

for x_val in [1, -1]:
    
    slope = slope_sym.subs(x, x_val)
    
    y_val = f(x_val)
    c = y_val - slope * x_val

    print("Slope at x = {} is {}".format(x_val, slope))
    print("Tangent line is y = {}*x + {:.2f}".format(slope, c))


import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def f(x):
    return 1 / x

x = sp.symbols('x')

f_sym = 1 / x

f_prime_sym = sp.diff(f_sym, x)

x_values = np.linspace(-3, 3, 400)
y_values = f(x_values)

plt.figure(figsize=(8, 6))

plt.plot(x_values, y_values, label='f(x) = 1/x')

for x_val in [1, -1]:
    slope = f_prime_sym.subs(x, x_val)
    tangent_line = slope * (x_values - x_val) + f(x_val)
    plt.plot(x_values, tangent_line, label=f'Tangent at x={x_val}')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graph of f(x) = 1/x and Tangent Lines')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()
