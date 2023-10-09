import numpy as np
import mygauss
from sympy import *

EPS = 1e-9

def newton(func, x):
	y = func(x)
	dx = 0.001
	k = (func(x+dx)-func(x-dx))/(2*dx)
	return x if abs(y/k)<EPS else newton(func, x - y/k)

def getFuncByPolynome(coeffs):
    return lambda x: sum(coeff * x**(len(coeffs) -1 - i) for i, coeff in enumerate(coeffs))

def gorner(pol, a):
	new_pol = [pol[0]]
	for i in range(1, len(pol)-1):
		new_pol = new_pol + [a*new_pol[i-1]+pol[i]]
	pol = new_pol
	return pol

A = [[2.2, 1.0, 0.5, 2.0],
	 [1.0, 1.3, 2.0, 1.0],
	 [0.5, 2.0, 0.5, 1.6],
	 [2.0, 1.0, 1.6, 2.0]]
n = len(A)
left_x = min([2*A[i][i] - sum([abs(A[i][j]) for j in range(0, n)]) for i in range(0, n)])
right_x = max([sum([abs(A[i][j]) for j in range(0, n)]) for i in range(0, n)])
print("lambdaы will be searched on the segment [" + str(left_x) + " .. " + str(right_x) + "]")
c = []
c.append([1] + [0 for i in range(0, n-1)]) # начальный с
for i in range(1, n + 1):
	c.append(np.dot(A, c[i - 1])) # рекурсивно вычисляем c
C = np.array(c) # копия С
cn = c.pop() # выделяем столбец свободных членов cn
c = np.array(c).transpose() 
for i in range(n): # транспонируем матрицу коэффициентов C
	c[i] = list(reversed(c[i]))
p = mygauss.gauss(c.copy(), cn.copy()) # решаем систему C*p = cn
pol = [1.] + list(-p)
lambdas = []
start = (right_x - left_x)/2
while len(pol) > 1:
	pol_func = getFuncByPolynome(pol)
	lmb = newton(pol_func, start)
	lambdas += [lmb]
	pol = gorner(pol, lmb)
vecs = []
for k, l in enumerate(lambdas):
	print('lambda ' + str(k) + ' = ' + str(l))
	b = ones(n)
	for i in range(1, n):
		b[i] = b[i - 1] * l - p[i - 1]
	x = np.sum([b[i] * C[n - i - 1] for i in range(n)], axis=0) # вычисляем собственный вектор x
	vecs += [x]
	print('eigenvector ' + str(k) +' = ' + str(x))

assert all([(np.dot(vecs[i], vecs[j]) < EPS if i!=j else True) for j in range(0, n) for i in range(0, n)])