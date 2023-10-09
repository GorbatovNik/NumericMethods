import numpy as np
import mygauss
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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

def genSim(k, l = -10, r = 10):
	MAT = np.random.uniform(l,r, (k,k))
	mat = np.array(MAT)
	At = mat.transpose()
	mat = np.dot(At, mat)
	print(mat)
	return mat

def Gershgorin(matrix):
    circles = []
    for i in range(len(matrix)):
        a = matrix[i][i]
        r = sum(map(abs, matrix[i]))
        circles.append([a, r])
    return circles

def ShowCircles(circles, lambds):
    _, axes = plt.subplots()
    print(lambds)
    for i in lambds:
        plt.plot(i, 0, 'ro')
    l, r = 0, 0

    for i in circles:
        print(i)
        d = plt.Circle((i[0], 0), i[1], fill=False)
        radius = i[1]
        l = min(l, i[0] - 2*radius)
        r = max(r, i[0] + 2*radius)
        axes.add_patch(d)

    axes.set_aspect("equal", "box")
    plt.axis('scaled')
    x = np.linspace(l, r, 2000)
    axes.plot(x, 0*x)
    plt.title("Gershgorin circles")
    plt.show()

A = [[2.2, 1.0, 0.5, 2.0],
	 [1.0, 1.3, 2.0, 1.0],
	 [0.5, 2.0, 0.5, 1.6],
	 [2.0, 1.0, 1.6, 2.0]]
# A = genSim(400)
n = len(A)
left_x = min([2*A[i][i] - sum([abs(A[i][j]) for j in range(0, n)]) for i in range(0, n)])
right_x = max([sum([abs(A[i][j]) for j in range(0, n)]) for i in range(0, n)])
print("lambdaы will be searched on the segment [" + str(left_x) + " .. " + str(right_x) + "]")

f = np.array(A)
s = np.identity(n)
for i in range(n - 1):
	m = np.identity(n)
	m[n - 2 - i][:] = f[n - 1 - i][:] # выделяем M^(-1)
	f = np.dot(m, f) # умножаем A на M^(-1) слева
	f = np.dot(f, mygauss.matrix_inverse(m.copy())) # умножаем A на M справа
	s = np.dot(s, mygauss.matrix_inverse(m.copy())) # находим S
p = f[0][:] # выделяем p

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
	y = [l ** i for i in range(n - 1, -1, -1)] # строим y
	x = np.dot(s, y)
	vecs += [x]
	print('eigenvector ' + str(k) +' = ' + str(x))


ShowCircles(Gershgorin(A), lambdas)
print('Vieta\'s check: ' + str(abs(sum(lambdas) - sum(A[i][i] for i in range(0, n)))<EPS))
print('ortho check: ' + str(all([(np.dot(vecs[i], vecs[j]) < EPS if i!=j else True) for j in range(0, n) for i in range(0, n)])))