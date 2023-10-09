import numpy as np
import math

EPS = 1e-6

def perm_if_zero(mat, f, i):
	if abs(mat[i][i]) > EPS:
		return 0

	n = mat.shape[0]
	for x in range(i + 1, n):
		if abs(mat[x][i]) > EPS:
			mat[i], mat[x] = mat[x].copy(), mat[i].copy()
			f[i], f[x] = f[x], f[i]
			return 0
	return -1

def gauss(mat, f):
	assert mat.shape[0] == mat.shape[1]
	assert mat.shape[1] == f.shape[0]

	n = mat.shape[0]
	for i in range(n):
		if perm_if_zero(mat, f, i) == 0:
			for k in range(i + 1, n):
				l = -mat[k][i]/mat[i][i]
				mat[k] += mat[i]*l
				f[k] += f[i]*l
	x = np.zeros(n)
	x[n-1] = f[n-1]/mat[n-1][n-1]
	for i in range(n-2, -1, -1):
		assert abs(mat[i][i]) > EPS
		to_right = np.dot(mat[i][i+1:], x[i+1:])
		x[i] = (f[i] - to_right)/mat[i][i]
	return x		

n = 15
a, b = -10, 100
predominance = 0.5 #[0.0 .. 1.0]
mat = np.random.rand(n,n)*(b-a) + a
if predominance > 0.0:
	for i in range(n):
		for j in range(n):
			if i == j:
				mat[i][j] += (b-mat[i][j])*predominance if abs(b)>abs(a) else -(mat[i][j]-a)*predominance
			else:
				mat[i][j] /= n

x = np.random.rand(n)*(b-a) + a

print("generated x:")
print(x)
f = np.dot(mat,x) 
bib_solution = np.linalg.solve(mat, f)
print("\nnp.linalg solution:")
print(bib_solution)
absol = math.sqrt(np.sum((x - bib_solution)**2))
rel = absol / math.sqrt(np.sum(bib_solution**2))
print("\nnp.linalg absolute error: " + str(absol))
print("np.linalg relative error: " + str(rel))
my_solution = gauss(mat, f)
print("\ngauss method solution:")
print(my_solution)
absol = math.sqrt(np.sum((x - my_solution)**2))
rel = absol / math.sqrt(np.sum(my_solution**2))
print("\ngauss method absolute error: " + str(absol))
print("gauss method relative error: " + str(rel))