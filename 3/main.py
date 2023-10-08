import numpy as np
import matplotlib.pyplot as plt
import math

EPS = 1e-6

def diagonal_dominance_degree(matrix):
    assert matrix.shape[0] == matrix.shape[1]
    n = matrix.shape[0]  # Размерность матрицы (число строк или столбцов)
    max_difference = -1e9
    for i in range(n):
        diagonal_value = abs(matrix[i, i])
        row_sum = np.sum(np.abs(matrix[i])) - diagonal_value 
        if diagonal_value - row_sum > max_difference:
            max_difference = diagonal_value - row_sum

    return max_difference

def perm_if_zero(mat, f, i, p):
	if abs(mat[i][i]) > EPS:
		return 0

	n = mat.shape[0]
	for x in range(i + 1, n):
		if abs(mat[x][i]) > EPS:
			mat[i], mat[x] = mat[x].copy(), mat[i].copy()
			f[i], f[x] = f[x], f[i]

def perm_v(mat, f, i, p):
	mxv, mxi = 0, 0
	n = mat.shape[0]
	for x in range(i, n):
		if abs(mat[x][i]) > mxv:
			mxv = abs(mat[x][i])
			mxi = x
	mat[i], mat[mxi] = mat[mxi].copy(), mat[i].copy()
	f[i], f[mxi] = f[mxi], f[i]

def perm_h(mat, f, i, p):
	mxv, mxj = 0, 0
	n = mat.shape[0]
	for y in range(i, n):
		if abs(mat[i][y])> mxv:
			mxv = abs(mat[i][y])
			mxj = y
	p[i], p[mxj] = p[mxj], p[i]
	for x in range(0, n):
		mat[x][i], mat[x][mxj] = mat[x][mxj], mat[x][i]

def perm_vh(mat, f, i, p):
	perm_v(mat, f, i, p)
	perm_h(mat, f, i, p)

def x_after_p(x, p):
	px = np.zeros(len(x))
	for i in range(len(x)):
		px[p[i]] = x[i]
	return px

def gauss(mat, f, perm_func):
	assert mat.shape[0] == mat.shape[1]
	assert mat.shape[1] == f.shape[0]

	n = mat.shape[0]
	p = list(range(0, n))
	for i in range(n):
		perm_func(mat, f, i, p)
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
	px = x_after_p(x, p)
	return px

n = 4
a, b = 0, 10
bib_sols = []
v_gauss_sols = []
h_gauss_sols = []
vh_gauss_sols = []
ref_mat = np.random.rand(n,n)*(b-a) + a
ref_x = np.random.rand(n)*(b-a) + a
for pr in [i / 10 for i in range(-100, 101)]:
	predominance = pr #[0.0 .. 1.0]
	mat = np.random.rand(n,n)*(b-a) + a
	for i in range(n):
		for j in range(n):
			if i == j:
				mat[i][j] *=(n-1)/2
				mat[i][j] += predominance
	x = np.random.rand(n)*(b-a) + a
	f = np.dot(mat,x) 
	dom = diagonal_dominance_degree(mat)

	def getRel(sol, x):
		absol = math.sqrt(np.sum((x - sol)**2))
		return absol / math.sqrt(np.sum(sol**2))

	bib_solution = np.linalg.solve(mat, f)
	bib_sols = bib_sols + [[dom, getRel(bib_solution, x)]]

	gauss_solution = gauss(mat.copy(), f.copy(), perm_v)
	v_gauss_sols = v_gauss_sols + [[dom, getRel(gauss_solution, x)]]

	gauss_solution = gauss(mat.copy(), f.copy(), perm_h)
	h_gauss_sols = h_gauss_sols + [[dom, getRel(gauss_solution, x)]]

	gauss_solution = gauss(mat.copy(), f.copy(), perm_vh)
	vh_gauss_sols = vh_gauss_sols + [[dom, getRel(gauss_solution, x)]]
	
sorted_bib_sols = sorted(bib_sols, key=lambda x: x[0])
sorted_v_gauss_sols = sorted(v_gauss_sols, key=lambda x: x[0])
sorted_h_gauss_sols = sorted(h_gauss_sols, key=lambda x: x[0])
sorted_vh_gauss_sols = sorted(vh_gauss_sols, key=lambda x: x[0])

def drawSols(sols, lbl):
	x = [item[0] for item in sols]
	y = [item[1] for item in sols]
	plt.plot(x, y, label=lbl)

drawSols(sorted_v_gauss_sols, 'Гаусс с верт. перест-ми')
drawSols(sorted_h_gauss_sols, 'Гаусс с гориз. перест-ми')
drawSols(sorted_vh_gauss_sols, 'Гаусс с гориз. и с верт. перест-ми')
drawSols(sorted_bib_sols, 'numpy.linalg.solve')
plt.legend()
plt.xlabel('Степень преобладания')
plt.ylabel('Относительная погрешность')
plt.show()



# print("generated x:")
# print(x)
# f = np.dot(mat,x) 

# bib_solution = np.linalg.solve(mat, f)
# print("\nnp.linalg solution:")
# print(bib_solution)
# absol = math.sqrt(np.sum((x - bib_solution)**2))
# rel = absol / math.sqrt(np.sum(bib_solution**2))
# print("\nnp.linalg absolute error: " + str(absol))
# print("np.linalg relative error: " + str(rel))
# my_solution = gauss(mat, f)
# print("\ngauss method solution:")
# print(my_solution)
# absol = math.sqrt(np.sum((x - my_solution)**2))
# rel = absol / math.sqrt(np.sum(my_solution**2))
# print("\ngauss method absolute error: " + str(absol))
# print("gauss method relative error: " + str(rel))