import numpy as np

EPS = 1e-9

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

def gauss(mat, f, perm_func = perm_vh):
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

def matrix_inverse(matrix):
    n = len(matrix)

    identity_matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    for col in range(n):
        pivot = matrix[col][col]
        for j in range(n):
            matrix[col][j] /= pivot
            identity_matrix[col][j] /= pivot

        for i in range(n):
            if i != col:
                factor = matrix[i][col]
                for j in range(n):
                    matrix[i][j] -= factor * matrix[col][j]
                    identity_matrix[i][j] -= factor * identity_matrix[col][j]
    
    return identity_matrix