import tarfile

import numpy as np
import math
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy import linalg
import warnings
import time
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def initial_matrix(n):
    N = n

    def f(x, y):  # delta U = f on interior and delta U = 0 on exterior
        return math.exp(-(x ** 2 + y ** 2))

    dx = 1 / N
    A11 = np.zeros((8 * N * N + 4 * N, 8 * N * N + 4 * N))
    A13 = np.zeros((8 * N * N - 8 * N, 4 * N))
    A22 = np.zeros(((N - 1) ** 2, (N - 1) ** 2))
    A23 = np.zeros(((N - 1) ** 2, 4 * N))
    A31 = np.zeros((4 * N, 8 * N * N - 8 * N))
    A32 = np.zeros((4 * N, (N - 1) ** 2))
    A33 = np.zeros((4 * N, 4 * N))

    f1 = np.zeros(8 * N * N - 8 * N)
    f2 = np.zeros((N - 1) ** 2)
    f3 = np.zeros(4 * N)

    for i in range(N):
        f3[i] = dx ** 2 * f((N + i) * dx, 2 * N * dx)
        f3[N + i] = dx ** 2 * f(2 * N * dx, (2 * N - i) * dx)
        f3[2 * N + i] = dx ** 2 * f((2 * N - i) * dx, N * dx)
        f3[3 * N + i] = dx ** 2 * f(N * dx, (N + i) * dx)

    for i in range(N - 1):
        for j in range(N - 1):
            f2[i * (N - 1) + j] = dx ** 2 * f((N + i + 1) * dx, (2 * N - j - 1) * dx)

    # A11
    for k in range(1, N):  # k in the number of square from inside to outside
        # the inside square is boundary of [1,2]^2, outside square is boundary of [0,3]^2
        start = 4 * (k - 1) * (N + k)

        # use 5-pencil fomula for four special points, topleft,topright,leftbottom,rightbottom.
        A11[start, start] = -4
        A11[start, start + 1] = 1
        A11[start, start + 4 * (N + 2 * k) - 1] = 1
        A11[start, start + 4 * (N + 2 * k) + 1] = 1
        A11[start, start + 4 * (N + 2 * k) + 4 * (N + 2 * (k + 1)) - 1] = 1

        A11[start + N + 2 * k, start + N + 2 * k] = -4
        A11[start + N + 2 * k, start + N + 2 * k + 1] = 1
        A11[start + N + 2 * k, start + N + 2 * k - 1] = 1
        A11[start + N + 2 * k, start + N + 2 * k + 4 * (N + 2 * k) + 1] = 1
        A11[start + N + 2 * k, start + N + 2 * k + 4 * (N + 2 * k) + 3] = 1

        A11[start + 2 * (N + 2 * k), start + 2 * (N + 2 * k)] = -4
        A11[start + 2 * (N + 2 * k), start + 2 * (N + 2 * k) + 1] = 1
        A11[start + 2 * (N + 2 * k), start + 2 * (N + 2 * k) - 1] = 1
        A11[start + 2 * (N + 2 * k), start + 2 * (N + 2 * k) + 4 * (N + 2 * k) + 3] = 1
        A11[start + 2 * (N + 2 * k), start + 2 * (N + 2 * k) + 4 * (N + 2 * k) + 5] = 1

        A11[start + 3 * (N + 2 * k), start + 3 * (N + 2 * k)] = -4
        A11[start + 3 * (N + 2 * k), start + 3 * (N + 2 * k) + 1] = 1
        A11[start + 3 * (N + 2 * k), start + 3 * (N + 2 * k) - 1] = 1
        A11[start + 3 * (N + 2 * k), start + 3 * (N + 2 * k) + 4 * (N + 2 * k) + 5] = 1
        A11[start + 3 * (N + 2 * k), start + 3 * (N + 2 * k) + 4 * (N + 2 * k) + 7] = 1

        # use 5-pencil fomula for each edge of the square
        for j in range(1, N + 2 * k):
            for i in range(4):
                A11[start + j + i * (N + 2 * k), start + j + i * (N + 2 * k)] = -4
                A11[start + j + i * (N + 2 * k), start + j + i * (N + 2 * k) + 1] = 1
                A11[start + j + i * (N + 2 * k), start + j + i * (N + 2 * k) - 1] = 1
                A11[start + j + i * (N + 2 * k), start + j + i * (N + 2 * k) + 4 * (N + 2 * k) + 1 + 2 * i] = 1
                if k != 1:
                    A11[start + j + i * (N + 2 * k), start + j + i * (N + 2 * k) - 4 * (
                            N + 2 * (k - 1)) - 1 - 2 * i] = 1
    A11 = A11[:(8 * N * N - 8 * N), :(8 * N * N - 8 * N)]

    # A13
    for i in range(N + 1):
        for j in range(4):
            A13[1 + j * (N + 2) + i, j * N + i - j] = 1

    # A22
    D = np.identity(N - 1)
    T = -2 * np.identity(N - 1)
    for i in range(N - 2):
        T[i, i + 1] = 1
        T[i + 1, i] = 1
    A22 = np.kron(T, D) + np.kron(D, T)

    # A23
    for i in range(N - 1):
        A23[i, 1 + i] = 1
        A23[-i, 2 * N + 1 + i] = 1
        A23[i * (N - 1), -(1 + i)] = 1
        A23[(i + 1) * (N - 1) - 1, N + 1 + i] = 1

    # A31
    A31 = np.transpose(A13)

    # A32
    A32 = np.transpose(A23)

    # A33
    A33 = -4 * np.identity(4 * N)
    for i in range(4 * N - 2):
        A33[i + 1, i] = 1
        A33[i + 1, i + 2] = 1
    A33[0, 1] = 1
    A33[0, -1] = 1
    A33[-1, 0] = 1
    A33[-1, -2] = 1
    #A11_sp = csc_matrix(A11, dtype=float)
    A11_inv = np.linalg.solve(A11, np.identity(np.shape(A11)[0]))
    #A22_sp = csc_matrix(A22, dtype=float)
    A22_inv = np.linalg.solve(A22, np.identity(np.shape(A22)[0]))
    C = A33 - np.dot(np.dot(A31, A11_inv), A13) - A32 @ A22_inv @ A23
    g = f3 - A32 @ A22_inv @ f2
    return C, g

class low_rank_matrix():
    def __init__(self):
        self.U = None
        self.Vh = None
        self.eigenvalue_mat = None
        self.dim = None

    def construct(self, A, epsilon=0.4):  #
        U, s, Vh = linalg.svd(A)
        self.dim = np.shape(A)[0]
        r = int(epsilon * self.dim)
        if r <= 1:
            r = 1
        self.U = U[:, :r]
        self.Vh = Vh[:r, :]
        self.eigenvalue_mat = np.diag(s[:r])
        #A = self.U @ self.eigenvalue @ self.Vh
        return self

    def add(self, B):
        assert self.dim == B.dim
        C = low_rank_matrix()
        ZU = np.transpose(self.U) @ B.U
        ZV = self.Vh @ np.transpose(B.Vh)
        YU = B.U - self.U @ ZU
        YV = np.transpose(B.Vh) - np.transpose(self.Vh) @ ZV
        QU, RU = np.linalg.qr(YU)
        QV, RV = np.linalg.qr(YV)
        C.U = np.concatenate((self.U, QU), axis=1)
        C.Vh = np.transpose(np.concatenate((np.transpose(self.Vh), QV), axis=1))
        row1 = np.concatenate(
            (self.eigenvalue_mat + ZU @ B.eigenvalue_mat @ np.transpose(ZV), ZU @ B.eigenvalue_mat @ np.transpose(RV)),
            axis=1)
        row2 = np.concatenate((RU @ B.eigenvalue_mat @ np.transpose(ZV), RU @ B.eigenvalue_mat @ np.transpose(RV)),
                              axis=1)
        C.eigenvalue_mat = np.concatenate((row1, row2), axis=0)
        C.dim = self.dim
        '''        C=low_rank_matrix()
        C.construct(self.matrix_form()+B.matrix_form())'''
        return C

    def Opposite(self):
        C = low_rank_matrix()
        C.eigenvalue_mat = -self.eigenvalue_mat
        C.dim = self.dim
        C.U = self.U
        C.Vh = self.Vh
        return C

    def matrix_form(self):
        return self.U @ self.eigenvalue_mat @ self.Vh

    def product(self, B):
        C = low_rank_matrix()
        C.dim = self.dim
        C.U = self.U
        C.Vh = B.Vh
        C.eigenvalue_mat = self.eigenvalue_mat @ self.Vh @ B.U @ B.eigenvalue_mat
        return C

    def product_vec(self, b):
        return self.U @ (self.eigenvalue_mat @ (self.Vh @ b))

    def split(self):
        k = self.dim >> 1
        B11 = low_rank_matrix()
        B11.eigenvalue_mat = self.eigenvalue_mat
        B11.U = self.U[:k, :]
        B11.Vh = self.Vh[:, :k]
        B11.dim = k
        B12 = low_rank_matrix()
        B12.eigenvalue_mat = self.eigenvalue_mat
        B12.U = self.U[:k, :]
        B12.Vh = self.Vh[:, k:]
        B12.dim = k
        B21 = low_rank_matrix()
        B21.eigenvalue_mat = self.eigenvalue_mat
        B21.U = self.U[k:, :]
        B21.Vh = self.Vh[:, :k]
        B21.dim = k
        B22 = low_rank_matrix()
        B22.eigenvalue_mat = self.eigenvalue_mat
        B22.U = self.U[k:, :]
        B22.Vh = self.Vh[:, k:]
        B22.dim = k
        return B11, B12, B21, B22

    @property
    def transpose(self):
        A = low_rank_matrix()
        A.dim = self.dim
        A.U = np.transpose(self.Vh)
        A.Vh = np.transpose(self.U)
        A.eigenvalue_mat = np.transpose(self.eigenvalue_mat)
        return A

def combine(B11: low_rank_matrix, B12: low_rank_matrix, B21: low_rank_matrix, B22: low_rank_matrix):
    k = 2 * B11.dim
    B = low_rank_matrix()
    B.dim = k
    B.eigenvalue_mat = B11.eigenvalue_mat
    B.U = np.concatenate((B11.U, B21.U), axis=0)
    B.Vh = np.concatenate((B11.Vh, B12.Vh), axis=1)
    return B

class hierarchical_matrix():
    def __init__(self):
        self.lb = None
        self.rt = None
        self.lt = None
        self.rb = None
        self.shape = None
        self.numpy_form = None

    def construct(self, A):
        k = np.shape(A)[0]
        t = k >> 1
        if k == 2:
            self.numpy_form = A
            self.shape = 2
            return 0
        else:
            # left_bottom
            self.lb = low_rank_matrix()
            self.lb.construct(A[t:, :t])
            # right_top
            self.rt = low_rank_matrix()
            self.rt.construct(A[:t, t:])
            # left_top
            self.lt = hierarchical_matrix()
            self.lt.construct(A[:t, :t])
            # right_bottom
            self.rb = hierarchical_matrix()
            self.rb.construct(A[t:, t:])
            self.shape = k
        return 0

    def minus(self, H):
        assert self.shape == H.shape
        D = hierarchical_matrix()
        D.shape = self.shape
        if self.shape == 2:
            D.numpy_form = self.numpy_form - H.numpy_form
        else:
            D.lt = self.lt.minus(H.lt)
            D.lb = self.lb.add(H.lb.opposite())
            D.rt = self.rt.add(H.rt.opposite())
            D.rb = self.rb.minus(H.rb)
        return D

    def minus_low_rank_matrix(self, X: low_rank_matrix):
        B = hierarchical_matrix()
        B.shape = self.shape
        if self.shape == 2:
            B.numpy_form = self.numpy_form - X.matrix_form()

            return B
        x11, x12, x21, x22 = X.split()
        B.lt = self.lt.minus_low_rank_matrix(x11)
        B.lb = self.lb.add(x21.Opposite())
        B.rt = self.rt.add(x12.Opposite())
        B.rb = self.rb.minus_low_rank_matrix(x22)
        return B

    '''only been used when self is low_triangle H-matrix in our problem
    def product_H(self, X: hierarchical_matrix):
        assert self.shape == H.shape
        D = hierarchical_matrix()
        D.shape = self.shape
        if self.shape==2:
            D.numpy_form = self.numpy_form @ X.numpy_form
            return D
        else:
            D.lt = self.lt.product_H(X.lt)
            D.rt = self.lt.product_H(X.rt).'''

    def product_vec(self, b):
        if self.shape == 2:
            return self.numpy_form @ b
        else:
            k = int(self.shape / 2)
            b11 = self.lt.product_vec(b[:k])
            b12 = self.rt.product_vec(b[k:])
            b21 = self.lb.product_vec(b[:k])
            b22 = self.rb.product_vec(b[k:])
            return np.concatenate((b11 + b12, b21 + b22), axis=0)

def H_matrix_solver_Lower_triangle(L: hierarchical_matrix, B: low_rank_matrix):
    if L.shape == 2:
        D = low_rank_matrix()
        D.dim = 2
        D.construct(linalg.solve(L.numpy_form, B.matrix_form()))
        return D
    B11, B12, B21, B22 = B.split()
    x11 = H_matrix_solver_Lower_triangle(L.lt, B11)
    x12 = H_matrix_solver_Lower_triangle(L.lt, B12)
    temp21 = B21.add((L.lb.product(x11)).Opposite())
    x21 = H_matrix_solver_Lower_triangle(L.rb, temp21)
    temp22 = B22.add((L.lb.product(x12)).Opposite())
    x22 = H_matrix_solver_Lower_triangle(L.rb, temp22)
    return combine(x11, x12, x21, x22)

def H_matrix_solver_Upper_triangle(U: hierarchical_matrix, B: low_rank_matrix):
    if U.shape == 2:
        D = low_rank_matrix()
        D.dim = 2
        D.construct(linalg.solve(U.numpy_form, B.matrix_form()))
        return D
    B11, B12, B21, B22 = B.split()
    x11 = H_matrix_solver_Upper_triangle(U.lt, B11)
    x21 = H_matrix_solver_Upper_triangle(U.lt, B21)
    temp21 = B12.add((x11.product(U.rt)).Opposite())
    x12 = H_matrix_solver_Upper_triangle(U.rb, temp21)
    temp22 = B22.add((x21.product(U.rt)).Opposite())
    x22 = H_matrix_solver_Upper_triangle(U.rb, temp22)
    return combine(x11, x12, x21, x22)

def H_matrix_solver_LU_decomposition(A: hierarchical_matrix):
    if A.shape == 2:
        P, L1, U1 = linalg.lu(A.numpy_form)
        D = hierarchical_matrix()
        D.shape = 2
        D.numpy_form = L1
        F = hierarchical_matrix()
        F.shape = 2
        F.numpy_form = U1
        return D, F
    L11, U11 = H_matrix_solver_LU_decomposition(A.lt)
    U12 = H_matrix_solver_Lower_triangle(L11, A.rt)
    L21 = H_matrix_solver_Upper_triangle(U11, A.lb)
    temp22 = L21.product(U12)
    L22, U22 = H_matrix_solver_LU_decomposition(A.rb.minus_low_rank_matrix(temp22))
    L = hierarchical_matrix()
    L.shape = 2 * L11.shape
    L.lt = L11
    L.rb = L22
    L.lb = L21
    L.rt = np.zeros((L11.shape, L11.shape))
    U = hierarchical_matrix()
    U.shape = 2 * U11.shape
    U.lt = U11
    U.rt = U12
    U.rb = U22
    U.lb = np.zeros((U11.shape, U11.shape))
    return L, U

def H_linear_eq_solver_Lower_triangle(A: hierarchical_matrix, g):
    if A.shape == 2:
        return linalg.solve(A.numpy_form, g)
    k = A.shape >> 1
    x1 = H_linear_eq_solver_Lower_triangle(A.lt, g[:k])
    x2 = H_linear_eq_solver_Lower_triangle(A.rb, g[k:] - A.lb.product_vec(x1))
    return np.concatenate((x1, x2), axis=0)

def H_linear_eq_solver_upper_triangle(U: hierarchical_matrix, g):
    if U.shape == 2:
        return linalg.solve(U.numpy_form, g)
    k = U.shape >> 1
    x2 = H_linear_eq_solver_upper_triangle(U.rb, g[k:])
    x1 = H_linear_eq_solver_upper_triangle(U.lt, g[:k] - U.rt.product_vec(x2))
    return np.concatenate((x1, x2), axis=0)


def gauss_seidel(A, b, tolerance=1e-9, max_iterations=10000):
    iter1 = 0
    x = np.zeros(b.shape)
    for k in range(max_iterations):
        iter1 = iter1 + 1
        x_old = x.copy()
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, (i + 1):], x_old[(i + 1):])) / A[i, i]
        LnormInf = max(abs((x - x_old))) / max(abs(x_old))
        if LnormInf < tolerance:
            break
    return x


def LU_decomposition(A):
    n = len(A[0])
    L = np.zeros([n, n])
    U = np.zeros([n, n])
    for i in range(n):
        L[i][i] = 1
        if i == 0:
            U[0][0] = A[0][0]
            for j in range(1, n):
                U[0][j] = A[0][j]
                L[j][0] = A[j][0] / U[0][0]
        else:
            for j in range(i, n):
                temp = 0
                for k in range(0, i):
                    temp = temp + L[i][k] * U[k][j]
                U[i][j] = A[i][j] - temp
            for j in range(i + 1, n):
                temp = 0
                for k in range(0, i):
                    temp = temp + L[j][k] * U[k][i]
                L[j][i] = (A[j][i] - temp) / U[i][i]
    return L, U


def backsub(A, B):
    n = B.size
    X = np.zeros([n, 1])
    X[n - 1] = B[n - 1] / A[n - 1, n - 1]
    for k in range(n - 2, -1, -1):
        X[k] = (B[k] - A[k, k + 1:] @ X[k + 1:]) / A[k, k]
    return X

def forwardsub(A, B):
    n = B.size
    X = np.zeros([n, 1])
    X[0] = B[0]/A[0, 0]
    for k in range(1, n):
        X[k] = (B[k] - A[k, :k] @ X[:k]) / A[k, k]
    return X

if __name__ == "__main__":
    T1 = []
    T2 = []
    E1 = []
    E2 = []
    U1 = []
    U2 = []
    for i in range(2, 6):
        C, g = initial_matrix(2**i)
        start = time.time()
        L, U = LU_decomposition(C)
        temp = forwardsub(L, g)
        U1.append(backsub(U, temp))
        end = time.time()
        T1.append(np.log2(end-start))
        H_C = hierarchical_matrix()
        H_C.construct(C)
        start = time.time()
        L, U = H_matrix_solver_LU_decomposition(H_C)
        U2.append(H_linear_eq_solver_upper_triangle(U, H_linear_eq_solver_Lower_triangle(L, g)))
        end = time.time()
        T2.append(np.log2(end-start))
    plt.figure()
    plt.plot(T1)
    plt.plot(T2)
    plt.legend(["regular method", "h-matrix"])
    plt.title("Time cost")
    plt.savefig("Time cost.png")
    plt.show()
    for i in range(2, 5):
        E1.append(np.linalg.norm(U1[i - 2][::2**(i-2)] - U1[-1][::2 ** 3], ord=np.inf))
        E2.append(np.linalg.norm(U2[i - 2][::2**(i-2)] - U2[-1][::2 ** 3], ord=np.inf))
    plt.figure()
    plt.plot(E1)
    plt.plot(E2)
    plt.legend(["regular method", "h-matrix"])
    plt.title("Error rate")
    plt.savefig("Error rate.png")
    plt.show()
    print(T1,T2)
    print(E1,E2)