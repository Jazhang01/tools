from operator import add, sub, mul


class Vector:
    def __init__(self, elements):
        self.v = tuple(e for e in elements)
        self.dim = len(self.v)

    def __str__(self):
        return str(self.v)

    def __repr__(self):
        return str(self.v)

    def __len__(self):
        return self.dim

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self.v)))
            return Vector([self.v[i] for i in indices])
        return self.v[key]

    def __iter__(self):
        yield from self.v

    def __add__(self, w):
        assert self.dim == w.dim
        return Vector._func_vectors((self.v, w), add)

    def __sub__(self, w):
        assert self.dim == w.dim
        return Vector._func_vectors((self.v, w), sub)

    def __mul__(self, a):
        assert type(a) == float or type(a) == int
        return Vector([a*e for e in self.v])

    def __rmul__(self, a):
        return self*a

    def __truediv__(self, a):
        return self*(1/a)

    def dot(self, w):
        assert self.dim == w.dim
        return sum(Vector._func_vectors((self.v, w), mul))

    def norm(self):
        """ Returns the norm of the vector """
        return self.dot(self)**(1/2)

    def copy(self):
        return Vector(self.v)

    @staticmethod
    def _func_vectors(vectors, f):
        """ Returns an n argument function 'f' applied over the corresponding elements in n Vectors """
        return Vector([f(*args) for args in zip(*vectors)])
    
    @staticmethod
    def concat(v, w):
        """ Returns Vector 'v' concatenated with Vector 'w' """
        return Vector(v.v + w.v)

    @staticmethod
    def zero(n):
        """ Returns a zero Vector of size 'n' """
        return Vector([0]*n)

class Matrix:
    def __init__(self, rows):
        self.rows = [Vector(r) for r in rows]
        self.dim = (len(self.rows), len(self.rows[0]))

    def __str__(self):
        return '\n'.join([str(r) for r in self.rows])

    def __repr__(self):
        return str(self.rows)

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self.rows)))
            return [self.rows[i] for i in indices]
        return self.rows[key]

    def _columns(self):
        """ Returns the columns of the matrix as a list of Vectors"""
        return [Vector(self.rows[i][j] for i in range(self.dim[0])) for j in range(self.dim[1])]

    def transpose(self):
        """ Returns the transpose of the matrix """
        return Matrix(self._columns())

    def swap_rows(self, r1, r2):
        """ Swaps row 'r1' and row 'r2' of the matrix """
        self.rows[r1], self.rows[r2] = self.rows[r2], self.rows[r1]

    def scale_row(self, r, a):
        """ Scales row 'r' by 'a' """
        self.rows[r] = a*self.rows[r]

    def add_row_to(self, r1, r2, scale=1):
        """ Adds row 'r1' scaled by 'scale' to row 'r2' """
        self.rows[r2] = scale*self.rows[r1] + self.rows[r2]

    def mult_vector(self, v):
        return Vector([r.dot(v) for r in self.rows])

    def mult_matrix(self, m):
        return Matrix([[r.dot(c) for c in m._columns()] for r in self.rows])

    def dot(self, x):
        if isinstance(x, Vector):
            return self.mult_vector(x)
        if isinstance(x, Matrix):
            return self.mult_matrix(x)
        raise TypeError('x needs to be a Vector or a Matrix')

    def copy(self):
        return Matrix(self.rows)

    def square(self):
        """ Returns whether the matrix is square """
        return self.dim[0] == self.dim[1]

    def trace(self):
        """ Returns trace of matrix, if square """
        assert self.square()
        return sum(self.rows[i][i] for i in range(self.dim[0]))

    @staticmethod
    def identity(dim):
        return Matrix([[int(i==j) for i in range(dim)] for j in range(dim)])

    @staticmethod
    def concat(m1, m2, dim=0):
        assert dim == 0 or dim == 1
        if dim == 0:
            return Matrix(m1.rows + m2.rows)
        return Matrix([Vector.concat(r1, r2) for r1, r2 in zip(m1.rows, m2.rows)])


def row_reduce(A, full=True, get_rank_nullity=False, get_determinant=False):
    """ If 'full' is True, returns the reduced row echelon form of Matrix 'A'.
    Otherwise, returns the row echelon form of Matrix 'A' 

    If 'get_rank_nullity' is True, then it only returns the rank and nullity as (rank, nullity)
    If 'get_determinant' is True, then it only returns the determinant """

    A = A.copy()
    N, M = A.dim

    rank, nullity = 0, 0
    determinant = 1

    # keep track of (row, column) of pivots
    pivots = []

    # upper triangle / row echelon form
    for j in range(min(A.dim)):
        # pivot row
        p = j - nullity
        
        # find pivot
        i = p
        while i < N and A[i][j] == 0:
            i += 1
        
        # no pivot
        if i >= N:
            nullity += 1
            continue
        
        rank += 1
        pivots.append((p, j))

        if i != p:
            A.swap_rows(i, p)  # put pivot in place
            determinant *= -1

        determinant *= A[p][j]
        A.scale_row(p, 1/A[p][j])  # set pivot to 1
        
        # zero elements below pivot
        for i in range(p+1, N):
            A.add_row_to(p, i, scale=-A[i][j])

    if get_rank_nullity:
        return (rank, nullity)

    if get_determinant:
        return determinant

    if not full:
        return A

    # fully reduce
    for p, j in pivots:
        for i in range(p-1, -1, -1):
            A.add_row_to(p, i, scale=-A[i][j])

    return A


def rank(A):
    """ Returns the rank of Matrix A i.e. the dimension of the columnspace of A """
    return row_reduce(A, get_rank_nullity=True)[0]


def nullity(A):
    """ Returns the nullity of Matrix A i.e. the dimension of the nullspace of A """
    return row_reduce(A, get_rank_nullity=True)[1]


def determinant(A):
    """ Returns the determinant of Matrix A, if A is square """
    assert A.square()
    return row_reduce(A, get_determinant=True)


def gaussian_elimination(A, B):
    """ Matrix 'A' must be square. 'B' is a list of Vectors 'b'. 

    Returns a list of Vectors, 'X', that corresponds to the solutions to Ax = b
    Returns -1 if A is over or underdetermined 
    
    >>> A = Matrix([[1,-1,1],[2,3,-1],[3,-2,-9]])
    >>> B = [Vector([8, -2, 9])]
    >>> gaussian_elimination(A, B)
    [(4.0, -3.0, 1.0)]
    >>> A = Matrix([[1,2,3],[2,4,6],[1,2,1]])
    >>> B = [Vector([6, 12, 4])]
    >>> gaussian_elimination(A, B)
    -1
    """

    N = A.dim[0]

    assert A.square()
    for b in B: assert b.dim == N  # check dimensions

    # augmented matrix
    B = Matrix(B).transpose()
    M = Matrix.concat(A, B, dim=1)
    
    # reduced row echelon form
    M = row_reduce(M, full=True)

    # check last row's pivot to see if over or underdetermined
    if M[N-1][N-1] != 1:
        return -1

    return M.transpose()[N:]


def inverse(A):
    """ Matrix A must be square. Returns the inverse of A if it exists, otherwise -1 """ 
    assert A.square()
    GE = gaussian_elimination(A, Matrix.identity(A.dim[0]))
    if GE == -1:
        return -1
    return Matrix(GE).transpose()


def pseudoinverse(A):
    """ Matrix A must be full rank """
    AT = A.transpose()
    if A.dim[0] >= A.dim[1]:
        ATA = AT.dot(A)
        return inverse(ATA).dot(AT)
    else:
        AAT = A.dot(AT)
        return AT.dot(inverse(AAT))


def one_dim_projection(v, w):
    """ Returns the projection of Vector v onto the subspace spanned by Vector w """
    return (v.dot(w) / w.norm()) * w


def projection(A, v):
    """ Returns the projection of Vector v onto the subspace spanned by the columns of matrix A """
    AT = A.transpose()
    ATA = AT.dot(A)
    return A.dot(inverse(ATA).dot(AT)).dot(v)


def gram_schmidt(V):
    """ Applies the gram-schmidt algorithm to the set of vectors, V.
    Returns a list of orthonormal vectors spanning the span of V """
    U = []
    for v in V:
        assert isinstance(v, Vector)
        for u in U:
            v -= one_dim_projection(v, u)
        if v.norm() != 0:
            U.append(v / v.norm())
    return U
