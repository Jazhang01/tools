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

    def concat(self, v):
        """ Returns the vector concatenated with Vector 'v' """
        return Vector(self.v + v.v)

    @staticmethod
    def _func_vectors(vectors, f):
        """ Returns an n argument function 'f' applied over the corresponding elements in n Vectors """
        return Vector([f(*args) for args in zip(*vectors)])


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

    def concat(self, m, dim=0):
        """ Returns the matrix concatenated with Matrix 'm' """ 
        assert dim == 0 or dim == 1
        if dim == 0:
            return Matrix(self.rows + m.rows)
        return Matrix([r.concat(mr) for r, mr in zip(self.rows, m.rows)]) 

    @staticmethod
    def identity(dim):
        return Matrix([[int(i==j) for i in range(dim)] for j in range(dim)])


def gaussian_elimination(A, B):
    """ Matrix 'A' must be square. 'B' is a list of Vectors 'b'. 

    Returns a list of Vectors, 'X', that corresponds to the solutions to Ax = b
    Returns -1 if A is over or underdetermined """

    N = A.dim[0]

    assert A.dim[0] == A.dim[1]  # check square
    for b in B: assert b.dim == N  # check dimensions

    # augmented matrix
    B = Matrix(B).transpose()
    M = A.concat(B, dim=1)

    # upper triangle
    for j in range(N):
        # find pivot row
        i = j
        while M[i][j] == 0:
            i += 1
            # zero column --> no solution
            if i >= N:
                return -1
 
        # put pivot row in place
        M.swap_rows(i, j)

        # set pivot element to 1
        M.scale_row(j, 1/M[j][j])

        # zero elements below pivot
        for i in range(j+1, N):
            M.add_row_to(j, i, scale=-M[i][j])

    # create identity
    for j in range(N-1, -1, -1):
        for i in range(j-1, -1, -1):
            M.add_row_to(j, i, scale=-M[i][j])

    return M.transpose()[N:]


def inverse(A):
    """ Matrix A must be square. Returns the inverse of A if it exists, otherwise -1 """ 
    assert A.dim[0] == A.dim[1]
    return Matrix(gaussian_elimination(A, Matrix.identity(A.dim[0]))).transpose()


def one_dim_projection(w, v):
    """ Returns the projection of Vector v onto the subspace spanned by Vector w """
    return (v.dot(w) / w.norm()) * w


def projection(A, v):
    """ Returns the projection of Vector v onto the subspace spanned by the columns of matrix A """
    AT = A.transpose()
    ATA = AT.dot(A)
    return A.dot(inverse(ATA).dot(AT)).dot(v)



    
