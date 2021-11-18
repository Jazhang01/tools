import linalg
from linalg import Vector
from linalg import Matrix

M = Matrix([[-1, 2, 100, 123],
            [2, -83, 9, 38],
            [213, 2, -12, 9],
            [2, 11, 213, 3]])

print(linalg.determinant(M))
print(linalg.row_reduce(M))
