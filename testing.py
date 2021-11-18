import linalg
from linalg import Vector
from linalg import Matrix

M = Matrix([[1, 0, 0],
            [0, 1, 0]])
M2 = Matrix([[0, 4, 1],
             [1, 0, 3]])
v = Vector([1, 1])

print(linalg.pseudoinverse(M))
print(M2.dot(linalg.pseudoinverse(M2)))