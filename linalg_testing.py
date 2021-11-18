import linalg
from linalg import Vector
from linalg import Matrix

v1 = Vector([1, 0, 1, 0, 0])
v2 = Vector([0, 1, 0, 0, 0])
v3 = Vector([0, 0, 1, 1, 0])
v4 = Vector([1, 1, 1, 0, 0])
V = [v1, v2, v3, v4]

print(linalg.gram_schmidt(V))