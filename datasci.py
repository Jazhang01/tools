import linalg

def linear_regression(x, y, degree=1):
    """ Fits a polynomial of the form a + bx + ... + yx^(n-1) + zx^n to the data (x, y) and returns the coefficients (a, b, ..., z) """

    assert len(x) == len(y)

    M = []
    for xi in x:
        pow_x = 1
        row = [pow_x]
        for i in range(n):
            pow_x *= xi
            row.append(pow_x)
        M.append(row)

    M = linalg.Matrix(M)
    MT = M.transpose()

    return tuple(inverse(MT.dot(M)).dot(MT).dot(y))

