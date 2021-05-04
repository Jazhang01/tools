import linalg

def linear_regression(x, y, n=1):
    """ Fits a 'n' degree polynomial of the form a + bx + ... + yx^(n-1) + zx^n to the data (x, y) and returns the coefficients (a, b, ..., z) """

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

    y = linalg.Vector(y)
    
    return tuple(linalg.inverse(MT.dot(M)).dot(MT).dot(y))

