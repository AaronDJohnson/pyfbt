import numpy as np
from mpmath import *

# TODO: Change the variable names to be more descriptive.
# TODO: use eltype in cont_frac to get rid of complex function.
# TODO: contfracK needs to be more descriptive
# TODO: use eltype in confracK
mp.dps = 100


def complex_cont_frac(a, b, tol=1e-20):
    n = len(a)
    A = np.zeros(n + 1) * mpc("1")
    B = np.zeros(n + 1) * mpc("1")
    x = np.zeros(n + 1) * mpc("1")
    A[0] = mpc("1")
    A[1] = b[0]
    B[0] = mpc("0")
    B[1] = mpc("1")
    x[1] = A[1] / B[1]
    # print(A[1])
    A[2] = b[1] * A[1] + a[1] * A[0]
    B[2] = b[1] * B[1] + a[1] * B[0]
    x[2] = A[2] / B[2]
    # print(A[2])
    A[3] = b[2] * A[2] + a[2] * A[1]
    B[3] = b[2] * B[2] + a[2] * B[1]
    x[3] = A[3] / B[3]
    # print(A[3])

    for i in range(3, n - 1):
        A[i + 1] = b[i] * A[i] + a[i] * A[i - 1]
        # print("A = ", A[i + 1])
        B[i + 1] = b[i] * B[i] + a[i] * B[i - 1]
        # print(B[i + 1])
        x[i + 1] = A[i + 1] / B[i + 1]
        # print(i)
        # print(x[i + 1])
        # print(x[i])
        re_err = abs(re(x[i + 1]) - re(x[i]))
        im_err = abs(im(x[i + 1]) - im(x[i]))
        # print(tol)
        if re_err < tol and im_err < tol:
            return x[i + 1]
    print('cont_frac failed to converge')
    print('re_err=', re_err)
    print('im_err=', im_err)
    return x[i + 1]

