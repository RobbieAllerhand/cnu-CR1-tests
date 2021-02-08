# Two different example solutions.
import numpy as np

# First: using a loop
def sub_anti_diagonal(A):
    '''
    Returns the sub-anti-diagonal of the matrix A,
    as a NumPy vector.
    '''
    # Find the size of the matrix
    M = A.shape[0]

    # Create a vector of zeros
    diag = np.zeros(M-1)

    # Loop starting from the second row to fill values
    for i in range(1, M):
        # We can use a negative index to start in the last column
        diag[i-1] = A[i, -i]

    return diag


# Second: using Numpy functions
def sub_anti_diagonal(A):
    '''
    Returns the sub-anti-diagonal of the matrix A,
    as a NumPy vector.
    '''
    # Flip A left to right, use np.diag() to obtain the sub-diagonal
    # of the flipped matrix
    return np.diag(np.fliplr(A), k=-1)



# Test cases. We can use functions from np.testing
# Test case 1
A = np.array([[6.0, 3.0, 9.0, 10.0],
              [3.0, -1.0, 6.0, 9.0],
              [-3.0, 6.0, 1.0, -1.0],
              [5.0, 3.0, 4.0, 3.0]])
expected = np.array([9.0, 1.0, 3.0])
result = sub_anti_diagonal(A)

# This does nothing if the result is correct (if numbers match to 6 significant digits),
# but triggers an AssertionError if the result is incorrect.
# (Check the documentation!)
np.testing.assert_allclose(expected, result, rtol=1e-6)


# Test case 2
A = np.array([[1.1, 2.2],
              [3.3, 4.4]])
expected = np.array([4.4])
result = sub_anti_diagonal(A)
np.testing.assert_allclose(expected, result, rtol=1e-6)

# Test case 3
A = np.array([[5.1, 6.2],
              [7.3, 8.4]])
expected = np.array([8.4])
result = sub_anti_diagonal(A)
np.testing.assert_allclose(expected, result, rtol=1e-6)

# Test case 4
A = np.array([[-9.1, 10.2, 11.3],
              [-12.4, 13.5, 14.6],
              [-15.7, 16.8, 17.9]])
expected = np.array([14.6, 16.8])
result = sub_anti_diagonal(A)
np.testing.assert_allclose(expected, result, rtol=1e-6)

# Test case 5: matrix of random size, full of zeros
M = np.random.randint(2, 50)
A = np.zeros([M, M])
expected = np.zeros([M-1])
result = sub_anti_diagonal(A)
np.testing.assert_allclose(expected, result, rtol=1e-6)

# Test case 6: matrix of random size, full of ones
M = np.random.randint(7, 80)
A = np.ones([M, M])
expected = np.ones([M-1])
result = sub_anti_diagonal(A)
np.testing.assert_allclose(expected, result, rtol=1e-6)

# Test case 7
A = np.zeros([6, 6])

A[0, :] = np.arange(1, A.shape[1] + 1)
A[1, :] = np.arange(1, A.shape[1] + 1)
for i in range(2, A.shape[0]):
    A[i, :] = (A[i - 1, :] + A[i - 2, :]) / i

expected = np.array([6.0, 5.0, 2.666667, 1.25, 0.433333])
result = sub_anti_diagonal(A)
np.testing.assert_allclose(expected, result, rtol=1e-6)
