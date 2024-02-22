import numpy as np

def psphere(x):
    """
    Normalization of projective points.

    Args:
    x (numpy.ndarray): Matrix in which each column is a point.

    Returns:
    y (numpy.ndarray): Result after normalization.
    alpha (numpy.ndarray): Depth.
    """
    a, n = [len(x), len(x[0])]
    alpha = np.sqrt(np.sum(x ** 2, axis=0))
    y = x / (np.ones((a, 1)) * alpha)

    return y, alpha

# Example usage:
# x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Example input matrix
# y, alpha = psphere(x)
# print("Normalized matrix y:\n", y)
# print("Depth alpha:", alpha)
