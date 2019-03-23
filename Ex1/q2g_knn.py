import numpy as np
from q2e_word2vec import normalizeRows


def knn(vector, matrix, k=10):
    """
    Finds the k-nearest rows in the matrix with comparison to the vector.
    Use the cosine cos_similarity as a distance metric.

    Arguments:
    vector -- A D dimensional vector
    matrix -- V x D dimensional numpy matrix.

    Return:
    nearest_idx -- A numpy vector consists of the rows indices of the k-nearest neighbors in the matrix
    """

    nearest_idx = []

    m_dot_v = matrix.dot(vector)
    row_magnitude = np.sqrt(np.sum(matrix ** 2, axis=1))  # in case vectors are not normalized
    v_magnitude = np.sqrt(np.sum(vector ** 2))  # in case vectors are not normalized

    cos_similarity = m_dot_v / (row_magnitude * v_magnitude)

    index_magnitude_tuples = [(index, magnitude) for index, magnitude in enumerate(cos_similarity)]
    index_magnitude_tuples.sort(key=lambda tup: tup[1], reverse=True)

    nearest_idx = [tup[0] for tup in index_magnitude_tuples[:k]]

    return nearest_idx

def test_knn():
    """
    Use this space to test your knn implementation by running:
        python knn.py
    This function will not be called by the autograder, nor will
        your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE

    indices = knn(np.array([0.2,0.5]), np.array([[0,0.5],[0.1,0.1],[0,0.5],[2,2],[4,4],[3,3]]), k=2)
    assert 0 in indices and 2 in indices and len(indices) == 2

    ### END YOUR CODE

if __name__ == "__main__":
    test_knn()


